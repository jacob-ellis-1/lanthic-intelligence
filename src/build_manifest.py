#!/usr/bin/env python3
"""
Manifest compiler for Lanthic Intelligence.

This module owns:
- scanning folders for supported local source files
- reading curated source/URL lists
- registering discovered sources in SourceRegistry
- applying dedupe and acquisition-status filters
- writing manifest.jsonl
- writing manifest_summary.json

It must not:
- fetch URLs
- parse documents
- call OpenAI
- call Neo4j
- run ingestion/extraction/PostRAG/KG-IRAG

Run:
    python src/build_manifest.py --self-test
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pipeline_contracts import (
    ACQUISITION_STATUSES,
    ContractError,
    ManifestRow,
    SourceRecord,
    canonicalize_source,
    is_url,
    read_json,
    validate_tags,
    write_json,
    write_manifest_jsonl,
)

from source_registry import (
    BLOCKING_ACQUISITION_STATUSES,
    SourceRegistry,
)


JSONDict = Dict[str, Any]


SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".xlsx",
    ".xls",
}

DEFAULT_INGESTIBLE_ACQUISITION_STATUSES: Set[str] = {
    "manual",
    "acquired",
    "cached",
    "not_acquired",
}


# ============================================================
# Options / result records
# ============================================================

@dataclass
class BuildManifestOptions:
    sources_dirs: List[Path] = field(default_factory=list)
    sources_files: List[Path] = field(default_factory=list)
    registry_path: Path = Path("data/source_registry.json")
    output_path: Path = Path("manifest.jsonl")
    summary_output_path: Optional[Path] = None

    corpus_id: str = "default_corpus"
    branch_id: str = "staging"
    tags: List[str] = field(default_factory=list)
    filter_tags: List[str] = field(default_factory=list)

    include_ext: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    recursive: bool = True

    from_registry: bool = False
    dedupe: bool = True
    include_duplicates: bool = False
    include_blocked: bool = False
    include_failed_acquisition: bool = False
    include_failed_pipeline: bool = False
    limit: Optional[int] = None

    metadata: JSONDict = field(default_factory=dict)


@dataclass
class DiscoveredSource:
    source: str
    origin: str
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass
class RegistrationEvent:
    source: str
    source_id: str
    status: str
    origin: str
    duplicate_of: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ============================================================
# Small helpers
# ============================================================

def normalize_extension(ext: str) -> str:
    ext = str(ext or "").strip().lower()

    if not ext:
        raise ContractError("Empty extension supplied.")

    if not ext.startswith("."):
        ext = "." + ext

    return ext


def effective_extensions(include_ext: Sequence[str]) -> Set[str]:
    if not include_ext:
        return set(SUPPORTED_EXTENSIONS)

    return {normalize_extension(ext) for ext in include_ext}


def is_sidecar_metadata(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".metadata.json") or name.endswith(".meta.json")


def is_supported_source_file(path: Path, include_ext: Set[str]) -> bool:
    if not path.is_file():
        return False

    if is_sidecar_metadata(path):
        return False

    return path.suffix.lower() in include_ext


def matches_exclude(path: Path, patterns: Sequence[str]) -> bool:
    text = str(path)

    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(text, pattern):
            return True

    return False


def read_sources_file(path: Path) -> List[str]:
    sources: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        sources.append(line)

    return sources


def resolve_source_from_file_line(source: str, *, sources_file: Path) -> str:
    source = str(source).strip()

    if is_url(source):
        return source

    path = Path(source).expanduser()

    if path.is_absolute():
        return str(path)

    candidate = sources_file.parent / path

    if candidate.exists():
        return str(candidate)

    return source


def sidecar_metadata_paths(path: Path) -> List[Path]:
    """
    Supported sidecar names:
    - report.pdf.metadata.json
    - report.metadata.json
    - report.pdf.meta.json
    - report.meta.json
    """
    return [
        Path(str(path) + ".metadata.json"),
        path.with_suffix(".metadata.json"),
        Path(str(path) + ".meta.json"),
        path.with_suffix(".meta.json"),
    ]


def read_sidecar_metadata(path: Path) -> JSONDict:
    for sidecar in sidecar_metadata_paths(path):
        if sidecar.exists() and sidecar.is_file():
            data = read_json(sidecar)

            if not isinstance(data, Mapping):
                raise ContractError(f"Sidecar metadata must be a JSON object: {sidecar}")

            return dict(data)

    return {}


def merge_metadata(*parts: Optional[Mapping[str, Any]]) -> JSONDict:
    out: JSONDict = {}

    for part in parts:
        if not part:
            continue

        for key, value in dict(part).items():
            if value not in (None, ""):
                out[key] = value

    return out


def acquisition_status_filter(
    *,
    include_blocked: bool,
    include_failed_acquisition: bool,
) -> Set[str]:
    statuses = set(DEFAULT_INGESTIBLE_ACQUISITION_STATUSES)

    if include_blocked:
        statuses.update({"blocked", "paywalled"})

    if include_failed_acquisition:
        statuses.update(set(ACQUISITION_STATUSES))

    return statuses


# ============================================================
# Discovery
# ============================================================

def discover_from_dir(
    directory: Path,
    *,
    include_ext: Set[str],
    exclude: Sequence[str],
    recursive: bool,
    base_metadata: Optional[Mapping[str, Any]] = None,
) -> List[DiscoveredSource]:
    if not directory.exists() or not directory.is_dir():
        raise ContractError(f"sources-dir does not exist or is not a directory: {directory}")

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    discovered: List[DiscoveredSource] = []

    for path in sorted(iterator):
        if matches_exclude(path, exclude):
            continue

        if not is_supported_source_file(path, include_ext):
            continue

        metadata = merge_metadata(base_metadata, read_sidecar_metadata(path))

        discovered.append(
            DiscoveredSource(
                source=str(path),
                origin=f"dir:{directory}",
                metadata=metadata,
            )
        )

    return discovered


def discover_from_sources_file(
    path: Path,
    *,
    base_metadata: Optional[Mapping[str, Any]] = None,
) -> List[DiscoveredSource]:
    if not path.exists() or not path.is_file():
        raise ContractError(f"sources-file does not exist or is not a file: {path}")

    discovered: List[DiscoveredSource] = []

    for source in read_sources_file(path):
        resolved = resolve_source_from_file_line(source, sources_file=path)

        metadata = dict(base_metadata or {})

        if not is_url(resolved):
            p = Path(resolved)
            if p.exists() and p.is_file():
                metadata = merge_metadata(metadata, read_sidecar_metadata(p))

        discovered.append(
            DiscoveredSource(
                source=resolved,
                origin=f"sources_file:{path}",
                metadata=metadata,
            )
        )

    return discovered


def discover_sources(options: BuildManifestOptions) -> List[DiscoveredSource]:
    discovered: List[DiscoveredSource] = []
    include_ext = effective_extensions(options.include_ext)

    for directory in options.sources_dirs:
        discovered.extend(
            discover_from_dir(
                directory,
                include_ext=include_ext,
                exclude=options.exclude,
                recursive=options.recursive,
                base_metadata=options.metadata,
            )
        )

    for sources_file in options.sources_files:
        discovered.extend(
            discover_from_sources_file(
                sources_file,
                base_metadata=options.metadata,
            )
        )

    return discovered


# ============================================================
# Registration
# ============================================================

def acquisition_status_for_discovered(source: str) -> str:
    if is_url(source):
        return "not_acquired"

    path = Path(source).expanduser()

    if path.exists() and path.is_file():
        return "manual"

    return "not_found"


def local_path_for_discovered(source: str) -> Optional[str]:
    if is_url(source):
        return None

    path = Path(source).expanduser()

    if path.exists() and path.is_file():
        return str(path.resolve())

    return None


def register_discovered_sources(
    registry: SourceRegistry,
    discovered: Sequence[DiscoveredSource],
    *,
    corpus_id: str,
    branch_id: str,
    tags: Sequence[str],
    dedupe: bool,
) -> List[RegistrationEvent]:
    events: List[RegistrationEvent] = []
    clean_tags = validate_tags(tags)

    for item in discovered:
        try:
            acq_status = acquisition_status_for_discovered(item.source)
            local_path = local_path_for_discovered(item.source)

            record = registry.register_source(
                item.source,
                corpus_id=corpus_id,
                branch_id=branch_id,
                tags=clean_tags,
                acquisition_status=acq_status,
                acquisition_method="local_file" if acq_status == "manual" else None,
                local_path=local_path,
                metadata=item.metadata,
                save=False,
            )

            duplicate_of: Optional[str] = None

            if dedupe:
                duplicate_of = registry.find_duplicate(record)

                if duplicate_of:
                    registry.mark_duplicate(record.source_id, duplicate_of, save=False)

            events.append(
                RegistrationEvent(
                    source=item.source,
                    source_id=record.source_id,
                    status="registered_duplicate" if duplicate_of else "registered",
                    origin=item.origin,
                    duplicate_of=duplicate_of,
                )
            )

        except Exception as exc:
            # Keep compiling the rest of the manifest.
            try:
                canonical = canonicalize_source(item.source)
                source_id = SourceRecord.from_source(
                    canonical,
                    corpus_id=corpus_id,
                    branch_id=branch_id,
                    tags=clean_tags,
                ).source_id
            except Exception:
                source_id = "unknown"

            events.append(
                RegistrationEvent(
                    source=item.source,
                    source_id=source_id,
                    status="failed",
                    origin=item.origin,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    registry.save()
    return events


# ============================================================
# Manifest export
# ============================================================

def export_manifest_rows(
    registry: SourceRegistry,
    *,
    corpus_id: str,
    branch_id: str,
    filter_tags: Sequence[str],
    include_duplicates: bool,
    include_blocked: bool,
    include_failed_acquisition: bool,
    include_failed_pipeline: bool,
    limit: Optional[int],
) -> List[ManifestRow]:
    statuses = acquisition_status_filter(
        include_blocked=include_blocked,
        include_failed_acquisition=include_failed_acquisition,
    )

    rows = registry.to_manifest_rows(
        corpus_id=corpus_id,
        branch_id=branch_id,
        tags=filter_tags or None,
        acquisition_status=sorted(statuses),
        include_blocked=include_blocked or include_failed_acquisition,
        include_failed=include_failed_pipeline,
        include_duplicates=include_duplicates,
        limit=limit,
    )

    rows.sort(key=lambda row: (row.source_id, row.source))
    return rows


# ============================================================
# Summary
# ============================================================

def build_summary(
    *,
    options: BuildManifestOptions,
    discovered: Sequence[DiscoveredSource],
    registration_events: Sequence[RegistrationEvent],
    rows: Sequence[ManifestRow],
    registry: SourceRegistry,
) -> JSONDict:
    discovered_by_origin = Counter(item.origin.split(":", 1)[0] for item in discovered)
    discovered_by_ext = Counter()

    for item in discovered:
        if is_url(item.source):
            discovered_by_ext["url"] += 1
        else:
            discovered_by_ext[Path(item.source).suffix.lower() or "none"] += 1

    registration_statuses = Counter(event.status for event in registration_events)

    row_kinds = Counter(row.source_kind for row in rows)
    row_acq = Counter(row.acquisition_status for row in rows)

    registered_source_ids = {event.source_id for event in registration_events if event.source_id != "unknown"}
    manifest_source_ids = {row.source_id for row in rows}

    excluded_registered = registered_source_ids - manifest_source_ids

    excluded_duplicates = 0
    excluded_blocked_or_failed_acq = 0
    excluded_failed_pipeline = 0

    for source_id in excluded_registered:
        if not registry.has_source(source_id):
            continue

        raw = registry.raw_record(source_id)

        if raw.get("duplicate_of"):
            excluded_duplicates += 1

        if raw.get("acquisition_status") in BLOCKING_ACQUISITION_STATUSES:
            excluded_blocked_or_failed_acq += 1

        if raw.get("last_status") == "failed":
            excluded_failed_pipeline += 1

    return {
        "schema_version": 1,
        "manifest_path": str(options.output_path),
        "summary_path": str(options.summary_output_path) if options.summary_output_path else None,
        "registry_path": str(options.registry_path),
        "corpus_id": options.corpus_id,
        "branch_id": options.branch_id,
        "tags_added": list(options.tags),
        "filter_tags": list(options.filter_tags),
        "from_registry": options.from_registry,
        "recursive": options.recursive,
        "include_duplicates": options.include_duplicates,
        "include_blocked": options.include_blocked,
        "include_failed_acquisition": options.include_failed_acquisition,
        "include_failed_pipeline": options.include_failed_pipeline,
        "sources_discovered": len(discovered),
        "sources_discovered_by_origin": dict(discovered_by_origin),
        "sources_discovered_by_ext": dict(discovered_by_ext),
        "sources_registered": sum(1 for e in registration_events if e.status.startswith("registered")),
        "registration_events_by_status": dict(registration_statuses),
        "sources_written_to_manifest": len(rows),
        "manifest_rows_by_kind": dict(row_kinds),
        "manifest_rows_by_acquisition_status": dict(row_acq),
        "sources_excluded_registered": len(excluded_registered),
        "sources_excluded_duplicate": excluded_duplicates,
        "sources_excluded_blocked_or_failed_acquisition": excluded_blocked_or_failed_acq,
        "sources_excluded_failed_pipeline": excluded_failed_pipeline,
        "registry_summary": registry.summary(),
        "registration_failures": [
            event.to_dict()
            for event in registration_events
            if event.status == "failed"
        ],
    }


# ============================================================
# Main compile function
# ============================================================

def build_manifest(options: BuildManifestOptions) -> JSONDict:
    registry = SourceRegistry(options.registry_path)

    discovered: List[DiscoveredSource] = []
    registration_events: List[RegistrationEvent] = []

    if not options.from_registry:
        discovered = discover_sources(options)
        registration_events = register_discovered_sources(
            registry,
            discovered,
            corpus_id=options.corpus_id,
            branch_id=options.branch_id,
            tags=options.tags,
            dedupe=options.dedupe,
        )

    rows = export_manifest_rows(
        registry,
        corpus_id=options.corpus_id,
        branch_id=options.branch_id,
        filter_tags=options.filter_tags,
        include_duplicates=options.include_duplicates,
        include_blocked=options.include_blocked,
        include_failed_acquisition=options.include_failed_acquisition,
        include_failed_pipeline=options.include_failed_pipeline,
        limit=options.limit,
    )

    write_manifest_jsonl(options.output_path, rows)

    summary = build_summary(
        options=options,
        discovered=discovered,
        registration_events=registration_events,
        rows=rows,
        registry=registry,
    )

    if options.summary_output_path:
        write_json(options.summary_output_path, summary)

    return summary


# ============================================================
# Self-test
# ============================================================

def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> int:
    print("[build_manifest self-test] starting")

    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)

        sources_dir = tmp / "sources"
        sources_dir.mkdir()

        pdf_a = sources_dir / "a_report.pdf"
        pdf_b_dup = sources_dir / "b_duplicate.pdf"
        html = sources_dir / "saved_reuters.html"
        txt = sources_dir / "notes.txt"
        unsupported = sources_dir / "ignore.exe"

        pdf_a.write_text("same pdf content", encoding="utf-8")
        pdf_b_dup.write_text("same pdf content", encoding="utf-8")
        html.write_text("<html><title>Saved Reuters</title></html>", encoding="utf-8")
        txt.write_text("notes", encoding="utf-8")
        unsupported.write_text("unsupported", encoding="utf-8")

        # Sidecar metadata should be read and preserved.
        write_json(
            Path(str(html) + ".metadata.json"),
            {
                "title": "Saved Reuters Article",
                "publisher": "Reuters",
                "original_url": "https://www.reuters.com/world/asia/myanmar-rare-earths/",
                "published_at": "2025-01-02",
            },
        )

        registry_path = tmp / "registry.json"
        manifest_path = tmp / "manifest.jsonl"
        summary_path = tmp / "manifest_summary.json"

        # 1-6. Scan folder, ignore unsupported, register, dedupe, exclude duplicates.
        summary = build_manifest(
            BuildManifestOptions(
                sources_dirs=[sources_dir],
                registry_path=registry_path,
                output_path=manifest_path,
                summary_output_path=summary_path,
                corpus_id="eval1",
                branch_id="staging_eval1",
                tags=["eval1", "myanmar"],
                recursive=True,
                dedupe=True,
            )
        )

        _assert(manifest_path.exists(), "manifest.jsonl not written")
        _assert(summary_path.exists(), "manifest_summary.json not written")
        _assert(registry_path.exists(), "registry not written")
        _assert(summary["sources_discovered"] == 4, "unsupported files should be ignored")
        _assert(summary["sources_written_to_manifest"] == 3, "duplicate should be excluded by default")
        _assert(summary["sources_excluded_duplicate"] == 1, "duplicate exclusion not counted")

        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        _assert(len(rows) == 3, "wrong manifest row count")
        _assert(all(row["corpus_id"] == "eval1" for row in rows), "corpus_id not preserved")
        _assert(all(row["branch_id"] == "staging_eval1" for row in rows), "branch_id not preserved")
        _assert(all("eval1" in row["tags"] for row in rows), "tag not preserved")

        html_rows = [row for row in rows if row["source"].endswith("saved_reuters.html")]
        _assert(len(html_rows) == 1, "saved HTML row missing")
        _assert(html_rows[0]["metadata"]["title"] == "Saved Reuters Article", "sidecar title not preserved")
        _assert(html_rows[0]["metadata"]["publisher"] == "Reuters", "sidecar publisher not preserved")
        _assert(
            html_rows[0]["metadata"]["original_url"] == "https://www.reuters.com/world/asia/myanmar-rare-earths/",
            "sidecar original_url not preserved",
        )

        # 7. Include duplicates when requested.
        manifest_with_dups = tmp / "manifest_with_dups.jsonl"
        summary_with_dups = build_manifest(
            BuildManifestOptions(
                from_registry=True,
                registry_path=registry_path,
                output_path=manifest_with_dups,
                corpus_id="eval1",
                branch_id="staging_eval1",
                include_duplicates=True,
            )
        )

        rows_with_dups = [
            json.loads(line)
            for line in manifest_with_dups.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        _assert(summary_with_dups["sources_written_to_manifest"] == 4, "include_duplicates failed")
        _assert(len(rows_with_dups) == 4, "duplicate row not written when requested")

        # 8-9. Blocked/paywalled excluded by default, included when requested.
        registry = SourceRegistry(registry_path)
        blocked = registry.register_source(
            "https://www.reuters.com/world/blocked/",
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["eval1", "blocked"],
            acquisition_status="blocked",
            acquisition_method="requests",
            metadata={"title": "Blocked Reuters Article", "publisher": "Reuters"},
        )
        registry.update_acquisition(
            blocked.source_id,
            acquisition_status="blocked",
            acquisition_method="requests",
            error="HTTP 403",
        )

        manifest_no_blocked = tmp / "manifest_no_blocked.jsonl"
        summary_no_blocked = build_manifest(
            BuildManifestOptions(
                from_registry=True,
                registry_path=registry_path,
                output_path=manifest_no_blocked,
                corpus_id="eval1",
                branch_id="staging_eval1",
                include_duplicates=True,
                include_blocked=False,
            )
        )

        no_blocked_rows = [
            json.loads(line)
            for line in manifest_no_blocked.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        _assert(
            all(row["acquisition_status"] != "blocked" for row in no_blocked_rows),
            "blocked source leaked into default manifest",
        )

        manifest_blocked = tmp / "manifest_blocked.jsonl"
        summary_blocked = build_manifest(
            BuildManifestOptions(
                from_registry=True,
                registry_path=registry_path,
                output_path=manifest_blocked,
                corpus_id="eval1",
                branch_id="staging_eval1",
                include_duplicates=True,
                include_blocked=True,
            )
        )

        blocked_rows = [
            json.loads(line)
            for line in manifest_blocked.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        _assert(
            any(row["acquisition_status"] == "blocked" for row in blocked_rows),
            "include_blocked did not include blocked source",
        )

        # 10-12. Read sources-file with comments/blank lines and write valid ManifestRows.
        extra_file = tmp / "extra.md"
        extra_file.write_text("# extra", encoding="utf-8")

        sources_file = tmp / "sources.txt"
        sources_file.write_text(
            f"""
# comment
{extra_file.name}

https://example.com/report.pdf
""".strip()
            + "\n",
            encoding="utf-8",
        )

        manifest_from_list = tmp / "manifest_from_list.jsonl"
        list_registry_path = tmp / "list_registry.json"

        summary_list = build_manifest(
            BuildManifestOptions(
                sources_files=[sources_file],
                registry_path=list_registry_path,
                output_path=manifest_from_list,
                corpus_id="eval1",
                branch_id="staging_eval1",
                tags=["list"],
            )
        )

        list_rows = [
            json.loads(line)
            for line in manifest_from_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        _assert(summary_list["sources_discovered"] == 2, "sources-file parsing failed")
        _assert(len(list_rows) == 2, "sources-file manifest row count wrong")

        for row in list_rows:
            ManifestRow.from_dict(row)

        _assert(
            any(row["source_kind"] == "markdown" for row in list_rows),
            "relative local source was not resolved/typed",
        )
        _assert(
            any(row["source_kind"] == "url" for row in list_rows),
            "URL source was not preserved",
        )

        # 13. Registry-only export.
        registry_only_manifest = tmp / "registry_only_manifest.jsonl"
        summary_registry_only = build_manifest(
            BuildManifestOptions(
                from_registry=True,
                registry_path=list_registry_path,
                output_path=registry_only_manifest,
                corpus_id="eval1",
                branch_id="staging_eval1",
            )
        )

        _assert(
            summary_registry_only["sources_written_to_manifest"] == 2,
            "registry-only export failed",
        )

        # 14. Extension filtering.
        pdf_only_manifest = tmp / "pdf_only_manifest.jsonl"
        pdf_only_registry = tmp / "pdf_only_registry.json"

        summary_pdf_only = build_manifest(
            BuildManifestOptions(
                sources_dirs=[sources_dir],
                registry_path=pdf_only_registry,
                output_path=pdf_only_manifest,
                corpus_id="eval1",
                branch_id="staging_eval1",
                include_ext=[".pdf"],
            )
        )

        _assert(summary_pdf_only["sources_discovered"] == 2, "include_ext did not restrict scan to PDFs")

    print("[build_manifest self-test] all tests passed")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_metadata_args(args: argparse.Namespace) -> JSONDict:
    metadata: JSONDict = {}

    for key in [
        "title",
        "author",
        "publisher",
        "published_at",
        "source_url",
        "original_url",
        "canonical_url",
    ]:
        value = getattr(args, key, None)
        if value not in (None, ""):
            metadata[key] = value

    if args.metadata_json:
        decoded = json.loads(args.metadata_json)
        if not isinstance(decoded, Mapping):
            raise SystemExit("--metadata-json must decode to a JSON object.")
        metadata.update(dict(decoded))

    if args.metadata_file:
        decoded = read_json(args.metadata_file)
        if not isinstance(decoded, Mapping):
            raise SystemExit("--metadata-file must contain a JSON object.")
        metadata.update(dict(decoded))

    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile registry-backed manifest.jsonl files.")

    parser.add_argument("--self-test", action="store_true")

    parser.add_argument("--sources-dir", type=Path, action="append", default=[])
    parser.add_argument("--sources-file", type=Path, action="append", default=[])
    parser.add_argument("--from-registry", action="store_true")

    parser.add_argument("--registry", type=Path, required=False, default=Path("data/source_registry.json"))
    parser.add_argument("--output", type=Path, required=False, default=Path("manifest.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=None)

    parser.add_argument("--corpus-id", default="default_corpus")
    parser.add_argument("--branch-id", default="staging")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--filter-tag", action="append", default=[])

    parser.add_argument("--include-ext", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")

    parser.add_argument("--dedupe", dest="dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--include-failed-acquisition", action="store_true")
    parser.add_argument("--include-failed-pipeline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--published-at", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--original-url", default=None)
    parser.add_argument("--canonical-url", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)

    return parser.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> BuildManifestOptions:
    return BuildManifestOptions(
        sources_dirs=args.sources_dir,
        sources_files=args.sources_file,
        registry_path=args.registry,
        output_path=args.output,
        summary_output_path=args.summary_output,
        corpus_id=args.corpus_id,
        branch_id=args.branch_id,
        tags=validate_tags(args.tag),
        filter_tags=validate_tags(args.filter_tag),
        include_ext=args.include_ext,
        exclude=args.exclude,
        recursive=args.recursive,
        from_registry=args.from_registry,
        dedupe=args.dedupe,
        include_duplicates=args.include_duplicates,
        include_blocked=args.include_blocked,
        include_failed_acquisition=args.include_failed_acquisition,
        include_failed_pipeline=args.include_failed_pipeline,
        limit=args.limit,
        metadata=parse_metadata_args(args),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.from_registry and not args.sources_dir and not args.sources_file:
        raise SystemExit("Provide --sources-dir, --sources-file, or --from-registry.")

    summary = build_manifest(options_from_args(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())