#!/usr/bin/env python3
"""
Shared pipeline contracts for Lanthic Intelligence.

This module is intentionally boring and deterministic.

It defines:
- canonical pipeline stage names
- source/acquisition/status enums
- stable ID and hashing helpers
- shared metadata records
- source record and manifest row contracts
- per-source status contract
- cost/event ledger contract
- artifact path conventions
- lightweight validation
- built-in self-test

It must not:
- fetch URLs
- parse HTML/PDFs
- call OpenAI
- call Neo4j
- run KG-IRAG/SARG
- orchestrate ingestion

Run:
    python src/pipeline_contracts.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import posixpath
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


JSONDict = Dict[str, Any]


# ============================================================
# Constants / allowed values
# ============================================================

SCHEMA_VERSION = 1

PIPELINE_STAGES: Tuple[str, ...] = (
    "discover",
    "acquire",
    "ingest",
    "extract",
    "postrag",
    "neo4j_fusion",
    "kg_irag",
    "sarg",
)

SOURCE_KINDS: Tuple[str, ...] = (
    "url",
    "file",
    "manual",
    "pdf",
    "html",
    "csv",
    "spreadsheet",
    "json",
    "text",
    "markdown",
    "unknown",
)

PIPELINE_STATUSES: Tuple[str, ...] = (
    "pending",
    "success",
    "failed",
    "skipped_duplicate",
    "skipped_cached",
    "blocked",
    "dry_run",
)

ACQUISITION_STATUSES: Tuple[str, ...] = (
    "not_acquired",
    "acquired",
    "cached",
    "manual",
    "blocked",
    "paywalled",
    "not_found",
    "fetch_failed",
    "unsupported",
)

CACHE_STATUSES: Tuple[str, ...] = (
    "not_applicable",
    "hit",
    "miss",
    "write",
    "bypass",
    "error",
)

ARTIFACT_FILENAMES: Mapping[str, str] = {
    "manifest_resolved": "manifest_resolved.jsonl",
    "source_registry": "source_registry.json",
    "run_summary": "run_summary.json",
    "cost_ledger": "cost_ledger.jsonl",
    "document": "document.json",
    "extraction": "extraction.json",
    "postrag": "postrag.json",
    "neo4j_summary": "neo4j_summary.json",
    "kg_irag": "kg_irag.json",
    "sarg": "sarg.json",
    "status": "status.json",
}

TRACKING_QUERY_PREFIXES: Tuple[str, ...] = (
    "utm_",
)

TRACKING_QUERY_KEYS: Tuple[str, ...] = (
    "fbclid",
    "gclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
)


# ============================================================
# Exceptions
# ============================================================

class ContractError(ValueError):
    """Raised when a pipeline contract is invalid."""


# ============================================================
# Generic helpers
# ============================================================

def utc_now() -> str:
    """Return an ISO-like UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )


def json_dumps_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def stable_hash(value: Any, length: int = 20) -> str:
    text = json_dumps_stable(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def normalize_id_component(value: str, *, field_name: str = "id") -> str:
    """
    Normalize a label-like ID component.

    This is for corpus_id / branch_id / run_id components, not hashes.
    """
    value = str(value or "").strip()

    if not value:
        raise ContractError(f"{field_name} cannot be empty.")

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")

    if not value:
        raise ContractError(f"{field_name} has no valid characters.")

    return value


def validate_id_component(value: str, *, field_name: str = "id") -> str:
    value = str(value or "").strip()

    if not value:
        raise ContractError(f"{field_name} cannot be empty.")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ContractError(
            f"{field_name} must start with an alphanumeric character and contain "
            f"only letters, numbers, underscores, hyphens, or periods: {value!r}"
        )

    return value


def make_run_id(prefix: str = "run") -> str:
    prefix = normalize_id_component(prefix, field_name="run_id_prefix")
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def is_url(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def strip_tracking_query(query: str) -> str:
    if not query:
        return ""

    kept: List[Tuple[str, str]] = []

    for key, value in parse_qsl(query, keep_blank_values=True):
        key_lower = key.lower()

        if key_lower in TRACKING_QUERY_KEYS:
            continue

        if any(key_lower.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue

        kept.append((key, value))

    kept.sort(key=lambda item: (item[0], item[1]))
    return urlencode(kept, doseq=True)


def canonicalize_url(url: str) -> str:
    """
    Canonicalize URLs for dedupe.

    It:
    - lowercases scheme and hostname
    - removes fragments
    - removes default ports
    - strips trailing slash except root
    - removes common tracking query params
    - sorts remaining query params
    """
    raw = str(url or "").strip()
    parsed = urlparse(raw)

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ContractError(f"Not a supported URL: {url!r}")

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        raise ContractError(f"URL has no hostname: {url!r}")

    port = parsed.port
    netloc = hostname

    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"

    path = parsed.path or "/"
    path = posixpath.normpath(path)

    if not path.startswith("/"):
        path = "/" + path

    if path != "/":
        path = path.rstrip("/")

    query = strip_tracking_query(parsed.query)

    return urlunparse((scheme, netloc, path, "", query, ""))


def canonicalize_local_path(path: Path, *, base_dir: Optional[Path] = None) -> str:
    path = Path(path).expanduser()

    if not path.is_absolute() and base_dir is not None:
        candidate = (base_dir / path).expanduser()
        if candidate.exists():
            path = candidate

    if path.exists():
        return str(path.resolve())

    return str(path)


def canonicalize_source(source: str, *, base_dir: Optional[Path] = None) -> str:
    source = str(source or "").strip()

    if not source:
        raise ContractError("source cannot be empty.")

    if is_url(source):
        return canonicalize_url(source)

    return canonicalize_local_path(Path(source), base_dir=base_dir)


def infer_domain(source: str) -> Optional[str]:
    if not is_url(source):
        return None

    parsed = urlparse(source)
    return parsed.netloc.lower() or None


def infer_source_kind(source: str, *, local_path: Optional[str] = None) -> str:
    candidate = local_path or source

    if is_url(candidate):
        return "url"

    suffix = Path(candidate).suffix.lower()

    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".csv":
        return "csv"
    if suffix in {".xlsx", ".xls"}:
        return "spreadsheet"
    if suffix == ".json":
        return "json"
    if suffix == ".txt":
        return "text"
    if suffix in {".md", ".markdown"}:
        return "markdown"

    if candidate:
        return "file"

    return "unknown"


def infer_mime_type(source: str) -> Optional[str]:
    if is_url(source):
        guessed, _ = mimetypes.guess_type(urlparse(source).path)
    else:
        guessed, _ = mimetypes.guess_type(source)

    return guessed


def source_id_from_canonical(canonical_source: str) -> str:
    return f"src_{stable_hash(canonical_source, 16)}"


def document_id_from_source_id(source_id: str) -> str:
    validate_id_component(source_id, field_name="source_id")
    return f"doc_{stable_hash(source_id, 16)}"


def artifact_id(*parts: Any, prefix: str = "art") -> str:
    prefix = normalize_id_component(prefix, field_name="artifact_prefix")
    return f"{prefix}_{stable_hash(parts, 16)}"


def source_fingerprint(source: str, *, base_dir: Optional[Path] = None) -> JSONDict:
    canonical = canonicalize_source(source, base_dir=base_dir)

    if is_url(canonical):
        return {
            "fingerprint_type": "url",
            "canonical_source": canonical,
            "content_hash": None,
            "source_hash": stable_hash(canonical),
            "size_bytes": None,
            "mtime": None,
        }

    path = Path(canonical)

    if path.exists() and path.is_file():
        return {
            "fingerprint_type": "file_sha256",
            "canonical_source": str(path.resolve()),
            "content_hash": file_sha256(path),
            "source_hash": stable_hash(str(path.resolve())),
            "size_bytes": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }

    return {
        "fingerprint_type": "unresolved_path",
        "canonical_source": canonical,
        "content_hash": None,
        "source_hash": stable_hash(canonical),
        "size_bytes": None,
        "mtime": None,
    }


# ============================================================
# Validation helpers
# ============================================================

def validate_choice(value: str, allowed: Sequence[str], *, field_name: str) -> str:
    value = str(value or "").strip()

    if value not in allowed:
        raise ContractError(
            f"Invalid {field_name}: {value!r}. Allowed values: {', '.join(allowed)}"
        )

    return value


def validate_stage_name(stage: str) -> str:
    return validate_choice(stage, PIPELINE_STAGES, field_name="stage")


def validate_status(status: str) -> str:
    return validate_choice(status, PIPELINE_STATUSES, field_name="status")


def validate_acquisition_status(status: str) -> str:
    return validate_choice(status, ACQUISITION_STATUSES, field_name="acquisition_status")


def validate_cache_status(status: str) -> str:
    return validate_choice(status, CACHE_STATUSES, field_name="cache_status")


def validate_source_kind(source_kind: str) -> str:
    return validate_choice(source_kind, SOURCE_KINDS, field_name="source_kind")


def validate_stage_list(stages: Iterable[str], *, field_name: str) -> List[str]:
    out = []

    for stage in stages:
        out.append(validate_stage_name(stage))

    return out


def validate_tags(tags: Iterable[Any]) -> List[str]:
    out = []

    for tag in tags or []:
        text = str(tag).strip()
        if text:
            out.append(text)

    return out


def clean_metadata(metadata: Optional[Mapping[str, Any]]) -> JSONDict:
    if metadata is None:
        return {}

    if not isinstance(metadata, Mapping):
        raise ContractError("metadata must be a mapping.")

    return dict(metadata)


# ============================================================
# Dataclass helpers
# ============================================================

def _dataclass_from_mapping(cls: Any, data: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    kwargs = {key: value for key, value in dict(data).items() if key in allowed}
    return cls(**kwargs)


# ============================================================
# Shared metadata contract
# ============================================================

@dataclass
class PipelineMetadata:
    run_id: str
    corpus_id: str
    branch_id: str
    source_id: Optional[str] = None
    document_id: Optional[str] = None
    canonical_source: Optional[str] = None
    source_kind: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "PipelineMetadata":
        validate_id_component(self.run_id, field_name="run_id")
        validate_id_component(self.corpus_id, field_name="corpus_id")
        validate_id_component(self.branch_id, field_name="branch_id")

        if self.source_id is not None:
            validate_id_component(self.source_id, field_name="source_id")

        if self.document_id is not None:
            validate_id_component(self.document_id, field_name="document_id")

        if self.source_kind is not None:
            validate_source_kind(self.source_kind)

        self.tags = validate_tags(self.tags)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineMetadata":
        return _dataclass_from_mapping(cls, data).validate()


def attach_pipeline_metadata(
    artifact: Mapping[str, Any],
    metadata: PipelineMetadata,
    *,
    key: str = "pipeline_metadata",
) -> JSONDict:
    out = dict(artifact)
    out[key] = metadata.to_dict()
    return out


# ============================================================
# Source record contract
# ============================================================

@dataclass
class SourceRecord:
    source_id: str
    source: str
    canonical_source: str
    source_kind: str

    domain: Optional[str] = None
    content_hash: Optional[str] = None
    source_hash: Optional[str] = None
    fingerprint_type: str = "unknown"

    acquisition_status: str = "not_acquired"
    acquisition_method: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: Optional[str] = None

    title: Optional[str] = None
    published_at: Optional[str] = None
    fetched_at: Optional[str] = None
    first_seen_at: str = field(default_factory=utc_now)
    last_seen_at: str = field(default_factory=utc_now)
    last_error: Optional[str] = None
    duplicate_of: Optional[str] = None

    corpus_id: str = "default_corpus"
    branch_id: str = "staging"
    tags: List[str] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "SourceRecord":
        validate_id_component(self.source_id, field_name="source_id")
        validate_source_kind(self.source_kind)
        validate_acquisition_status(self.acquisition_status)
        validate_id_component(self.corpus_id, field_name="corpus_id")
        validate_id_component(self.branch_id, field_name="branch_id")

        if not self.source:
            raise ContractError("SourceRecord.source cannot be empty.")

        if not self.canonical_source:
            raise ContractError("SourceRecord.canonical_source cannot be empty.")

        if self.duplicate_of is not None:
            validate_id_component(self.duplicate_of, field_name="duplicate_of")

        self.tags = validate_tags(self.tags)
        self.metadata = clean_metadata(self.metadata)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRecord":
        return _dataclass_from_mapping(cls, data).validate()

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        corpus_id: str,
        branch_id: str,
        tags: Optional[Sequence[str]] = None,
        base_dir: Optional[Path] = None,
        acquisition_status: str = "not_acquired",
        acquisition_method: Optional[str] = None,
        local_path: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SourceRecord":
        fingerprint = source_fingerprint(source, base_dir=base_dir)
        canonical = str(fingerprint["canonical_source"])
        source_id = source_id_from_canonical(canonical)
        kind = infer_source_kind(source, local_path=local_path)

        if local_path and Path(local_path).exists():
            kind = infer_source_kind(local_path)

        return cls(
            source_id=source_id,
            source=str(source),
            canonical_source=canonical,
            source_kind=kind,
            domain=infer_domain(canonical),
            content_hash=fingerprint.get("content_hash"),
            source_hash=fingerprint.get("source_hash"),
            fingerprint_type=str(fingerprint.get("fingerprint_type") or "unknown"),
            acquisition_status=acquisition_status,
            acquisition_method=acquisition_method,
            local_path=local_path,
            mime_type=infer_mime_type(local_path or canonical),
            corpus_id=corpus_id,
            branch_id=branch_id,
            tags=validate_tags(tags or []),
            metadata=clean_metadata(metadata),
        ).validate()

    def to_manifest_row(self) -> "ManifestRow":
        return ManifestRow(
            source=self.local_path or self.source,
            source_id=self.source_id,
            source_kind=self.source_kind,
            canonical_source=self.canonical_source,
            local_path=self.local_path,
            corpus_id=self.corpus_id,
            branch_id=self.branch_id,
            tags=list(self.tags),
            acquisition_status=self.acquisition_status,
            metadata={
                **self.metadata,
                "domain": self.domain,
                "content_hash": self.content_hash,
                "source_hash": self.source_hash,
                "fingerprint_type": self.fingerprint_type,
                "title": self.title,
                "published_at": self.published_at,
                "fetched_at": self.fetched_at,
                "mime_type": self.mime_type,
            },
        )


# ============================================================
# Manifest row contract
# ============================================================

@dataclass
class ManifestRow:
    source: str
    source_id: str
    source_kind: str
    canonical_source: str
    local_path: Optional[str] = None

    corpus_id: str = "default_corpus"
    branch_id: str = "staging"
    tags: List[str] = field(default_factory=list)
    acquisition_status: str = "not_acquired"
    metadata: JSONDict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "ManifestRow":
        if not self.source:
            raise ContractError("ManifestRow.source cannot be empty.")

        validate_id_component(self.source_id, field_name="source_id")
        validate_source_kind(self.source_kind)
        validate_id_component(self.corpus_id, field_name="corpus_id")
        validate_id_component(self.branch_id, field_name="branch_id")
        validate_acquisition_status(self.acquisition_status)

        if not self.canonical_source:
            raise ContractError("ManifestRow.canonical_source cannot be empty.")

        self.tags = validate_tags(self.tags)
        self.metadata = clean_metadata(self.metadata)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManifestRow":
        return _dataclass_from_mapping(cls, data).validate()

    @classmethod
    def from_source_record(cls, record: SourceRecord) -> "ManifestRow":
        return record.to_manifest_row().validate()

    def pipeline_metadata(self, *, run_id: str, document_id: Optional[str] = None) -> PipelineMetadata:
        return PipelineMetadata(
            run_id=run_id,
            corpus_id=self.corpus_id,
            branch_id=self.branch_id,
            source_id=self.source_id,
            document_id=document_id,
            canonical_source=self.canonical_source,
            source_kind=self.source_kind,
            tags=list(self.tags),
        ).validate()


# ============================================================
# Artifact layout contract
# ============================================================

@dataclass(frozen=True)
class ArtifactPaths:
    output_root: str
    run_id: str
    source_id: Optional[str] = None

    def validate(self) -> "ArtifactPaths":
        validate_id_component(self.run_id, field_name="run_id")

        if self.source_id is not None:
            validate_id_component(self.source_id, field_name="source_id")

        return self

    @property
    def run_dir(self) -> Path:
        self.validate()
        return Path(self.output_root) / self.run_id

    @property
    def sources_dir(self) -> Path:
        return self.run_dir / "sources"

    @property
    def source_dir(self) -> Path:
        if self.source_id is None:
            raise ContractError("source_id is required for source artifact paths.")
        return self.sources_dir / self.source_id

    def run_artifact(self, name: str) -> Path:
        if name not in {"manifest_resolved", "source_registry", "run_summary", "cost_ledger"}:
            raise ContractError(f"{name!r} is not a run-level artifact.")

        return self.run_dir / ARTIFACT_FILENAMES[name]

    def source_artifact(self, name: str) -> Path:
        if name not in {
            "document",
            "extraction",
            "postrag",
            "neo4j_summary",
            "kg_irag",
            "sarg",
            "status",
        }:
            raise ContractError(f"{name!r} is not a source-level artifact.")

        return self.source_dir / ARTIFACT_FILENAMES[name]

    def as_dict(self) -> JSONDict:
        out: JSONDict = {
            "run_dir": str(self.run_dir),
            "sources_dir": str(self.sources_dir),
            "manifest_resolved": str(self.run_artifact("manifest_resolved")),
            "source_registry": str(self.run_artifact("source_registry")),
            "run_summary": str(self.run_artifact("run_summary")),
            "cost_ledger": str(self.run_artifact("cost_ledger")),
        }

        if self.source_id is not None:
            out.update(
                {
                    "source_dir": str(self.source_dir),
                    "document": str(self.source_artifact("document")),
                    "extraction": str(self.source_artifact("extraction")),
                    "postrag": str(self.source_artifact("postrag")),
                    "neo4j_summary": str(self.source_artifact("neo4j_summary")),
                    "kg_irag": str(self.source_artifact("kg_irag")),
                    "sarg": str(self.source_artifact("sarg")),
                    "status": str(self.source_artifact("status")),
                }
            )

        return out


def artifact_paths(
    *,
    output_root: str | Path,
    run_id: str,
    source_id: Optional[str] = None,
) -> ArtifactPaths:
    return ArtifactPaths(str(output_root), run_id, source_id).validate()


# ============================================================
# Status contract
# ============================================================

@dataclass
class StatusRecord:
    source_id: str
    status: str = "pending"

    completed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[str] = field(default_factory=list)
    failed_stage: Optional[str] = None

    error: Optional[str] = None
    traceback: Optional[str] = None

    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    metrics: JSONDict = field(default_factory=dict)

    run_id: Optional[str] = None
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "StatusRecord":
        validate_id_component(self.source_id, field_name="source_id")
        validate_status(self.status)

        self.completed_stages = validate_stage_list(
            self.completed_stages,
            field_name="completed_stages",
        )
        self.skipped_stages = validate_stage_list(
            self.skipped_stages,
            field_name="skipped_stages",
        )

        if self.failed_stage is not None:
            validate_stage_name(self.failed_stage)

        if self.run_id is not None:
            validate_id_component(self.run_id, field_name="run_id")

        if self.corpus_id is not None:
            validate_id_component(self.corpus_id, field_name="corpus_id")

        if self.branch_id is not None:
            validate_id_component(self.branch_id, field_name="branch_id")

        if self.duplicate_of is not None:
            validate_id_component(self.duplicate_of, field_name="duplicate_of")

        self.metrics = clean_metadata(self.metrics)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StatusRecord":
        return _dataclass_from_mapping(cls, data).validate()


# ============================================================
# Cost/event ledger contract
# ============================================================

@dataclass
class LedgerEvent:
    event_id: str
    run_id: str
    source_id: Optional[str]
    stage: str
    operation: str

    model: Optional[str] = None
    input_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    cache_status: str = "not_applicable"

    estimated_input_tokens: Optional[int] = None
    estimated_output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    runtime_seconds: Optional[float] = None

    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    metadata: JSONDict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> "LedgerEvent":
        validate_id_component(self.event_id, field_name="event_id")
        validate_id_component(self.run_id, field_name="run_id")
        validate_stage_name(self.stage)
        validate_cache_status(self.cache_status)

        if self.source_id is not None:
            validate_id_component(self.source_id, field_name="source_id")

        if not self.operation:
            raise ContractError("LedgerEvent.operation cannot be empty.")

        self.metadata = clean_metadata(self.metadata)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LedgerEvent":
        return _dataclass_from_mapping(cls, data).validate()

    @classmethod
    def make(
        cls,
        *,
        run_id: str,
        stage: str,
        operation: str,
        source_id: Optional[str] = None,
        model: Optional[str] = None,
        input_hash: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        cache_status: str = "not_applicable",
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        runtime_seconds: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "LedgerEvent":
        return cls(
            event_id=artifact_id(
                run_id,
                source_id,
                stage,
                operation,
                model,
                input_hash,
                prompt_hash,
                time.time(),
                prefix="evt",
            ),
            run_id=run_id,
            source_id=source_id,
            stage=stage,
            operation=operation,
            model=model,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            cache_status=cache_status,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            runtime_seconds=runtime_seconds,
            error=error,
            metadata=clean_metadata(metadata),
        ).validate()


# ============================================================
# Manifest JSONL helpers
# ============================================================

def write_manifest_jsonl(path: Path, rows: Sequence[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(row.to_jsonl() for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def read_manifest_jsonl(path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []

    text = path.read_text(encoding="utf-8")

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid manifest JSONL at line {line_number}: {exc}") from exc

        if not isinstance(data, Mapping):
            raise ContractError(f"Manifest line {line_number} must be a JSON object.")

        rows.append(ManifestRow.from_dict(data))

    return rows


# ============================================================
# High-level validation aliases
# ============================================================

def validate_pipeline_metadata(data: Mapping[str, Any]) -> PipelineMetadata:
    return PipelineMetadata.from_dict(data)


def validate_source_record(data: Mapping[str, Any]) -> SourceRecord:
    return SourceRecord.from_dict(data)


def validate_manifest_row(data: Mapping[str, Any]) -> ManifestRow:
    return ManifestRow.from_dict(data)


def validate_status_record(data: Mapping[str, Any]) -> StatusRecord:
    return StatusRecord.from_dict(data)


def validate_ledger_event(data: Mapping[str, Any]) -> LedgerEvent:
    return LedgerEvent.from_dict(data)


# ============================================================
# Self-test
# ============================================================

def _assert_raises(expected_exception: type[BaseException], fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return

    raise AssertionError(f"Expected {expected_exception.__name__} from {getattr(fn, '__name__', fn)!r}")


def run_self_test() -> int:
    print("[pipeline_contracts self-test] starting")

    # 1. Equivalent URLs canonicalize to same source_id.
    url_a = "https://Example.COM/reports/rare-earths/?utm_source=newsletter#section"
    url_b = "https://example.com/reports/rare-earths"
    canonical_a = canonicalize_source(url_a)
    canonical_b = canonicalize_source(url_b)

    assert canonical_a == canonical_b, (canonical_a, canonical_b)
    assert source_id_from_canonical(canonical_a) == source_id_from_canonical(canonical_b)

    # 2. Local file can produce a content hash.
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        sample = tmp / "sample.txt"
        sample.write_text("rare earth supply-chain evidence", encoding="utf-8")

        fingerprint = source_fingerprint(str(sample))
        assert fingerprint["fingerprint_type"] == "file_sha256"
        assert fingerprint["content_hash"] == file_sha256(sample)

        # 3. SourceRecord -> ManifestRow.
        record = SourceRecord.from_source(
            str(sample),
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["myanmar", "hree"],
            acquisition_status="manual",
            acquisition_method="local_file",
            local_path=str(sample),
            metadata={"note": "self-test"},
        )

        assert record.source_id.startswith("src_")
        assert record.content_hash == file_sha256(sample)
        assert record.acquisition_status == "manual"

        manifest_row = record.to_manifest_row()
        manifest_row.validate()

        assert manifest_row.source_id == record.source_id
        assert manifest_row.corpus_id == "eval1"
        assert manifest_row.branch_id == "staging_eval1"
        assert manifest_row.acquisition_status == "manual"

        manifest_path = tmp / "manifest.jsonl"
        write_manifest_jsonl(manifest_path, [manifest_row])
        loaded_rows = read_manifest_jsonl(manifest_path)

        assert len(loaded_rows) == 1
        assert loaded_rows[0].source_id == manifest_row.source_id

        # 4. Pipeline metadata can be attached to arbitrary artifact dict.
        metadata = manifest_row.pipeline_metadata(
            run_id="run_selftest",
            document_id=document_id_from_source_id(manifest_row.source_id),
        )
        artifact = attach_pipeline_metadata({"hello": "world"}, metadata)

        assert artifact["hello"] == "world"
        assert artifact["pipeline_metadata"]["run_id"] == "run_selftest"
        assert artifact["pipeline_metadata"]["corpus_id"] == "eval1"
        assert artifact["pipeline_metadata"]["branch_id"] == "staging_eval1"

        # 5. Invalid stage/status values are rejected.
        validate_stage_name("extract")
        validate_status("success")
        validate_acquisition_status("paywalled")
        validate_cache_status("hit")

        _assert_raises(ContractError, validate_stage_name, "made_up_stage")
        _assert_raises(ContractError, validate_status, "half_success")
        _assert_raises(ContractError, validate_acquisition_status, "secretly_scraped")
        _assert_raises(ContractError, validate_cache_status, "maybe")

        # 6. Artifact paths are deterministic.
        paths_a = artifact_paths(
            output_root=tmp / "runs",
            run_id="run_selftest",
            source_id=manifest_row.source_id,
        ).as_dict()
        paths_b = artifact_paths(
            output_root=tmp / "runs",
            run_id="run_selftest",
            source_id=manifest_row.source_id,
        ).as_dict()

        assert paths_a == paths_b
        assert paths_a["document"].endswith(f"sources/{manifest_row.source_id}/document.json")
        assert paths_a["run_summary"].endswith("run_selftest/run_summary.json")
        assert paths_a["cost_ledger"].endswith("run_selftest/cost_ledger.jsonl")

        # 7. StatusRecord contract.
        status = StatusRecord(
            source_id=manifest_row.source_id,
            status="success",
            completed_stages=["acquire", "ingest", "extract", "postrag"],
            skipped_stages=[],
            run_id="run_selftest",
            corpus_id="eval1",
            branch_id="staging_eval1",
            metrics={"table_blocks": 2},
        ).validate()

        assert status.metrics["table_blocks"] == 2
        _assert_raises(
            ContractError,
            StatusRecord(source_id=manifest_row.source_id, status="success", completed_stages=["bad_stage"]).validate,
        )

        # 8. LedgerEvent contract.
        event = LedgerEvent.make(
            run_id="run_selftest",
            source_id=manifest_row.source_id,
            stage="extract",
            operation="openai.chat.completions.create",
            model="gpt-4.1-mini",
            input_hash=text_sha256("input"),
            prompt_hash=text_sha256("prompt"),
            cache_status="miss",
            estimated_input_tokens=100,
            estimated_output_tokens=50,
            estimated_cost_usd=0.001,
            runtime_seconds=0.25,
        )

        assert event.event_id.startswith("evt_")
        assert event.stage == "extract"
        assert event.cache_status == "miss"

        event_roundtrip = LedgerEvent.from_dict(event.to_dict())
        assert event_roundtrip.event_id == event.event_id

        # 9. All acquisition statuses validate.
        for status_name in ACQUISITION_STATUSES:
            validate_acquisition_status(status_name)

        # 10. SourceRecord JSON roundtrip.
        record_roundtrip = SourceRecord.from_dict(record.to_dict())
        assert record_roundtrip.source_id == record.source_id
        assert record_roundtrip.content_hash == record.content_hash

    print("[pipeline_contracts self-test] all tests passed")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared pipeline contracts for Lanthic Intelligence.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in contract tests.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    print("pipeline_contracts.py defines shared contracts. Run with --self-test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())