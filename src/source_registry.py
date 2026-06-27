#!/usr/bin/env python3
"""
Persistent source registry for Lanthic Intelligence.

This module owns:
- source identity
- deduplication
- source lifecycle state
- acquisition status tracking
- pipeline status tracking
- source metadata enrichment for extract.json document blocks
- manifest-row export
- registry summaries

It must not:
- fetch URLs
- parse documents
- call OpenAI
- call Neo4j
- run ingestion/extraction/PostRAG/KG-IRAG

Run:
    python src/source_registry.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pipeline_contracts import (
    ACQUISITION_STATUSES,
    PIPELINE_STATUSES,
    ContractError,
    ManifestRow,
    PipelineMetadata,
    SourceRecord,
    StatusRecord,
    canonicalize_source,
    canonicalize_url,
    infer_domain,
    is_url,
    read_json,
    source_fingerprint,
    source_id_from_canonical,
    utc_now,
    validate_acquisition_status,
    validate_id_component,
    validate_source_kind,
    validate_status,
    validate_tags,
    write_json,
)


JSONDict = Dict[str, Any]


HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "csis.org",
    "usgs.gov",
    "iea.org",
    "energy.gov",
    "commerce.gov",
    "sec.gov",
    "bis.doc.gov",
    "worldbank.org",
    "imf.org",
    "crisisgroup.org",
    "internationalcrisisgroup.org",
}

MEDIUM_CREDIBILITY_DOMAINS = {
    "stimson.org",
    "brookings.edu",
    "rand.org",
    "rusi.org",
    "chathamhouse.org",
    "spglobal.com",
    "mining.com",
    "argusmedia.com",
    "fastmarkets.com",
    "theconversation.com",
    "ft.com",
    "economist.com",
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
}


BLOCKING_ACQUISITION_STATUSES = {
    "blocked",
    "paywalled",
    "not_found",
    "fetch_failed",
    "unsupported",
}


# ============================================================
# Small helpers
# ============================================================

def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as f:
        tmp_path = Path(f.name)
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        f.write("\n")

    os.replace(tmp_path, path)


def _none_if_blank(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _metadata_get(metadata: Mapping[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _domain_matches(domain: Optional[str], known_domains: Iterable[str]) -> Optional[str]:
    if not domain:
        return None

    domain = domain.lower().strip()

    for known in known_domains:
        known = known.lower().strip()
        if domain == known or domain.endswith("." + known):
            return known

    return None


def _filename_title(path_or_url: str) -> Optional[str]:
    if not path_or_url:
        return None

    if is_url(path_or_url):
        from urllib.parse import urlparse

        name = Path(urlparse(path_or_url).path).name
    else:
        name = Path(path_or_url).name

    if not name:
        return None

    stem = Path(name).stem
    return stem.replace("_", " ").replace("-", " ").strip() or None


def _merge_non_null(base: JSONDict, updates: Mapping[str, Any]) -> JSONDict:
    out = dict(base)

    for key, value in updates.items():
        if value not in (None, ""):
            out[key] = value

    return out


def _append_unique(items: Sequence[str], extra: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in list(items) + list(extra):
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)

    return out


# ============================================================
# Plan decision
# ============================================================

@dataclass
class PlanDecision:
    action: str
    reason: str
    source_id: str
    duplicate_of: Optional[str] = None
    existing_status: Optional[str] = None
    acquisition_status: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ============================================================
# Source registry
# ============================================================

class SourceRegistry:
    """
    Persistent registry of known sources.

    The registry stores SourceRecord-shaped objects plus registry-specific
    lifecycle fields such as last_status, last_error, and run history.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data: JSONDict = self._empty_registry()
        self.content_hash_index: Dict[str, str] = {}
        self.canonical_source_index: Dict[str, str] = {}
        self.load()

    # --------------------------------------------------------
    # Load/save/indexing
    # --------------------------------------------------------

    @staticmethod
    def _empty_registry() -> JSONDict:
        now = utc_now()
        return {
            "schema_version": 1,
            "created_at": now,
            "updated_at": now,
            "sources": {},
            "content_hash_index": {},
            "canonical_source_index": {},
        }

    def load(self) -> "SourceRegistry":
        if not self.path.exists():
            self.rebuild_indexes()
            return self

        loaded = read_json(self.path)

        if not isinstance(loaded, Mapping):
            raise ContractError(f"Registry file must be a JSON object: {self.path}")

        self.data = dict(self._empty_registry())
        self.data.update(dict(loaded))
        self.data.setdefault("sources", {})
        self.data.setdefault("content_hash_index", {})
        self.data.setdefault("canonical_source_index", {})

        self.rebuild_indexes()
        return self

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        self.rebuild_indexes()
        self.data["content_hash_index"] = dict(self.content_hash_index)
        self.data["canonical_source_index"] = dict(self.canonical_source_index)
        _atomic_write_json(self.path, self.data)

    def rebuild_indexes(self) -> None:
        """
        Rebuild lookup indexes.

        Important rule:
        duplicate records must not become canonical index targets.

        If src_B is marked duplicate_of src_A, then content_hash_index/hash
        must point to src_A, not src_B. Otherwise the original source can later
        be incorrectly treated as a duplicate of its duplicate.
        """
        self.content_hash_index = {}
        self.canonical_source_index = {}

        sources = self.data.get("sources", {})

        # First pass: index non-duplicate canonical records.
        for source_id, record in sources.items():
            if record.get("duplicate_of"):
                continue

            canonical = record.get("canonical_source")
            content_hash = record.get("content_hash")

            if canonical and canonical not in self.canonical_source_index:
                self.canonical_source_index[str(canonical)] = str(source_id)

            if content_hash and content_hash not in self.content_hash_index:
                self.content_hash_index[str(content_hash)] = str(source_id)

        # Second pass: if only duplicate records exist for a key, point the key
        # to their declared canonical source where possible.
        for source_id, record in sources.items():
            duplicate_of = record.get("duplicate_of")
            if not duplicate_of:
                continue

            canonical = record.get("canonical_source")
            content_hash = record.get("content_hash")

            if canonical and canonical not in self.canonical_source_index:
                self.canonical_source_index[str(canonical)] = str(duplicate_of)

            if content_hash and content_hash not in self.content_hash_index:
                self.content_hash_index[str(content_hash)] = str(duplicate_of)
    # --------------------------------------------------------
    # Basic access
    # --------------------------------------------------------

    def source_ids(self) -> List[str]:
        return sorted(self.data.get("sources", {}).keys())

    def has_source(self, source_id: str) -> bool:
        return source_id in self.data.get("sources", {})

    def raw_record(self, source_id: str) -> JSONDict:
        validate_id_component(source_id, field_name="source_id")

        try:
            return dict(self.data["sources"][source_id])
        except KeyError as exc:
            raise ContractError(f"Unknown source_id: {source_id}") from exc

    def get_source_record(self, source_id: str) -> SourceRecord:
        return SourceRecord.from_dict(self.raw_record(source_id))

    # --------------------------------------------------------
    # Registration / upsert
    # --------------------------------------------------------

    def register_source(
        self,
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
        save: bool = True,
    ) -> SourceRecord:
        """
        Register a raw source path/URL and return its canonical SourceRecord.

        Metadata may include:
        - title
        - author
        - publisher
        - published_at
        - source_url
        - original_url
        - canonical_url
        - credibility / credibility_score / credibility_tier
        """

        metadata_dict = dict(metadata or {})

        record = SourceRecord.from_source(
            source,
            corpus_id=corpus_id,
            branch_id=branch_id,
            tags=tags or [],
            base_dir=base_dir,
            acquisition_status=acquisition_status,
            acquisition_method=acquisition_method,
            local_path=local_path,
            metadata=metadata_dict,
        )

        record = self._enrich_record_from_metadata(record, metadata_dict)
        self.upsert_source(record, save=save)
        return record

    def upsert_source(self, record: SourceRecord, *, save: bool = True) -> SourceRecord:
        record.validate()
        now = utc_now()

        sources = self.data.setdefault("sources", {})
        existing = dict(sources.get(record.source_id, {}))

        first_seen = existing.get("first_seen_at") or record.first_seen_at or now
        runs = list(existing.get("runs") or [])

        merged = {
            **existing,
            **record.to_dict(),
            "first_seen_at": first_seen,
            "last_seen_at": now,
            "runs": runs,
            "last_status": existing.get("last_status"),
            "last_error": existing.get("last_error"),
            "last_failed_stage": existing.get("last_failed_stage"),
            "last_run_id": existing.get("last_run_id"),
            "last_output_dir": existing.get("last_output_dir"),
            "last_ingested_at": existing.get("last_ingested_at"),
        }

        if existing.get("metadata") or record.metadata:
            merged["metadata"] = _merge_non_null(
                dict(existing.get("metadata") or {}),
                dict(record.metadata or {}),
            )

        sources[record.source_id] = merged
        self.rebuild_indexes()

        if save:
            self.save()

        return SourceRecord.from_dict(merged)

    def _enrich_record_from_metadata(
        self,
        record: SourceRecord,
        metadata: Mapping[str, Any],
    ) -> SourceRecord:
        metadata = dict(metadata or {})

        source_url = _metadata_get(metadata, "source_url", "original_url", "url")
        canonical_url = _metadata_get(metadata, "canonical_url")

        if source_url and not canonical_url and is_url(str(source_url)):
            try:
                canonical_url = canonicalize_url(str(source_url))
                metadata["canonical_url"] = canonical_url
            except ContractError:
                pass

        if not record.domain:
            domain_source = canonical_url or source_url
            if domain_source and is_url(str(domain_source)):
                record.domain = infer_domain(str(domain_source))

        record.title = _none_if_blank(_metadata_get(metadata, "title")) or record.title
        record.published_at = _none_if_blank(_metadata_get(metadata, "published_at", "date")) or record.published_at
        record.mime_type = _none_if_blank(_metadata_get(metadata, "mime_type")) or record.mime_type
        record.metadata = _merge_non_null(record.metadata, metadata)

        return record.validate()

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    def find_duplicate(self, record: SourceRecord) -> Optional[str]:
        """
        Return duplicate source_id if this record duplicates an existing source.

        Priority:
        1. explicit duplicate_of
        2. content_hash match
        3. canonical_source match

        Duplicate records are never treated as canonical targets.
        """
        record.validate()
        self.rebuild_indexes()

        if record.duplicate_of:
            return record.duplicate_of

        if record.content_hash:
            existing = self.content_hash_index.get(record.content_hash)
            if existing and existing != record.source_id:
                existing_record = self.data.get("sources", {}).get(existing, {})
                return existing_record.get("duplicate_of") or existing

        existing = self.canonical_source_index.get(record.canonical_source)
        if existing and existing != record.source_id:
            existing_record = self.data.get("sources", {}).get(existing, {})
            return existing_record.get("duplicate_of") or existing

        return None
    
    def mark_duplicate(
        self,
        source_id: str,
        duplicate_of: str,
        *,
        save: bool = True,
    ) -> None:
        validate_id_component(source_id, field_name="source_id")
        validate_id_component(duplicate_of, field_name="duplicate_of")

        record = self.raw_record(source_id)
        record["duplicate_of"] = duplicate_of
        record["last_seen_at"] = utc_now()
        self.data["sources"][source_id] = record

        if save:
            self.save()

    # --------------------------------------------------------
    # Acquisition status
    # --------------------------------------------------------

    def update_acquisition(
        self,
        source_id: str,
        *,
        acquisition_status: str,
        acquisition_method: Optional[str] = None,
        local_path: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        save: bool = True,
    ) -> SourceRecord:
        validate_acquisition_status(acquisition_status)
        record_dict = self.raw_record(source_id)

        record_dict["acquisition_status"] = acquisition_status
        record_dict["acquisition_method"] = acquisition_method or record_dict.get("acquisition_method")
        record_dict["last_error"] = error

        if acquisition_status in {"acquired", "cached", "manual"}:
            record_dict["fetched_at"] = record_dict.get("fetched_at") or utc_now()

        if local_path:
            local_path_obj = Path(local_path)
            record_dict["local_path"] = str(local_path_obj)

            try:
                fingerprint = source_fingerprint(str(local_path_obj))
                record_dict["content_hash"] = fingerprint.get("content_hash") or record_dict.get("content_hash")
                record_dict["fingerprint_type"] = fingerprint.get("fingerprint_type") or record_dict.get("fingerprint_type")
            except Exception as exc:
                record_dict["last_error"] = f"Could not fingerprint local_path {local_path}: {exc}"

        if metadata:
            record_dict["metadata"] = _merge_non_null(
                dict(record_dict.get("metadata") or {}),
                dict(metadata),
            )

        record = SourceRecord.from_dict(record_dict)
        record = self._enrich_record_from_metadata(record, record.metadata)
        self.data["sources"][source_id] = record.to_dict()

        if save:
            self.save()

        return record

    # --------------------------------------------------------
    # Pipeline status
    # --------------------------------------------------------

    def update_pipeline_status(
        self,
        source_id: str,
        status: StatusRecord,
        *,
        output_dir: Optional[str] = None,
        save: bool = True,
    ) -> None:
        status.validate()
        record = self.raw_record(source_id)

        record["last_status"] = status.status
        record["last_error"] = status.error
        record["last_failed_stage"] = status.failed_stage
        record["last_run_id"] = status.run_id
        record["last_output_dir"] = output_dir or record.get("last_output_dir")
        record["last_seen_at"] = utc_now()

        if status.status == "success":
            record["last_ingested_at"] = status.finished_at or utc_now()

        event = {
            "run_id": status.run_id,
            "corpus_id": status.corpus_id,
            "branch_id": status.branch_id,
            "status": status.status,
            "completed_stages": list(status.completed_stages),
            "skipped_stages": list(status.skipped_stages),
            "failed_stage": status.failed_stage,
            "error": status.error,
            "started_at": status.started_at,
            "finished_at": status.finished_at,
            "runtime_seconds": status.runtime_seconds,
            "metrics": dict(status.metrics or {}),
        }

        runs = list(record.get("runs") or [])
        runs.append(event)
        record["runs"] = runs[-100:]

        self.data["sources"][source_id] = record

        if save:
            self.save()

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    def plan_source(
        self,
        record: SourceRecord,
        *,
        resume: bool = True,
        force: bool = False,
        dedupe: bool = True,
        retry_failed: bool = False,
        allow_blocked: bool = False,
    ) -> PlanDecision:
        record.validate()

        if force:
            return PlanDecision(
                action="force_reprocess",
                reason="force=True",
                source_id=record.source_id,
                acquisition_status=record.acquisition_status,
            )

        if dedupe:
            duplicate_of = self.find_duplicate(record)
            if duplicate_of:
                return PlanDecision(
                    action="skip_duplicate",
                    reason="duplicate content_hash or canonical_source",
                    source_id=record.source_id,
                    duplicate_of=duplicate_of,
                    acquisition_status=record.acquisition_status,
                )

        if record.acquisition_status in BLOCKING_ACQUISITION_STATUSES and not allow_blocked:
            return PlanDecision(
                action="blocked",
                reason=f"acquisition_status={record.acquisition_status}",
                source_id=record.source_id,
                acquisition_status=record.acquisition_status,
            )

        existing = self.data.get("sources", {}).get(record.source_id)
        if not existing:
            return PlanDecision(
                action="process",
                reason="new source",
                source_id=record.source_id,
                acquisition_status=record.acquisition_status,
            )

        last_status = existing.get("last_status")

        if resume and last_status == "success":
            return PlanDecision(
                action="skip_cached",
                reason="previously succeeded and resume=True",
                source_id=record.source_id,
                existing_status=last_status,
                acquisition_status=existing.get("acquisition_status"),
            )

        if last_status == "failed" and retry_failed:
            return PlanDecision(
                action="retry_failed",
                reason="previously failed and retry_failed=True",
                source_id=record.source_id,
                existing_status=last_status,
                acquisition_status=existing.get("acquisition_status"),
            )

        return PlanDecision(
            action="process",
            reason="no skip rule matched",
            source_id=record.source_id,
            existing_status=last_status,
            acquisition_status=existing.get("acquisition_status"),
        )

    # --------------------------------------------------------
    # Manifest export
    # --------------------------------------------------------

    def to_manifest_rows(
        self,
        *,
        corpus_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        acquisition_status: Optional[Sequence[str]] = None,
        source_kind: Optional[Sequence[str]] = None,
        include_blocked: bool = False,
        include_failed: bool = False,
        include_duplicates: bool = False,
        limit: Optional[int] = None,
    ) -> List[ManifestRow]:
        wanted_tags = set(validate_tags(tags or []))
        wanted_acq = set(acquisition_status or [])
        wanted_kind = set(source_kind or [])

        for status in wanted_acq:
            validate_acquisition_status(status)

        for kind in wanted_kind:
            validate_source_kind(kind)

        rows: List[ManifestRow] = []

        for source_id in self.source_ids():
            record_dict = self.raw_record(source_id)

            if corpus_id and record_dict.get("corpus_id") != corpus_id:
                continue

            if branch_id and record_dict.get("branch_id") != branch_id:
                continue

            if wanted_tags and not wanted_tags.intersection(set(record_dict.get("tags") or [])):
                continue

            if wanted_acq and record_dict.get("acquisition_status") not in wanted_acq:
                continue

            if wanted_kind and record_dict.get("source_kind") not in wanted_kind:
                continue

            if not include_blocked and record_dict.get("acquisition_status") in BLOCKING_ACQUISITION_STATUSES:
                continue

            if not include_failed and record_dict.get("last_status") == "failed":
                continue

            if not include_duplicates and record_dict.get("duplicate_of"):
                continue

            rows.append(SourceRecord.from_dict(record_dict).to_manifest_row())

            if limit is not None and len(rows) >= limit:
                break

        return rows

    # --------------------------------------------------------
    # Source metadata for extract.json
    # --------------------------------------------------------

    def credibility_for_record(self, record_or_id: SourceRecord | str) -> JSONDict:
        record = (
            self.get_source_record(record_or_id)
            if isinstance(record_or_id, str)
            else record_or_id
        )

        metadata = dict(record.metadata or {})
        explicit = metadata.get("credibility")

        if isinstance(explicit, Mapping):
            score = explicit.get("score")
            tier = explicit.get("tier")
            rationale = explicit.get("rationale")

            if score is not None and tier:
                return {
                    "score": float(score),
                    "tier": str(tier),
                    "rationale": str(rationale or "Credibility supplied by manifest/source metadata."),
                }

        if metadata.get("credibility_score") is not None and metadata.get("credibility_tier"):
            return {
                "score": float(metadata["credibility_score"]),
                "tier": str(metadata["credibility_tier"]),
                "rationale": str(
                    metadata.get("credibility_rationale")
                    or "Credibility supplied by manifest/source metadata."
                ),
            }

        source_url = _metadata_get(metadata, "canonical_url", "source_url", "original_url", "url")
        domain = record.domain

        if not domain and source_url and is_url(str(source_url)):
            domain = infer_domain(str(source_url))

        high_match = _domain_matches(domain, HIGH_CREDIBILITY_DOMAINS)
        if high_match:
            return {
                "score": 0.9,
                "tier": "high",
                "rationale": f"Source domain matched high-credibility domain rule: {high_match}.",
            }

        medium_match = _domain_matches(domain, MEDIUM_CREDIBILITY_DOMAINS)
        if medium_match:
            return {
                "score": 0.7,
                "tier": "medium",
                "rationale": f"Source domain matched medium-credibility domain rule: {medium_match}.",
            }

        publisher = _metadata_get(metadata, "publisher", "site_name", "publication")
        if publisher:
            return {
                "score": 0.6,
                "tier": "medium",
                "rationale": "Source has explicit publisher metadata but no curated domain rule.",
            }

        return {
            "score": 0.5,
            "tier": "unknown",
            "rationale": "No curated credibility rule or publisher metadata available.",
        }

    def document_metadata_for(
        self,
        source_id: str,
        *,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> JSONDict:
        """
        Return the document metadata shape expected by extract.json.

        This is designed for the later patch where extract.py or mass_ingest.py
        merges registry metadata into extraction records.
        """

        record = self.get_source_record(source_id)
        metadata = dict(record.metadata or {})

        source_url = _metadata_get(metadata, "source_url", "original_url", "url")
        canonical_url = _metadata_get(metadata, "canonical_url")

        if source_url and not canonical_url and is_url(str(source_url)):
            try:
                canonical_url = canonicalize_url(str(source_url))
            except ContractError:
                canonical_url = None

        if not source_url and is_url(record.source):
            source_url = record.source

        if not canonical_url:
            canonical_url = record.canonical_source

        title = (
            _none_if_blank(_metadata_get(metadata, "title"))
            or record.title
            or _filename_title(record.local_path or record.source)
        )

        author = _none_if_blank(_metadata_get(metadata, "author", "byline", "creator"))
        publisher = (
            _none_if_blank(_metadata_get(metadata, "publisher", "site_name", "publication"))
            or record.domain
        )
        published_at = (
            _none_if_blank(_metadata_get(metadata, "published_at", "date", "publication_date"))
            or record.published_at
        )

        return {
            "document_id": document_id,
            "source_type": source_type or record.source_kind,
            "title": title,
            "author": author,
            "publisher": publisher,
            "published_at": published_at,
            "source_url": source_url,
            "canonical_url": canonical_url,
            "credibility": self.credibility_for_record(record),
        }

    def merge_document_metadata(
        self,
        source_id: str,
        existing_document: Mapping[str, Any],
    ) -> JSONDict:
        """
        Merge registry metadata into an existing extract.json document block.

        Merge rule:
        - registry wins for source_url, canonical_url, publisher, credibility
        - parser/existing metadata wins for title/author if registry lacks it
        - never overwrite a real value with null
        """

        existing = dict(existing_document or {})
        registry_doc = self.document_metadata_for(
            source_id,
            document_id=existing.get("document_id"),
            source_type=existing.get("source_type"),
        )

        merged = dict(existing)

        for key in ["source_url", "canonical_url", "publisher", "published_at"]:
            if registry_doc.get(key) not in (None, ""):
                merged[key] = registry_doc[key]

        for key in ["title", "author"]:
            if merged.get(key) in (None, "") and registry_doc.get(key) not in (None, ""):
                merged[key] = registry_doc[key]

        if registry_doc.get("credibility"):
            merged["credibility"] = registry_doc["credibility"]

        if not merged.get("source_type") and registry_doc.get("source_type"):
            merged["source_type"] = registry_doc["source_type"]

        return merged

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def summary(self) -> JSONDict:
        by_kind = Counter()
        by_acquisition = Counter()
        by_last_status = Counter()
        by_domain = Counter()
        by_tag = Counter()
        duplicates_total = 0
        blocked_total = 0
        failed_total = 0
        successful_total = 0

        for source_id in self.source_ids():
            record = self.raw_record(source_id)

            by_kind[str(record.get("source_kind") or "unknown")] += 1
            by_acquisition[str(record.get("acquisition_status") or "not_acquired")] += 1
            by_last_status[str(record.get("last_status") or "none")] += 1

            domain = record.get("domain")
            if domain:
                by_domain[str(domain)] += 1

            for tag in record.get("tags") or []:
                by_tag[str(tag)] += 1

            if record.get("duplicate_of"):
                duplicates_total += 1

            if record.get("acquisition_status") in BLOCKING_ACQUISITION_STATUSES:
                blocked_total += 1

            if record.get("last_status") == "failed":
                failed_total += 1

            if record.get("last_status") == "success":
                successful_total += 1

        return {
            "schema_version": 1,
            "registry_path": str(self.path),
            "sources_total": len(self.source_ids()),
            "sources_by_kind": dict(by_kind),
            "sources_by_acquisition_status": dict(by_acquisition),
            "sources_by_last_status": dict(by_last_status),
            "duplicates_total": duplicates_total,
            "blocked_total": blocked_total,
            "failed_total": failed_total,
            "successful_total": successful_total,
            "by_domain": dict(by_domain),
            "by_tag": dict(by_tag),
            "updated_at": utc_now(),
        }


# ============================================================
# Self-test
# ============================================================

def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> int:
    print("[source_registry self-test] starting")

    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        registry_path = tmp / "source_registry.json"

        # 1. Registry creates a new file if none exists.
        registry = SourceRegistry(registry_path)
        _assert(registry.summary()["sources_total"] == 0, "new registry should be empty")
        registry.save()
        _assert(registry_path.exists(), "registry file was not created")

        # 2. Local file registration with content_hash and stable source_id.
        file_a = tmp / "a.pdf"
        file_a.write_text("same content", encoding="utf-8")

        rec_a = registry.register_source(
            str(file_a),
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["myanmar", "hree"],
            acquisition_status="manual",
            acquisition_method="local_file",
            metadata={
                "title": "Conflict Economy: Myanmar Rare Earths",
                "publisher": "International Crisis Group",
                "source_url": "https://www.crisisgroup.org/asia/south-east-asia/myanmar/report.pdf",
                "published_at": "2024-01-01",
            },
        )

        _assert(rec_a.source_id.startswith("src_"), "source_id not created")
        _assert(rec_a.content_hash is not None, "content_hash not created")
        _assert(registry.has_source(rec_a.source_id), "source not registered")

        # 3. Two files with identical content are detected as duplicates.
        file_b = tmp / "b.pdf"
        file_b.write_text("same content", encoding="utf-8")

        rec_b = SourceRecord.from_source(
            str(file_b),
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["duplicate"],
            acquisition_status="manual",
            acquisition_method="local_file",
        )

        duplicate_of = registry.find_duplicate(rec_b)
        _assert(duplicate_of == rec_a.source_id, "duplicate content hash not detected")

        registry.upsert_source(rec_b)
        registry.mark_duplicate(rec_b.source_id, duplicate_of)

        # 4. Equivalent URLs canonicalize to same source_id.
        url_1 = "https://Reuters.com/world/asia/myanmar-rare-earths/?utm_source=x#section"
        url_2 = "https://reuters.com/world/asia/myanmar-rare-earths"

        rec_url_1 = registry.register_source(
            url_1,
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["reuters"],
            acquisition_status="blocked",
            acquisition_method="requests",
            metadata={"title": "Myanmar rare earths article"},
        )

        rec_url_2 = SourceRecord.from_source(
            url_2,
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["reuters"],
        )

        _assert(
            rec_url_1.source_id == rec_url_2.source_id,
            "equivalent URLs did not produce same source_id",
        )

        # 5. Acquisition status update.
        registry.update_acquisition(
            rec_url_1.source_id,
            acquisition_status="paywalled",
            acquisition_method="requests",
            error="HTTP 403",
        )

        updated_url = registry.get_source_record(rec_url_1.source_id)
        _assert(updated_url.acquisition_status == "paywalled", "acquisition status not updated")
        _assert(updated_url.last_error == "HTTP 403", "acquisition error not stored")

        # 6. Pipeline status updates.
        status_success = StatusRecord(
            source_id=rec_a.source_id,
            status="success",
            completed_stages=["acquire", "ingest", "extract", "postrag"],
            skipped_stages=[],
            run_id="run_test",
            corpus_id="eval1",
            branch_id="staging_eval1",
            started_at=utc_now(),
            finished_at=utc_now(),
            runtime_seconds=12.5,
            metrics={"table_blocks": 2},
        )

        registry.update_pipeline_status(
            rec_a.source_id,
            status_success,
            output_dir="runs/run_test/sources/src_x",
        )

        raw_a = registry.raw_record(rec_a.source_id)
        _assert(raw_a["last_status"] == "success", "last_status not updated")
        _assert(raw_a["last_run_id"] == "run_test", "last_run_id not updated")
        _assert(raw_a["last_ingested_at"] is not None, "last_ingested_at not updated")
        _assert(len(raw_a["runs"]) == 1, "run history not appended")

        # 7. Resume policy skips previously successful sources.
        plan_resume = registry.plan_source(rec_a, resume=True, force=False, dedupe=True)
        _assert(plan_resume.action == "skip_cached", f"resume policy failed: {plan_resume}")

        # 8. Force policy reprocesses previously successful sources.
        plan_force = registry.plan_source(rec_a, resume=True, force=True, dedupe=True)
        _assert(plan_force.action == "force_reprocess", "force policy failed")

        # 9. Retry-failed policy reprocesses failed sources.
        file_fail = tmp / "fail.txt"
        file_fail.write_text("bad source", encoding="utf-8")

        rec_fail = registry.register_source(
            str(file_fail),
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["fail"],
            acquisition_status="manual",
        )

        status_failed = StatusRecord(
            source_id=rec_fail.source_id,
            status="failed",
            completed_stages=["acquire"],
            skipped_stages=[],
            failed_stage="ingest",
            error="intentional failure",
            run_id="run_test",
            corpus_id="eval1",
            branch_id="staging_eval1",
        )

        registry.update_pipeline_status(rec_fail.source_id, status_failed)

        plan_retry = registry.plan_source(
            rec_fail,
            resume=True,
            force=False,
            dedupe=True,
            retry_failed=True,
        )
        _assert(plan_retry.action == "retry_failed", "retry_failed policy failed")

        # 10. Manifest export.
        rows = registry.to_manifest_rows(
            corpus_id="eval1",
            branch_id="staging_eval1",
            include_blocked=False,
            include_failed=True,
            include_duplicates=False,
        )

        _assert(all(isinstance(row, ManifestRow) for row in rows), "manifest rows invalid")
        _assert(any(row.source_id == rec_a.source_id for row in rows), "registered source missing from manifest export")
        _assert(all(row.acquisition_status != "paywalled" for row in rows), "blocked source leaked into manifest export")

        # 11. Summary counts.
        summary = registry.summary()
        _assert(summary["sources_total"] >= 4, "summary source count wrong")
        _assert(summary["duplicates_total"] == 1, "duplicate count wrong")
        _assert(summary["successful_total"] == 1, "success count wrong")
        _assert(summary["failed_total"] == 1, "failed count wrong")
        _assert(summary["by_tag"]["myanmar"] == 1, "tag count wrong")

        # 12. Document metadata enrichment for extract.json.
        doc_meta = registry.document_metadata_for(
            rec_a.source_id,
            document_id="doc_test",
            source_type="pdf",
        )

        _assert(doc_meta["document_id"] == "doc_test", "document_id not preserved")
        _assert(doc_meta["source_type"] == "pdf", "source_type not preserved")
        _assert(doc_meta["title"] == "Conflict Economy: Myanmar Rare Earths", "title not filled")
        _assert(doc_meta["publisher"] == "International Crisis Group", "publisher not filled")
        _assert(doc_meta["source_url"] is not None, "source_url not filled")
        _assert(doc_meta["canonical_url"] is not None, "canonical_url not filled")
        _assert(doc_meta["credibility"]["tier"] in {"high", "medium"}, "credibility not enriched")

        existing_extract_doc = {
            "document_id": "doc_test",
            "source_type": "pdf",
            "title": None,
            "author": None,
            "publisher": None,
            "published_at": None,
            "source_url": None,
            "canonical_url": None,
            "credibility": {
                "score": 0.5,
                "tier": "unknown",
                "rationale": "",
            },
        }

        merged_doc = registry.merge_document_metadata(rec_a.source_id, existing_extract_doc)
        _assert(merged_doc["title"] == "Conflict Economy: Myanmar Rare Earths", "merge did not fill title")
        _assert(merged_doc["publisher"] == "International Crisis Group", "merge did not fill publisher")
        _assert(merged_doc["credibility"]["tier"] in {"high", "medium"}, "merge did not fill credibility")

        # 13. Blocked plan.
        plan_blocked = registry.plan_source(
            updated_url,
            resume=True,
            force=False,
            dedupe=True,
            allow_blocked=False,
        )
        _assert(plan_blocked.action == "blocked", "blocked acquisition plan failed")

        plan_allow_blocked = registry.plan_source(
            updated_url,
            resume=True,
            force=False,
            dedupe=True,
            allow_blocked=True,
        )
        _assert(plan_allow_blocked.action in {"process", "skip_cached"}, "allow_blocked policy failed")

    print("[source_registry self-test] all tests passed")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent source registry for Lanthic Intelligence.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in source registry tests.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    print("source_registry.py defines SourceRegistry. Run with --self-test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())