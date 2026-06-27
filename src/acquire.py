#!/usr/bin/env python3
"""
Source acquisition layer for Lanthic Intelligence.

This module owns:
- local/manual source acquisition
- URL fetching into local artifacts
- acquisition status classification
- basic source metadata capture
- SourceRegistry updates
- acquisition.json records

It must not:
- perform web search
- extract claims
- call OpenAI
- call Neo4j
- run PostRAG
- run KG-IRAG/SARG

Run:
    python src/acquire.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from pipeline_contracts import (
    ContractError,
    SourceRecord,
    canonicalize_url,
    infer_domain,
    infer_mime_type,
    infer_source_kind,
    is_url,
    read_json,
    source_id_from_canonical,
    utc_now,
    validate_acquisition_status,
    validate_tags,
    write_json,
)

from source_registry import SourceRegistry


JSONDict = Dict[str, Any]


SUPPORTED_MIME_PREFIXES = (
    "text/html",
    "application/xhtml+xml",
    "application/pdf",
    "text/plain",
    "application/json",
    "text/csv",
    "application/csv",
)

SUPPORTED_EXTENSIONS = {
    ".html",
    ".htm",
    ".pdf",
    ".txt",
    ".json",
    ".csv",
}


PAYWALL_PATTERNS = (
    "subscribe to continue",
    "subscription required",
    "sign in to continue",
    "register to continue",
    "paywall",
    "premium article",
    "for subscribers",
)

BLOCK_PATTERNS = (
    "access denied",
    "request blocked",
    "temporarily unavailable",
    "enable javascript",
    "verify you are human",
    "captcha",
    "cloudflare",
)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; LanthicIntelligence/1.0; "
    "+https://example.local/research-agent)"
)


# ============================================================
# Result objects
# ============================================================

@dataclass
class FetchResponse:
    url: str
    status_code: Optional[int]
    headers: JSONDict = field(default_factory=dict)
    content: bytes = b""
    final_url: Optional[str] = None
    error: Optional[str] = None
    method: str = "requests"

    @property
    def ok(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300 and not self.error


@dataclass
class AcquisitionResult:
    source_id: str
    source: str
    canonical_source: str
    acquisition_status: str
    acquisition_method: str
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    error: Optional[str] = None
    metadata: JSONDict = field(default_factory=dict)
    acquired_at: str = field(default_factory=utc_now)

    def validate(self) -> "AcquisitionResult":
        validate_acquisition_status(self.acquisition_status)
        return self

    def to_dict(self) -> JSONDict:
        self.validate()
        return asdict(self)


# ============================================================
# Basic helpers
# ============================================================

def _none_if_blank(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _headers_to_dict(headers: Any) -> JSONDict:
    out: JSONDict = {}

    if not headers:
        return out

    if hasattr(headers, "items"):
        for key, value in headers.items():
            out[str(key).lower()] = str(value)

    return out


def _content_type(headers: Mapping[str, Any]) -> Optional[str]:
    for key in ("content-type", "Content-Type"):
        value = headers.get(key)
        if value:
            return str(value).split(";")[0].strip().lower()

    return None


def _charset(headers: Mapping[str, Any]) -> str:
    content_type = ""

    for key in ("content-type", "Content-Type"):
        if headers.get(key):
            content_type = str(headers[key])
            break

    match = re.search(r"charset=([A-Za-z0-9_.-]+)", content_type, flags=re.I)
    if match:
        return match.group(1)

    return "utf-8"


def _decode_content(content: bytes, headers: Mapping[str, Any]) -> str:
    charset = _charset(headers)

    try:
        return content.decode(charset, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _is_supported_mime(mime_type: Optional[str], source: str) -> bool:
    if mime_type:
        mime_type = mime_type.lower().strip()
        if any(mime_type.startswith(prefix) for prefix in SUPPORTED_MIME_PREFIXES):
            return True

    suffix = Path(urlparse(source).path if is_url(source) else source).suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def _extension_for(mime_type: Optional[str], source: str) -> str:
    source_suffix = Path(urlparse(source).path if is_url(source) else source).suffix.lower()

    if source_suffix in SUPPORTED_EXTENSIONS:
        if source_suffix == ".htm":
            return ".html"
        return source_suffix

    mime_type = (mime_type or "").lower().strip()

    if mime_type in {"text/html", "application/xhtml+xml"}:
        return ".html"
    if mime_type == "application/pdf":
        return ".pdf"
    if mime_type == "application/json":
        return ".json"
    if mime_type in {"text/csv", "application/csv"}:
        return ".csv"
    if mime_type.startswith("text/plain"):
        return ".txt"

    return ".bin"


def _safe_filename(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return text or "source"


def _source_dir(output_dir: Path, source_id: str) -> Path:
    return output_dir / source_id


def _acquisition_json_path(output_dir: Path, source_id: str) -> Path:
    return _source_dir(output_dir, source_id) / "acquisition.json"


def _existing_acquired_file(output_dir: Path, source_id: str) -> Optional[Path]:
    directory = _source_dir(output_dir, source_id)

    if not directory.exists():
        return None

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "acquisition.json" and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return path

    return None


def _write_acquisition_record(output_dir: Path, result: AcquisitionResult) -> None:
    path = _acquisition_json_path(output_dir, result.source_id)
    write_json(path, result.to_dict())


# ============================================================
# Metadata extraction
# ============================================================

def _extract_attrs(tag_text: str) -> JSONDict:
    attrs: JSONDict = {}

    for match in re.finditer(
        r"""([A-Za-z_:.-]+)\s*=\s*["']([^"']*)["']""",
        tag_text,
        flags=re.I,
    ):
        attrs[match.group(1).lower()] = match.group(2).strip()

    return attrs


def _first_meta_value(meta: Mapping[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        value = meta.get(key.lower())
        if value:
            return value
    return None


def extract_html_metadata(html: str) -> JSONDict:
    meta: Dict[str, str] = {}

    for match in re.finditer(r"<meta\s+([^>]+)>", html, flags=re.I):
        attrs = _extract_attrs(match.group(1))
        key = attrs.get("property") or attrs.get("name") or attrs.get("itemprop")
        value = attrs.get("content")

        if key and value:
            meta[key.lower()] = value.strip()

    title = _first_meta_value(
        meta,
        "og:title",
        "twitter:title",
        "title",
        "headline",
        "dc.title",
        "citation_title",
    )

    if not title:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()

    author = _first_meta_value(
        meta,
        "author",
        "article:author",
        "dc.creator",
        "citation_author",
        "byl",
    )

    publisher = _first_meta_value(
        meta,
        "og:site_name",
        "application-name",
        "dc.publisher",
        "citation_publisher",
        "publisher",
    )

    published_at = _first_meta_value(
        meta,
        "article:published_time",
        "article:modified_time",
        "date",
        "dc.date",
        "dc.date.issued",
        "citation_publication_date",
        "pubdate",
    )

    canonical_url = _first_meta_value(meta, "og:url", "twitter:url", "canonical")

    return {
        "title": _none_if_blank(title),
        "author": _none_if_blank(author),
        "publisher": _none_if_blank(publisher),
        "published_at": _none_if_blank(published_at),
        "canonical_url": _none_if_blank(canonical_url),
    }


def detect_paywall_or_block(html: str) -> Optional[str]:
    lowered = html.lower()

    for pattern in PAYWALL_PATTERNS:
        if pattern in lowered:
            return "paywalled"

    for pattern in BLOCK_PATTERNS:
        if pattern in lowered:
            return "blocked"

    return None


def filename_title(source: str) -> Optional[str]:
    if is_url(source):
        name = Path(urlparse(source).path).name
    else:
        name = Path(source).name

    if not name:
        return None

    title = Path(name).stem.replace("_", " ").replace("-", " ").strip()
    return title or None


def base_metadata_for_source(
    source: str,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> JSONDict:
    extra = dict(extra_metadata or {})

    source_url = extra.get("source_url") or extra.get("original_url") or extra.get("url")
    canonical_url = extra.get("canonical_url")

    if not source_url and is_url(source):
        source_url = source

    if not canonical_url and source_url and is_url(str(source_url)):
        try:
            canonical_url = canonicalize_url(str(source_url))
        except ContractError:
            canonical_url = None

    domain = None
    if canonical_url and is_url(str(canonical_url)):
        domain = infer_domain(str(canonical_url))
    elif source_url and is_url(str(source_url)):
        domain = infer_domain(str(source_url))
    elif is_url(source):
        domain = infer_domain(source)

    metadata = {
        "title": extra.get("title") or filename_title(source),
        "author": extra.get("author") or extra.get("byline") or extra.get("creator"),
        "publisher": extra.get("publisher") or extra.get("site_name") or extra.get("publication"),
        "published_at": extra.get("published_at") or extra.get("date") or extra.get("publication_date"),
        "source_url": source_url,
        "original_url": extra.get("original_url"),
        "canonical_url": canonical_url,
        "domain": domain,
        "mime_type": extra.get("mime_type") or infer_mime_type(source),
    }

    for key, value in extra.items():
        if value not in (None, ""):
            metadata[key] = value

    return {key: value for key, value in metadata.items() if value not in (None, "")}


# ============================================================
# Fetchers
# ============================================================

def fetch_url_urllib(
    url: str,
    *,
    timeout: int = 20,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResponse:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain,application/json,text/csv,*/*;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            headers = _headers_to_dict(response.headers)
            final_url = response.geturl()

            return FetchResponse(
                url=url,
                status_code=int(response.status),
                headers=headers,
                content=content,
                final_url=final_url,
                method="requests",
            )

    except urllib.error.HTTPError as exc:
        try:
            content = exc.read()
        except Exception:
            content = b""

        return FetchResponse(
            url=url,
            status_code=int(exc.code),
            headers=_headers_to_dict(exc.headers),
            content=content,
            final_url=exc.geturl(),
            error=f"HTTPError: {exc.code} {exc.reason}",
            method="requests",
        )

    except urllib.error.URLError as exc:
        return FetchResponse(
            url=url,
            status_code=None,
            headers={},
            content=b"",
            final_url=None,
            error=f"URLError: {exc.reason}",
            method="requests",
        )

    except TimeoutError as exc:
        return FetchResponse(
            url=url,
            status_code=None,
            headers={},
            content=b"",
            final_url=None,
            error=f"TimeoutError: {exc}",
            method="requests",
        )

    except Exception as exc:
        return FetchResponse(
            url=url,
            status_code=None,
            headers={},
            content=b"",
            final_url=None,
            error=f"{type(exc).__name__}: {exc}",
            method="requests",
        )


def fetch_url_browser(
    url: str,
    *,
    timeout: int = 20,
) -> FetchResponse:
    """
    Optional Playwright-based renderer.

    This is intentionally optional. If Playwright is unavailable, acquisition
    records a fetch_failed result rather than crashing.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return FetchResponse(
            url=url,
            status_code=None,
            headers={},
            content=b"",
            final_url=None,
            error=f"browser acquisition requested but Playwright is unavailable: {exc}",
            method="browser",
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=DEFAULT_USER_AGENT)
            response = page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            html = page.content()
            final_url = page.url
            status_code = response.status if response else None
            headers = response.headers if response else {}
            browser.close()

        return FetchResponse(
            url=url,
            status_code=status_code,
            headers={str(k).lower(): str(v) for k, v in dict(headers).items()},
            content=html.encode("utf-8", errors="replace"),
            final_url=final_url,
            method="browser",
        )

    except Exception as exc:
        return FetchResponse(
            url=url,
            status_code=None,
            headers={},
            content=b"",
            final_url=None,
            error=f"{type(exc).__name__}: {exc}",
            method="browser",
        )


def classify_fetch_response(response: FetchResponse) -> str:
    if response.status_code in {401, 403}:
        return "blocked"

    if response.status_code == 402:
        return "paywalled"

    if response.status_code in {404, 410}:
        return "not_found"

    if response.status_code is None:
        return "fetch_failed"

    if not (200 <= response.status_code < 300):
        return "fetch_failed"

    mime_type = _content_type(response.headers)

    if not _is_supported_mime(mime_type, response.final_url or response.url):
        return "unsupported"

    if mime_type in {"text/html", "application/xhtml+xml"} or response.content[:100].lower().find(b"<html") >= 0:
        html = _decode_content(response.content, response.headers)
        detected = detect_paywall_or_block(html)
        if detected:
            return detected

    return "acquired"


# ============================================================
# Acquisition implementation
# ============================================================

def acquire_local_file(
    source: str,
    *,
    registry: SourceRegistry,
    output_dir: Path,
    corpus_id: str,
    branch_id: str,
    tags: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
    copy_local: bool = False,
    force: bool = False,
) -> AcquisitionResult:
    path = Path(source).expanduser()

    if not path.exists() or not path.is_file():
        source_record = registry.register_source(
            source,
            corpus_id=corpus_id,
            branch_id=branch_id,
            tags=tags,
            acquisition_status="not_found",
            acquisition_method="local_file",
            metadata=base_metadata_for_source(source, extra_metadata=metadata),
        )

        result = AcquisitionResult(
            source_id=source_record.source_id,
            source=source,
            canonical_source=source_record.canonical_source,
            acquisition_status="not_found",
            acquisition_method="local_file",
            error=f"Local file not found: {source}",
            metadata=source_record.metadata,
        )

        registry.update_acquisition(
            source_record.source_id,
            acquisition_status="not_found",
            acquisition_method="local_file",
            error=result.error,
            metadata=result.metadata,
        )
        _write_acquisition_record(output_dir, result)
        return result

    base_metadata = base_metadata_for_source(str(path), extra_metadata=metadata)

    local_path = str(path.resolve())

    source_record = registry.register_source(
        str(path),
        corpus_id=corpus_id,
        branch_id=branch_id,
        tags=tags,
        acquisition_status="manual",
        acquisition_method="local_file",
        local_path=local_path,
        metadata=base_metadata,
    )

    acquired_path = Path(local_path)

    if copy_local:
        ext = acquired_path.suffix or ".bin"
        target_dir = _source_dir(output_dir, source_record.source_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"source{ext}"

        if force or not target.exists():
            shutil.copy2(acquired_path, target)

        acquired_path = target

    updated = registry.update_acquisition(
        source_record.source_id,
        acquisition_status="manual",
        acquisition_method="local_file_copy" if copy_local else "local_file",
        local_path=str(acquired_path),
        metadata=base_metadata,
    )

    result = AcquisitionResult(
        source_id=updated.source_id,
        source=str(path),
        canonical_source=updated.canonical_source,
        acquisition_status="manual",
        acquisition_method="local_file_copy" if copy_local else "local_file",
        local_path=str(acquired_path),
        mime_type=updated.mime_type,
        metadata=updated.metadata,
    )

    _write_acquisition_record(output_dir, result)
    return result


def acquire_url(
    source: str,
    *,
    registry: SourceRegistry,
    output_dir: Path,
    corpus_id: str,
    branch_id: str,
    tags: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
    force: bool = False,
    use_browser: bool = False,
    timeout: int = 20,
    fetcher: Optional[Callable[[str], FetchResponse]] = None,
) -> AcquisitionResult:
    canonical = canonicalize_url(source)

    base_metadata = base_metadata_for_source(
        canonical,
        extra_metadata={
            **dict(metadata or {}),
            "source_url": source,
            "canonical_url": canonical,
        },
    )

    source_record = registry.register_source(
        canonical,
        corpus_id=corpus_id,
        branch_id=branch_id,
        tags=tags,
        acquisition_status="not_acquired",
        acquisition_method=None,
        metadata=base_metadata,
    )

    existing_file = _existing_acquired_file(output_dir, source_record.source_id)

    if existing_file and not force:
        updated = registry.update_acquisition(
            source_record.source_id,
            acquisition_status="cached",
            acquisition_method="cache",
            local_path=str(existing_file),
            metadata={
                **base_metadata,
                "mime_type": infer_mime_type(str(existing_file)),
            },
        )

        result = AcquisitionResult(
            source_id=updated.source_id,
            source=source,
            canonical_source=updated.canonical_source,
            acquisition_status="cached",
            acquisition_method="cache",
            local_path=str(existing_file),
            mime_type=updated.mime_type,
            final_url=updated.metadata.get("final_url"),
            metadata=updated.metadata,
        )

        _write_acquisition_record(output_dir, result)
        return result

    if fetcher is not None:
        response = fetcher(canonical)
    else:
        response = fetch_url_urllib(canonical, timeout=timeout)

    status = classify_fetch_response(response)

    if status != "acquired" and use_browser:
        browser_response = fetch_url_browser(canonical, timeout=timeout)
        browser_status = classify_fetch_response(browser_response)

        if browser_status == "acquired" or status in {"fetch_failed", "blocked"}:
            response = browser_response
            status = browser_status

    mime_type = _content_type(response.headers)
    final_url = response.final_url or canonical

    metadata_out: JSONDict = {
        **base_metadata,
        "source_url": source,
        "canonical_url": canonical,
        "final_url": final_url,
        "domain": infer_domain(final_url) if is_url(final_url) else infer_domain(canonical),
        "http_status": response.status_code,
        "mime_type": mime_type,
        "fetched_at": utc_now(),
        "acquisition_method": response.method,
    }

    local_path: Optional[str] = None
    error = response.error

    if status == "acquired":
        if mime_type in {"text/html", "application/xhtml+xml"} or response.content[:100].lower().find(b"<html") >= 0:
            html = _decode_content(response.content, response.headers)
            html_metadata = extract_html_metadata(html)

            metadata_out.update(
                {
                    key: value
                    for key, value in html_metadata.items()
                    if value not in (None, "")
                }
            )

            if metadata_out.get("canonical_url") and is_url(str(metadata_out["canonical_url"])):
                try:
                    metadata_out["canonical_url"] = canonicalize_url(str(metadata_out["canonical_url"]))
                except ContractError:
                    metadata_out["canonical_url"] = canonical

        ext = _extension_for(mime_type, final_url)
        target_dir = _source_dir(output_dir, source_record.source_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"source{ext}"
        target.write_bytes(response.content)
        local_path = str(target)

    elif status == "unsupported":
        error = error or f"Unsupported content type: {mime_type or 'unknown'}"

    elif status == "blocked":
        error = error or f"Blocked or access denied. HTTP status: {response.status_code}"

    elif status == "paywalled":
        error = error or f"Paywall or subscription page detected. HTTP status: {response.status_code}"

    elif status == "not_found":
        error = error or f"Source not found. HTTP status: {response.status_code}"

    elif status == "fetch_failed":
        error = error or f"Fetch failed. HTTP status: {response.status_code}"

    updated = registry.update_acquisition(
        source_record.source_id,
        acquisition_status=status,
        acquisition_method=response.method,
        local_path=local_path,
        error=error,
        metadata=metadata_out,
    )

    result = AcquisitionResult(
        source_id=updated.source_id,
        source=source,
        canonical_source=updated.canonical_source,
        acquisition_status=status,
        acquisition_method=response.method,
        local_path=local_path,
        mime_type=mime_type,
        http_status=response.status_code,
        final_url=final_url,
        error=error,
        metadata=updated.metadata,
    )

    _write_acquisition_record(output_dir, result)
    return result


def acquire_source(
    source: str,
    *,
    registry: SourceRegistry,
    output_dir: Path,
    corpus_id: str,
    branch_id: str,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    force: bool = False,
    use_browser: bool = False,
    timeout: int = 20,
    copy_local: bool = False,
    fetcher: Optional[Callable[[str], FetchResponse]] = None,
) -> AcquisitionResult:
    tags_clean = validate_tags(tags or [])

    if is_url(source):
        return acquire_url(
            source,
            registry=registry,
            output_dir=output_dir,
            corpus_id=corpus_id,
            branch_id=branch_id,
            tags=tags_clean,
            metadata=metadata,
            force=force,
            use_browser=use_browser,
            timeout=timeout,
            fetcher=fetcher,
        )

    return acquire_local_file(
        source,
        registry=registry,
        output_dir=output_dir,
        corpus_id=corpus_id,
        branch_id=branch_id,
        tags=tags_clean,
        metadata=metadata,
        copy_local=copy_local,
        force=force,
    )


def acquire_many(
    sources: Sequence[str],
    *,
    registry: SourceRegistry,
    output_dir: Path,
    corpus_id: str,
    branch_id: str,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    force: bool = False,
    use_browser: bool = False,
    timeout: int = 20,
    copy_local: bool = False,
    fetcher: Optional[Callable[[str], FetchResponse]] = None,
) -> List[AcquisitionResult]:
    results: List[AcquisitionResult] = []

    for i, source in enumerate(sources, start=1):
        print(f"[acquire] {i}/{len(sources)} {source}")

        try:
            result = acquire_source(
                source,
                registry=registry,
                output_dir=output_dir,
                corpus_id=corpus_id,
                branch_id=branch_id,
                tags=tags,
                metadata=metadata,
                force=force,
                use_browser=use_browser,
                timeout=timeout,
                copy_local=copy_local,
                fetcher=fetcher,
            )
            results.append(result)
            print(f"[acquire] {result.acquisition_status}: {source}")

        except Exception as exc:
            canonical_source = source
            try:
                if is_url(source):
                    canonical_source = canonicalize_url(source)
            except Exception:
                pass

            source_id = source_id_from_canonical(canonical_source)

            result = AcquisitionResult(
                source_id=source_id,
                source=source,
                canonical_source=canonical_source,
                acquisition_status="fetch_failed",
                acquisition_method="internal_error",
                error=f"{type(exc).__name__}: {exc}",
                metadata={"traceback": traceback.format_exc()},
            )
            results.append(result)
            _write_acquisition_record(output_dir, result)
            print(f"[acquire] fetch_failed: {source}: {exc}")

    registry.save()
    return results


# ============================================================
# Sources file
# ============================================================

def read_sources_file(path: Path) -> List[str]:
    sources: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        sources.append(line)

    return sources


def acquisition_summary(results: Sequence[AcquisitionResult]) -> JSONDict:
    counts: Dict[str, int] = {}

    for result in results:
        counts[result.acquisition_status] = counts.get(result.acquisition_status, 0) + 1

    return {
        "sources_total": len(results),
        "by_acquisition_status": counts,
        "results": [result.to_dict() for result in results],
    }


# ============================================================
# Self-test
# ============================================================

def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fake_html_response(url: str) -> FetchResponse:
    html = """
    <html>
      <head>
        <title>Rare Earth Test Article</title>
        <meta property="og:site_name" content="Example Publisher">
        <meta name="author" content="Jane Reporter">
        <meta property="article:published_time" content="2025-01-02">
        <meta property="og:url" content="https://example.com/reports/rare-earth-test">
      </head>
      <body>Rare earth supply chain evidence.</body>
    </html>
    """.encode("utf-8")

    return FetchResponse(
        url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=html,
        final_url=url,
        method="fake",
    )


def _fake_pdf_response(url: str) -> FetchResponse:
    return FetchResponse(
        url=url,
        status_code=200,
        headers={"content-type": "application/pdf"},
        content=b"%PDF-1.4 fake pdf bytes",
        final_url=url,
        method="fake",
    )


def _fake_403_response(url: str) -> FetchResponse:
    return FetchResponse(
        url=url,
        status_code=403,
        headers={"content-type": "text/html"},
        content=b"<html>Access denied</html>",
        final_url=url,
        error="HTTPError: 403 Forbidden",
        method="fake",
    )


def _fake_404_response(url: str) -> FetchResponse:
    return FetchResponse(
        url=url,
        status_code=404,
        headers={"content-type": "text/html"},
        content=b"<html>Not found</html>",
        final_url=url,
        error="HTTPError: 404 Not Found",
        method="fake",
    )


def _fake_timeout_response(url: str) -> FetchResponse:
    return FetchResponse(
        url=url,
        status_code=None,
        headers={},
        content=b"",
        final_url=None,
        error="TimeoutError: test timeout",
        method="fake",
    )


def _fake_unsupported_response(url: str) -> FetchResponse:
    return FetchResponse(
        url=url,
        status_code=200,
        headers={"content-type": "application/octet-stream"},
        content=b"\x00\x01\x02",
        final_url=url,
        method="fake",
    )


def run_self_test() -> int:
    print("[acquire self-test] starting")

    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        registry_path = tmp / "source_registry.json"
        output_dir = tmp / "acquired"
        registry = SourceRegistry(registry_path)

        # 1. Local file acquisition.
        pdf = tmp / "Conflict_Economy_Myanmar_Rare_Earth_ENG.pdf"
        pdf.write_text("local pdf placeholder", encoding="utf-8")

        local_result = acquire_source(
            str(pdf),
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["myanmar", "hree"],
        )

        _assert(local_result.acquisition_status == "manual", "local file should be manual")
        _assert(local_result.local_path is not None, "local path not preserved")
        _assert(registry.get_source_record(local_result.source_id).local_path is not None, "registry local_path missing")

        # 2. Saved HTML with original_url metadata preserves local_path + source_url.
        saved_html = tmp / "reuters_saved.html"
        saved_html.write_text("<html><title>Saved Reuters Article</title></html>", encoding="utf-8")

        saved_result = acquire_source(
            str(saved_html),
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            tags=["reuters"],
            metadata={
                "original_url": "https://www.reuters.com/world/asia/myanmar-rare-earths/",
                "publisher": "Reuters",
                "title": "Saved Reuters Article",
            },
        )

        saved_doc_meta = registry.document_metadata_for(saved_result.source_id, document_id="doc_saved")
        _assert(saved_result.acquisition_status == "manual", "saved HTML should be manual")
        _assert(saved_doc_meta["source_url"] == "https://www.reuters.com/world/asia/myanmar-rare-earths/", "source_url not preserved")
        _assert(saved_doc_meta["canonical_url"] is not None, "canonical_url not filled")
        _assert(saved_doc_meta["publisher"] == "Reuters", "publisher not preserved")

        # 3. Equivalent URLs canonicalize to same source_id.
        result_url_1 = acquire_source(
            "https://Example.com/reports/rare-earth-test?utm_source=x#frag",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_html_response,
        )

        result_url_2 = acquire_source(
            "https://example.com/reports/rare-earth-test",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_html_response,
        )

        _assert(result_url_1.source_id == result_url_2.source_id, "equivalent URLs did not share source_id")

        # 4. Mocked successful HTML writes source.html and metadata.
        html_result = result_url_1
        _assert(html_result.acquisition_status == "acquired", "HTML fetch should be acquired")
        _assert(html_result.local_path is not None and html_result.local_path.endswith(".html"), "HTML artifact missing")
        _assert(Path(html_result.local_path).exists(), "HTML artifact file does not exist")
        _assert(html_result.metadata.get("title") == "Rare Earth Test Article", "HTML title metadata missing")
        _assert(html_result.metadata.get("publisher") == "Example Publisher", "HTML publisher metadata missing")
        _assert(html_result.metadata.get("author") == "Jane Reporter", "HTML author metadata missing")

        # 5. Mocked successful PDF writes source.pdf.
        pdf_url_result = acquire_source(
            "https://example.com/report.pdf",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_pdf_response,
            force=True,
        )

        _assert(pdf_url_result.acquisition_status == "acquired", "PDF fetch should be acquired")
        _assert(pdf_url_result.local_path is not None and pdf_url_result.local_path.endswith(".pdf"), "PDF artifact missing")
        _assert(Path(pdf_url_result.local_path).exists(), "PDF artifact file does not exist")

        # 6. Mocked 403 -> blocked.
        blocked_result = acquire_source(
            "https://reuters.com/blocked-article",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_403_response,
            force=True,
        )

        _assert(blocked_result.acquisition_status == "blocked", "403 should be blocked")
        _assert(blocked_result.error is not None, "blocked error missing")

        # 7. Mocked 404 -> not_found.
        not_found_result = acquire_source(
            "https://example.com/missing",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_404_response,
            force=True,
        )

        _assert(not_found_result.acquisition_status == "not_found", "404 should be not_found")

        # 8. Mocked timeout -> fetch_failed.
        timeout_result = acquire_source(
            "https://example.com/timeout",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_timeout_response,
            force=True,
        )

        _assert(timeout_result.acquisition_status == "fetch_failed", "timeout should be fetch_failed")

        # 9. Unsupported content type -> unsupported.
        unsupported_result = acquire_source(
            "https://example.com/blob",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_unsupported_response,
            force=True,
        )

        _assert(unsupported_result.acquisition_status == "unsupported", "unsupported MIME should be unsupported")

        # 10. Re-acquiring an already saved URL returns cached unless force.
        cached_result = acquire_source(
            "https://example.com/reports/rare-earth-test",
            registry=registry,
            output_dir=output_dir,
            corpus_id="eval1",
            branch_id="staging_eval1",
            fetcher=_fake_html_response,
            force=False,
        )

        _assert(cached_result.acquisition_status == "cached", "second URL acquisition should be cached")

        # 11. Registry updated after every attempt.
        summary = registry.summary()
        _assert(summary["sources_total"] >= 7, "registry source count too low")
        _assert(summary["sources_by_acquisition_status"].get("blocked", 0) >= 1, "blocked not in registry summary")
        _assert(summary["sources_by_acquisition_status"].get("not_found", 0) >= 1, "not_found not in registry summary")
        _assert(summary["sources_by_acquisition_status"].get("fetch_failed", 0) >= 1, "fetch_failed not in registry summary")
        _assert(summary["sources_by_acquisition_status"].get("unsupported", 0) >= 1, "unsupported not in registry summary")

        # 12. acquisition.json written.
        acquisition_json = _acquisition_json_path(output_dir, html_result.source_id)
        _assert(acquisition_json.exists(), "acquisition.json not written")
        acquisition_data = read_json(acquisition_json)
        _assert(acquisition_data["source_id"] == html_result.source_id, "acquisition.json wrong source_id")

    print("[acquire self-test] all tests passed")
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
        extra = json.loads(args.metadata_json)
        if not isinstance(extra, Mapping):
            raise SystemExit("--metadata-json must decode to a JSON object.")
        metadata.update(dict(extra))

    if args.metadata_file:
        extra = read_json(args.metadata_file)
        if not isinstance(extra, Mapping):
            raise SystemExit("--metadata-file must contain a JSON object.")
        metadata.update(dict(extra))

    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire local files or URLs into registry-backed local artifacts.")

    parser.add_argument("--self-test", action="store_true")

    parser.add_argument("--source", default=None, help="Single source path or URL to acquire.")
    parser.add_argument("--sources-file", type=Path, default=None, help="Text file of source paths/URLs, one per line.")

    parser.add_argument("--registry", type=Path, default=Path("data/source_registry.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/acquired"))

    parser.add_argument("--corpus-id", default="default_corpus")
    parser.add_argument("--branch-id", default="staging")
    parser.add_argument("--tag", action="append", default=[])

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-local", action="store_true")
    parser.add_argument("--use-browser", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)

    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--published-at", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--original-url", default=None)
    parser.add_argument("--canonical-url", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)

    parser.add_argument("--summary-output", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.source and not args.sources_file:
        raise SystemExit("Provide --source or --sources-file.")

    sources: List[str] = []

    if args.source:
        sources.append(args.source)

    if args.sources_file:
        sources.extend(read_sources_file(args.sources_file))

    registry = SourceRegistry(args.registry)
    metadata = parse_metadata_args(args)

    results = acquire_many(
        sources,
        registry=registry,
        output_dir=args.output_dir,
        corpus_id=args.corpus_id,
        branch_id=args.branch_id,
        tags=args.tag,
        metadata=metadata,
        force=args.force,
        use_browser=args.use_browser,
        timeout=args.timeout,
        copy_local=args.copy_local,
    )

    summary = acquisition_summary(results)

    if args.summary_output:
        write_json(args.summary_output, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())