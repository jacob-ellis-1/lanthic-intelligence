#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
import trafilatura

try:
    from .document import (
        DocumentMetadata,
        HTMLWebpageDocument,
        SourceCredibility,
        SourceCredibilityTier,
        TextBlock,
        stable_id,
    )
except ImportError:
    from document import (
        DocumentMetadata,
        HTMLWebpageDocument,
        SourceCredibility,
        SourceCredibilityTier,
        TextBlock,
        stable_id,
    )


JSONDict = Dict[str, Any]


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


# ============================================================
# Fetching / cleaning
# ============================================================

def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_clean_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )

    if not text or not text.strip():
        raise ValueError(f"trafilatura failed to extract article text from {url}")

    return normalize_article_text(text)


def extract_metadata(html: str, url: str) -> DocumentMetadata:
    metadata = trafilatura.extract_metadata(html)

    title = None
    author = None
    publisher = None
    published_at = None
    language = None
    canonical_url = url

    if metadata is not None:
        title = metadata.title
        author = metadata.author
        publisher = metadata.sitename
        published_at = metadata.date
        language = metadata.language
        canonical_url = metadata.url or url

    return DocumentMetadata(
        title=title,
        author=author,
        publisher=publisher,
        published_at=published_at,
        language=language,
        source_url=url,
        canonical_url=canonical_url,
    )


def normalize_article_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# Source credibility
# ============================================================

def assess_source_credibility(url: str, metadata: DocumentMetadata) -> SourceCredibility:
    domain = urlparse(url).netloc.lower()
    domain = domain.removeprefix("www.")

    high_credibility_domains = {
        "csis.org",
        "reuters.com",
        "apnews.com",
        "sec.gov",
        "bis.doc.gov",
        "commerce.gov",
        "energy.gov",
        "usgs.gov",
        "imf.org",
        "worldbank.org",
        "iea.org",
    }

    medium_credibility_suffixes = (
        ".edu",
        ".gov",
        ".int",
        ".org",
    )

    if domain in high_credibility_domains:
        return SourceCredibility(
            score=0.92,
            tier=SourceCredibilityTier.HIGH,
            rationale=f"{domain} is treated as a high-credibility institutional/source domain.",
            factors={
                "domain_reputation": 0.9,
                "institutional_source": 0.9,
                "publisher": metadata.publisher,
            },
        )

    if domain.endswith(medium_credibility_suffixes):
        return SourceCredibility(
            score=0.75,
            tier=SourceCredibilityTier.MEDIUM,
            rationale=f"{domain} has an institutional-looking domain suffix.",
            factors={
                "domain_suffix": domain,
                "publisher": metadata.publisher,
            },
        )

    return SourceCredibility(
        score=0.5,
        tier=SourceCredibilityTier.UNKNOWN,
        rationale="No credibility heuristic matched; source credibility is unknown.",
        factors={
            "domain": domain,
            "publisher": metadata.publisher,
        },
    )


# ============================================================
# Paragraph spans
# ============================================================

def split_into_paragraph_spans(text: str) -> List[JSONDict]:
    """
    Split cleaned article text into paragraph-like units while preserving
    deterministic character spans.

    Important: trafilatura often returns article text with single newlines,
    especially for Q/A articles. So we split on non-empty lines, not only on
    blank-line paragraph boundaries.
    """
    paragraphs: List[JSONDict] = []

    for match in re.finditer(r"[^\n]+", text):
        paragraph = match.group(0).strip()

        if not paragraph:
            continue

        # Skip tiny navigation-ish fragments, but keep Q1/A1 style headings.
        if len(paragraph) < 3:
            continue

        paragraphs.append({
            "paragraph_id": len(paragraphs),
            "text": paragraph,
            "start_char": match.start(),
            "end_char": match.end(),
        })

    return paragraphs


# ============================================================
# Ollama semantic chunking
# ============================================================

def build_chunking_prompt(paragraphs: Sequence[JSONDict]) -> str:
    paragraph_listing = "\n\n".join(
        f"[{p['paragraph_id']}]\n{p['text']}"
        for p in paragraphs
    )

    return f"""
You are segmenting an article into coherent evidence blocks.

You will receive numbered paragraphs.
Group all paragraphs into contiguous semantic chunks.

Rules:
- Every paragraph id must appear exactly once.
- Chunks must be contiguous ranges.
- Preserve original order.
- Do not omit introductory or concluding paragraphs.
- Do not create relevance scores.
- Do not decide whether a paragraph is important.
- The goal is complete document coverage, not top-k selection.
- Prefer chunks of 2-6 paragraphs unless a section naturally requires otherwise.

Return only valid JSON in this exact shape:
{{
  "chunks": [
    {{
      "title": "brief semantic title",
      "start_paragraph": 0,
      "end_paragraph": 3,
      "rationale": "why these paragraphs belong together"
    }}
  ]
}}

Paragraphs:
{paragraph_listing}
""".strip()


def call_ollama_json(
    prompt: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
) -> JSONDict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }

    response = requests.post(ollama_url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    raw = data.get("response", "")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(extract_json_object(raw))


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in Ollama response:\n{text}")

    return text[start:end + 1]


def deterministic_chunk_ranges(
    paragraphs: Sequence[JSONDict],
    *,
    target_paragraphs_per_chunk: int = 4,
) -> List[JSONDict]:
    ranges: List[JSONDict] = []

    for start in range(0, len(paragraphs), target_paragraphs_per_chunk):
        end = min(start + target_paragraphs_per_chunk - 1, len(paragraphs) - 1)

        ranges.append({
            "title": f"Paragraphs {start}-{end}",
            "start_paragraph": start,
            "end_paragraph": end,
            "rationale": "Deterministic fallback chunking.",
        })

    return ranges


def validate_and_normalize_chunk_ranges(
    raw_chunks: Sequence[JSONDict],
    paragraphs: Sequence[JSONDict],
) -> List[JSONDict]:
    if not paragraphs:
        return []

    n = len(paragraphs)
    normalized: List[JSONDict] = []

    for raw in raw_chunks:
        try:
            start = int(raw["start_paragraph"])
            end = int(raw["end_paragraph"])
        except Exception:
            raise ValueError(f"Invalid chunk range: {raw}")

        if start < 0 or end < 0 or start >= n or end >= n or start > end:
            raise ValueError(f"Chunk range out of bounds: {raw}")

        normalized.append({
            "title": str(raw.get("title") or f"Paragraphs {start}-{end}"),
            "start_paragraph": start,
            "end_paragraph": end,
            "rationale": str(raw.get("rationale") or ""),
        })

    normalized.sort(key=lambda item: item["start_paragraph"])

    expected = 0
    for chunk in normalized:
        if chunk["start_paragraph"] != expected:
            raise ValueError(
                f"Chunks do not cover paragraph ids exactly. "
                f"Expected start {expected}, got {chunk['start_paragraph']}"
            )
        expected = chunk["end_paragraph"] + 1

    if expected != n:
        raise ValueError(
            f"Chunks do not cover all paragraphs. Covered through {expected - 1}, "
            f"but final paragraph is {n - 1}."
        )

    return normalized


def semantic_chunk_ranges_with_ollama(
    paragraphs: Sequence[JSONDict],
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> List[JSONDict]:
    prompt = build_chunking_prompt(paragraphs)
    result = call_ollama_json(prompt, model=model, ollama_url=ollama_url)

    chunks = result.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"Ollama response did not contain a list of chunks: {result}")

    return validate_and_normalize_chunk_ranges(chunks, paragraphs)


# ============================================================
# Block construction
# ============================================================

def paragraph_range_to_text_block(
    *,
    document_id: str,
    source_url: str,
    cleaned_text: str,
    paragraphs: Sequence[JSONDict],
    chunk_range: JSONDict,
    chunk_index: int,
    extraction_method: str,
) -> TextBlock:
    start_paragraph = chunk_range["start_paragraph"]
    end_paragraph = chunk_range["end_paragraph"]

    start_char = paragraphs[start_paragraph]["start_char"]
    end_char = paragraphs[end_paragraph]["end_char"]

    chunk_text = cleaned_text[start_char:end_char].strip()

    block_id = stable_id("blk", {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "start_char": start_char,
        "end_char": end_char,
        "source_url": source_url,
    })

    return TextBlock(
        block_id=block_id,
        document_id=document_id,
        source_url=source_url,
        text=chunk_text,
        start_char=start_char,
        end_char=end_char,
        section_title=chunk_range.get("title"),
        extraction_method=extraction_method,
        extraction_confidence=1.0,
        metadata={
            "chunk_index": chunk_index,
            "start_paragraph": start_paragraph,
            "end_paragraph": end_paragraph,
            "chunking_rationale": chunk_range.get("rationale"),
        },
    )


def build_text_blocks(
    *,
    document_id: str,
    source_url: str,
    cleaned_text: str,
    paragraphs: Sequence[JSONDict],
    chunk_ranges: Sequence[JSONDict],
    extraction_method: str,
) -> List[TextBlock]:
    return [
        paragraph_range_to_text_block(
            document_id=document_id,
            source_url=source_url,
            cleaned_text=cleaned_text,
            paragraphs=paragraphs,
            chunk_range=chunk_range,
            chunk_index=i,
            extraction_method=extraction_method,
        )
        for i, chunk_range in enumerate(chunk_ranges)
    ]


# ============================================================
# Main acquisition function
# ============================================================

def html_webpage_document_from_url(
    url: str,
    *,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    use_ollama_chunking: bool = True,
) -> HTMLWebpageDocument:
    html = fetch_html(url)
    metadata = extract_metadata(html, url)
    cleaned_text = extract_clean_text(html, url)
    credibility = assess_source_credibility(url, metadata)

    document = HTMLWebpageDocument(
        source_url=url,
        html=html,
        cleaned_text=cleaned_text,
        metadata=metadata,
        credibility=credibility,
    )

    paragraphs = split_into_paragraph_spans(cleaned_text)

    if not paragraphs:
        raise ValueError(f"No paragraphs extracted from {url}")

    extraction_method = "html_trafilatura_ollama_semantic_chunking"

    if use_ollama_chunking:
        try:
            chunk_ranges = semantic_chunk_ranges_with_ollama(
                paragraphs,
                model=ollama_model,
                ollama_url=ollama_url,
            )
        except Exception as error:
            print(f"[warn] Ollama semantic chunking failed: {error}")
            print("[warn] Falling back to deterministic paragraph chunking.")
            chunk_ranges = deterministic_chunk_ranges(paragraphs)
            extraction_method = "html_trafilatura_deterministic_paragraph_chunking"
    else:
        chunk_ranges = deterministic_chunk_ranges(paragraphs)
        extraction_method = "html_trafilatura_deterministic_paragraph_chunking"

    document.blocks = build_text_blocks(
        document_id=document.document_id,
        source_url=url,
        cleaned_text=cleaned_text,
        paragraphs=paragraphs,
        chunk_ranges=chunk_ranges,
        extraction_method=extraction_method,
    )

    document.retrieval_metadata = {
        "paragraph_count": len(paragraphs),
        "block_count": len(document.blocks),
        "chunking_method": extraction_method,
        "ollama_model": ollama_model if use_ollama_chunking else None,
    }

    return document


# ============================================================
# Test / demo
# ============================================================

def test() -> None:
    url = "https://www.csis.org/analysis/consequences-chinas-new-rare-earths-export-restrictions"

    document = html_webpage_document_from_url(
        url,
        ollama_model=DEFAULT_OLLAMA_MODEL,
        use_ollama_chunking=True,
    )

    print(json.dumps(document.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test()