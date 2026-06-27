#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse
import requests

try:
    from .document import (
        CSVDocument,
        EvidenceBlockType,
        EvidenceDocument,
        HTMLWebpageDocument,
        JSONDocument,
        PDFDocument,
        SourceType,
        SpreadsheetDocument,
        TableBlock,
        TextFileDocument,
        TextBlock,
        TimeSeriesBlock,
    )
    from .webpage import (
        DEFAULT_OLLAMA_MODEL,
        DEFAULT_OLLAMA_URL,
        assess_source_credibility,
        extract_clean_text,
        extract_metadata,
        html_webpage_document_from_url,
    )
except ImportError:
    from document import (
        CSVDocument,
        EvidenceBlockType,
        EvidenceDocument,
        HTMLWebpageDocument,
        JSONDocument,
        PDFDocument,
        SourceType,
        SpreadsheetDocument,
        TableBlock,
        TextFileDocument,
        TextBlock,
        TimeSeriesBlock,
    )
    from webpage import (
        DEFAULT_OLLAMA_MODEL,
        DEFAULT_OLLAMA_URL,
        assess_source_credibility,
        extract_clean_text,
        extract_metadata,
        html_webpage_document_from_url,
    )


JSONDict = Dict[str, Any]


# ============================================================
# Options / errors
# ============================================================

@dataclass
class IngestionOptions:
    extract_blocks: bool = True

    use_ollama_chunking: bool = False
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL

    delimiter: str = ","
    sheet_name: Optional[str] = None
    encoding: str = "utf-8"


class UnsupportedSourceError(ValueError):
    pass


# ============================================================
# Source detection
# ============================================================

def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def infer_source_type_from_path(path: Path) -> SourceType:
    suffix = path.suffix.lower()

    if suffix in {".html", ".htm"}:
        return SourceType.HTML_WEBPAGE

    if suffix == ".pdf":
        return SourceType.PDF

    if suffix == ".csv":
        return SourceType.CSV

    if suffix in {".xlsx", ".xls"}:
        return SourceType.SPREADSHEET

    if suffix == ".json":
        return SourceType.JSON_API

    if suffix in {".txt", ".text"}:
        return SourceType.TEXT_FILE

    if suffix in {".md", ".markdown"}:
        return SourceType.MARKDOWN

    raise UnsupportedSourceError(
        f"Unsupported file extension `{suffix}` for source `{path}`. "
        "Supported local types: .html, .htm, .pdf, .csv, .xlsx, .xls, .json, .txt, .md"
    )


def supported_source_types() -> JSONDict:
    return {
        "urls": ["http://...", "https://..."],
        "local_files": {
            ".html": SourceType.HTML_WEBPAGE.value,
            ".htm": SourceType.HTML_WEBPAGE.value,
            ".pdf": SourceType.PDF.value,
            ".csv": SourceType.CSV.value,
            ".xlsx": SourceType.SPREADSHEET.value,
            ".xls": SourceType.SPREADSHEET.value,
            ".json": SourceType.JSON_API.value,
            ".txt": SourceType.TEXT_FILE.value,
            ".md": SourceType.MARKDOWN.value,
        },
    }

def download_url_to_temp_file(url: str, suffix: str) -> Path:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        timeout=60,
    )
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()

    return Path(tmp.name)

# ============================================================
# Main API
# ============================================================

def ingest_source(
    source: str | Path,
    *,
    options: Optional[InestionOptions] = None,
    url_loader: Optional[Callable[..., EvidenceDocument]] = None,
) -> EvidenceDocument:
    """
    Universal ingestion entrypoint.

    Dispatches:
      URL         -> webpage.py
      .html/.htm  -> HTMLWebpageDocument
      .pdf        -> PDFDocument
      .csv        -> CSVDocument
      .xlsx/.xls  -> SpreadsheetDocument
      .json       -> JSONDocument
      .txt        -> TextFileDocument
      .md         -> TextFileDocument with SourceType.MARKDOWN
    """
    options = options or IngestionOptions()
    source_str = str(source)

    if is_url(source_str):
        document = ingest_url(
            source_str,
            options=options,
            url_loader=url_loader,
        )
    else:
        document = ingest_path(
            Path(source_str),
            options=options,
        )

    if options.extract_blocks:
        document.ensure_blocks()

    return document


def ingest_url(
    url: str,
    *,
    options: Optional[IngestionOptions] = None,
    url_loader: Optional[Callable[..., EvidenceDocument]] = None,
) -> EvidenceDocument:
    options = options or IngestionOptions()

    if url.lower().split("?", 1)[0].endswith(".pdf"):
        tmp_path = download_url_to_temp_file(url, ".pdf")
        document = PDFDocument(file_path=tmp_path)
        document.metadata.source_url = url
        document.metadata.canonical_url = url
        document.discovered_via = "ingest_url_pdf"
        document.retrieval_metadata = {
            **document.retrieval_metadata,
            "ingestion_entrypoint": "ingest_url_pdf",
            "source": url,
            "downloaded_temp_path": str(tmp_path),
            "inferred_source_type": SourceType.PDF.value,
        }
        return document

    loader = url_loader or html_webpage_document_from_url

    document = loader(
        url,
        ollama_model=options.ollama_model,
        ollama_url=options.ollama_url,
        use_ollama_chunking=options.use_ollama_chunking,
    )

    document.discovered_via = document.discovered_via or "ingest_url"
    document.retrieval_metadata = {
        **document.retrieval_metadata,
        "ingestion_entrypoint": "ingest_url",
        "source": url,
    }

    return document

def ingest_html_path(
    path: Path,
    *,
    options: Optional[IngestionOptions] = None,
) -> EvidenceDocument:
    options = options or IngestionOptions()
    path = Path(path)

    html = path.read_text(encoding=options.encoding)
    source_url = path.resolve().as_uri()

    metadata = extract_metadata(html, source_url)
    cleaned_text = extract_clean_text(html, source_url)
    credibility = assess_source_credibility(source_url, metadata)

    document = HTMLWebpageDocument(
        source_url=source_url,
        html=html,
        cleaned_text=cleaned_text,
        metadata=metadata,
        credibility=credibility,
    )

    document.discovered_via = "ingest_html_path"
    document.retrieval_metadata = {
        **document.retrieval_metadata,
        "ingestion_entrypoint": "ingest_html_path",
        "source": str(path),
        "inferred_source_type": SourceType.HTML_WEBPAGE.value,
    }

    return document


def ingest_path(
    path: Path,
    *,
    options: Optional[IngestionOptions] = None,
) -> EvidenceDocument:
    options = options or IngestionOptions()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot ingest missing file: {path}")

    source_type = infer_source_type_from_path(path)

    if source_type == SourceType.HTML_WEBPAGE:
        document = ingest_html_path(
            path,
            options=options,
        )

    elif source_type == SourceType.PDF:
        document: EvidenceDocument = PDFDocument(file_path=path)

    elif source_type == SourceType.CSV:
        document = CSVDocument(
            file_path=path,
            delimiter=options.delimiter,
        )

    elif source_type == SourceType.SPREADSHEET:
        document = SpreadsheetDocument(
            file_path=path,
            sheet_name=options.sheet_name,
        )

    elif source_type == SourceType.JSON_API:
        document = JSONDocument(file_path=path)

    elif source_type == SourceType.TEXT_FILE:
        document = TextFileDocument(
            file_path=path,
            encoding=options.encoding,
            source_type=SourceType.TEXT_FILE,
        )

    elif source_type == SourceType.MARKDOWN:
        document = TextFileDocument(
            file_path=path,
            encoding=options.encoding,
            source_type=SourceType.MARKDOWN,
        )

    else:
        raise UnsupportedSourceError(f"Unsupported source type: {source_type}")

    document.discovered_via = document.discovered_via or "ingest_path"
    document.retrieval_metadata = {
        **document.retrieval_metadata,
        "ingestion_entrypoint": "ingest_path",
        "source": str(path),
        "inferred_source_type": source_type.value,
    }

    return document


def ingest_json_data(
    data: Any,
    *,
    source_url: Optional[str] = None,
    options: Optional[IngestionOptions] = None,
) -> EvidenceDocument:
    options = options or IngestionOptions()

    document = JSONDocument(
        data=data,
        source_url=source_url,
    )

    document.discovered_via = "ingest_json_data"
    document.retrieval_metadata = {
        **document.retrieval_metadata,
        "ingestion_entrypoint": "ingest_json_data",
        "source": source_url,
    }

    if options.extract_blocks:
        document.ensure_blocks()

    return document


# ============================================================
# Summary / persistence
# ============================================================

def summarize_document(document: EvidenceDocument) -> str:
    blocks = document.ensure_blocks()

    block_counts: Dict[str, int] = {}
    for block in blocks:
        key = block.block_type.value
        block_counts[key] = block_counts.get(key, 0) + 1

    lines = []
    lines.append("INGESTED DOCUMENT")
    lines.append("=" * 60)
    lines.append(f"document_id: {document.document_id}")
    lines.append(f"source_type: {document.source_type.value}")
    lines.append(f"title: {document.metadata.title}")
    lines.append(f"source_url: {document.metadata.source_url}")
    lines.append(f"raw_content_ref: {document.metadata.raw_content_ref}")
    lines.append(f"credibility: {document.credibility.tier.value} ({document.credibility.score})")
    lines.append(f"blocks: {len(blocks)}")
    lines.append(f"block_counts: {json.dumps(block_counts, sort_keys=True)}")

    if blocks:
        lines.append("")
        lines.append("First block:")
        first = blocks[0]
        lines.append(f"  block_id: {first.block_id}")
        lines.append(f"  block_type: {first.block_type.value}")
        lines.append(f"  extraction_method: {first.extraction_method}")
        lines.append("  preview:")
        preview = first.to_text()
        if len(preview) > 600:
            preview = preview[:597].rstrip() + "..."
        lines.append(indent(preview, "    "))

    return "\n".join(lines)


def save_document_json(document: EvidenceDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ============================================================
# Tests
# ============================================================

def fake_url_loader(
    url: str,
    *,
    ollama_model: str,
    ollama_url: str,
    use_ollama_chunking: bool,
) -> EvidenceDocument:
    # Avoids network access while still testing URL dispatch.
    try:
        from .document import HTMLWebpageDocument, DocumentMetadata, SourceCredibility
    except ImportError:
        from document import HTMLWebpageDocument, DocumentMetadata, SourceCredibility

    document = HTMLWebpageDocument(
        source_url=url,
        cleaned_text=(
            "China imposed export restrictions on rare earth elements.\n"
            "The restrictions affected downstream magnet supply chains."
        ),
        metadata=DocumentMetadata(
            title="Fake webpage",
            publisher="Example Publisher",
            source_url=url,
            canonical_url=url,
        ),
        credibility=SourceCredibility(score=0.7),
    )
    document.blocks = document.extract_blocks()
    return document


def test_url_detection() -> None:
    print("[test] URL detection and path source-type inference")

    assert is_url("https://example.com/article")
    assert is_url("http://example.com/article")
    assert not is_url("/tmp/example.csv")
    assert not is_url("example.csv")

    assert infer_source_type_from_path(Path("x.html")) == SourceType.HTML_WEBPAGE
    assert infer_source_type_from_path(Path("x.htm")) == SourceType.HTML_WEBPAGE
    assert infer_source_type_from_path(Path("x.pdf")) == SourceType.PDF
    assert infer_source_type_from_path(Path("x.csv")) == SourceType.CSV
    assert infer_source_type_from_path(Path("x.xlsx")) == SourceType.SPREADSHEET
    assert infer_source_type_from_path(Path("x.xls")) == SourceType.SPREADSHEET
    assert infer_source_type_from_path(Path("x.json")) == SourceType.JSON_API
    assert infer_source_type_from_path(Path("x.txt")) == SourceType.TEXT_FILE
    assert infer_source_type_from_path(Path("x.md")) == SourceType.MARKDOWN

    try:
        infer_source_type_from_path(Path("x.exe"))
        raise AssertionError("Expected UnsupportedSourceError for .exe")
    except UnsupportedSourceError:
        pass


def test_missing_file() -> None:
    print("[test] Missing local file gives FileNotFoundError")

    try:
        ingest_source("/definitely/not/present.csv")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_url_dispatch_without_network() -> None:
    print("[test] URL dispatch uses webpage loader without network")

    document = ingest_source(
        "https://example.com/fake-article",
        options=IngestionOptions(use_ollama_chunking=False),
        url_loader=fake_url_loader,
    )

    assert document.source_type == SourceType.HTML_WEBPAGE
    assert document.metadata.source_url == "https://example.com/fake-article"
    assert document.retrieval_metadata["ingestion_entrypoint"] == "ingest_url"
    assert len(document.ensure_blocks()) >= 1
    assert any(block.block_type == EvidenceBlockType.TEXT for block in document.blocks)


def test_html_dispatch_and_blocks(tmp_path: Path) -> None:
    print("[test] HTML dispatch creates HTMLWebpageDocument and TextBlock evidence")

    html_path = tmp_path / "article.html"
    html_path.write_text(
        """
        <html>
          <head><title>Rare earth article</title></head>
          <body>
            <article>
              <h1>Rare earth disruption</h1>
              <p>China imposed export restrictions on rare earth elements.</p>
              <p>The restrictions affected downstream magnet supply chains.</p>
            </article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = ingest_source(html_path)

    assert document.source_type == SourceType.HTML_WEBPAGE
    assert document.retrieval_metadata["inferred_source_type"] == SourceType.HTML_WEBPAGE.value
    assert document.metadata.source_url == html_path.resolve().as_uri()
    assert any(isinstance(block, TextBlock) for block in document.ensure_blocks())
    assert "rare earth" in document.ensure_blocks()[0].to_text().lower()


def test_csv_dispatch_and_blocks(tmp_path: Path) -> None:
    print("[test] CSV dispatch creates TableBlock and detected TimeSeriesBlock")

    csv_path = tmp_path / "production.csv"
    csv_path.write_text(
        "date,country,production_tonnes\n"
        "2025-01-01,China,240000\n"
        "2025-02-01,United States,43000\n"
        "2025-03-01,Australia,18000\n",
        encoding="utf-8",
    )

    document = ingest_source(csv_path)

    assert document.source_type == SourceType.CSV
    assert document.retrieval_metadata["inferred_source_type"] == SourceType.CSV.value

    blocks = document.ensure_blocks()
    assert any(isinstance(block, TableBlock) for block in blocks)
    assert any(isinstance(block, TimeSeriesBlock) for block in blocks)

    table = next(block for block in blocks if isinstance(block, TableBlock))
    assert table.row_count == 3
    assert "production_tonnes" in table.columns


def test_text_and_markdown_dispatch(tmp_path: Path) -> None:
    print("[test] TXT and Markdown dispatch create TextBlock evidence")

    txt_path = tmp_path / "note.txt"
    txt_path.write_text(
        "China restricted several critical minerals.\n"
        "This may affect supply chains.",
        encoding="utf-8",
    )

    txt_doc = ingest_source(txt_path)
    assert txt_doc.source_type == SourceType.TEXT_FILE
    assert any(isinstance(block, TextBlock) for block in txt_doc.ensure_blocks())

    md_path = tmp_path / "note.md"
    md_path.write_text(
        "# Export controls\n\nChina restricted several rare earth elements.",
        encoding="utf-8",
    )

    md_doc = ingest_source(md_path)
    assert md_doc.source_type == SourceType.MARKDOWN
    assert any(isinstance(block, TextBlock) for block in md_doc.ensure_blocks())


def test_json_dispatch_and_json_data(tmp_path: Path) -> None:
    print("[test] JSON file and in-memory JSON create table/time-series blocks")

    payload = {
        "records": [
            {"date": "2025-01-01", "price": 10.5, "commodity": "dysprosium"},
            {"date": "2025-02-01", "price": 11.2, "commodity": "dysprosium"},
            {"date": "2025-03-01", "price": 12.1, "commodity": "dysprosium"},
        ]
    }

    json_path = tmp_path / "records.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    file_doc = ingest_source(json_path)
    assert file_doc.source_type == SourceType.JSON_API
    assert any(isinstance(block, TableBlock) for block in file_doc.ensure_blocks())
    assert any(isinstance(block, TimeSeriesBlock) for block in file_doc.ensure_blocks())

    memory_doc = ingest_json_data(payload, source_url="memory://records")
    assert memory_doc.source_type == SourceType.JSON_API
    assert memory_doc.metadata.source_url == "memory://records"
    assert any(isinstance(block, TableBlock) for block in memory_doc.ensure_blocks())


def test_spreadsheet_dispatch_if_available(tmp_path: Path) -> None:
    print("[test] XLSX dispatch creates table/time-series blocks if pandas Excel support is available")

    try:
        import pandas as pd
    except Exception as error:
        print(f"[skip] pandas unavailable: {error}")
        return

    xlsx_path = tmp_path / "capacity.xlsx"

    try:
        pd.DataFrame(
            [
                {"date": "2025-01-01", "capacity": 100, "facility": "A"},
                {"date": "2025-02-01", "capacity": 120, "facility": "A"},
                {"date": "2025-03-01", "capacity": 130, "facility": "A"},
            ]
        ).to_excel(xlsx_path, index=False, sheet_name="capacity")
    except Exception as error:
        print(f"[skip] Excel writer unavailable: {error}")
        return

    document = ingest_source(xlsx_path)
    assert document.source_type == SourceType.SPREADSHEET

    blocks = document.ensure_blocks()
    assert any(isinstance(block, TableBlock) for block in blocks)
    assert any(isinstance(block, TimeSeriesBlock) for block in blocks)


def test_pdf_dispatch_if_available(tmp_path: Path) -> None:
    print("[test] PDF dispatch creates text blocks if PDF generation/parsing dependencies are available")

    pdf_path = tmp_path / "sample.pdf"

    try:
        from reportlab.pdfgen import canvas
    except Exception as error:
        print(f"[skip] reportlab unavailable for PDF fixture generation: {error}")
        return

    try:
        c = canvas.Canvas(str(pdf_path))
        c.drawString(72, 720, "China imposed export restrictions on rare earth elements.")
        c.drawString(72, 700, "This is a simple PDF ingestion test.")
        c.save()
    except Exception as error:
        print(f"[skip] PDF fixture generation failed: {error}")
        return

    document = ingest_source(pdf_path)
    assert document.source_type == SourceType.PDF

    blocks = document.ensure_blocks()
    assert any(isinstance(block, TextBlock) for block in blocks)
    assert any("rare earth" in block.to_text().lower() for block in blocks)


def test_output_json(tmp_path: Path) -> None:
    print("[test] Ingested document can be saved to JSON")

    text_path = tmp_path / "save_test.txt"
    text_path.write_text("A short text document.", encoding="utf-8")

    document = ingest_source(text_path)
    output_path = tmp_path / "document.json"

    save_document_json(document, output_path)

    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["document_id"] == document.document_id
    assert data["source_type"] == SourceType.TEXT_FILE.value
    assert len(data["blocks"]) >= 1


def run_tests() -> None:
    print("Running ingest.py tests...")

    test_url_detection()
    test_missing_file()
    test_url_dispatch_without_network()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        test_html_dispatch_and_blocks(tmp_path)
        test_csv_dispatch_and_blocks(tmp_path)
        test_text_and_markdown_dispatch(tmp_path)
        test_json_dispatch_and_json_data(tmp_path)
        test_spreadsheet_dispatch_if_available(tmp_path)
        test_pdf_dispatch_if_available(tmp_path)
        test_output_json(tmp_path)

    print("All ingest.py tests passed.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal ingestion entrypoint")

    parser.add_argument("--source", help="URL or local file path to ingest")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--summary", action="store_true")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--supported", action="store_true")

    parser.add_argument("--use-ollama-chunking", action="store_true")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)

    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--encoding", default="utf-8")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test:
        run_tests()
        return 0

    if args.supported:
        print(json.dumps(supported_source_types(), indent=2, ensure_ascii=False))
        return 0

    if not args.source:
        raise SystemExit("Provide --source, --supported, or --test.")

    options = IngestionOptions(
        use_ollama_chunking=args.use_ollama_chunking,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        delimiter=args.delimiter,
        sheet_name=args.sheet_name,
        encoding=args.encoding,
    )

    document = ingest_source(args.source, options=options)

    if args.output:
        save_document_json(document, args.output)
        print(f"[done] wrote {args.output}")

    if args.summary or not args.output:
        print(summarize_document(document))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())