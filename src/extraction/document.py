from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


JSONDict = Dict[str, Any]


# ============================================================
# Enums
# ============================================================

class SourceType(str, Enum):
    HTML_WEBPAGE = "html_webpage"
    PDF = "pdf"
    CSV = "csv"
    SPREADSHEET = "spreadsheet"
    JSON_API = "json_api"
    PRICE_SERIES = "price_series"
    SEC_FILING = "sec_filing"
    NEWS_API = "news_api"
    TEXT_FILE = "text_file"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class EvidenceBlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    METADATA = "metadata"


class SourceCredibilityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# ============================================================
# Utility
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{prefix}_{digest}"


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compact_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def clean_cell(value: Any) -> Any:
    if value is None:
        return None

    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # pandas / numpy scalar compatibility without importing globally.
    if hasattr(value, "item"):
        try:
            return clean_cell(value.item())
        except Exception:
            pass

    text = str(value).strip()

    if text.lower() in {"nan", "nat", "none", "null"}:
        return None

    return value


def make_unique_columns(columns: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []

    for i, column in enumerate(columns):
        name = compact_ws(column) or f"column_{i + 1}"
        name = re.sub(r"\s+", "_", name)

        count = seen.get(name, 0)
        seen[name] = count + 1

        if count:
            name = f"{name}_{count + 1}"

        out.append(name)

    return out


def parse_date_like(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    text = str(value).strip()

    if not text:
        return None

    # Keep this conservative to avoid treating arbitrary integers as dates.
    if not re.search(r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text):
        if not re.fullmatch(r"\d{4}", text):
            return None

    try:
        from dateutil.parser import parse

        parsed = parse(text, fuzzy=False)
        return parsed.date().isoformat()
    except Exception:
        return None


def is_numeric_like(value: Any) -> bool:
    if value is None or value == "":
        return False

    try:
        float(str(value).replace(",", ""))
        return True
    except Exception:
        return False


def infer_column_schema(columns: Sequence[str], rows: Sequence[JSONDict]) -> JSONDict:
    schema: JSONDict = {}

    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) not in {None, ""}]
        sample = values[:20]
        col_norm = column.lower()

        numeric_hits = sum(1 for value in sample if is_numeric_like(value))
        date_hits = sum(1 for value in sample if parse_date_like(value))

        if sample and date_hits >= max(1, int(0.8 * len(sample))) and any(
            hint in col_norm for hint in ["date", "time", "year", "month", "period"]
        ):
            semantic_type = "date"
        elif sample and numeric_hits >= max(1, int(0.8 * len(sample))):
            semantic_type = "numeric"
        else:
            semantic_type = "categorical_or_text"

        unit = None
        unit_match = re.search(r"\(([^)]+)\)|_([a-zA-Z%$]+)$", column)
        if unit_match:
            unit = unit_match.group(1) or unit_match.group(2)

        schema[column] = {
            "semantic_type": semantic_type,
            "unit": unit,
            "non_null_count": len(values),
            "sample_values": [clean_cell(v) for v in sample[:5]],
        }

    return schema


def detect_time_series(columns: Sequence[str], rows: Sequence[JSONDict]) -> Optional[JSONDict]:
    if not rows:
        return None

    schema = infer_column_schema(columns, rows)

    date_columns = [
        column for column, info in schema.items()
        if info.get("semantic_type") == "date"
    ]

    numeric_columns = [
        column for column, info in schema.items()
        if info.get("semantic_type") == "numeric"
    ]

    if not date_columns or not numeric_columns:
        return None

    date_column = date_columns[0]
    parsed_dates = [
        parse_date_like(row.get(date_column))
        for row in rows
        if parse_date_like(row.get(date_column))
    ]

    if len(parsed_dates) < 2:
        return None

    return {
        "date_column": date_column,
        "value_columns": numeric_columns,
        "start_date": min(parsed_dates),
        "end_date": max(parsed_dates),
        "frequency": infer_frequency(parsed_dates),
    }


def infer_frequency(dates: Sequence[str]) -> Optional[str]:
    unique = sorted(set(dates))

    if len(unique) < 3:
        return None

    try:
        parsed = [datetime.fromisoformat(x).date() for x in unique[:20]]
        deltas = [(parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))]
    except Exception:
        return None

    if not deltas:
        return None

    median = sorted(deltas)[len(deltas) // 2]

    if median <= 1:
        return "daily"
    if 6 <= median <= 8:
        return "weekly"
    if 27 <= median <= 32:
        return "monthly"
    if 80 <= median <= 100:
        return "quarterly"
    if 350 <= median <= 380:
        return "annual"

    return None


def dataframe_to_rows(df: Any) -> List[JSONDict]:
    df = df.dropna(how="all")
    df.columns = make_unique_columns(df.columns)
    rows: List[JSONDict] = []

    for raw in df.to_dict(orient="records"):
        rows.append({str(k): clean_cell(v) for k, v in raw.items()})

    return rows


def rows_from_csv(path: Path, delimiter: str = ",") -> tuple[List[str], List[JSONDict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = make_unique_columns(reader.fieldnames or [])
        rows = []

        for raw in reader:
            row = {}
            for old_key, new_key in zip(reader.fieldnames or [], columns):
                row[new_key] = clean_cell(raw.get(old_key))
            rows.append(row)

    return columns, rows


# ============================================================
# Metadata / credibility
# ============================================================

@dataclass
class SourceCredibility:
    score: float = 0.5
    tier: SourceCredibilityTier = SourceCredibilityTier.UNKNOWN
    rationale: str = ""
    assessed_by: str = "system"
    factors: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp_score(self.score)

        if isinstance(self.tier, str):
            self.tier = SourceCredibilityTier(self.tier)


@dataclass
class DocumentMetadata:
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None

    published_at: Optional[str] = None
    fetched_at: str = field(default_factory=utc_now_iso)

    language: Optional[str] = None
    country: Optional[str] = None

    source_url: Optional[str] = None
    canonical_url: Optional[str] = None

    raw_content_ref: Optional[str] = None
    extra: JSONDict = field(default_factory=dict)


# ============================================================
# Evidence blocks
# ============================================================

@dataclass
class EvidenceBlock(ABC):
    block_id: str
    document_id: str
    block_type: EvidenceBlockType

    source_url: Optional[str] = None
    extraction_method: Optional[str] = None
    extraction_confidence: float = 1.0
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.block_type, str):
            self.block_type = EvidenceBlockType(self.block_type)

        self.extraction_confidence = clamp_score(self.extraction_confidence)

    @abstractmethod
    def to_text(self) -> str:
        raise NotImplementedError

    def to_embedding_text(self) -> str:
        return self.to_text()

    def to_llm_context(self) -> JSONDict:
        return {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "block_type": self.block_type.value,
            "source_url": self.source_url,
            "text": self.to_text(),
            "metadata": self.metadata,
        }

    def to_dict(self) -> JSONDict:
        data = asdict(self)
        data["block_type"] = self.block_type.value
        data["text"] = self.to_text()
        return data


@dataclass
class TextBlock(EvidenceBlock):
    text: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    def __init__(
        self,
        *,
        block_id: str,
        document_id: str,
        text: str,
        source_url: Optional[str] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        extraction_method: Optional[str] = None,
        extraction_confidence: float = 1.0,
        metadata: Optional[JSONDict] = None,
    ) -> None:
        super().__init__(
            block_id=block_id,
            document_id=document_id,
            block_type=EvidenceBlockType.TEXT,
            source_url=source_url,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
            metadata=metadata or {},
        )
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.page_number = page_number
        self.section_title = section_title

    def to_text(self) -> str:
        prefix_parts = []

        if self.section_title:
            prefix_parts.append(f"Section: {self.section_title}")

        if self.page_number is not None:
            prefix_parts.append(f"Page: {self.page_number}")

        prefix = "\n".join(prefix_parts)

        if prefix:
            return f"{prefix}\n\n{self.text}"

        return self.text


@dataclass
class TableBlock(EvidenceBlock):
    caption: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[JSONDict] = field(default_factory=list)

    page_number: Optional[int] = None
    table_number: Optional[str] = None

    column_schema: JSONDict = field(default_factory=dict)
    row_count: int = 0
    data_ref: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)

    def __init__(
        self,
        *,
        block_id: str,
        document_id: str,
        columns: Sequence[str],
        rows: Sequence[JSONDict],
        source_url: Optional[str] = None,
        caption: Optional[str] = None,
        page_number: Optional[int] = None,
        table_number: Optional[str] = None,
        extraction_method: Optional[str] = None,
        extraction_confidence: float = 1.0,
        metadata: Optional[JSONDict] = None,
        column_schema: Optional[JSONDict] = None,
        data_ref: Optional[str] = None,
        quality_flags: Optional[Sequence[str]] = None,
    ) -> None:
        rows_list = list(rows)
        columns_list = list(columns)

        super().__init__(
            block_id=block_id,
            document_id=document_id,
            block_type=EvidenceBlockType.TABLE,
            source_url=source_url,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
            metadata=metadata or {},
        )

        self.caption = caption
        self.columns = columns_list
        self.rows = rows_list
        self.page_number = page_number
        self.table_number = table_number
        self.column_schema = column_schema or infer_column_schema(columns_list, rows_list)
        self.row_count = len(rows_list)
        self.data_ref = data_ref
        self.quality_flags = list(quality_flags or [])

    def to_text(self) -> str:
        lines = []

        if self.caption:
            lines.append(f"Table caption: {self.caption}")

        if self.page_number is not None:
            lines.append(f"Page: {self.page_number}")

        if self.table_number:
            lines.append(f"Table number: {self.table_number}")

        lines.append("Columns: " + ", ".join(self.columns))
        lines.append(f"Row count: {self.row_count}")

        typed = [
            f"{column}={info.get('semantic_type')}"
            for column, info in self.column_schema.items()
        ]

        if typed:
            lines.append("Column schema: " + "; ".join(typed))

        for i, row in enumerate(self.rows[:20], start=1):
            row_text = "; ".join(f"{key}: {value}" for key, value in row.items())
            lines.append(f"Row {i}: {row_text}")

        if len(self.rows) > 20:
            lines.append(f"... {len(self.rows) - 20} additional rows omitted")

        if self.data_ref:
            lines.append(f"Full data reference: {self.data_ref}")

        if self.quality_flags:
            lines.append("Quality flags: " + ", ".join(self.quality_flags))

        return "\n".join(lines)


@dataclass
class ChartBlock(EvidenceBlock):
    title: Optional[str] = None
    description: Optional[str] = None
    image_ref: Optional[str] = None
    extracted_data: Optional[JSONDict] = None
    page_number: Optional[int] = None

    def __init__(
        self,
        *,
        block_id: str,
        document_id: str,
        source_url: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        image_ref: Optional[str] = None,
        extracted_data: Optional[JSONDict] = None,
        page_number: Optional[int] = None,
        extraction_method: Optional[str] = None,
        extraction_confidence: float = 1.0,
        metadata: Optional[JSONDict] = None,
    ) -> None:
        super().__init__(
            block_id=block_id,
            document_id=document_id,
            block_type=EvidenceBlockType.CHART,
            source_url=source_url,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
            metadata=metadata or {},
        )
        self.title = title
        self.description = description
        self.image_ref = image_ref
        self.extracted_data = extracted_data
        self.page_number = page_number

    def to_text(self) -> str:
        parts = []

        if self.title:
            parts.append(f"Chart title: {self.title}")

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.extracted_data:
            parts.append("Extracted data: " + json.dumps(self.extracted_data, ensure_ascii=False, sort_keys=True))

        if self.image_ref:
            parts.append(f"Image reference: {self.image_ref}")

        return "\n".join(parts)


@dataclass
class TimeSeriesBlock(EvidenceBlock):
    instrument: str = ""
    fields: List[str] = field(default_factory=list)
    observations: List[JSONDict] = field(default_factory=list)
    data_ref: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    summary: Optional[str] = None

    date_field: Optional[str] = None
    value_fields: List[str] = field(default_factory=list)
    frequency: Optional[str] = None
    units: JSONDict = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)

    def __init__(
        self,
        *,
        block_id: str,
        document_id: str,
        instrument: str,
        fields: Sequence[str],
        source_url: Optional[str] = None,
        observations: Optional[Sequence[JSONDict]] = None,
        data_ref: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        summary: Optional[str] = None,
        extraction_method: Optional[str] = None,
        extraction_confidence: float = 1.0,
        metadata: Optional[JSONDict] = None,
        date_field: Optional[str] = None,
        value_fields: Optional[Sequence[str]] = None,
        frequency: Optional[str] = None,
        units: Optional[JSONDict] = None,
        quality_flags: Optional[Sequence[str]] = None,
    ) -> None:
        obs = list(observations or [])

        super().__init__(
            block_id=block_id,
            document_id=document_id,
            block_type=EvidenceBlockType.TIME_SERIES,
            source_url=source_url,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
            metadata=metadata or {},
        )
        self.instrument = instrument
        self.fields = list(fields)
        self.observations = obs
        self.data_ref = data_ref
        self.start_date = start_date
        self.end_date = end_date
        self.summary = summary
        self.date_field = date_field
        self.value_fields = list(value_fields or [])
        self.frequency = frequency
        self.units = dict(units or {})
        self.quality_flags = list(quality_flags or [])

    def to_text(self) -> str:
        lines = [
            f"Time series instrument: {self.instrument}",
            f"Fields: {', '.join(self.fields)}",
        ]

        if self.date_field:
            lines.append(f"Date field: {self.date_field}")

        if self.value_fields:
            lines.append(f"Value fields: {', '.join(self.value_fields)}")

        if self.frequency:
            lines.append(f"Frequency: {self.frequency}")

        if self.start_date or self.end_date:
            lines.append(f"Date range: {self.start_date or '?'} to {self.end_date or '?'}")

        if self.summary:
            lines.append(f"Summary: {self.summary}")

        for i, row in enumerate(self.observations[:10], start=1):
            row_text = "; ".join(f"{key}: {value}" for key, value in row.items())
            lines.append(f"Observation {i}: {row_text}")

        if len(self.observations) > 10:
            lines.append(f"... {len(self.observations) - 10} additional observations omitted")

        if self.data_ref:
            lines.append(f"Full data reference: {self.data_ref}")

        if self.quality_flags:
            lines.append("Quality flags: " + ", ".join(self.quality_flags))

        return "\n".join(lines)


# ============================================================
# Evidence documents
# ============================================================

@dataclass
class EvidenceDocument(ABC):
    document_id: str
    source_type: SourceType
    metadata: DocumentMetadata

    credibility: SourceCredibility = field(default_factory=SourceCredibility)
    blocks: List[EvidenceBlock] = field(default_factory=list)

    discovered_via: Optional[str] = None
    retrieval_metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)

    @abstractmethod
    def extract_blocks(self) -> List[EvidenceBlock]:
        raise NotImplementedError

    def add_block(self, block: EvidenceBlock) -> None:
        self.blocks.append(block)

    def ensure_blocks(self) -> List[EvidenceBlock]:
        if not self.blocks:
            self.blocks = self.extract_blocks()
        return self.blocks

    def to_evidence_store(self) -> JSONDict:
        self.ensure_blocks()

        return {
            block.block_id: block.to_dict()
            for block in self.blocks
        }

    def to_dict(self) -> JSONDict:
        self.ensure_blocks()

        return {
            "document_id": self.document_id,
            "source_type": self.source_type.value,
            "metadata": asdict(self.metadata),
            "credibility": {
                **asdict(self.credibility),
                "tier": self.credibility.tier.value,
            },
            "discovered_via": self.discovered_via,
            "retrieval_metadata": self.retrieval_metadata,
            "blocks": [block.to_dict() for block in self.blocks],
        }


# ============================================================
# Shared document block builders
# ============================================================

def make_table_blocks_from_rows(
    *,
    document_id: str,
    source_url: Optional[str],
    rows: Sequence[JSONDict],
    columns: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    table_number: Optional[str] = None,
    page_number: Optional[int] = None,
    extraction_method: str,
    data_ref: Optional[str] = None,
) -> List[EvidenceBlock]:
    rows_list = list(rows)

    if not rows_list:
        return []

    columns_list = list(columns or rows_list[0].keys())
    column_schema = infer_column_schema(columns_list, rows_list)

    blocks: List[EvidenceBlock] = [
        TableBlock(
            block_id=stable_id("blk", {
                "document_id": document_id,
                "kind": "table",
                "caption": caption,
                "table_number": table_number,
                "columns": columns_list,
                "n": len(rows_list),
            }),
            document_id=document_id,
            source_url=source_url,
            caption=caption,
            columns=columns_list,
            rows=rows_list,
            page_number=page_number,
            table_number=table_number,
            extraction_method=extraction_method,
            column_schema=column_schema,
            data_ref=data_ref,
        )
    ]

    ts = detect_time_series(columns_list, rows_list)

    if ts:
        value_columns = ts["value_columns"]
        date_column = ts["date_column"]

        observations = [
            {
                date_column: parse_date_like(row.get(date_column)) or row.get(date_column),
                **{column: row.get(column) for column in value_columns},
            }
            for row in rows_list
            if row.get(date_column) is not None
        ]

        blocks.append(
            TimeSeriesBlock(
                block_id=stable_id("blk", {
                    "document_id": document_id,
                    "kind": "time_series",
                    "date_column": date_column,
                    "value_columns": value_columns,
                    "n": len(observations),
                }),
                document_id=document_id,
                source_url=source_url,
                instrument=caption or table_number or "tabular_time_series",
                fields=[date_column] + value_columns,
                observations=observations,
                start_date=ts["start_date"],
                end_date=ts["end_date"],
                date_field=date_column,
                value_fields=value_columns,
                frequency=ts["frequency"],
                data_ref=data_ref,
                summary=f"Detected time series with date field `{date_column}` and value fields: {', '.join(value_columns)}.",
                extraction_method=f"{extraction_method}_time_series_detection",
            )
        )

    return blocks


def chunk_text_blocks(
    *,
    document_id: str,
    source_url: Optional[str],
    text: str,
    extraction_method: str,
    section_title: Optional[str] = None,
    page_number: Optional[int] = None,
    max_chars: int = 2500,
) -> List[EvidenceBlock]:
    text = text.strip()

    if not text:
        return []

    blocks: List[EvidenceBlock] = []

    start = 0
    index = 0

    while start < len(text):
        end = min(len(text), start + max_chars)

        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + 500:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            blocks.append(
                TextBlock(
                    block_id=stable_id("blk", {
                        "document_id": document_id,
                        "kind": "text_chunk",
                        "index": index,
                        "start": start,
                        "sample": chunk[:120],
                    }),
                    document_id=document_id,
                    source_url=source_url,
                    text=chunk,
                    start_char=start,
                    end_char=end,
                    page_number=page_number,
                    section_title=section_title,
                    extraction_method=extraction_method,
                )
            )
            index += 1

        start = max(end, start + 1)

    return blocks


# ============================================================
# Document subclasses
# ============================================================

@dataclass
class HTMLWebpageDocument(EvidenceDocument):
    html: Optional[str] = None
    cleaned_text: Optional[str] = None

    def __init__(
        self,
        *,
        source_url: str,
        html: Optional[str] = None,
        cleaned_text: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
    ) -> None:
        metadata = metadata or DocumentMetadata(source_url=source_url, canonical_url=source_url)
        metadata.source_url = metadata.source_url or source_url

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.HTML_WEBPAGE.value,
            "source_url": source_url,
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.HTML_WEBPAGE,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.html = html
        self.cleaned_text = cleaned_text

    def extract_blocks(self) -> List[EvidenceBlock]:
        if not self.cleaned_text:
            return []

        return [
            TextBlock(
                block_id=stable_id("blk", {
                    "document_id": self.document_id,
                    "kind": "full_text",
                    "text": self.cleaned_text[:200],
                }),
                document_id=self.document_id,
                source_url=self.metadata.source_url,
                text=self.cleaned_text,
                start_char=0,
                end_char=len(self.cleaned_text),
                extraction_method="html_clean_text",
            )
        ]


@dataclass
class PDFDocument(EvidenceDocument):
    file_path: Optional[Path] = None
    page_count: Optional[int] = None

    def __init__(
        self,
        *,
        file_path: Path,
        source_url: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
    ) -> None:
        file_path = Path(file_path)

        metadata = metadata or DocumentMetadata(source_url=source_url, raw_content_ref=str(file_path))

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.PDF.value,
            "source_url": source_url,
            "file_path": str(file_path),
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.PDF,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.file_path = file_path

    def extract_blocks(self) -> List[EvidenceBlock]:
        if not self.file_path or not self.file_path.exists():
            return []

        try:
            return self._extract_with_pdfplumber()
        except Exception as pdfplumber_error:
            self.retrieval_metadata["pdfplumber_error"] = str(pdfplumber_error)

        try:
            return self._extract_with_pypdf()
        except Exception as pypdf_error:
            self.retrieval_metadata["pypdf_error"] = str(pypdf_error)
            return []

    def _extract_with_pdfplumber(self) -> List[EvidenceBlock]:
        import pdfplumber

        blocks: List[EvidenceBlock] = []

        with pdfplumber.open(str(self.file_path)) as pdf:
            self.page_count = len(pdf.pages)
            self.metadata.extra["page_count"] = self.page_count

            for page_index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    blocks.append(
                        TextBlock(
                            block_id=stable_id("blk", {
                                "document_id": self.document_id,
                                "kind": "pdf_page_text",
                                "page": page_index,
                                "sample": text[:120],
                            }),
                            document_id=self.document_id,
                            source_url=self.metadata.source_url,
                            text=text.strip(),
                            page_number=page_index,
                            section_title=f"PDF page {page_index}",
                            extraction_method="pdfplumber_extract_text",
                        )
                    )

                for table_index, table in enumerate(page.extract_tables() or [], start=1):
                    if not table or len(table) < 2:
                        continue

                    columns = make_unique_columns(table[0])
                    rows = []

                    for raw_row in table[1:]:
                        if not raw_row or all(cell in {None, ""} for cell in raw_row):
                            continue

                        padded = list(raw_row) + [None] * max(0, len(columns) - len(raw_row))
                        rows.append({
                            column: clean_cell(value)
                            for column, value in zip(columns, padded[:len(columns)])
                        })

                    blocks.extend(
                        make_table_blocks_from_rows(
                            document_id=self.document_id,
                            source_url=self.metadata.source_url,
                            rows=rows,
                            columns=columns,
                            caption=f"PDF page {page_index} table {table_index}",
                            table_number=f"p{page_index}_t{table_index}",
                            page_number=page_index,
                            extraction_method="pdfplumber_extract_table",
                            data_ref=str(self.file_path),
                        )
                    )

        return blocks

    def _extract_with_pypdf(self) -> List[EvidenceBlock]:
        from pypdf import PdfReader

        reader = PdfReader(str(self.file_path))
        self.page_count = len(reader.pages)
        self.metadata.extra["page_count"] = self.page_count

        blocks: List[EvidenceBlock] = []

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            blocks.append(
                TextBlock(
                    block_id=stable_id("blk", {
                        "document_id": self.document_id,
                        "kind": "pdf_page_text",
                        "page": page_index,
                        "sample": text[:120],
                    }),
                    document_id=self.document_id,
                    source_url=self.metadata.source_url,
                    text=text.strip(),
                    page_number=page_index,
                    section_title=f"PDF page {page_index}",
                    extraction_method="pypdf_extract_text",
                    extraction_confidence=0.85,
                )
            )

        return blocks


@dataclass
class CSVDocument(EvidenceDocument):
    file_path: Optional[Path] = None
    delimiter: str = ","

    def __init__(
        self,
        *,
        file_path: Path,
        source_url: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
        delimiter: str = ",",
    ) -> None:
        file_path = Path(file_path)

        metadata = metadata or DocumentMetadata(source_url=source_url, raw_content_ref=str(file_path))

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.CSV.value,
            "source_url": source_url,
            "file_path": str(file_path),
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.CSV,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.file_path = file_path
        self.delimiter = delimiter

    def extract_blocks(self) -> List[EvidenceBlock]:
        if not self.file_path or not self.file_path.exists():
            return []

        columns, rows = rows_from_csv(self.file_path, self.delimiter)

        return make_table_blocks_from_rows(
            document_id=self.document_id,
            source_url=self.metadata.source_url,
            rows=rows,
            columns=columns,
            caption=self.metadata.title or self.file_path.name,
            table_number="csv_1",
            extraction_method="csv_dict_reader",
            data_ref=str(self.file_path),
        )


@dataclass
class SpreadsheetDocument(EvidenceDocument):
    file_path: Optional[Path] = None
    sheet_name: Optional[str] = None

    def __init__(
        self,
        *,
        file_path: Path,
        source_url: Optional[str] = None,
        sheet_name: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
    ) -> None:
        file_path = Path(file_path)

        metadata = metadata or DocumentMetadata(source_url=source_url, raw_content_ref=str(file_path))

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.SPREADSHEET.value,
            "source_url": source_url,
            "file_path": str(file_path),
            "sheet_name": sheet_name,
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.SPREADSHEET,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.file_path = file_path
        self.sheet_name = sheet_name

    def extract_blocks(self) -> List[EvidenceBlock]:
        if not self.file_path or not self.file_path.exists():
            return []

        import pandas as pd

        xls = pd.ExcelFile(self.file_path)
        sheets = [self.sheet_name] if self.sheet_name else xls.sheet_names

        blocks: List[EvidenceBlock] = []

        for sheet in sheets:
            df = pd.read_excel(self.file_path, sheet_name=sheet)
            rows = dataframe_to_rows(df)

            if not rows:
                continue

            blocks.extend(
                make_table_blocks_from_rows(
                    document_id=self.document_id,
                    source_url=self.metadata.source_url,
                    rows=rows,
                    columns=list(rows[0].keys()),
                    caption=f"{self.file_path.name} / {sheet}",
                    table_number=f"sheet_{sheet}",
                    extraction_method="pandas_read_excel",
                    data_ref=str(self.file_path),
                )
            )

        return blocks


@dataclass
class TextFileDocument(EvidenceDocument):
    file_path: Optional[Path] = None
    encoding: str = "utf-8"

    def __init__(
        self,
        *,
        file_path: Path,
        source_url: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
        encoding: str = "utf-8",
        source_type: SourceType = SourceType.TEXT_FILE,
    ) -> None:
        file_path = Path(file_path)

        metadata = metadata or DocumentMetadata(source_url=source_url, raw_content_ref=str(file_path))

        document_id = document_id or stable_id("doc", {
            "source_type": source_type.value,
            "source_url": source_url,
            "file_path": str(file_path),
        })

        super().__init__(
            document_id=document_id,
            source_type=source_type,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.file_path = file_path
        self.encoding = encoding

    def extract_blocks(self) -> List[EvidenceBlock]:
        if not self.file_path or not self.file_path.exists():
            return []

        text = self.file_path.read_text(encoding=self.encoding, errors="replace")

        return chunk_text_blocks(
            document_id=self.document_id,
            source_url=self.metadata.source_url,
            text=text,
            extraction_method="text_file_read",
            section_title=self.metadata.title or self.file_path.name,
        )


@dataclass
class JSONDocument(EvidenceDocument):
    file_path: Optional[Path] = None
    data: Any = None

    def __init__(
        self,
        *,
        file_path: Optional[Path] = None,
        data: Any = None,
        source_url: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
    ) -> None:
        file_path = Path(file_path) if file_path else None

        metadata = metadata or DocumentMetadata(
            source_url=source_url,
            raw_content_ref=str(file_path) if file_path else None,
        )

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.JSON_API.value,
            "source_url": source_url,
            "file_path": str(file_path) if file_path else None,
            "data_sample": str(data)[:120] if data is not None else None,
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.JSON_API,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.file_path = file_path
        self.data = data

    def extract_blocks(self) -> List[EvidenceBlock]:
        data = self.data

        if data is None and self.file_path and self.file_path.exists():
            data = json.loads(self.file_path.read_text(encoding="utf-8"))

        if data is None:
            return []

        records = self._records_from_json(data)

        if records:
            columns = make_unique_columns(records[0].keys())

            normalized_rows = [
                {column: clean_cell(record.get(column)) for column in columns}
                for record in records
            ]

            return make_table_blocks_from_rows(
                document_id=self.document_id,
                source_url=self.metadata.source_url,
                rows=normalized_rows,
                columns=columns,
                caption=self.metadata.title or "JSON records",
                table_number="json_records",
                extraction_method="json_records",
                data_ref=str(self.file_path) if self.file_path else None,
            )

        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        return chunk_text_blocks(
            document_id=self.document_id,
            source_url=self.metadata.source_url,
            text=text,
            extraction_method="json_text_fallback",
            section_title=self.metadata.title or "JSON document",
        )

    def _records_from_json(self, data: Any) -> Optional[List[JSONDict]]:
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data

        if isinstance(data, dict):
            for key in ["records", "data", "observations", "results", "items"]:
                value = data.get(key)
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    return value

        return None


@dataclass
class PriceSeriesDocument(EvidenceDocument):
    instrument: str = ""
    provider: Optional[str] = None
    observations: List[JSONDict] = field(default_factory=list)

    def __init__(
        self,
        *,
        instrument: str,
        observations: Sequence[JSONDict],
        provider: Optional[str] = None,
        source_url: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        credibility: Optional[SourceCredibility] = None,
        document_id: Optional[str] = None,
    ) -> None:
        metadata = metadata or DocumentMetadata(source_url=source_url, publisher=provider)

        document_id = document_id or stable_id("doc", {
            "source_type": SourceType.PRICE_SERIES.value,
            "instrument": instrument,
            "provider": provider,
            "source_url": source_url,
        })

        super().__init__(
            document_id=document_id,
            source_type=SourceType.PRICE_SERIES,
            metadata=metadata,
            credibility=credibility or SourceCredibility(),
        )

        self.instrument = instrument
        self.provider = provider
        self.observations = list(observations)

    def extract_blocks(self) -> List[EvidenceBlock]:
        fields = sorted({key for row in self.observations for key in row.keys()})
        ts = detect_time_series(fields, self.observations)

        dates = [
            parse_date_like(row.get("date"))
            for row in self.observations
            if parse_date_like(row.get("date"))
        ]

        return [
            TimeSeriesBlock(
                block_id=stable_id("blk", {
                    "document_id": self.document_id,
                    "instrument": self.instrument,
                    "n": len(self.observations),
                }),
                document_id=self.document_id,
                source_url=self.metadata.source_url,
                instrument=self.instrument,
                fields=fields,
                observations=self.observations,
                start_date=(ts or {}).get("start_date") or (min(dates) if dates else None),
                end_date=(ts or {}).get("end_date") or (max(dates) if dates else None),
                date_field=(ts or {}).get("date_column") or "date",
                value_fields=(ts or {}).get("value_columns") or [f for f in fields if f != "date"],
                frequency=(ts or {}).get("frequency"),
                extraction_method="structured_price_series",
            )
        ]


# ============================================================
# Tests
# ============================================================

def test() -> None:
    import tempfile

    print("Running document.py tests...")

    assert clamp_score(-1.0) == 0.0
    assert clamp_score(0.4) == 0.4
    assert clamp_score(2.0) == 1.0

    id_a = stable_id("doc", {"a": 1, "b": 2})
    id_b = stable_id("doc", {"b": 2, "a": 1})
    id_c = stable_id("doc", {"a": 1, "b": 3})

    assert id_a == id_b
    assert id_a != id_c
    assert id_a.startswith("doc_")

    credibility = SourceCredibility(score=1.7, tier=SourceCredibilityTier.HIGH)
    assert credibility.score == 1.0
    assert credibility.tier == SourceCredibilityTier.HIGH

    text_block = TextBlock(
        block_id="blk_text_1",
        document_id="doc_test",
        source_url="https://example.com/report",
        text="China imposed export restrictions on rare earth elements.",
        start_char=0,
        end_char=58,
        section_title="Export controls",
        extraction_method="unit_test",
        extraction_confidence=1.5,
    )

    assert text_block.block_type == EvidenceBlockType.TEXT
    assert text_block.extraction_confidence == 1.0
    assert "Export controls" in text_block.to_text()
    assert text_block.to_dict()["block_type"] == "text"

    table_block = TableBlock(
        block_id="blk_table_1",
        document_id="doc_test",
        source_url="https://example.com/table",
        caption="Rare earth production by country",
        columns=["country", "production_tonnes"],
        rows=[
            {"country": "China", "production_tonnes": 240000},
            {"country": "United States", "production_tonnes": 43000},
        ],
        page_number=4,
        extraction_method="unit_test_table",
        extraction_confidence=0.9,
    )

    table_text = table_block.to_text()

    assert table_block.block_type == EvidenceBlockType.TABLE
    assert "Rare earth production" in table_text
    assert "production_tonnes=numeric" in table_text
    assert table_block.row_count == 2
    assert table_block.column_schema["production_tonnes"]["semantic_type"] == "numeric"

    time_series_block = TimeSeriesBlock(
        block_id="blk_ts_1",
        document_id="doc_test",
        source_url="https://example.com/prices.csv",
        instrument="MP",
        fields=["date", "close", "volume"],
        observations=[
            {"date": "2025-04-01", "close": 20.0, "volume": 1000},
            {"date": "2025-04-02", "close": 21.5, "volume": 1500},
        ],
        start_date="2025-04-01",
        end_date="2025-04-02",
        summary="MP price increased over the sample period.",
        extraction_method="unit_test_price_series",
    )

    assert time_series_block.block_type == EvidenceBlockType.TIME_SERIES
    assert "MP" in time_series_block.to_text()
    assert time_series_block.to_dict()["block_type"] == "time_series"

    html_doc = HTMLWebpageDocument(
        source_url="https://example.com/article",
        html="<html><body>Example</body></html>",
        cleaned_text=(
            "China imposed export restrictions on several rare earth elements. "
            "The restrictions affected magnet supply chains."
        ),
        metadata=DocumentMetadata(
            title="Rare earth article",
            publisher="Example News",
            source_url="https://example.com/article",
        ),
        credibility=SourceCredibility(score=0.8, tier=SourceCredibilityTier.HIGH),
    )

    html_blocks = html_doc.ensure_blocks()
    assert html_doc.source_type == SourceType.HTML_WEBPAGE
    assert len(html_blocks) == 1
    assert isinstance(html_blocks[0], TextBlock)

    pdf_doc = PDFDocument(file_path=Path("missing.pdf"))
    assert pdf_doc.source_type == SourceType.PDF
    assert pdf_doc.ensure_blocks() == []

    csv_doc_missing = CSVDocument(file_path=Path("missing.csv"))
    assert csv_doc_missing.source_type == SourceType.CSV
    assert csv_doc_missing.ensure_blocks() == []

    price_doc = PriceSeriesDocument(
        instrument="MP",
        provider="Example Market Data",
        source_url="https://example.com/mp.csv",
        observations=[
            {"date": "2025-04-01", "close": 20.0, "volume": 1000},
            {"date": "2025-04-03", "close": 23.0, "volume": 1800},
            {"date": "2025-04-02", "close": 21.0, "volume": 1400},
        ],
        credibility=SourceCredibility(score=0.95, tier=SourceCredibilityTier.HIGH),
    )

    price_blocks = price_doc.ensure_blocks()
    assert len(price_blocks) == 1
    assert isinstance(price_blocks[0], TimeSeriesBlock)
    assert price_blocks[0].start_date == "2025-04-01"
    assert price_blocks[0].end_date == "2025-04-03"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        csv_path = tmp_path / "production.csv"
        csv_path.write_text(
            "date,country,production_tonnes\n"
            "2025-01-01,China,240000\n"
            "2025-02-01,United States,43000\n"
            "2025-03-01,Australia,18000\n",
            encoding="utf-8",
        )

        csv_doc = CSVDocument(file_path=csv_path)
        csv_blocks = csv_doc.ensure_blocks()

        assert any(isinstance(block, TableBlock) for block in csv_blocks)
        assert any(isinstance(block, TimeSeriesBlock) for block in csv_blocks)

        table = next(block for block in csv_blocks if isinstance(block, TableBlock))
        assert table.row_count == 3
        assert table.column_schema["production_tonnes"]["semantic_type"] == "numeric"

        text_path = tmp_path / "note.txt"
        text_path.write_text("Section A\n\nChina restricted critical minerals.\n", encoding="utf-8")

        text_doc = TextFileDocument(file_path=text_path)
        text_blocks = text_doc.ensure_blocks()
        assert len(text_blocks) == 1
        assert isinstance(text_blocks[0], TextBlock)
        assert "critical minerals" in text_blocks[0].to_text()

        json_path = tmp_path / "records.json"
        json_path.write_text(
            json.dumps({
                "records": [
                    {"date": "2025-01-01", "price": 10.5, "commodity": "dysprosium"},
                    {"date": "2025-02-01", "price": 11.2, "commodity": "dysprosium"},
                ]
            }),
            encoding="utf-8",
        )

        json_doc = JSONDocument(file_path=json_path)
        json_blocks = json_doc.ensure_blocks()
        assert any(isinstance(block, TableBlock) for block in json_blocks)
        assert any(isinstance(block, TimeSeriesBlock) for block in json_blocks)

        try:
            import pandas as pd

            xlsx_path = tmp_path / "workbook.xlsx"
            pd.DataFrame(
                [
                    {"date": "2025-01-01", "capacity": 100, "facility": "A"},
                    {"date": "2025-02-01", "capacity": 120, "facility": "A"},
                ]
            ).to_excel(xlsx_path, index=False, sheet_name="capacity")

            xlsx_doc = SpreadsheetDocument(file_path=xlsx_path)
            xlsx_blocks = xlsx_doc.ensure_blocks()
            assert any(isinstance(block, TableBlock) for block in xlsx_blocks)
            assert any(isinstance(block, TimeSeriesBlock) for block in xlsx_blocks)
        except Exception as error:
            print(f"[warn] spreadsheet test skipped: {error}")

    serialized = json.dumps(price_doc.to_dict(), ensure_ascii=False)
    reparsed = json.loads(serialized)

    assert reparsed["document_id"] == price_doc.document_id
    assert reparsed["blocks"][0]["instrument"] == "MP"

    print("All document.py tests passed.")


if __name__ == "__main__":
    test()