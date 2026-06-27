#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JSONDict = Dict[str, Any]


# ============================================================
# Data classes
# ============================================================

@dataclass
class RiskFactor:
    name: str
    mechanism: str
    severity: float
    likelihood: float
    exposure: float
    mitigation: float = 1.0
    confidence: float = 3.0
    evidence_ids: List[str] = field(default_factory=list)
    required_variables: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class QuantitativeVariable:
    name: str
    value: float
    unit: Optional[str] = None
    entity: Optional[str] = None
    period: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class TableAnalysisRequest:
    block_id: str
    operation: str
    columns: List[str] = field(default_factory=list)
    group_by: Optional[str] = None
    value_column: Optional[str] = None
    n: int = 5
    baseline_start: Optional[str] = None
    baseline_end: Optional[str] = None
    shock_start: Optional[str] = None
    shock_end: Optional[str] = None
    date_column: Optional[str] = None
    window: int = 3


@dataclass
class Scenario:
    name: str
    exposure_value: Optional[float] = None
    shock_pct: Optional[float] = None
    pass_through_pct: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)
    affected_factor_names: List[str] = field(default_factory=list)


@dataclass
class NumericColumnSummary:
    column: str
    count: int
    missing: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]


@dataclass
class ConcentrationMetrics:
    value_column: str
    group_column: Optional[str]
    total: float
    top1_share_pct: float
    top3_share_pct: float
    hhi: float
    concentration_level: str
    rows_used: int


@dataclass
class TimeSeriesChange:
    value_column: str
    start_date: Optional[str]
    end_date: Optional[str]
    start_value: Optional[float]
    end_value: Optional[float]
    absolute_change: Optional[float]
    percent_change: Optional[float]


@dataclass
class RiskScore:
    factor_name: str
    raw_score: float
    adjusted_score: float
    risk_level: str
    severity: float
    likelihood: float
    exposure: float
    mitigation: float
    confidence: float


@dataclass
class ScenarioResult:
    name: str
    impacted_value: Optional[float]
    formula: str
    assumptions: List[str]
    affected_factor_names: List[str]


@dataclass
class RiskAnalysis:
    question: str
    risk_level: str
    risk_score: float
    confidence_level: str
    confidence_score: float
    risk_factors: List[JSONDict]
    quantitative_variables: List[JSONDict]
    table_results: List[JSONDict]
    scenario_results: List[JSONDict]
    missing_variables: List[str]
    evidence_basis: List[JSONDict]
    memo: str
    methodology: JSONDict


# ============================================================
# Generic utilities
# ============================================================

def clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        x = default
    return max(low, min(high, x))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except Exception:
        return None


def parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        from dateutil.parser import parse
        return parse(text, fuzzy=False).date()
    except Exception:
        return None


def compact(text: Any, max_chars: int = 700) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def risk_level(score: float) -> str:
    if score >= 71:
        return "severe"
    if score >= 41:
        return "high"
    if score >= 21:
        return "moderate"
    return "low"


def confidence_level(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 45:
        return "moderate"
    return "low"


def concentration_level(top1_share_pct: float, hhi: float) -> str:
    if top1_share_pct >= 80 or hhi >= 5000:
        return "severe"
    if top1_share_pct >= 60 or hhi >= 2500:
        return "high"
    if top1_share_pct >= 40 or hhi >= 1500:
        return "moderate"
    return "low"


def unique_keep_order(items: Iterable[Any]) -> List[Any]:
    out = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ============================================================
# Block loading from document / SARG JSON
# ============================================================

def load_blocks_from_document_json(document_json: JSONDict) -> List[JSONDict]:
    blocks = document_json.get("blocks")
    if isinstance(blocks, list):
        return blocks
    return []


def load_blocks_from_paths(paths: Sequence[Path]) -> List[JSONDict]:
    blocks: List[JSONDict] = []
    for path in paths:
        data = read_json(path)
        blocks.extend(load_blocks_from_document_json(data))
    return blocks


def table_blocks(blocks: Sequence[JSONDict]) -> List[JSONDict]:
    return [b for b in blocks if b.get("block_type") == "table"]


def time_series_blocks(blocks: Sequence[JSONDict]) -> List[JSONDict]:
    return [b for b in blocks if b.get("block_type") == "time_series"]


def find_block(blocks: Sequence[JSONDict], block_id: str) -> Optional[JSONDict]:
    for block in blocks:
        if block.get("block_id") == block_id:
            return block
    return None


def sarg_evidence_basis(sarg_result: Optional[JSONDict]) -> List[JSONDict]:
    if not sarg_result:
        return []
    ctx = sarg_result.get("sarg_context") or sarg_result.get("context") or {}
    out = []
    for item in ctx.get("evidence") or []:
        out.append({
            "evidence_id": item.get("evidence_id"),
            "claim_key": item.get("claim_key"),
            "source_title": item.get("source_title"),
            "source_url": item.get("source_url"),
            "text": compact(item.get("text"), 500),
        })
    return out


# ============================================================
# Table tools
# ============================================================

def table_summary(block: JSONDict) -> JSONDict:
    rows = block.get("rows") or []
    columns = block.get("columns") or list(rows[0].keys()) if rows else []
    column_schema = block.get("column_schema") or {}

    missing_counts = {}
    numeric_columns = []
    date_columns = []

    for col in columns:
        values = [row.get(col) for row in rows]
        missing_counts[col] = sum(1 for v in values if v in {None, ""})

        numeric_values = [safe_float(v) for v in values]
        numeric_non_null = [v for v in numeric_values if v is not None]

        date_values = [parse_date(v) for v in values]
        date_non_null = [v for v in date_values if v is not None]

        if numeric_non_null and len(numeric_non_null) >= max(1, int(0.6 * len([v for v in values if v not in {None, ""}]))):
            numeric_columns.append(col)

        if date_non_null and len(date_non_null) >= max(1, int(0.6 * len([v for v in values if v not in {None, ""}]))):
            date_columns.append(col)

        if col in column_schema:
            semantic = column_schema[col].get("semantic_type")
            if semantic == "numeric" and col not in numeric_columns:
                numeric_columns.append(col)
            if semantic == "date" and col not in date_columns:
                date_columns.append(col)

    return {
        "block_id": block.get("block_id"),
        "caption": block.get("caption"),
        "row_count": len(rows),
        "columns": list(columns),
        "numeric_columns": numeric_columns,
        "date_columns": date_columns,
        "missing_counts": missing_counts,
    }


def column_stats(block: JSONDict, column: str) -> NumericColumnSummary:
    rows = block.get("rows") or []
    values = [safe_float(row.get(column)) for row in rows]
    nums = [v for v in values if v is not None]

    if not nums:
        return NumericColumnSummary(column, 0, len(values), None, None, None, None, None)

    std = statistics.stdev(nums) if len(nums) >= 2 else 0.0

    return NumericColumnSummary(
        column=column,
        count=len(nums),
        missing=len(values) - len(nums),
        minimum=min(nums),
        maximum=max(nums),
        mean=statistics.mean(nums),
        median=statistics.median(nums),
        std=std,
    )


def group_sum(block: JSONDict, group_by: str, value_column: str) -> List[JSONDict]:
    rows = block.get("rows") or []
    groups: Dict[str, float] = {}

    for row in rows:
        group = str(row.get(group_by))
        value = safe_float(row.get(value_column))
        if value is None:
            continue
        groups[group] = groups.get(group, 0.0) + value

    return [
        {group_by: key, value_column: value}
        for key, value in sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    ]


def group_share(block: JSONDict, group_by: str, value_column: str) -> List[JSONDict]:
    totals = group_sum(block, group_by, value_column)
    total = sum(safe_float(row[value_column]) or 0.0 for row in totals)

    if total == 0:
        return [
            {**row, "share_pct": None}
            for row in totals
        ]

    return [
        {**row, "share_pct": 100.0 * (safe_float(row[value_column]) or 0.0) / total}
        for row in totals
    ]


def top_n(block: JSONDict, value_column: str, n: int = 5, group_by: Optional[str] = None) -> List[JSONDict]:
    if group_by:
        rows = group_sum(block, group_by, value_column)
    else:
        rows = block.get("rows") or []

    return sorted(
        rows,
        key=lambda row: safe_float(row.get(value_column)) if safe_float(row.get(value_column)) is not None else float("-inf"),
        reverse=True,
    )[:n]


def concentration_metrics(
    block: JSONDict,
    value_column: str,
    group_column: Optional[str] = None,
) -> ConcentrationMetrics:
    if group_column:
        rows = group_sum(block, group_column, value_column)
    else:
        rows = block.get("rows") or []

    values = [safe_float(row.get(value_column)) for row in rows]
    nums = [v for v in values if v is not None and v >= 0]
    total = sum(nums)

    if not nums or total <= 0:
        return ConcentrationMetrics(value_column, group_column, 0.0, 0.0, 0.0, 0.0, "low", 0)

    shares = sorted([100.0 * v / total for v in nums], reverse=True)
    top1 = shares[0]
    top3 = sum(shares[:3])
    hhi = sum(s * s for s in shares)

    return ConcentrationMetrics(
        value_column=value_column,
        group_column=group_column,
        total=total,
        top1_share_pct=top1,
        top3_share_pct=top3,
        hhi=hhi,
        concentration_level=concentration_level(top1, hhi),
        rows_used=len(nums),
    )


# ============================================================
# Time-series tools
# ============================================================

def ordered_observations(block: JSONDict, date_column: Optional[str] = None) -> List[JSONDict]:
    date_column = date_column or block.get("date_field") or "date"
    rows = block.get("observations") or block.get("rows") or []

    enriched = []
    for row in rows:
        d = parse_date(row.get(date_column))
        if d is None:
            continue
        enriched.append((d, row))

    enriched.sort(key=lambda pair: pair[0])
    return [row for _, row in enriched]


def time_series_summary(block: JSONDict) -> JSONDict:
    date_column = block.get("date_field") or "date"
    value_fields = block.get("value_fields") or [
        col for col in block.get("fields") or []
        if col != date_column
    ]

    rows = ordered_observations(block, date_column)
    dates = [parse_date(row.get(date_column)) for row in rows]
    dates = [d for d in dates if d is not None]

    return {
        "block_id": block.get("block_id"),
        "instrument": block.get("instrument"),
        "date_field": date_column,
        "value_fields": value_fields,
        "observation_count": len(rows),
        "start_date": min(dates).isoformat() if dates else None,
        "end_date": max(dates).isoformat() if dates else None,
        "frequency": block.get("frequency"),
    }


def value_at_or_after(rows: Sequence[JSONDict], date_column: str, value_column: str, target: str) -> Tuple[Optional[str], Optional[float]]:
    target_date = parse_date(target)
    if target_date is None:
        return None, None

    for row in rows:
        d = parse_date(row.get(date_column))
        v = safe_float(row.get(value_column))
        if d is not None and v is not None and d >= target_date:
            return d.isoformat(), v

    return None, None


def value_at_or_before(rows: Sequence[JSONDict], date_column: str, value_column: str, target: str) -> Tuple[Optional[str], Optional[float]]:
    target_date = parse_date(target)
    if target_date is None:
        return None, None

    best_d = None
    best_v = None

    for row in rows:
        d = parse_date(row.get(date_column))
        v = safe_float(row.get(value_column))
        if d is not None and v is not None and d <= target_date:
            best_d = d
            best_v = v

    return (best_d.isoformat(), best_v) if best_d else (None, None)


def percent_change(
    block: JSONDict,
    value_column: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: Optional[str] = None,
) -> TimeSeriesChange:
    date_column = date_column or block.get("date_field") or "date"
    rows = ordered_observations(block, date_column)

    if not rows:
        return TimeSeriesChange(value_column, None, None, None, None, None, None)

    if start_date:
        actual_start, start_value = value_at_or_after(rows, date_column, value_column, start_date)
    else:
        actual_start = parse_date(rows[0].get(date_column)).isoformat()
        start_value = safe_float(rows[0].get(value_column))

    if end_date:
        actual_end, end_value = value_at_or_before(rows, date_column, value_column, end_date)
    else:
        actual_end = parse_date(rows[-1].get(date_column)).isoformat()
        end_value = safe_float(rows[-1].get(value_column))

    if start_value is None or end_value is None:
        return TimeSeriesChange(value_column, actual_start, actual_end, start_value, end_value, None, None)

    absolute = end_value - start_value
    pct = None if start_value == 0 else 100.0 * absolute / start_value

    return TimeSeriesChange(value_column, actual_start, actual_end, start_value, end_value, absolute, pct)


def window_mean(
    block: JSONDict,
    value_column: str,
    start_date: str,
    end_date: str,
    date_column: Optional[str] = None,
) -> Optional[float]:
    date_column = date_column or block.get("date_field") or "date"
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        return None

    values = []
    for row in ordered_observations(block, date_column):
        d = parse_date(row.get(date_column))
        v = safe_float(row.get(value_column))
        if d is not None and v is not None and start <= d <= end:
            values.append(v)

    return statistics.mean(values) if values else None


def window_change(
    block: JSONDict,
    value_column: str,
    baseline_start: str,
    baseline_end: str,
    shock_start: str,
    shock_end: str,
    date_column: Optional[str] = None,
) -> JSONDict:
    base = window_mean(block, value_column, baseline_start, baseline_end, date_column)
    shock = window_mean(block, value_column, shock_start, shock_end, date_column)

    if base is None or shock is None:
        return {
            "value_column": value_column,
            "baseline_mean": base,
            "shock_mean": shock,
            "absolute_change": None,
            "percent_change": None,
        }

    absolute = shock - base
    pct = None if base == 0 else 100.0 * absolute / base

    return {
        "value_column": value_column,
        "baseline_window": [baseline_start, baseline_end],
        "shock_window": [shock_start, shock_end],
        "baseline_mean": base,
        "shock_mean": shock,
        "absolute_change": absolute,
        "percent_change": pct,
    }


def period_pct_changes(block: JSONDict, value_column: str, date_column: Optional[str] = None) -> List[float]:
    date_column = date_column or block.get("date_field") or "date"
    rows = ordered_observations(block, date_column)
    changes = []

    previous = None
    for row in rows:
        value = safe_float(row.get(value_column))
        if value is None:
            continue
        if previous is not None and previous != 0:
            changes.append(100.0 * (value - previous) / previous)
        previous = value

    return changes


def volatility(block: JSONDict, value_column: str, date_column: Optional[str] = None) -> JSONDict:
    changes = period_pct_changes(block, value_column, date_column)
    if len(changes) < 2:
        return {
            "value_column": value_column,
            "period_pct_change_std": None,
            "observations": len(changes),
        }

    return {
        "value_column": value_column,
        "period_pct_change_std": statistics.stdev(changes),
        "period_pct_change_mean": statistics.mean(changes),
        "observations": len(changes),
    }


def max_drawdown(block: JSONDict, value_column: str, date_column: Optional[str] = None) -> JSONDict:
    date_column = date_column or block.get("date_field") or "date"
    rows = ordered_observations(block, date_column)

    peak = None
    peak_date = None
    worst = 0.0
    worst_date = None

    for row in rows:
        d = parse_date(row.get(date_column))
        v = safe_float(row.get(value_column))
        if d is None or v is None:
            continue

        if peak is None or v > peak:
            peak = v
            peak_date = d

        if peak and peak != 0:
            drawdown = 100.0 * (v - peak) / peak
            if drawdown < worst:
                worst = drawdown
                worst_date = d

    return {
        "value_column": value_column,
        "max_drawdown_pct": worst,
        "peak_date": peak_date.isoformat() if peak_date else None,
        "trough_date": worst_date.isoformat() if worst_date else None,
    }


# ============================================================
# Risk calculations
# ============================================================

def score_risk_factor(factor: RiskFactor) -> RiskScore:
    severity = clamp(factor.severity, 1, 5, 1)
    likelihood = clamp(factor.likelihood, 1, 5, 1)
    exposure = clamp(factor.exposure, 1, 5, 1)
    mitigation = clamp(factor.mitigation, 1, 5, 1)
    confidence = clamp(factor.confidence, 1, 5, 3)

    raw = severity * likelihood * exposure / 125.0 * 100.0
    mitigation_discount = (mitigation - 1.0) / 5.0
    adjusted = raw * (1.0 - mitigation_discount)

    return RiskScore(
        factor_name=factor.name,
        raw_score=round(raw, 2),
        adjusted_score=round(adjusted, 2),
        risk_level=risk_level(adjusted),
        severity=severity,
        likelihood=likelihood,
        exposure=exposure,
        mitigation=mitigation,
        confidence=confidence,
    )


def score_risk_model(factors: Sequence[RiskFactor]) -> JSONDict:
    scores = [score_risk_factor(f) for f in factors]

    if not scores:
        return {
            "risk_score": 0.0,
            "risk_level": "low",
            "factor_scores": [],
            "method": "No risk factors supplied.",
        }

    adjusted = [s.adjusted_score for s in scores]
    top = max(adjusted)
    top_three = sorted(adjusted, reverse=True)[:3]
    overall = 0.6 * top + 0.4 * statistics.mean(top_three)

    return {
        "risk_score": round(overall, 2),
        "risk_level": risk_level(overall),
        "factor_scores": [asdict(s) for s in scores],
        "method": "0.6 * max adjusted factor risk + 0.4 * average of top three adjusted factor risks.",
    }


def scenario_impact(scenario: Scenario) -> ScenarioResult:
    formula = "exposure_value * shock_pct * pass_through_pct"

    if scenario.exposure_value is None or scenario.shock_pct is None or scenario.pass_through_pct is None:
        impacted = None
    else:
        impacted = (
            float(scenario.exposure_value)
            * float(scenario.shock_pct)
            * float(scenario.pass_through_pct)
        )

    return ScenarioResult(
        name=scenario.name,
        impacted_value=impacted,
        formula=formula,
        assumptions=scenario.assumptions,
        affected_factor_names=scenario.affected_factor_names,
    )


def confidence_score(
    *,
    factors: Sequence[RiskFactor],
    quantitative_variables: Sequence[QuantitativeVariable],
    table_results: Sequence[JSONDict],
    missing_variables: Sequence[str],
    evidence_count: int,
) -> JSONDict:
    score = 20.0
    score += min(25.0, evidence_count * 5.0)
    score += min(20.0, len(factors) * 4.0)
    score += min(20.0, len(quantitative_variables) * 5.0)
    score += min(20.0, len(table_results) * 4.0)
    score -= min(35.0, len(missing_variables) * 4.0)

    if factors:
        score += 5.0 * statistics.mean([clamp(f.confidence, 1, 5, 3) for f in factors]) / 5.0

    score = max(0.0, min(100.0, score))

    return {
        "confidence_score": round(score, 2),
        "confidence_level": confidence_level(score),
        "method": "Evidence count, risk factors, quantitative variables, table/time-series computations, and missing-variable penalties.",
    }


# ============================================================
# Risk model parsing layer
# ============================================================

# Risk model extraction/orchestration is intentionally owned by sarg.py.
# This module remains deterministic: it parses a supplied risk model and
# executes table, time-series, scoring, and scenario calculations.

def parse_risk_model(raw: JSONDict) -> Tuple[List[RiskFactor], List[QuantitativeVariable], List[TableAnalysisRequest], List[Scenario], List[str]]:
    factors = []
    variables = []
    requests = []
    scenarios = []
    missing = []

    for item in raw.get("risk_factors") or []:
        if not isinstance(item, dict):
            continue
        factors.append(
            RiskFactor(
                name=str(item.get("name") or "unnamed factor"),
                mechanism=str(item.get("mechanism") or ""),
                severity=clamp(item.get("severity"), 1, 5, 1),
                likelihood=clamp(item.get("likelihood"), 1, 5, 1),
                exposure=clamp(item.get("exposure"), 1, 5, 1),
                mitigation=clamp(item.get("mitigation"), 1, 5, 1),
                confidence=clamp(item.get("confidence"), 1, 5, 3),
                evidence_ids=[str(x) for x in item.get("evidence_ids") or []],
                required_variables=[str(x) for x in item.get("required_variables") or []],
                notes=item.get("notes"),
            )
        )

    for item in raw.get("quantitative_variables") or []:
        if not isinstance(item, dict):
            continue
        value = safe_float(item.get("value"))
        if value is None:
            continue
        variables.append(
            QuantitativeVariable(
                name=str(item.get("name") or "unnamed variable"),
                value=value,
                unit=item.get("unit"),
                entity=item.get("entity"),
                period=item.get("period"),
                evidence_ids=[str(x) for x in item.get("evidence_ids") or []],
                notes=item.get("notes"),
            )
        )

    for item in raw.get("table_analysis_requests") or []:
        if not isinstance(item, dict):
            continue
        requests.append(
            TableAnalysisRequest(
                block_id=str(item.get("block_id") or ""),
                operation=str(item.get("operation") or ""),
                columns=[str(x) for x in item.get("columns") or []],
                group_by=item.get("group_by"),
                value_column=item.get("value_column"),
                n=int(item.get("n") or 5),
                baseline_start=item.get("baseline_start"),
                baseline_end=item.get("baseline_end"),
                shock_start=item.get("shock_start"),
                shock_end=item.get("shock_end"),
                date_column=item.get("date_column"),
                window=int(item.get("window") or 3),
            )
        )

    for item in raw.get("scenarios") or []:
        if not isinstance(item, dict):
            continue
        scenarios.append(
            Scenario(
                name=str(item.get("name") or "unnamed scenario"),
                exposure_value=safe_float(item.get("exposure_value")),
                shock_pct=safe_float(item.get("shock_pct")),
                pass_through_pct=safe_float(item.get("pass_through_pct")),
                assumptions=[str(x) for x in item.get("assumptions") or []],
                affected_factor_names=[str(x) for x in item.get("affected_factor_names") or []],
            )
        )

    for item in raw.get("missing_variables") or []:
        if str(item).strip():
            missing.append(str(item).strip())

    for factor in factors:
        missing.extend(factor.required_variables)

    return factors, variables, requests, scenarios, unique_keep_order(missing)


# ============================================================
# Table/time-series request execution
# ============================================================

def execute_table_request(request: TableAnalysisRequest, blocks: Sequence[JSONDict]) -> JSONDict:
    block = find_block(blocks, request.block_id)

    if not block:
        return {
            "operation": request.operation,
            "block_id": request.block_id,
            "ok": False,
            "error": "Block not found.",
        }

    try:
        if request.operation == "table_summary":
            result = table_summary(block)

        elif request.operation == "column_stats":
            column = request.value_column or (request.columns[0] if request.columns else "")
            result = asdict(column_stats(block, column))

        elif request.operation == "group_sum":
            result = group_sum(block, request.group_by or "", request.value_column or "")

        elif request.operation == "group_share":
            result = group_share(block, request.group_by or "", request.value_column or "")

        elif request.operation == "top_n":
            result = top_n(block, request.value_column or "", request.n, request.group_by)

        elif request.operation == "concentration":
            result = asdict(concentration_metrics(block, request.value_column or "", request.group_by))

        elif request.operation == "time_series_summary":
            result = time_series_summary(block)

        elif request.operation == "percent_change":
            result = asdict(
                percent_change(
                    block,
                    request.value_column or "",
                    start_date=request.baseline_start,
                    end_date=request.shock_end or request.baseline_end,
                    date_column=request.date_column,
                )
            )

        elif request.operation == "window_change":
            result = window_change(
                block,
                request.value_column or "",
                baseline_start=request.baseline_start or "",
                baseline_end=request.baseline_end or "",
                shock_start=request.shock_start or "",
                shock_end=request.shock_end or "",
                date_column=request.date_column,
            )

        elif request.operation == "volatility":
            result = volatility(block, request.value_column or "", request.date_column)

        elif request.operation == "max_drawdown":
            result = max_drawdown(block, request.value_column or "", request.date_column)

        else:
            return {
                "operation": request.operation,
                "block_id": request.block_id,
                "ok": False,
                "error": f"Unsupported operation: {request.operation}",
            }

        return {
            "operation": request.operation,
            "block_id": request.block_id,
            "ok": True,
            "result": result,
        }

    except Exception as error:
        return {
            "operation": request.operation,
            "block_id": request.block_id,
            "ok": False,
            "error": str(error),
        }


# ============================================================
# Analysis / memo
# ============================================================

def make_risk_memo(analysis: RiskAnalysis) -> str:
    lines = []
    lines.append("Bottom line")
    lines.append(
        f"Risk is assessed as {analysis.risk_level.upper()} "
        f"({analysis.risk_score:.1f}/100), with {analysis.confidence_level} confidence "
        f"({analysis.confidence_score:.1f}/100)."
    )

    lines.append("")
    lines.append("Risk factors")
    for item in analysis.risk_factors:
        score = item.get("score", {})
        lines.append(
            f"- {item['name']}: {score.get('risk_level')} "
            f"({score.get('adjusted_score')}/100). {item.get('mechanism')}"
        )

    if analysis.quantitative_variables:
        lines.append("")
        lines.append("Quantitative variables")
        for var in analysis.quantitative_variables:
            unit = f" {var.get('unit')}" if var.get("unit") else ""
            entity = f" for {var.get('entity')}" if var.get("entity") else ""
            period = f" ({var.get('period')})" if var.get("period") else ""
            lines.append(f"- {var.get('name')}: {var.get('value')}{unit}{entity}{period}")

    if analysis.table_results:
        lines.append("")
        lines.append("Table/time-series computations")
        for result in analysis.table_results[:8]:
            lines.append(f"- {result.get('operation')} on {result.get('block_id')}: {compact(result.get('result') or result.get('error'), 300)}")

    if analysis.scenario_results:
        lines.append("")
        lines.append("Scenarios")
        for scenario in analysis.scenario_results:
            lines.append(
                f"- {scenario.get('name')}: impacted_value={scenario.get('impacted_value')} "
                f"using {scenario.get('formula')}"
            )

    if analysis.missing_variables:
        lines.append("")
        lines.append("Missing variables")
        for item in analysis.missing_variables[:8]:
            lines.append(f"- {item}")

    return "\n".join(lines)


def analyze(
    *,
    question: str = "",
    sarg_result: Optional[JSONDict] = None,
    document_jsons: Optional[Sequence[JSONDict]] = None,
    risk_model: Optional[JSONDict] = None,
    use_llm: bool = True,
    model: str = "gpt-4.1-mini",
) -> RiskAnalysis:
    document_jsons = list(document_jsons or [])
    blocks: List[JSONDict] = []

    for doc in document_jsons:
        blocks.extend(load_blocks_from_document_json(doc))

    if risk_model is None:
        risk_model = {
            "risk_factors": [],
            "quantitative_variables": [],
            "table_analysis_requests": [],
            "scenarios": [],
            "missing_variables": [
                "No risk model supplied. Risk model extraction is handled by sarg.py."
            ],
        }

    factors, variables, requests, scenarios, missing = parse_risk_model(risk_model)

    table_results = [execute_table_request(req, blocks) for req in requests]
    scenario_results = [asdict(scenario_impact(s)) for s in scenarios]

    score = score_risk_model(factors)

    evidence = sarg_evidence_basis(sarg_result)
    confidence = confidence_score(
        factors=factors,
        quantitative_variables=variables,
        table_results=[r for r in table_results if r.get("ok")],
        missing_variables=missing,
        evidence_count=len(evidence),
    )

    factor_payload = []
    factor_scores = {item["factor_name"]: item for item in score.get("factor_scores", [])}

    for factor in factors:
        data = asdict(factor)
        data["score"] = factor_scores.get(factor.name, {})
        factor_payload.append(data)

    analysis = RiskAnalysis(
        question=question or (sarg_result or {}).get("question", ""),
        risk_level=score["risk_level"],
        risk_score=score["risk_score"],
        confidence_level=confidence["confidence_level"],
        confidence_score=confidence["confidence_score"],
        risk_factors=factor_payload,
        quantitative_variables=[asdict(v) for v in variables],
        table_results=table_results,
        scenario_results=scenario_results,
        missing_variables=missing,
        evidence_basis=evidence,
        memo="",
        methodology={
            "risk_score": score.get("method"),
            "factor_formula": "severity * likelihood * exposure / 125 * 100, discounted by mitigation.",
            "scenario_formula": "exposure_value * shock_pct * pass_through_pct.",
            "confidence": confidence.get("method"),
            "note": "Quantitative calculations are deterministic. Risk model extraction is separated from calculation.",
        },
    )
    analysis.memo = make_risk_memo(analysis)
    return analysis


def summarize_analysis(analysis: RiskAnalysis) -> str:
    lines = []
    lines.append("RISK ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Question: {analysis.question}")
    lines.append(f"Risk: {analysis.risk_level} ({analysis.risk_score}/100)")
    lines.append(f"Confidence: {analysis.confidence_level} ({analysis.confidence_score}/100)")
    lines.append("")
    lines.append(analysis.memo)
    return "\n".join(lines)


# ============================================================
# Tests
# ============================================================

def synthetic_document_json() -> JSONDict:
    return {
        "document_id": "doc_synthetic",
        "source_type": "csv",
        "blocks": [
            {
                "block_id": "tbl_imports",
                "block_type": "table",
                "caption": "Imports by source country",
                "columns": ["country", "imports_tonnes"],
                "rows": [
                    {"country": "A", "imports_tonnes": 80},
                    {"country": "B", "imports_tonnes": 15},
                    {"country": "C", "imports_tonnes": 5},
                ],
            },
            {
                "block_id": "ts_price",
                "block_type": "time_series",
                "instrument": "Material price index",
                "date_field": "date",
                "value_fields": ["price"],
                "fields": ["date", "price"],
                "observations": [
                    {"date": "2025-01-01", "price": 100},
                    {"date": "2025-02-01", "price": 110},
                    {"date": "2025-03-01", "price": 121},
                    {"date": "2025-04-01", "price": 90},
                ],
            },
        ],
    }


def synthetic_risk_model() -> JSONDict:
    return {
        "risk_factors": [
            {
                "name": "Input supply exposure",
                "mechanism": "High source concentration increases disruption sensitivity.",
                "severity": 5,
                "likelihood": 4,
                "exposure": 5,
                "mitigation": 2,
                "confidence": 4,
                "evidence_ids": ["ev_1"],
                "required_variables": ["Inventory coverage in days"],
            }
        ],
        "quantitative_variables": [
            {
                "name": "Source A import share",
                "value": 0.80,
                "unit": "share",
                "entity": "input supply",
                "period": "2025",
                "evidence_ids": ["ev_1"],
            }
        ],
        "table_analysis_requests": [
            {
                "block_id": "tbl_imports",
                "operation": "concentration",
                "group_by": "country",
                "value_column": "imports_tonnes",
            },
            {
                "block_id": "ts_price",
                "operation": "percent_change",
                "value_column": "price",
                "baseline_start": "2025-01-01",
                "shock_end": "2025-03-01",
                "date_column": "date",
            },
            {
                "block_id": "ts_price",
                "operation": "max_drawdown",
                "value_column": "price",
                "date_column": "date",
            },
        ],
        "scenarios": [
            {
                "name": "25% supply shock with 50% pass-through",
                "exposure_value": 0.80,
                "shock_pct": 0.25,
                "pass_through_pct": 0.50,
                "assumptions": ["80% exposed supply share", "25% shock", "50% pass-through"],
                "affected_factor_names": ["Input supply exposure"],
            }
        ],
        "missing_variables": ["Substitution cost curve"],
    }


def test_table_tools() -> None:
    print("[test] table tools: summary, stats, group share, concentration")

    block = synthetic_document_json()["blocks"][0]

    summary = table_summary(block)
    assert summary["row_count"] == 3
    assert "imports_tonnes" in summary["numeric_columns"]

    stats = column_stats(block, "imports_tonnes")
    assert stats.count == 3
    assert stats.maximum == 80

    shares = group_share(block, "country", "imports_tonnes")
    assert round(shares[0]["share_pct"], 1) == 80.0

    concentration = concentration_metrics(block, "imports_tonnes", "country")
    assert round(concentration.top1_share_pct, 1) == 80.0
    assert concentration.concentration_level == "severe"


def test_time_series_tools() -> None:
    print("[test] time-series tools: summary, percent change, volatility, drawdown")

    block = synthetic_document_json()["blocks"][1]

    summary = time_series_summary(block)
    assert summary["observation_count"] == 4
    assert summary["start_date"] == "2025-01-01"

    change = percent_change(block, "price", "2025-01-01", "2025-03-01")
    assert round(change.percent_change or 0, 1) == 21.0

    vol = volatility(block, "price")
    assert vol["period_pct_change_std"] is not None

    dd = max_drawdown(block, "price")
    assert round(dd["max_drawdown_pct"], 1) == -25.6


def test_risk_calculations() -> None:
    print("[test] risk scoring and scenario arithmetic")

    factor = RiskFactor(
        name="Test factor",
        mechanism="test",
        severity=5,
        likelihood=4,
        exposure=5,
        mitigation=2,
        confidence=4,
    )

    score = score_risk_factor(factor)
    assert score.adjusted_score > 60
    assert score.risk_level in {"high", "severe"}

    scenario = Scenario(
        name="shock",
        exposure_value=0.80,
        shock_pct=0.25,
        pass_through_pct=0.50,
    )
    impact = scenario_impact(scenario)
    assert impact.impacted_value == 0.1


def test_end_to_end_without_llm() -> None:
    print("[test] end-to-end deterministic analysis with supplied risk model and document blocks")

    analysis = analyze(
        question="Synthetic risk test",
        document_jsons=[synthetic_document_json()],
        risk_model=synthetic_risk_model(),
        use_llm=False,
    )

    assert analysis.risk_level in {"high", "severe"}
    assert analysis.table_results
    assert len(analysis.scenario_results) == 1
    assert analysis.scenario_results[0]["impacted_value"] == 0.1
    assert "Bottom line" in analysis.memo or "Risk factors" in analysis.memo


def test_serialization() -> None:
    print("[test] analysis JSON serialization")

    analysis = analyze(
        question="Synthetic risk test",
        document_jsons=[synthetic_document_json()],
        risk_model=synthetic_risk_model(),
        use_llm=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "risk.json"
        write_json(path, asdict(analysis))
        data = read_json(path)
        assert data["question"] == "Synthetic risk test"
        assert data["table_results"]


def run_tests() -> None:
    print("Running risk_tools.py tests...")
    test_table_tools()
    test_time_series_tools()
    test_risk_calculations()
    test_end_to_end_without_llm()
    test_serialization()
    print("All risk_tools.py tests passed.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic quantitative risk tools")

    parser.add_argument("--sarg-json", type=Path)
    parser.add_argument("--document-json", type=Path, action="append", default=[])
    parser.add_argument("--risk-model-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test:
        run_tests()
        return 0

    sarg_result = read_json(args.sarg_json) if args.sarg_json else None
    document_jsons = [read_json(path) for path in args.document_json]
    risk_model = read_json(args.risk_model_json) if args.risk_model_json else None

    analysis = analyze(
        question=(sarg_result or {}).get("question", ""),
        sarg_result=sarg_result,
        document_jsons=document_jsons,
        risk_model=risk_model,
        use_llm=not args.no_llm,
        model=args.model,
    )

    if args.output:
        write_json(args.output, asdict(analysis))
        print(f"[done] wrote {args.output}")

    if args.summary or not args.output:
        print(summarize_analysis(analysis))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())