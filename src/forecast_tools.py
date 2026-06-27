#!/usr/bin/env python3
"""
Basic time-series forecasting tools for Lanthic Intelligence.

This module intentionally uses simple, dependency-light forecasting methods:
- naive
- moving average
- linear trend
- simple exponential smoothing
- seasonal naive where enough monthly/quarterly data exists
- ensemble over available methods

It does not invent forecasts from aggregate non-time-series values. At least three
numeric observations are required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

JSONDict = Dict[str, Any]


# ============================================================
# I/O helpers
# ============================================================

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ============================================================
# Period handling
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None

    try:
        number = float(match.group(0))
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def infer_frequency(periods: Sequence[str], explicit: Optional[str] = None) -> str:
    if explicit:
        value = explicit.lower().strip()
        aliases = {
            "annual": "year",
            "yearly": "year",
            "y": "year",
            "monthly": "month",
            "m": "month",
            "quarterly": "quarter",
            "q": "quarter",
            "daily": "day",
            "d": "day",
        }
        return aliases.get(value, value)

    sample = [str(p) for p in periods if str(p).strip()]
    if not sample:
        return "period"

    if all(re.match(r"^\d{4}$", p) for p in sample):
        return "year"
    if all(re.match(r"^\d{4}-\d{1,2}$", p) for p in sample):
        return "month"
    if all(re.match(r"^\d{4}[- ]?Q[1-4]$", p, re.I) for p in sample):
        return "quarter"
    if all(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", p) for p in sample):
        return "day"

    return "period"


def period_sort_key(period: str, frequency: str, fallback_index: int) -> Tuple[int, int, int, int]:
    text = str(period).strip()

    if frequency == "year":
        match = re.search(r"\d{4}", text)
        if match:
            return (int(match.group(0)), 0, 0, 0)

    if frequency == "month":
        match = re.match(r"^(\d{4})-(\d{1,2})$", text)
        if match:
            return (int(match.group(1)), int(match.group(2)), 0, 0)

    if frequency == "quarter":
        match = re.match(r"^(\d{4})[- ]?Q([1-4])$", text, re.I)
        if match:
            return (int(match.group(1)), int(match.group(2)), 0, 0)

    if frequency == "day":
        match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)), 0)

    return (10**9, fallback_index, 0, 0)


def next_period_label(last_period: str, step: int, frequency: str) -> str:
    text = str(last_period).strip()

    if frequency == "year":
        match = re.search(r"\d{4}", text)
        if match:
            return str(int(match.group(0)) + step)

    if frequency == "month":
        match = re.match(r"^(\d{4})-(\d{1,2})$", text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            total = year * 12 + (month - 1) + step
            return f"{total // 12:04d}-{total % 12 + 1:02d}"

    if frequency == "quarter":
        match = re.match(r"^(\d{4})[- ]?Q([1-4])$", text, re.I)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            total = year * 4 + (quarter - 1) + step
            return f"{total // 4:04d}-Q{total % 4 + 1}"

    return f"t+{step}"


def season_length_for_frequency(frequency: str) -> Optional[int]:
    if frequency == "month":
        return 12
    if frequency == "quarter":
        return 4
    return None


# ============================================================
# Series normalization
# ============================================================

@dataclass(frozen=True)
class Observation:
    period: str
    value: float
    raw: JSONDict


def observation_from_item(item: Any) -> Optional[Observation]:
    if not isinstance(item, Mapping):
        return None

    period = (
        item.get("period")
        or item.get("date")
        or item.get("year")
        or item.get("month")
        or item.get("quarter")
        or item.get("time")
    )

    value = item.get("value")
    if value is None:
        for candidate in ["amount", "volume", "count", "score", "metric", "y"]:
            if candidate in item:
                value = item.get(candidate)
                break

    number = safe_float(value)
    if period is None or number is None:
        return None

    return Observation(period=str(period), value=number, raw=dict(item))


def normalize_series(series: Sequence[Any], frequency: Optional[str] = None) -> Tuple[List[Observation], str, List[str]]:
    warnings: List[str] = []
    observations: List[Observation] = []

    for item in series or []:
        obs = observation_from_item(item)
        if obs is None:
            warnings.append(f"Skipped non-numeric or malformed observation: {item}")
            continue
        observations.append(obs)

    inferred_frequency = infer_frequency([obs.period for obs in observations], explicit=frequency)

    indexed = list(enumerate(observations))
    indexed.sort(key=lambda pair: period_sort_key(pair[1].period, inferred_frequency, pair[0]))
    observations = [obs for _, obs in indexed]

    deduped: List[Observation] = []
    seen = set()
    for obs in reversed(observations):
        if obs.period in seen:
            warnings.append(f"Duplicate period {obs.period!r} found; kept the last occurrence.")
            continue
        seen.add(obs.period)
        deduped.append(obs)
    deduped.reverse()

    return deduped, inferred_frequency, warnings


# ============================================================
# Forecast methods
# ============================================================

def clamp_value(value: float, nonnegative: bool) -> float:
    if nonnegative:
        return max(0.0, float(value))
    return float(value)


def forecast_naive(values: Sequence[float], horizon: int, *, nonnegative: bool = True) -> List[float]:
    if not values:
        return []
    return [clamp_value(values[-1], nonnegative) for _ in range(horizon)]


def forecast_moving_average(values: Sequence[float], horizon: int, *, window: int = 3, nonnegative: bool = True) -> List[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    value = mean(values[-window:])
    return [clamp_value(value, nonnegative) for _ in range(horizon)]


def linear_fit(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(values[0]), 0.0

    xs = list(range(n))
    x_bar = mean(xs)
    y_bar = mean(values)
    denom = sum((x - x_bar) ** 2 for x in xs)

    if denom == 0:
        return y_bar, 0.0

    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, values)) / denom
    intercept = y_bar - slope * x_bar
    return intercept, slope


def forecast_linear_trend(values: Sequence[float], horizon: int, *, nonnegative: bool = True) -> List[float]:
    if not values:
        return []
    intercept, slope = linear_fit(values)
    n = len(values)
    return [clamp_value(intercept + slope * (n + h), nonnegative) for h in range(horizon)]


def smooth_level(values: Sequence[float], alpha: float = 0.4) -> float:
    if not values:
        return 0.0

    alpha = min(1.0, max(0.0, alpha))
    level = float(values[0])

    for value in values[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level

    return level


def forecast_exponential_smoothing(values: Sequence[float], horizon: int, *, alpha: float = 0.4, nonnegative: bool = True) -> List[float]:
    if not values:
        return []

    level = smooth_level(values, alpha=alpha)
    return [clamp_value(level, nonnegative) for _ in range(horizon)]


def forecast_seasonal_naive(
    values: Sequence[float],
    horizon: int,
    *,
    season_length: int,
    nonnegative: bool = True,
) -> List[float]:
    if season_length <= 0 or len(values) < season_length:
        return []

    out: List[float] = []
    for h in range(1, horizon + 1):
        index = len(values) - season_length + ((h - 1) % season_length)
        out.append(clamp_value(values[index], nonnegative))

    return out


def method_forecast_values(
    method: str,
    values: Sequence[float],
    horizon: int,
    *,
    frequency: str,
    nonnegative: bool = True,
    moving_average_window: int = 3,
    alpha: float = 0.4,
) -> List[float]:
    method = method.lower().strip()

    if method == "naive":
        return forecast_naive(values, horizon, nonnegative=nonnegative)

    if method == "moving_average":
        return forecast_moving_average(values, horizon, window=moving_average_window, nonnegative=nonnegative)

    if method == "linear_trend":
        return forecast_linear_trend(values, horizon, nonnegative=nonnegative)

    if method == "exponential_smoothing":
        return forecast_exponential_smoothing(values, horizon, alpha=alpha, nonnegative=nonnegative)

    if method == "seasonal_naive":
        season_length = season_length_for_frequency(frequency)
        if season_length is None:
            return []
        return forecast_seasonal_naive(values, horizon, season_length=season_length, nonnegative=nonnegative)

    raise ValueError(f"Unknown forecast method: {method}")


def available_methods(values: Sequence[float], frequency: str) -> List[str]:
    methods = ["naive", "moving_average", "linear_trend", "exponential_smoothing"]
    season_length = season_length_for_frequency(frequency)

    if season_length is not None and len(values) >= season_length * 2:
        methods.append("seasonal_naive")

    return methods


def ensemble_forecast(methods: Mapping[str, Sequence[float]], horizon: int) -> List[float]:
    out: List[float] = []

    for i in range(horizon):
        values = [float(forecast[i]) for forecast in methods.values() if len(forecast) > i]
        if values:
            out.append(mean(values))

    return out


# ============================================================
# Diagnostics and intervals
# ============================================================

def mae(errors: Sequence[float]) -> Optional[float]:
    if not errors:
        return None
    return mean(abs(error) for error in errors)


def one_step_backtest_mae(
    method: str,
    values: Sequence[float],
    *,
    frequency: str,
    nonnegative: bool = True,
    moving_average_window: int = 3,
    alpha: float = 0.4,
) -> Optional[float]:
    if len(values) < 4:
        return None

    errors: List[float] = []

    if method == "seasonal_naive":
        season_length = season_length_for_frequency(frequency)
        if season_length is None or len(values) < season_length + 2:
            return None
        min_train = season_length
    else:
        min_train = 3

    for i in range(min_train, len(values)):
        train = values[:i]
        actual = values[i]

        try:
            forecast = method_forecast_values(
                method,
                train,
                1,
                frequency=frequency,
                nonnegative=nonnegative,
                moving_average_window=moving_average_window,
                alpha=alpha,
            )
        except Exception:
            continue

        if forecast:
            errors.append(forecast[0] - actual)

    return mae(errors)


def ensemble_backtest_mae(
    component_methods: Sequence[str],
    values: Sequence[float],
    *,
    frequency: str,
    nonnegative: bool = True,
    moving_average_window: int = 3,
    alpha: float = 0.4,
) -> Optional[float]:
    if len(values) < 4:
        return None

    errors: List[float] = []

    for i in range(3, len(values)):
        train = values[:i]
        actual = values[i]
        forecasts: List[float] = []

        for method in component_methods:
            try:
                pred = method_forecast_values(
                    method,
                    train,
                    1,
                    frequency=frequency,
                    nonnegative=nonnegative,
                    moving_average_window=moving_average_window,
                    alpha=alpha,
                )
            except Exception:
                pred = []

            if pred:
                forecasts.append(pred[0])

        if forecasts:
            errors.append(mean(forecasts) - actual)

    return mae(errors)


def pct_change_volatility(values: Sequence[float]) -> Optional[float]:
    changes: List[float] = []

    for prev, curr in zip(values, values[1:]):
        if prev == 0:
            continue
        changes.append((curr - prev) / abs(prev))

    if len(changes) < 2:
        return None

    avg = mean(changes)
    variance = sum((x - avg) ** 2 for x in changes) / (len(changes) - 1)
    return math.sqrt(variance)


def interval_bounds(
    forecast_value: float,
    *,
    horizon_step: int,
    error_scale: float,
    z: float = 1.64,
    nonnegative: bool = True,
) -> Tuple[float, float]:
    margin = z * error_scale * math.sqrt(max(1, horizon_step))
    lower = forecast_value - margin
    upper = forecast_value + margin

    if nonnegative:
        lower = max(0.0, lower)

    return lower, upper


def fallback_error_scale(values: Sequence[float]) -> float:
    if not values:
        return 1.0

    last = abs(values[-1])
    diffs = [abs(curr - prev) for prev, curr in zip(values, values[1:])]

    candidates = [0.10 * last]
    if diffs:
        candidates.append(mean(diffs))
    candidates.append(1.0)

    return max(candidates)


# ============================================================
# Main API
# ============================================================

def forecast_series(payload: Mapping[str, Any]) -> JSONDict:
    target = str(payload.get("target") or payload.get("name") or "time series")
    unit = payload.get("unit")
    requested_frequency = payload.get("frequency")

    horizon = int(payload.get("horizon") or payload.get("forecast_horizon") or 3)
    horizon = max(1, horizon)

    nonnegative = bool(payload.get("nonnegative", True))
    moving_average_window = int(payload.get("moving_average_window") or 3)
    alpha = float(payload.get("alpha") or 0.4)
    selected_method = str(payload.get("selected_method") or "ensemble").strip().lower()

    series = payload.get("series") or payload.get("history") or payload.get("observations") or []
    observations, frequency, warnings = normalize_series(series, frequency=requested_frequency)

    if len(observations) < 3:
        return {
            "status": "unavailable",
            "target": target,
            "unit": unit,
            "frequency": frequency,
            "horizon": horizon,
            "reason": "Need at least 3 numeric observations for forecasting.",
            "history": [obs.raw for obs in observations],
            "required_input_shape": "series: [{period: 'YYYY', value: number}, ...] with at least 3 observations",
            "warnings": warnings + [
                "Forecast not computed; missing or insufficient time-series observations were not imputed."
            ],
        }

    values = [obs.value for obs in observations]
    periods = [obs.period for obs in observations]
    future_periods = [
        next_period_label(periods[-1], step, frequency)
        for step in range(1, horizon + 1)
    ]

    method_names = available_methods(values, frequency)
    method_outputs: Dict[str, List[float]] = {}

    for method in method_names:
        forecast_values = method_forecast_values(
            method,
            values,
            horizon,
            frequency=frequency,
            nonnegative=nonnegative,
            moving_average_window=moving_average_window,
            alpha=alpha,
        )
        if len(forecast_values) == horizon:
            method_outputs[method] = forecast_values

    if method_outputs:
        method_outputs["ensemble"] = ensemble_forecast(method_outputs, horizon)

    if selected_method not in method_outputs:
        warnings.append(f"Selected method {selected_method!r} unavailable; using ensemble.")
        selected_method = "ensemble" if "ensemble" in method_outputs else next(iter(method_outputs))

    selected_values = method_outputs[selected_method]

    backtest: Dict[str, Optional[float]] = {}
    for method in method_outputs:
        if method == "ensemble":
            backtest[method] = ensemble_backtest_mae(
                [m for m in method_outputs if m != "ensemble"],
                values,
                frequency=frequency,
                nonnegative=nonnegative,
                moving_average_window=moving_average_window,
                alpha=alpha,
            )
        else:
            backtest[method] = one_step_backtest_mae(
                method,
                values,
                frequency=frequency,
                nonnegative=nonnegative,
                moving_average_window=moving_average_window,
                alpha=alpha,
            )

    error_scale = backtest.get(selected_method) or fallback_error_scale(values)

    forecasts: List[JSONDict] = []
    for i, (period, value) in enumerate(zip(future_periods, selected_values), start=1):
        lower, upper = interval_bounds(
            value,
            horizon_step=i,
            error_scale=error_scale,
            nonnegative=nonnegative,
        )
        forecasts.append({
            "period": period,
            "value": value,
            "lower": lower,
            "upper": upper,
            "interval": "heuristic_90_percent_interval",
        })

    method_forecasts: Dict[str, List[JSONDict]] = {}
    for method, values_out in method_outputs.items():
        method_forecasts[method] = [
            {"period": period, "value": value}
            for period, value in zip(future_periods, values_out)
        ]

    _, slope = linear_fit(values)
    volatility = pct_change_volatility(values)

    diagnostics: JSONDict = {
        "n_observations": len(observations),
        "first_period": periods[0],
        "last_period": periods[-1],
        "first_value": values[0],
        "last_value": values[-1],
        "trend_per_period": slope,
        "pct_change_volatility": volatility,
        "backtest_mae": backtest,
        "selected_error_scale": error_scale,
        "interval_method": (
            "selected method one-step backtest MAE; fallback uses 10% of last value "
            "and average absolute difference"
        ),
    }

    if len(observations) < 6:
        warnings.append(
            "Forecast is based on a short series; interpret as a simple extrapolation, not a high-confidence prediction."
        )

    return {
        "status": "computed",
        "target": target,
        "unit": unit,
        "frequency": frequency,
        "horizon": horizon,
        "selected_method": selected_method,
        "history": [
            {
                "period": obs.period,
                "value": obs.value,
                **{k: v for k, v in obs.raw.items() if k not in {"period", "value"}},
            }
            for obs in observations
        ],
        "forecasts": forecasts,
        "methods": method_forecasts,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def forecast_many(payloads: Sequence[Mapping[str, Any]]) -> JSONDict:
    results = [forecast_series(payload) for payload in payloads]
    computed = sum(1 for result in results if result.get("status") == "computed")
    unavailable = len(results) - computed

    if computed == len(results) and results:
        status = "computed"
    elif computed > 0:
        status = "partial"
    else:
        status = "unavailable"

    return {
        "status": status,
        "computed_count": computed,
        "unavailable_count": unavailable,
        "series_count": len(results),
        "forecasts": results,
    }


def forecast_payload(payload: Mapping[str, Any]) -> JSONDict:
    if "series" in payload or "history" in payload or "observations" in payload:
        return forecast_series(payload)

    series_list = payload.get("series_list") or payload.get("time_series") or payload.get("forecast_inputs")
    if isinstance(series_list, list):
        return forecast_many([item for item in series_list if isinstance(item, Mapping)])

    return {
        "status": "unavailable",
        "reason": "No usable time-series payload found.",
        "required_input_shape": "Either {series: [{period, value}, ...]} or {series_list: [...]}."
    }


def summarize_forecast(result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("TIME-SERIES FORECAST")
    lines.append("=" * 60)
    lines.append(f"Status: {result.get('status')}")

    if result.get("series_count") is not None:
        lines.append(f"Series count: {result.get('series_count')}")
        lines.append(f"Computed: {result.get('computed_count')}")
        lines.append(f"Unavailable: {result.get('unavailable_count')}")
        for item in result.get("forecasts") or []:
            if isinstance(item, Mapping):
                lines.append(f"- {item.get('target')}: {item.get('status')}")
        return "\n".join(lines)

    if result.get("status") != "computed":
        lines.append(f"Reason: {result.get('reason')}")
        warnings = result.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in warnings[:8]:
                lines.append(f"- {warning}")
        return "\n".join(lines)

    lines.append(f"Target: {result.get('target')}")
    lines.append(f"Frequency: {result.get('frequency')}")
    lines.append(f"Selected method: {result.get('selected_method')}")
    lines.append("")

    unit = result.get("unit") or ""
    for item in result.get("forecasts") or []:
        if not isinstance(item, Mapping):
            continue

        value = item.get("value")
        lower = item.get("lower")
        upper = item.get("upper")

        if isinstance(value, (int, float)) and isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            lines.append(f"- {item.get('period')}: {value:.4g} {unit} [{lower:.4g}, {upper:.4g}]")
        else:
            lines.append(f"- {item.get('period')}: {value} {unit}")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings[:8]:
            lines.append(f"- {warning}")

    return "\n".join(lines)


# ============================================================
# Tests
# ============================================================

def run_tests() -> None:
    annual = {
        "target": "Myanmar rare-earth exports to China",
        "unit": "tonnes",
        "frequency": "year",
        "horizon": 3,
        "series": [
            {"period": "2019", "value": 105000},
            {"period": "2020", "value": 120000},
            {"period": "2021", "value": 145000},
            {"period": "2022", "value": 170000},
            {"period": "2023", "value": 210000},
            {"period": "2024", "value": 240000},
        ],
    }

    result = forecast_series(annual)
    assert result["status"] == "computed"
    assert result["selected_method"] == "ensemble"
    assert len(result["forecasts"]) == 3
    assert result["forecasts"][0]["period"] == "2025"
    assert result["forecasts"][0]["value"] > 0
    assert "linear_trend" in result["methods"]

    insufficient = forecast_series({
        "target": "too short",
        "series": [
            {"period": "2023", "value": 1},
            {"period": "2024", "value": 2},
        ],
    })
    assert insufficient["status"] == "unavailable"

    monthly_series = []
    for year in [2023, 2024]:
        for month in range(1, 13):
            monthly_series.append({
                "period": f"{year}-{month:02d}",
                "value": 100 + month + (year - 2023) * 10,
            })

    monthly = forecast_series({
        "target": "monthly test",
        "frequency": "month",
        "horizon": 2,
        "series": monthly_series,
    })
    assert monthly["status"] == "computed"
    assert "seasonal_naive" in monthly["methods"]
    assert monthly["forecasts"][0]["period"] == "2025-01"

    multi = forecast_payload({
        "series_list": [
            annual,
            {"target": "bad", "series": []},
        ]
    })
    assert multi["status"] == "partial"
    assert multi["computed_count"] == 1
    assert multi["unavailable_count"] == 1

    print("forecast_tools tests passed")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic time-series forecasting tools for Lanthic Intelligence")
    parser.add_argument("--input", type=Path, help="Input JSON forecast payload")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--summary", action="store_true", help="Print text summary")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test:
        run_tests()
        return 0

    if not args.input:
        raise SystemExit("Provide --input or use --test.")

    payload = read_json(args.input)
    if not isinstance(payload, Mapping):
        raise SystemExit("Input JSON must be an object.")

    result = forecast_payload(payload)

    if args.output:
        write_json(args.output, result)
        print(f"[done] wrote {args.output}")

    if args.summary or not args.output:
        print(summarize_forecast(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())