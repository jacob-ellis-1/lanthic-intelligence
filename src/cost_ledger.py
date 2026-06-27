#!/usr/bin/env python3
"""
Cost, cache, and model-call ledger for Lanthic Intelligence.

This module owns:
- stable cache keys for chat and embedding calls
- JSONL event logging using LedgerEvent from pipeline_contracts.py
- token estimation
- optional model pricing estimates
- cache read/write
- safe wrappers around OpenAI-style chat and embedding clients
- ledger summarisation

It must not:
- know domain prompts
- run extraction/PostRAG/KG-IRAG itself
- require OpenAI during self-test

Run:
    python src/cost_ledger.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pipeline_contracts import (
    ContractError,
    LedgerEvent,
    json_dumps_stable,
    read_json,
    stable_hash,
    text_sha256,
    utc_now,
    validate_cache_status,
    validate_stage_name,
    write_json,
)


JSONDict = Dict[str, Any]


# Intentionally empty by default: model pricing changes over time.
# Provide a pricing file for cost estimates:
# {
#   "gpt-4.1-mini": {"input_per_1m": 0.0, "output_per_1m": 0.0},
#   "text-embedding-3-small": {"input_per_1m": 0.0}
# }
DEFAULT_PRICING: JSONDict = {}


# ============================================================
# Generic helpers
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


def _append_jsonl(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(data), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _nested_get(obj: Any, *keys: Any) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return None
        if isinstance(key, int):
            try:
                cur = cur[key]
            except Exception:
                return None
        elif isinstance(cur, Mapping):
            cur = cur.get(key)
        else:
            cur = getattr(cur, str(key), None)
    return cur


def _as_dict(obj: Any) -> JSONDict:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return dict(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"repr": repr(obj)}


def _normalise_messages(messages: Sequence[Mapping[str, Any]]) -> List[JSONDict]:
    out: List[JSONDict] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            out.append(dict(msg))
        else:
            out.append(_as_dict(msg))
    return out


def _safe_params(params: Mapping[str, Any]) -> JSONDict:
    out: JSONDict = {}
    for key, value in params.items():
        if value is None:
            continue
        try:
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            out[key] = value
        except Exception:
            out[key] = repr(value)
    return out


def _read_cache(path: Path) -> Optional[JSONDict]:
    if not path.exists() or not path.is_file():
        return None

    data = read_json(path)

    if not isinstance(data, Mapping):
        raise ContractError(f"Cache file is not a JSON object: {path}")

    return dict(data)


def _write_cache(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_json(path, dict(data))


def load_pricing_config(path: Optional[Path]) -> JSONDict:
    if path is None:
        return dict(DEFAULT_PRICING)

    data = read_json(path)

    if not isinstance(data, Mapping):
        raise ContractError(f"Pricing config must be a JSON object: {path}")

    return dict(data)


def _pricing_for_model(pricing: Mapping[str, Any], model: Optional[str]) -> JSONDict:
    if not model:
        return {}

    value = pricing.get(model)

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def estimate_cost_usd(
    *,
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    pricing: Mapping[str, Any],
) -> Optional[float]:
    model_pricing = _pricing_for_model(pricing, model)

    if not model_pricing:
        return None

    total = 0.0
    has_any = False

    if input_tokens is not None and model_pricing.get("input_per_1m") is not None:
        total += (float(input_tokens) / 1_000_000.0) * float(model_pricing["input_per_1m"])
        has_any = True

    if output_tokens is not None and model_pricing.get("output_per_1m") is not None:
        total += (float(output_tokens) / 1_000_000.0) * float(model_pricing["output_per_1m"])
        has_any = True

    return round(total, 10) if has_any else None


# ============================================================
# Token estimation
# ============================================================

@dataclass
class TokenEstimate:
    tokens: int
    method: str


def estimate_text_tokens(text: str, *, model: Optional[str] = None) -> TokenEstimate:
    text = text or ""

    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        return TokenEstimate(tokens=len(encoding.encode(text)), method="tiktoken")

    except Exception:
        return TokenEstimate(tokens=max(1, int(len(text) / 4)), method="chars_div_4")


def estimate_value_tokens(value: Any, *, model: Optional[str] = None) -> TokenEstimate:
    return estimate_text_tokens(json_dumps_stable(value), model=model)


def estimate_messages_tokens(messages: Sequence[Mapping[str, Any]], *, model: Optional[str] = None) -> TokenEstimate:
    return estimate_value_tokens(_normalise_messages(messages), model=model)


# ============================================================
# Response extraction helpers
# ============================================================

def extract_chat_content(response: Any) -> str:
    content = _nested_get(response, "choices", 0, "message", "content")

    if content is not None:
        return str(content)

    raise ContractError("Could not extract chat response content.")


def extract_usage(response: Any) -> JSONDict:
    usage = _nested_get(response, "usage")

    if usage is None:
        return {}

    usage_dict = _as_dict(usage)
    out: JSONDict = {}

    for source_key, target_key in [
        ("prompt_tokens", "input_tokens"),
        ("input_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ]:
        value = usage_dict.get(source_key)
        if isinstance(value, int):
            out[target_key] = value

    return out


def parse_json_content(content: str) -> Any:
    text = str(content or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def extract_embedding_response(response: Any) -> List[List[float]]:
    data = _nested_get(response, "data")

    if data is None:
        data = []

    items = list(data or [])

    def item_index(item: Any, fallback: int) -> int:
        idx = item.get("index") if isinstance(item, Mapping) else getattr(item, "index", None)
        return int(idx) if idx is not None else fallback

    indexed: List[Tuple[int, List[float]]] = []

    for i, item in enumerate(items):
        embedding = item.get("embedding") if isinstance(item, Mapping) else getattr(item, "embedding", None)

        if embedding is None:
            raise ContractError("Embedding response item missing embedding.")

        indexed.append((item_index(item, i), [float(x) for x in list(embedding)]))

    indexed.sort(key=lambda pair: pair[0])
    return [embedding for _, embedding in indexed]


# ============================================================
# Cost ledger
# ============================================================

@dataclass
class CostLedger:
    run_id: str
    source_id: Optional[str] = None
    ledger_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    pricing_config: JSONDict = field(default_factory=dict)
    enabled: bool = True
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.pricing_config is None:
            self.pricing_config = dict(DEFAULT_PRICING)

        if self.ledger_path is not None:
            self.ledger_path = Path(self.ledger_path)

        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    @classmethod
    def disabled(cls, *, run_id: str = "run_disabled", source_id: Optional[str] = None) -> "CostLedger":
        return cls(run_id=run_id, source_id=source_id, enabled=False, cache_enabled=False)

    @classmethod
    def from_env(cls) -> "CostLedger":
        run_id = os.environ.get("LANTHIC_RUN_ID") or "run_env"
        source_id = os.environ.get("LANTHIC_SOURCE_ID") or None
        ledger_path_raw = os.environ.get("LANTHIC_COST_LEDGER")
        cache_dir_raw = os.environ.get("LANTHIC_CACHE_DIR")
        disable_cache = os.environ.get("LANTHIC_DISABLE_CACHE", "").lower() in {"1", "true", "yes"}
        disable_ledger = os.environ.get("LANTHIC_DISABLE_LEDGER", "").lower() in {"1", "true", "yes"}
        pricing_file = os.environ.get("LANTHIC_PRICING_FILE")

        pricing = load_pricing_config(Path(pricing_file)) if pricing_file else dict(DEFAULT_PRICING)

        return cls(
            run_id=run_id,
            source_id=source_id,
            ledger_path=Path(ledger_path_raw) if ledger_path_raw else None,
            cache_dir=Path(cache_dir_raw) if cache_dir_raw else None,
            pricing_config=pricing,
            enabled=not disable_ledger and bool(ledger_path_raw),
            cache_enabled=not disable_cache and bool(cache_dir_raw),
        )

    def cache_key(
        self,
        *,
        stage: str,
        operation: str,
        model: Optional[str],
        payload: Any,
        params: Optional[Mapping[str, Any]] = None,
    ) -> str:
        validate_stage_name(stage)

        key_payload = {
            "schema_version": 1,
            "stage": stage,
            "operation": operation,
            "model": model,
            "payload": payload,
            "params": _safe_params(params or {}),
        }

        return f"cache_{stable_hash(key_payload, 32)}"

    def cache_path(self, *, stage: str, operation: str, cache_key: str) -> Path:
        validate_stage_name(stage)

        if self.cache_dir is None:
            raise ContractError("cache_dir is not configured.")

        safe_operation = str(operation).replace("/", "_").replace(" ", "_")
        return self.cache_dir / stage / safe_operation / f"{cache_key}.json"

    def log_event(
        self,
        *,
        stage: str,
        operation: str,
        cache_status: str,
        model: Optional[str] = None,
        input_hash: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        runtime_seconds: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LedgerEvent]:
        validate_stage_name(stage)
        validate_cache_status(cache_status)

        if not self.enabled or self.ledger_path is None:
            return None

        event = LedgerEvent.make(
            run_id=self.run_id,
            source_id=self.source_id,
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
            metadata=dict(metadata or {}),
        )

        _append_jsonl(self.ledger_path, event.to_dict())
        return event

    def _maybe_cost(
        self,
        *,
        model: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Optional[float]:
        return estimate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=self.pricing_config,
        )

    # --------------------------------------------------------
    # Chat wrappers
    # --------------------------------------------------------

    def chat_text(
        self,
        client: Any,
        *,
        stage: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        operation: str = "chat_text",
        response_format: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        validate_stage_name(stage)

        normalised_messages = _normalise_messages(messages)
        prompt_hash = stable_hash(normalised_messages, 32)
        input_estimate = estimate_messages_tokens(normalised_messages, model=model)

        params = {
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        cache_payload = {
            "messages": normalised_messages,
            "response_format": response_format,
        }
        input_hash = stable_hash(cache_payload, 32)

        cache_key = self.cache_key(
            stage=stage,
            operation=operation,
            model=model,
            payload=cache_payload,
            params=params,
        )

        cache_path: Optional[Path] = None

        if self.cache_enabled and self.cache_dir is not None:
            cache_path = self.cache_path(stage=stage, operation=operation, cache_key=cache_key)
            cached = _read_cache(cache_path)

            if cached is not None and "content" in cached:
                content = str(cached["content"])
                output_estimate = estimate_text_tokens(content, model=model)

                cost = self._maybe_cost(
                    model=model,
                    input_tokens=input_estimate.tokens,
                    output_tokens=output_estimate.tokens,
                )

                self.log_event(
                    stage=stage,
                    operation=operation,
                    model=model,
                    input_hash=input_hash,
                    prompt_hash=prompt_hash,
                    cache_status="hit",
                    estimated_input_tokens=input_estimate.tokens,
                    estimated_output_tokens=output_estimate.tokens,
                    estimated_cost_usd=cost,
                    runtime_seconds=0.0,
                    metadata={
                        "cache_key": cache_key,
                        "cache_path": str(cache_path),
                        "token_estimate_method": input_estimate.method,
                    },
                )

                return content

        create_kwargs = dict(kwargs)

        if response_format is not None:
            create_kwargs["response_format"] = response_format

        if temperature is not None:
            create_kwargs["temperature"] = temperature

        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        t0 = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                **create_kwargs,
            )

            runtime = round(time.time() - t0, 6)
            content = extract_chat_content(response)
            usage = extract_usage(response)

            input_tokens = usage.get("input_tokens") or input_estimate.tokens
            output_tokens = usage.get("output_tokens") or estimate_text_tokens(content, model=model).tokens
            cost = self._maybe_cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)

            cache_status = "miss" if self.cache_enabled else "bypass"
            metadata = {
                "cache_key": cache_key,
                "cache_path": str(cache_path) if cache_path else None,
                "cache_written": False,
                "token_estimate_method": input_estimate.method,
                "usage": usage,
            }

            if self.cache_enabled and cache_path is not None:
                _write_cache(
                    cache_path,
                    {
                        "schema_version": 1,
                        "created_at": utc_now(),
                        "stage": stage,
                        "operation": operation,
                        "model": model,
                        "cache_key": cache_key,
                        "content": content,
                    },
                )
                metadata["cache_written"] = True

            self.log_event(
                stage=stage,
                operation=operation,
                model=model,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                cache_status=cache_status,
                estimated_input_tokens=int(input_tokens) if input_tokens is not None else None,
                estimated_output_tokens=int(output_tokens) if output_tokens is not None else None,
                estimated_cost_usd=cost,
                runtime_seconds=runtime,
                metadata=metadata,
            )

            return content

        except Exception as exc:
            runtime = round(time.time() - t0, 6)
            cost = self._maybe_cost(model=model, input_tokens=input_estimate.tokens, output_tokens=None)

            self.log_event(
                stage=stage,
                operation=operation,
                model=model,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                cache_status="error",
                estimated_input_tokens=input_estimate.tokens,
                estimated_output_tokens=None,
                estimated_cost_usd=cost,
                runtime_seconds=runtime,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "cache_key": cache_key,
                    "cache_path": str(cache_path) if cache_path else None,
                    "traceback": traceback.format_exc(),
                    "token_estimate_method": input_estimate.method,
                },
            )

            raise

    def chat_json(
        self,
        client: Any,
        *,
        stage: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        operation: str = "chat_json",
        response_format: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if response_format is None:
            response_format = {"type": "json_object"}

        content = self.chat_text(
            client,
            stage=stage,
            model=model,
            messages=messages,
            operation=operation,
            response_format=response_format,
            **kwargs,
        )

        return parse_json_content(content)

    # --------------------------------------------------------
    # Embedding wrappers
    # --------------------------------------------------------

    def embed_text(
        self,
        client: Any,
        *,
        stage: str,
        model: str,
        text: str,
        operation: str = "embed_text",
        **kwargs: Any,
    ) -> List[float]:
        return self.embed_texts(
            client,
            stage=stage,
            model=model,
            texts=[text],
            operation=operation,
            **kwargs,
        )[0]

    def embed_texts(
        self,
        client: Any,
        *,
        stage: str,
        model: str,
        texts: Sequence[str],
        operation: str = "embed_texts",
        **kwargs: Any,
    ) -> List[List[float]]:
        validate_stage_name(stage)

        texts_list = [str(text) for text in texts]
        results: List[Optional[List[float]]] = [None] * len(texts_list)
        misses: List[Tuple[int, str, str, Optional[Path], TokenEstimate]] = []
        hit_count = 0

        for i, text in enumerate(texts_list):
            token_estimate = estimate_text_tokens(text, model=model)
            payload = {"text_hash": text_sha256(text), "text": text}
            params = _safe_params(kwargs)

            cache_key = self.cache_key(
                stage=stage,
                operation=operation,
                model=model,
                payload=payload,
                params=params,
            )

            cache_path = None

            if self.cache_enabled and self.cache_dir is not None:
                cache_path = self.cache_path(stage=stage, operation=operation, cache_key=cache_key)
                cached = _read_cache(cache_path)

                if cached is not None and "embedding" in cached:
                    results[i] = [float(x) for x in cached["embedding"]]
                    hit_count += 1
                    cost = self._maybe_cost(model=model, input_tokens=token_estimate.tokens, output_tokens=None)

                    self.log_event(
                        stage=stage,
                        operation=operation,
                        model=model,
                        input_hash=text_sha256(text),
                        prompt_hash=None,
                        cache_status="hit",
                        estimated_input_tokens=token_estimate.tokens,
                        estimated_output_tokens=None,
                        estimated_cost_usd=cost,
                        runtime_seconds=0.0,
                        metadata={
                            "cache_key": cache_key,
                            "cache_path": str(cache_path),
                            "text_index": i,
                            "batch_size": len(texts_list),
                            "token_estimate_method": token_estimate.method,
                        },
                    )

                    continue

            misses.append((i, text, cache_key, cache_path, token_estimate))

        if not misses:
            return [item if item is not None else [] for item in results]

        missing_texts = [item[1] for item in misses]
        t0 = time.time()
        input_estimate_total = sum(item[4].tokens for item in misses)

        try:
            response = client.embeddings.create(
                model=model,
                input=missing_texts,
                **kwargs,
            )

            runtime = round(time.time() - t0, 6)
            embeddings = extract_embedding_response(response)

            if len(embeddings) != len(misses):
                raise ContractError(
                    f"Embedding response length mismatch: expected {len(misses)}, got {len(embeddings)}"
                )

            usage = extract_usage(response)
            actual_total_input = usage.get("input_tokens") or input_estimate_total
            cost_total = self._maybe_cost(model=model, input_tokens=actual_total_input, output_tokens=None)

            for j, ((i, text, cache_key, cache_path, token_estimate), embedding) in enumerate(zip(misses, embeddings)):
                results[i] = embedding

                if self.cache_enabled and cache_path is not None:
                    _write_cache(
                        cache_path,
                        {
                            "schema_version": 1,
                            "created_at": utc_now(),
                            "stage": stage,
                            "operation": operation,
                            "model": model,
                            "cache_key": cache_key,
                            "text_hash": text_sha256(text),
                            "embedding": embedding,
                        },
                    )

                allocated_cost = None

                if cost_total is not None and input_estimate_total > 0:
                    allocated_cost = round(cost_total * (token_estimate.tokens / input_estimate_total), 10)

                self.log_event(
                    stage=stage,
                    operation=operation,
                    model=model,
                    input_hash=text_sha256(text),
                    prompt_hash=None,
                    cache_status="miss" if self.cache_enabled else "bypass",
                    estimated_input_tokens=token_estimate.tokens,
                    estimated_output_tokens=None,
                    estimated_cost_usd=allocated_cost,
                    runtime_seconds=runtime if j == 0 else 0.0,
                    metadata={
                        "cache_key": cache_key,
                        "cache_path": str(cache_path) if cache_path else None,
                        "cache_written": bool(self.cache_enabled and cache_path is not None),
                        "text_index": i,
                        "batch_size": len(texts_list),
                        "miss_count": len(misses),
                        "hit_count": hit_count,
                        "token_estimate_method": token_estimate.method,
                        "usage": usage,
                    },
                )

            return [item if item is not None else [] for item in results]

        except Exception as exc:
            runtime = round(time.time() - t0, 6)
            cost = self._maybe_cost(model=model, input_tokens=input_estimate_total, output_tokens=None)

            self.log_event(
                stage=stage,
                operation=operation,
                model=model,
                input_hash=stable_hash(missing_texts, 32),
                prompt_hash=None,
                cache_status="error",
                estimated_input_tokens=input_estimate_total,
                estimated_output_tokens=None,
                estimated_cost_usd=cost,
                runtime_seconds=runtime,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "miss_count": len(misses),
                    "hit_count": hit_count,
                    "traceback": traceback.format_exc(),
                },
            )

            raise


# ============================================================
# Convenience wrappers for minimal patches
# ============================================================

def tracked_chat_text(
    client: Any,
    *,
    stage: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    ledger: Optional[CostLedger] = None,
    **kwargs: Any,
) -> str:
    ledger = ledger or CostLedger.from_env()

    if not ledger.enabled and not ledger.cache_enabled:
        response = client.chat.completions.create(model=model, messages=list(messages), **kwargs)
        return extract_chat_content(response)

    return ledger.chat_text(client, stage=stage, model=model, messages=messages, **kwargs)


def tracked_chat_json(
    client: Any,
    *,
    stage: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    ledger: Optional[CostLedger] = None,
    **kwargs: Any,
) -> Any:
    ledger = ledger or CostLedger.from_env()

    if not ledger.enabled and not ledger.cache_enabled:
        response_format = kwargs.pop("response_format", {"type": "json_object"})
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format=response_format,
            **kwargs,
        )
        return parse_json_content(extract_chat_content(response))

    return ledger.chat_json(client, stage=stage, model=model, messages=messages, **kwargs)


def tracked_embed_text(
    client: Any,
    *,
    stage: str,
    model: str,
    text: str,
    ledger: Optional[CostLedger] = None,
    **kwargs: Any,
) -> List[float]:
    ledger = ledger or CostLedger.from_env()

    if not ledger.enabled and not ledger.cache_enabled:
        response = client.embeddings.create(model=model, input=[text], **kwargs)
        return extract_embedding_response(response)[0]

    return ledger.embed_text(client, stage=stage, model=model, text=text, **kwargs)


def tracked_embed_texts(
    client: Any,
    *,
    stage: str,
    model: str,
    texts: Sequence[str],
    ledger: Optional[CostLedger] = None,
    **kwargs: Any,
) -> List[List[float]]:
    ledger = ledger or CostLedger.from_env()

    if not ledger.enabled and not ledger.cache_enabled:
        response = client.embeddings.create(model=model, input=list(texts), **kwargs)
        return extract_embedding_response(response)

    return ledger.embed_texts(client, stage=stage, model=model, texts=texts, **kwargs)


# ============================================================
# Ledger summary
# ============================================================

def read_ledger_events(path: Path) -> List[JSONDict]:
    events: List[JSONDict] = []

    if not path.exists():
        return events

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid ledger JSONL at {path}:{line_number}: {exc}") from exc

        if not isinstance(data, Mapping):
            raise ContractError(f"Ledger line {line_number} is not a JSON object.")

        events.append(dict(data))

    return events


def summarize_ledger(path: Path) -> JSONDict:
    events = read_ledger_events(path)

    by_stage = Counter()
    by_model = Counter()
    by_cache_status = Counter()
    by_operation = Counter()
    cost_by_stage = defaultdict(float)
    cost_by_model = defaultdict(float)
    tokens_by_stage = defaultdict(lambda: {"input": 0, "output": 0})

    total_input = 0
    total_output = 0
    total_cost = 0.0
    cost_known = False
    errors: List[JSONDict] = []

    for event in events:
        stage = str(event.get("stage") or "unknown")
        model = str(event.get("model") or "unknown")
        operation = str(event.get("operation") or "unknown")
        cache_status = str(event.get("cache_status") or "unknown")

        by_stage[stage] += 1
        by_model[model] += 1
        by_operation[operation] += 1
        by_cache_status[cache_status] += 1

        input_tokens = int(event.get("estimated_input_tokens") or 0)
        output_tokens = int(event.get("estimated_output_tokens") or 0)

        total_input += input_tokens
        total_output += output_tokens
        tokens_by_stage[stage]["input"] += input_tokens
        tokens_by_stage[stage]["output"] += output_tokens

        if event.get("estimated_cost_usd") is not None:
            cost_known = True
            value = float(event["estimated_cost_usd"])
            total_cost += value
            cost_by_stage[stage] += value
            cost_by_model[model] += value

        if event.get("error"):
            errors.append(
                {
                    "stage": stage,
                    "operation": operation,
                    "model": model,
                    "error": event.get("error"),
                    "created_at": event.get("created_at"),
                }
            )

    return {
        "ledger_path": str(path),
        "events_total": len(events),
        "by_stage": dict(by_stage),
        "by_model": dict(by_model),
        "by_operation": dict(by_operation),
        "by_cache_status": dict(by_cache_status),
        "cache_hits": by_cache_status.get("hit", 0),
        "cache_misses": by_cache_status.get("miss", 0),
        "cache_bypasses": by_cache_status.get("bypass", 0),
        "errors_total": len(errors),
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_total_tokens": total_input + total_output,
        "estimated_cost_usd": round(total_cost, 10) if cost_known else None,
        "cost_by_stage": {k: round(v, 10) for k, v in cost_by_stage.items()} if cost_known else {},
        "cost_by_model": {k: round(v, 10) for k, v in cost_by_model.items()} if cost_known else {},
        "tokens_by_stage": dict(tokens_by_stage),
        "errors": errors,
    }


# ============================================================
# Self-test fakes
# ============================================================

class _Obj:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        messages = kwargs.get("messages") or []
        response_format = kwargs.get("response_format")

        if response_format:
            content = json.dumps({"answer": "ok", "call": self.calls})
        else:
            content = f"hello call {self.calls}: {messages[-1].get('content', '') if messages else ''}"

        return _Obj(
            choices=[_Obj(message=_Obj(content=content))],
            usage=_Obj(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        inputs = kwargs.get("input") or []

        if isinstance(inputs, str):
            inputs = [inputs]

        data = []

        for i, text in enumerate(inputs):
            base = float((sum(ord(ch) for ch in str(text)) % 100) / 100.0)
            data.append(_Obj(index=i, embedding=[base, base + 0.1, base + 0.2]))

        return _Obj(
            data=data,
            usage=_Obj(prompt_tokens=sum(max(1, int(len(str(t)) / 4)) for t in inputs), total_tokens=0),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _Obj(completions=_FakeChatCompletions())
        self.embeddings = _FakeEmbeddings()


class _FailingChatCompletions:
    def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("intentional chat failure")


class _FailingClient:
    def __init__(self) -> None:
        self.chat = _Obj(completions=_FailingChatCompletions())
        self.embeddings = _FakeEmbeddings()


# ============================================================
# Self-test
# ============================================================

def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> int:
    print("[cost_ledger self-test] starting")

    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        ledger_path = tmp / "cost_ledger.jsonl"
        cache_dir = tmp / "cache"

        pricing = {
            "fake-chat": {"input_per_1m": 1.0, "output_per_1m": 2.0},
            "fake-embed": {"input_per_1m": 0.25},
        }

        ledger = CostLedger(
            run_id="run_test",
            source_id="src_test",
            ledger_path=ledger_path,
            cache_dir=cache_dir,
            pricing_config=pricing,
            enabled=True,
            cache_enabled=True,
        )

        client = _FakeClient()

        messages = [
            {"role": "system", "content": "Return concise answers."},
            {"role": "user", "content": "Say hello."},
        ]

        # 1. chat_text logs miss then hit.
        text_1 = ledger.chat_text(
            client,
            stage="extract",
            model="fake-chat",
            messages=messages,
            operation="chat_text_test",
        )

        _assert("hello call 1" in text_1, "first chat_text call returned wrong content")
        _assert(client.chat.completions.calls == 1, "first chat_text did not call fake client")

        text_2 = ledger.chat_text(
            client,
            stage="extract",
            model="fake-chat",
            messages=messages,
            operation="chat_text_test",
        )

        _assert(text_2 == text_1, "cached chat_text content mismatch")
        _assert(client.chat.completions.calls == 1, "second chat_text should have hit cache")

        # 2. chat_json parses JSON and caches.
        json_1 = ledger.chat_json(
            client,
            stage="postrag",
            model="fake-chat",
            messages=messages,
            operation="chat_json_test",
        )

        _assert(json_1["answer"] == "ok", "chat_json parse failed")

        calls_after_json_1 = client.chat.completions.calls

        json_2 = ledger.chat_json(
            client,
            stage="postrag",
            model="fake-chat",
            messages=messages,
            operation="chat_json_test",
        )

        _assert(json_2 == json_1, "cached chat_json mismatch")
        _assert(client.chat.completions.calls == calls_after_json_1, "second chat_json should have hit cache")

        # 3. embed_text logs miss then hit.
        emb_1 = ledger.embed_text(
            client,
            stage="postrag",
            model="fake-embed",
            text="Myanmar rare earths",
            operation="embed_text_test",
        )

        _assert(len(emb_1) == 3, "embedding length wrong")
        _assert(client.embeddings.calls == 1, "first embedding did not call fake client")

        emb_2 = ledger.embed_text(
            client,
            stage="postrag",
            model="fake-embed",
            text="Myanmar rare earths",
            operation="embed_text_test",
        )

        _assert(emb_2 == emb_1, "cached embedding mismatch")
        _assert(client.embeddings.calls == 1, "second embedding should have hit cache")

        # 4. embed_texts only calls API for misses.
        batch = ledger.embed_texts(
            client,
            stage="neo4j_fusion",
            model="fake-embed",
            texts=["Myanmar rare earths", "China magnet supply"],
            operation="embed_batch_test",
        )

        _assert(len(batch) == 2, "batch embedding count wrong")
        _assert(client.embeddings.calls == 2, "batch embedding should call fake client once")

        batch_2 = ledger.embed_texts(
            client,
            stage="neo4j_fusion",
            model="fake-embed",
            texts=["Myanmar rare earths", "China magnet supply"],
            operation="embed_batch_test",
        )

        _assert(batch_2 == batch, "cached batch embedding mismatch")
        _assert(client.embeddings.calls == 2, "second batch embedding should be all cache hits")

        # 5. Error events are logged.
        failing = _FailingClient()

        try:
            ledger.chat_text(
                failing,
                stage="sarg",
                model="fake-chat",
                messages=messages,
                operation="failing_chat_test",
            )
            raise AssertionError("failing chat did not raise")
        except RuntimeError:
            pass

        _assert(ledger_path.exists(), "ledger JSONL was not written")

        events = read_ledger_events(ledger_path)

        _assert(len(events) >= 8, "not enough ledger events written")
        _assert(any(e["cache_status"] == "miss" for e in events), "miss event missing")
        _assert(any(e["cache_status"] == "hit" for e in events), "hit event missing")
        _assert(any(e["cache_status"] == "error" for e in events), "error event missing")
        _assert(any(e.get("estimated_cost_usd") is not None for e in events), "cost estimates missing")

        cache_files = list(cache_dir.rglob("*.json"))
        _assert(len(cache_files) >= 4, "cache files were not written")

        summary = summarize_ledger(ledger_path)

        _assert(summary["events_total"] == len(events), "summary event count mismatch")
        _assert(summary["cache_hits"] >= 3, "summary cache hits wrong")
        _assert(summary["cache_misses"] >= 4, "summary cache misses wrong")
        _assert(summary["errors_total"] == 1, "summary error count wrong")
        _assert(summary["estimated_input_tokens"] > 0, "summary input tokens missing")
        _assert(summary["estimated_cost_usd"] is not None, "summary cost missing")

        # 6. Disabled ledger degrades to direct calls.
        disabled_client = _FakeClient()
        disabled = CostLedger.disabled()

        disabled_text = disabled.chat_text(
            disabled_client,
            stage="extract",
            model="fake-chat",
            messages=messages,
        )

        _assert("hello call 1" in disabled_text, "disabled ledger direct chat failed")
        _assert(disabled_client.chat.completions.calls == 1, "disabled ledger did not call directly")

    print("[cost_ledger self-test] all tests passed")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost/cache ledger utilities for Lanthic Intelligence.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summary", type=Path, default=None, help="Summarise a cost_ledger.jsonl file.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional output JSON path for summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.summary:
        summary = summarize_ledger(args.summary)

        if args.summary_output:
            write_json(args.summary_output, summary)

        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0

    print("cost_ledger.py defines CostLedger. Run with --self-test or --summary path/to/cost_ledger.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())