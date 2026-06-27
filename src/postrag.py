from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from openai import OpenAI


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

for path in (CURRENT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from extraction.webpage import html_webpage_document_from_url
except ImportError:
    html_webpage_document_from_url = None

try:
    from cost_ledger import CostLedger, load_pricing_config
except Exception:
    CostLedger = None  # type: ignore
    load_pricing_config = None  # type: ignore


JSONDict = Dict[str, Any]
CandidateType = Literal["entity", "relation"]

PROMPT_VERSION = "postrag_blockwise_v1"


SUPPORT_VALUES = {
    "direct",
    "strong_indirect",
    "weak_indirect",
    "ambiguous",
    "contradicted",
    "absent",
}

SPECIFICITY_VALUES = {
    "exact",
    "slightly_broader",
    "much_broader",
    "over_specific",
}

ENTITY_GROUNDING_VALUES = {
    "explicit",
    "inferred",
    "mismatched",
    "absent",
}

TEMPORAL_GROUNDING_VALUES = {
    "explicit",
    "inferred",
    "missing",
    "wrong",
    "not_applicable",
}

RELATION_GROUNDING_VALUES = {
    "explicit",
    "inferred",
    "causal_overreach",
    "absent",
    "not_applicable",
}


# ============================================================
# CLI / I/O
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PostRAG validator for extracted KG candidates.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--source-url", default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")

    # Kept for CLI compatibility and candidate-level fallback.
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ollama-chunking", action="store_true")

    # Blockwise PostRAG controls.
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=5000,
        help="Maximum characters from one evidence block sent to the blockwise validator.",
    )
    parser.add_argument(
        "--max-block-candidates",
        type=int,
        default=30,
        help="Maximum compact candidates sent in one block validation call before splitting.",
    )
    parser.add_argument(
        "--fallback-candidate-validation",
        dest="fallback_candidate_validation",
        action="store_true",
        default=True,
        help="Use old candidate-level validation only for candidates with no usable block verdict.",
    )
    parser.add_argument(
        "--no-fallback-candidate-validation",
        dest="fallback_candidate_validation",
        action="store_false",
    )

    parser.add_argument("--run-id", default=os.getenv("LANTHIC_RUN_ID"))
    parser.add_argument("--source-id", default=os.getenv("LANTHIC_SOURCE_ID"))
    parser.add_argument("--corpus-id", default=os.getenv("LANTHIC_CORPUS_ID"))
    parser.add_argument("--branch-id", default=os.getenv("LANTHIC_BRANCH_ID"))
    parser.add_argument("--canonical-source", default=os.getenv("LANTHIC_CANONICAL_SOURCE"))

    parser.add_argument(
        "--cost-ledger",
        type=Path,
        default=Path(os.getenv("LANTHIC_COST_LEDGER")) if os.getenv("LANTHIC_COST_LEDGER") else None,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("LANTHIC_CACHE_DIR")) if os.getenv("LANTHIC_CACHE_DIR") else None,
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=os.getenv("LANTHIC_DISABLE_CACHE", "").lower() in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        default=Path(os.getenv("LANTHIC_PRICING_FILE")) if os.getenv("LANTHIC_PRICING_FILE") else None,
    )

    return parser.parse_args()


def read_json(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: JSONDict) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Pipeline metadata / cost ledger
# ============================================================

def clean_pipeline_metadata(metadata: Optional[JSONDict]) -> JSONDict:
    if not isinstance(metadata, dict):
        return {}

    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ""
    }


def pipeline_metadata_from_args(args: argparse.Namespace) -> JSONDict:
    return clean_pipeline_metadata({
        "run_id": args.run_id,
        "source_id": args.source_id,
        "corpus_id": args.corpus_id,
        "branch_id": args.branch_id,
        "canonical_source": args.canonical_source,
    })


def merge_pipeline_metadata(record: JSONDict, args: argparse.Namespace) -> JSONDict:
    metadata = {}

    existing = record.get("pipeline_metadata")
    if isinstance(existing, dict):
        metadata.update(clean_pipeline_metadata(existing))

    metadata.update(pipeline_metadata_from_args(args))

    return clean_pipeline_metadata(metadata)


def ledger_from_args(args: argparse.Namespace) -> Optional[Any]:
    if not args.cost_ledger and not args.cache_dir:
        return None

    if CostLedger is None:
        raise RuntimeError(
            "cost_ledger.py could not be imported, but cost/cache options were provided."
        )

    pricing_config = {}
    if args.pricing_file and load_pricing_config is not None:
        pricing_config = load_pricing_config(args.pricing_file)

    return CostLedger(
        run_id=args.run_id or "postrag_run",
        source_id=args.source_id,
        ledger_path=args.cost_ledger,
        cache_dir=args.cache_dir,
        pricing_config=pricing_config,
        enabled=bool(args.cost_ledger),
        cache_enabled=bool(args.cache_dir) and not args.disable_cache,
    )


# ============================================================
# Source / evidence handling
# ============================================================

def get_source_url(record: JSONDict, override_url: Optional[str]) -> str:
    if override_url:
        return override_url.strip()

    candidates = [
        record.get("source_url"),
        (record.get("document") or {}).get("source_url"),
        (record.get("document") or {}).get("canonical_url"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

    for section in ("entities", "relations"):
        for item in record.get(section, []) or []:
            if isinstance(item, dict) and isinstance(item.get("source_url"), str):
                return item["source_url"].strip()

    raise ValueError("No source_url found. Pass --source-url explicitly.")


def evidence_has_text(evidence_store: JSONDict) -> bool:
    if not evidence_store:
        return False

    return all(
        isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
        for item in evidence_store.values()
    )


def evidence_store_from_document(source_url: str, *, use_ollama_chunking: bool) -> JSONDict:
    if html_webpage_document_from_url is None:
        raise RuntimeError(
            "Input has no evidence_store text, and extraction.webpage could not be imported."
        )

    document = html_webpage_document_from_url(
        source_url,
        use_ollama_chunking=use_ollama_chunking,
    )

    evidence_store = {}

    for block in document.ensure_blocks():
        data = block.to_dict()
        data.setdefault("evidence_id", block.block_id)
        data.setdefault("source_url", block.source_url)
        data.setdefault("text", block.to_text())
        evidence_store[block.block_id] = data

    return evidence_store


def get_evidence_store(
    record: JSONDict,
    *,
    source_url: str,
    use_ollama_chunking: bool,
) -> JSONDict:
    """
    Priority:
      1. Use input evidence_store if it contains text.
      2. Reconstruct evidence blocks from source URL using webpage.py.
      3. Fail.

    evidence_manifest alone is not enough for PostRAG because the validator
    needs text to judge grounding.
    """
    existing = record.get("evidence_store")

    if isinstance(existing, dict) and evidence_has_text(existing):
        return normalize_evidence_store(existing, fallback_source_url=source_url)

    return normalize_evidence_store(
        evidence_store_from_document(source_url, use_ollama_chunking=use_ollama_chunking),
        fallback_source_url=source_url,
    )


def normalize_evidence_store(evidence_store: JSONDict, *, fallback_source_url: str) -> JSONDict:
    normalized = {}

    for evidence_id, raw in evidence_store.items():
        if not isinstance(raw, dict):
            continue

        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        normalized[evidence_id] = {
            "evidence_id": raw.get("evidence_id") or raw.get("block_id") or evidence_id,
            "document_id": raw.get("document_id"),
            "block_type": raw.get("block_type"),
            "source_url": raw.get("source_url") or fallback_source_url,
            "start_char": raw.get("start_char"),
            "end_char": raw.get("end_char"),
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            "text": text.strip(),
        }

    if not normalized:
        raise ValueError("Could not build evidence_store with text.")

    return normalized


# ============================================================
# Embeddings / retrieval fallback
# ============================================================

def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    model: str,
    *,
    ledger: Optional[Any] = None,
    operation: str = "postrag_embeddings",
) -> List[List[float]]:
    if not texts:
        return []

    max_chars = 8000
    bounded_texts = [
        str(text or "")[:max_chars]
        for text in texts
    ]

    if ledger is not None:
        return ledger.embed_texts(
            client,
            stage="postrag",
            model=model,
            texts=bounded_texts,
            operation=operation,
        )

    response = client.embeddings.create(model=model, input=bounded_texts)
    return [item.embedding for item in response.data]


def build_evidence_index(
    client: OpenAI,
    evidence_store: JSONDict,
    *,
    embed_model: str,
    ledger: Optional[Any] = None,
) -> Dict[str, JSONDict]:
    evidence_ids = list(evidence_store.keys())
    texts = [evidence_store[evidence_id]["text"] for evidence_id in evidence_ids]
    embeddings = embed_texts(
        client,
        texts,
        embed_model,
        ledger=ledger,
        operation="postrag_evidence_store_embeddings",
    )

    return {
        evidence_id: {
            "evidence_id": evidence_id,
            "embedding": embedding,
        }
        for evidence_id, embedding in zip(evidence_ids, embeddings)
    }


def provenance_quotes(candidate: JSONDict) -> List[str]:
    quotes = []

    for item in candidate.get("provenance", []) or []:
        if isinstance(item, dict) and isinstance(item.get("quote"), str):
            if item["quote"].strip():
                quotes.append(item["quote"].strip())

    return quotes[:5]


def candidate_query(candidate: JSONDict, candidate_type: CandidateType) -> str:
    if candidate_type == "entity":
        fields = {
            "kind": "entity",
            "canonical_name": candidate.get("canonical_name") or candidate.get("name"),
            "entity_type": candidate.get("entity_type") or candidate.get("type"),
            "description": candidate.get("description"),
            "aliases": candidate.get("aliases"),
            "attributes": candidate.get("attributes"),
            "temporal": candidate.get("temporal"),
            "evidence_quotes": provenance_quotes(candidate),
        }
    else:
        fields = {
            "kind": "relation",
            "subject": candidate.get("subject"),
            "subject_id": candidate.get("subject_id"),
            "relation_type": candidate.get("relation_type") or candidate.get("type"),
            "object": candidate.get("object"),
            "object_id": candidate.get("object_id"),
            "description": candidate.get("description"),
            "temporal": candidate.get("temporal"),
            "attributes": candidate.get("attributes"),
            "evidence_quotes": provenance_quotes(candidate),
        }

    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def candidate_evidence_ids(candidate: JSONDict) -> List[str]:
    ids = []

    for item in candidate.get("postrag_evidence", []) or []:
        if isinstance(item, dict) and isinstance(item.get("evidence_id"), str):
            ids.append(item["evidence_id"])

    for item in candidate.get("provenance", []) or []:
        if isinstance(item, dict) and isinstance(item.get("evidence_id"), str):
            ids.append(item["evidence_id"])

    deduped = []
    seen = set()

    for evidence_id in ids:
        if evidence_id not in seen:
            deduped.append(evidence_id)
            seen.add(evidence_id)

    return deduped


def pointer_refs(candidate: JSONDict, evidence_store: JSONDict) -> List[JSONDict]:
    refs = []
    seen: Set[str] = set()

    for evidence_id in candidate_evidence_ids(candidate):
        if evidence_id in evidence_store and evidence_id not in seen:
            refs.append({
                "evidence_id": evidence_id,
                "rank": 0,
                "retrieval_score": 1.0,
                "retrieval_source": "candidate_pointer",
            })
            seen.add(evidence_id)

    return refs


def retrieve_evidence(
    client: OpenAI,
    evidence_store: JSONDict,
    evidence_index: Dict[str, JSONDict],
    candidate: JSONDict,
    candidate_type: CandidateType,
    *,
    embed_model: str,
    top_k: int,
    ledger: Optional[Any] = None,
) -> List[JSONDict]:
    """
    Candidate-level fallback retrieval. The default blockwise validator avoids
    this path when candidates already carry evidence pointers.
    """
    refs: List[JSONDict] = []
    seen: Set[str] = set()

    for ref in pointer_refs(candidate, evidence_store):
        refs.append(ref)
        seen.add(ref["evidence_id"])

    query = candidate_query(candidate, candidate_type)
    query_embedding = embed_texts(
        client,
        [query],
        embed_model,
        ledger=ledger,
        operation="postrag_candidate_query_embedding",
    )[0]

    scored = []

    for evidence_id, indexed in evidence_index.items():
        score = cosine_similarity(query_embedding, indexed["embedding"])
        scored.append((score, evidence_id))

    scored.sort(reverse=True)

    rank = 1
    for score, evidence_id in scored:
        if evidence_id in seen:
            continue

        refs.append({
            "evidence_id": evidence_id,
            "rank": rank,
            "retrieval_score": round(float(score), 6),
            "retrieval_source": "embedding",
        })

        seen.add(evidence_id)
        rank += 1

        if len(refs) >= top_k:
            break

    return refs[:top_k]


def prompt_evidence(refs: Sequence[JSONDict], evidence_store: JSONDict, *, max_chars: Optional[int] = None) -> List[JSONDict]:
    out = []

    for ref in refs:
        evidence_id = ref["evidence_id"]
        evidence = evidence_store[evidence_id]
        text = evidence["text"]
        if isinstance(max_chars, int) and max_chars > 0:
            text = text[:max_chars]

        out.append({
            "evidence_id": evidence_id,
            "rank": ref.get("rank"),
            "retrieval_score": ref.get("retrieval_score"),
            "retrieval_source": ref.get("retrieval_source"),
            "block_type": evidence.get("block_type"),
            "document_id": evidence.get("document_id"),
            "source_url": evidence.get("source_url"),
            "metadata": evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {},
            "text": text,
        })

    return out


# ============================================================
# Validation prompt / scoring
# ============================================================

def compact_candidate(candidate: JSONDict, candidate_type: CandidateType) -> JSONDict:
    if candidate_type == "entity":
        aliases = candidate.get("aliases")
        if not isinstance(aliases, list):
            aliases = []

        return {
            "candidate_id": candidate_id(candidate, candidate_type),
            "canonical_name": candidate.get("canonical_name") or candidate.get("name"),
            "entity_type": candidate.get("entity_type") or candidate.get("type"),
            "aliases": aliases[:5],
            "description": candidate.get("description"),
            "attributes": candidate.get("attributes") if isinstance(candidate.get("attributes"), dict) else {},
            "temporal": candidate.get("temporal") if isinstance(candidate.get("temporal"), dict) else {},
            "provenance_quotes": provenance_quotes(candidate)[:3],
        }

    return {
        "candidate_id": candidate_id(candidate, candidate_type),
        "triple": {
            "subject": candidate.get("subject"),
            "relation_type": candidate.get("relation_type") or candidate.get("type"),
            "object": candidate.get("object"),
        },
        "subject_id": candidate.get("subject_id"),
        "object_id": candidate.get("object_id"),
        "description": candidate.get("description"),
        "confidence": candidate.get("confidence"),
        "attributes": candidate.get("attributes") if isinstance(candidate.get("attributes"), dict) else {},
        "temporal": candidate.get("temporal") if isinstance(candidate.get("temporal"), dict) else {},
        "provenance_quotes": provenance_quotes(candidate)[:3],
    }


def compact_block_evidence(evidence_id: str, evidence_store: JSONDict, *, max_chars: int) -> JSONDict:
    evidence = evidence_store[evidence_id]
    text = evidence.get("text") or ""
    if max_chars > 0:
        text = text[:max_chars]

    return {
        "evidence_id": evidence_id,
        "block_type": evidence.get("block_type"),
        "document_id": evidence.get("document_id"),
        "source_url": evidence.get("source_url"),
        "metadata": evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {},
        "text": text,
    }


def build_block_validation_prompt(
    evidence_id: str,
    evidence_store: JSONDict,
    entities: Sequence[JSONDict],
    relations: Sequence[JSONDict],
    *,
    max_evidence_chars: int,
) -> str:
    evidence = compact_block_evidence(evidence_id, evidence_store, max_chars=max_evidence_chars)
    compact_entities = [compact_candidate(entity, "entity") for entity in entities]
    compact_relations = [compact_candidate(relation, "relation") for relation in relations]

    return f"""
You are validating extracted knowledge-graph candidates against one evidence block.

Use only the evidence block below.
Do not use outside knowledge.
Do not invent evidence IDs.
Do not provide character offsets.
Every returned used_evidence_ids value must be ["{evidence_id}"] or [].

The candidates came from an extractor and may be wrong.
Your job is PostRAG validation:
- accept if this evidence block supports the candidate as written;
- coarsen if this evidence block supports a broader/less specific corrected version;
- reject if unsupported, contradicted, endpoint-corrupted, or over-inferred.

Return only valid JSON in this exact shape:
{{
  "entity_verdicts": [
    {{
      "candidate_id": "entity id from input",
      "decision": "accept|reject|coarsen",
      "epistemic_status": {{
        "support": "direct|strong_indirect|weak_indirect|ambiguous|contradicted|absent",
        "specificity": "exact|slightly_broader|much_broader|over_specific",
        "entity_grounding": "explicit|inferred|mismatched|absent",
        "temporal_grounding": "explicit|inferred|missing|wrong|not_applicable",
        "relation_grounding": "not_applicable"
      }},
      "used_evidence_ids": [],
      "uncertainty_factors": [],
      "reason": "brief reason",
      "corrected_candidate": null
    }}
  ],
  "relation_verdicts": [
    {{
      "candidate_id": "relation id from input",
      "decision": "accept|reject|coarsen",
      "epistemic_status": {{
        "support": "direct|strong_indirect|weak_indirect|ambiguous|contradicted|absent",
        "specificity": "exact|slightly_broader|much_broader|over_specific",
        "entity_grounding": "explicit|inferred|mismatched|absent",
        "temporal_grounding": "explicit|inferred|missing|wrong|not_applicable",
        "relation_grounding": "explicit|inferred|causal_overreach|absent|not_applicable"
      }},
      "used_evidence_ids": [],
      "uncertainty_factors": [],
      "reason": "brief reason",
      "corrected_candidate": null
    }}
  ]
}}

Important:
- Return exactly one verdict for every input entity candidate.
- Return exactly one verdict for every input relation candidate.
- If a relation uses the wrong subject or object, reject it.
- If a company-specific claim is inferred from a generic industry statement, reject it.
- If the date is wrong, use coarsen or reject depending on whether the rest is supported.
- If an object is too broad or too narrow, use coarsen.
- If the block is only generally related but does not ground the candidate, reject it with support="absent" or "ambiguous".
- If decision is "coarsen", corrected_candidate must contain the corrected entity or relation object. Do not return decision="coarsen" with corrected_candidate=null. If you cannot provide a corrected_candidate, return decision="reject".

Evidence block:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Entity candidates:
{json.dumps(compact_entities, indent=2, ensure_ascii=False)}

Relation candidates:
{json.dumps(compact_relations, indent=2, ensure_ascii=False)}
""".strip()


def build_validation_prompt(
    candidate: JSONDict,
    candidate_type: CandidateType,
    evidence: Sequence[JSONDict],
) -> str:
    """Candidate-level fallback prompt. Blockwise validation is the default path."""
    return f"""
You are validating one extracted {candidate_type} against retrieved evidence blocks.

Use only the evidence blocks below.
Do not use outside knowledge.
Do not invent evidence IDs.
Do not provide character offsets.

The candidate came from an extractor and may be wrong.
Your job is PostRAG validation:
- accept if the evidence supports it as written;
- coarsen if the evidence supports a broader/less specific corrected version;
- reject if unsupported, contradicted, endpoint-corrupted, or over-inferred.

Return only valid JSON in this exact shape:
{{
  "decision": "accept|reject|coarsen",
  "epistemic_status": {{
    "support": "direct|strong_indirect|weak_indirect|ambiguous|contradicted|absent",
    "specificity": "exact|slightly_broader|much_broader|over_specific",
    "entity_grounding": "explicit|inferred|mismatched|absent",
    "temporal_grounding": "explicit|inferred|missing|wrong|not_applicable",
    "relation_grounding": "explicit|inferred|causal_overreach|absent|not_applicable"
  }},
  "used_evidence_ids": [],
  "uncertainty_factors": [],
  "reason": "brief reason",
  "corrected_candidate": null
}}

Important:
- If a relation uses the wrong subject or object, reject it.
- If a company-specific claim is inferred from a generic industry statement, reject it.
- If the date is wrong, use coarsen or reject depending on whether the rest is supported.
- If an object is too broad or too narrow, use coarsen.
- used_evidence_ids must be selected only from the provided evidence IDs.
For every relation candidate, validate the literal triple:
SUBJECT -- RELATION_TYPE --> OBJECT

The description is secondary context only. Do not accept a relation because the description is supported if the literal subject-relation-object triple is wrong, reversed, too broad, too narrow, or semantically malformed.

Before deciding, rewrite the candidate internally as:
"<subject> <relation_type> <object>"

Then ask:
"Does this exact directed relation follow from this evidence block?"

If the evidence instead supports the reverse direction, reject or coarsen.

Candidate:
{json.dumps(compact_candidate(candidate, candidate_type), indent=2, ensure_ascii=False)}

Evidence blocks:
{json.dumps(evidence, indent=2, ensure_ascii=False)}
""".strip()


def call_llm_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    ledger: Optional[Any] = None,
    operation: str = "postrag_validate_candidate",
) -> JSONDict:
    messages = [
        {
            "role": "system",
            "content": "Return only valid JSON. No markdown. No commentary.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    if ledger is not None:
        return ledger.chat_json(
            client,
            stage="postrag",
            model=model,
            messages=messages,
            operation=operation,
            temperature=0,
        )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned empty content.")

    return json.loads(content)


def clean_enum(value: Any, allowed: Set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def derive_grounding_score(status: JSONDict, decision: str) -> float:
    support = status.get("support")
    specificity = status.get("specificity")
    entity = status.get("entity_grounding")
    temporal = status.get("temporal_grounding")
    relation = status.get("relation_grounding")

    base = {
        "direct": 0.96,
        "strong_indirect": 0.86,
        "weak_indirect": 0.66,
        "ambiguous": 0.45,
        "absent": 0.16,
        "contradicted": 0.02,
    }.get(support, 0.35)

    penalty = 0.0

    penalty += {
        "exact": 0.0,
        "slightly_broader": 0.05,
        "much_broader": 0.16,
        "over_specific": 0.25,
    }.get(specificity, 0.08)

    penalty += {
        "explicit": 0.0,
        "inferred": 0.08,
        "mismatched": 0.32,
        "absent": 0.35,
    }.get(entity, 0.08)

    penalty += {
        "explicit": 0.0,
        "inferred": 0.08,
        "missing": 0.10,
        "wrong": 0.42,
        "not_applicable": 0.0,
    }.get(temporal, 0.08)

    penalty += {
        "explicit": 0.0,
        "inferred": 0.10,
        "causal_overreach": 0.28,
        "absent": 0.35,
        "not_applicable": 0.0,
    }.get(relation, 0.08)

    if decision == "reject":
        penalty += 0.05
    elif decision == "coarsen":
        penalty += 0.03

    return round(max(0.0, min(1.0, base - penalty)), 3)


def enforce_coarsen_requires_correction(verdict: JSONDict) -> JSONDict:
    """
    A coarsen verdict is only safe downstream if the validator provides
    a concrete corrected_candidate. Otherwise postrag_filtered may import
    the original, known-imperfect candidate.
    """
    if verdict.get("decision") != "coarsen":
        return verdict

    corrected = verdict.get("corrected_candidate")
    if isinstance(corrected, dict) and corrected:
        return verdict

    status = verdict.get("epistemic_status")
    if not isinstance(status, dict):
        status = {}

    status = {
        "support": clean_enum(status.get("support"), SUPPORT_VALUES, "ambiguous"),
        "specificity": clean_enum(status.get("specificity"), SPECIFICITY_VALUES, "over_specific"),
        "entity_grounding": clean_enum(status.get("entity_grounding"), ENTITY_GROUNDING_VALUES, "inferred"),
        "temporal_grounding": clean_enum(status.get("temporal_grounding"), TEMPORAL_GROUNDING_VALUES, "not_applicable"),
        "relation_grounding": clean_enum(status.get("relation_grounding"), RELATION_GROUNDING_VALUES, "not_applicable"),
    }

    uncertainty = verdict.get("uncertainty_factors")
    if not isinstance(uncertainty, list):
        uncertainty = []

    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Coarsen verdict did not provide a corrected candidate."

    return {
        "decision": "reject",
        "grounding_score": derive_grounding_score(status, "reject"),
        "epistemic_status": status,
        "used_evidence_ids": [
            evidence_id
            for evidence_id in verdict.get("used_evidence_ids", []) or []
            if isinstance(evidence_id, str)
        ],
        "uncertainty_factors": [
            *[str(item) for item in uncertainty if str(item).strip()],
            "Coarsen verdict rejected because corrected_candidate was not provided.",
        ],
        "reason": (
            f"{reason.strip()} Rejected because decision='coarsen' requires "
            "a concrete corrected_candidate; otherwise the original candidate "
            "would be unsafe to import downstream."
        ),
        "corrected_candidate": None,
    }

def normalize_verdict(raw: JSONDict, allowed_evidence_ids: Set[str]) -> JSONDict:
    decision = raw.get("decision")
    if decision not in {"accept", "reject", "coarsen"}:
        decision = "reject"

    status = raw.get("epistemic_status")
    if not isinstance(status, dict):
        status = {}

    # Also accept flattened block-validator fields if they appear.
    status = {
        "support": clean_enum(raw.get("support", status.get("support")), SUPPORT_VALUES, "ambiguous"),
        "specificity": clean_enum(raw.get("specificity", status.get("specificity")), SPECIFICITY_VALUES, "exact"),
        "entity_grounding": clean_enum(raw.get("entity_grounding", status.get("entity_grounding")), ENTITY_GROUNDING_VALUES, "inferred"),
        "temporal_grounding": clean_enum(raw.get("temporal_grounding", status.get("temporal_grounding")), TEMPORAL_GROUNDING_VALUES, "not_applicable"),
        "relation_grounding": clean_enum(raw.get("relation_grounding", status.get("relation_grounding")), RELATION_GROUNDING_VALUES, "not_applicable"),
    }

    used = raw.get("used_evidence_ids")
    if not isinstance(used, list):
        used = []

    used = [
        evidence_id
        for evidence_id in used
        if isinstance(evidence_id, str) and evidence_id in allowed_evidence_ids
    ]

    uncertainty = raw.get("uncertainty_factors")
    if not isinstance(uncertainty, list):
        uncertainty = []

    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No valid reason returned."

    corrected = raw.get("corrected_candidate")
    if corrected is not None and not isinstance(corrected, dict):
        corrected = None

    verdict = {
        "decision": decision,
        "grounding_score": derive_grounding_score(status, decision),
        "epistemic_status": status,
        "used_evidence_ids": used,
        "uncertainty_factors": [str(x) for x in uncertainty if str(x).strip()],
        "reason": reason.strip(),
        "corrected_candidate": corrected,
    }

    return enforce_coarsen_requires_correction(verdict)


def fallback_verdict(reason: str, *, support: str = "ambiguous") -> JSONDict:
    status = {
        "support": support,
        "specificity": "exact",
        "entity_grounding": "inferred",
        "temporal_grounding": "not_applicable",
        "relation_grounding": "not_applicable",
    }
    return {
        "decision": "reject",
        "grounding_score": derive_grounding_score(status, "reject"),
        "epistemic_status": status,
        "used_evidence_ids": [],
        "uncertainty_factors": [reason],
        "reason": reason,
        "corrected_candidate": None,
    }


# ============================================================
# Blockwise validation
# ============================================================

def candidate_id(candidate: JSONDict, candidate_type: CandidateType) -> str:
    if candidate_type == "entity":
        value = candidate.get("entity_id") or candidate.get("id") or candidate.get("canonical_name") or candidate.get("name")
    else:
        value = candidate.get("relation_id") or candidate.get("id") or candidate.get("claim_id")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return json.dumps(compact_candidate_without_id(candidate, candidate_type), sort_keys=True, ensure_ascii=False)


def compact_candidate_without_id(candidate: JSONDict, candidate_type: CandidateType) -> JSONDict:
    if candidate_type == "entity":
        return {
            "canonical_name": candidate.get("canonical_name") or candidate.get("name"),
            "entity_type": candidate.get("entity_type") or candidate.get("type"),
        }

    return {
        "subject": candidate.get("subject"),
        "relation_type": candidate.get("relation_type") or candidate.get("type"),
        "object": candidate.get("object"),
    }


def chunked(items: Sequence[JSONDict], size: int) -> List[List[JSONDict]]:
    if size <= 0:
        return [list(items)]
    return [list(items[index:index + size]) for index in range(0, len(items), size)]


def build_block_candidate_index(
    evidence_store: JSONDict,
    entities: Sequence[JSONDict],
    relations: Sequence[JSONDict],
) -> Tuple[Dict[str, Dict[str, List[JSONDict]]], List[Tuple[CandidateType, JSONDict]]]:
    block_index: Dict[str, Dict[str, List[JSONDict]]] = {}
    without_pointer: List[Tuple[CandidateType, JSONDict]] = []

    for candidate_type, candidates in (("entity", entities), ("relation", relations)):
        for candidate in candidates:
            usable_ids = [evidence_id for evidence_id in candidate_evidence_ids(candidate) if evidence_id in evidence_store]

            if not usable_ids:
                without_pointer.append((candidate_type, candidate))
                continue

            for evidence_id in usable_ids:
                if evidence_id not in block_index:
                    block_index[evidence_id] = {"entities": [], "relations": []}

                key = "entities" if candidate_type == "entity" else "relations"
                block_index[evidence_id][key].append(candidate)

    return block_index, without_pointer


def parse_block_verdicts(
    raw: JSONDict,
    *,
    evidence_id: str,
    expected_entities: Sequence[JSONDict],
    expected_relations: Sequence[JSONDict],
) -> Dict[Tuple[CandidateType, str], JSONDict]:
    out: Dict[Tuple[CandidateType, str], JSONDict] = {}
    allowed = {evidence_id}

    expected_ids = {
        ("entity", candidate_id(candidate, "entity"))
        for candidate in expected_entities
    } | {
        ("relation", candidate_id(candidate, "relation"))
        for candidate in expected_relations
    }

    for candidate_type, field in (("entity", "entity_verdicts"), ("relation", "relation_verdicts")):
        items = raw.get(field)
        if not isinstance(items, list):
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue

            cid = item.get("candidate_id")
            if not isinstance(cid, str):
                continue

            key = (candidate_type, cid)
            if key not in expected_ids:
                continue

            out[key] = normalize_verdict(item, allowed)

    return out


def validate_block_batch(
    client: OpenAI,
    evidence_store: JSONDict,
    evidence_id: str,
    entities: Sequence[JSONDict],
    relations: Sequence[JSONDict],
    *,
    model: str,
    max_evidence_chars: int,
    ledger: Optional[Any] = None,
) -> Dict[Tuple[CandidateType, str], JSONDict]:
    prompt = build_block_validation_prompt(
        evidence_id,
        evidence_store,
        entities,
        relations,
        max_evidence_chars=max_evidence_chars,
    )

    raw = call_llm_json(
        client,
        model=model,
        prompt=prompt,
        ledger=ledger,
        operation="postrag_validate_block",
    )

    return parse_block_verdicts(
        raw,
        evidence_id=evidence_id,
        expected_entities=entities,
        expected_relations=relations,
    )


def validate_blocks(
    client: OpenAI,
    evidence_store: JSONDict,
    block_index: Dict[str, Dict[str, List[JSONDict]]],
    *,
    model: str,
    max_evidence_chars: int,
    max_block_candidates: int,
    ledger: Optional[Any] = None,
) -> Dict[Tuple[CandidateType, str], List[JSONDict]]:
    verdicts: Dict[Tuple[CandidateType, str], List[JSONDict]] = {}
    evidence_ids = list(block_index.keys())

    for index, evidence_id in enumerate(evidence_ids, start=1):
        grouped = block_index[evidence_id]
        entities = grouped.get("entities", [])
        relations = grouped.get("relations", [])
        all_candidates: List[Tuple[CandidateType, JSONDict]] = [
            *(("entity", entity) for entity in entities),
            *(("relation", relation) for relation in relations),
        ]

        batches = chunked(
            [{"candidate_type": candidate_type, "candidate": candidate} for candidate_type, candidate in all_candidates],
            max_block_candidates,
        )

        print(
            f"[postrag] block {index}/{len(evidence_ids)} {evidence_id}: "
            f"{len(entities)} entities, {len(relations)} relations, {len(batches)} batch(es)"
        )

        for batch_index, batch in enumerate(batches, start=1):
            batch_entities = [item["candidate"] for item in batch if item["candidate_type"] == "entity"]
            batch_relations = [item["candidate"] for item in batch if item["candidate_type"] == "relation"]

            try:
                block_verdicts = validate_block_batch(
                    client,
                    evidence_store,
                    evidence_id,
                    batch_entities,
                    batch_relations,
                    model=model,
                    max_evidence_chars=max_evidence_chars,
                    ledger=ledger,
                )
            except Exception as exc:
                print(f"[postrag] block validation failed for {evidence_id} batch {batch_index}: {exc}")
                block_verdicts = {}

            for candidate_type, candidate in [("entity", x) for x in batch_entities] + [("relation", x) for x in batch_relations]:
                cid = candidate_id(candidate, candidate_type)
                key = (candidate_type, cid)
                verdict = block_verdicts.get(key)

                if verdict is None:
                    verdict = fallback_verdict(
                        f"No usable block-level verdict returned for {candidate_type} {cid} on evidence block {evidence_id}.",
                    )

                verdicts.setdefault(key, []).append(verdict)

    return verdicts


def support_rank(verdict: JSONDict) -> int:
    status = verdict.get("epistemic_status") if isinstance(verdict.get("epistemic_status"), dict) else {}
    support = status.get("support")
    return {
        "direct": 6,
        "strong_indirect": 5,
        "weak_indirect": 4,
        "ambiguous": 3,
        "absent": 2,
        "contradicted": 1,
    }.get(support, 0)


def decision_rank(verdict: JSONDict) -> int:
    return {
        "accept": 3,
        "coarsen": 2,
        "reject": 1,
    }.get(verdict.get("decision"), 0)


def verdict_sort_key(verdict: JSONDict) -> Tuple[int, int, float]:
    score = verdict.get("grounding_score")
    if not isinstance(score, (int, float)):
        score = 0.0
    return (decision_rank(verdict), support_rank(verdict), float(score))


def aggregate_candidate_verdict(verdicts: Sequence[JSONDict]) -> JSONDict:
    if not verdicts:
        return fallback_verdict("No PostRAG verdicts were produced for this candidate.")

    supporting = [verdict for verdict in verdicts if verdict.get("decision") in {"accept", "coarsen"}]
    pool = supporting if supporting else list(verdicts)
    selected = max(pool, key=verdict_sort_key)

    used_ids: List[str] = []
    uncertainty: List[str] = []
    reasons: List[str] = []

    for verdict in pool:
        for evidence_id in verdict.get("used_evidence_ids", []) or []:
            if isinstance(evidence_id, str) and evidence_id not in used_ids:
                used_ids.append(evidence_id)

        for item in verdict.get("uncertainty_factors", []) or []:
            text = str(item).strip()
            if text and text not in uncertainty:
                uncertainty.append(text)

        reason = verdict.get("reason")
        if isinstance(reason, str) and reason.strip() and reason.strip() not in reasons:
            reasons.append(reason.strip())

    out = {
        "decision": selected.get("decision"),
        "grounding_score": selected.get("grounding_score"),
        "epistemic_status": selected.get("epistemic_status"),
        "used_evidence_ids": used_ids,
        "uncertainty_factors": uncertainty[:8],
        "reason": " / ".join(reasons[:3]) if reasons else selected.get("reason", "No valid reason returned."),
        "corrected_candidate": selected.get("corrected_candidate"),
    }

    allowed = set(used_ids)
    if not allowed:
        for verdict in verdicts:
            for evidence_id in verdict.get("used_evidence_ids", []) or []:
                if isinstance(evidence_id, str):
                    allowed.add(evidence_id)

    return normalize_verdict(out, allowed)


# ============================================================
# Candidate-level fallback validation
# ============================================================

def validate_candidate(
    client: OpenAI,
    evidence_store: JSONDict,
    evidence_index: Dict[str, JSONDict],
    candidate: JSONDict,
    candidate_type: CandidateType,
    *,
    model: str,
    embed_model: str,
    top_k: int,
    max_evidence_chars: int,
    ledger: Optional[Any] = None,
) -> JSONDict:
    refs = retrieve_evidence(
        client,
        evidence_store,
        evidence_index,
        candidate,
        candidate_type,
        embed_model=embed_model,
        top_k=top_k,
        ledger=ledger,
    )

    evidence = prompt_evidence(refs, evidence_store, max_chars=max_evidence_chars)
    prompt = build_validation_prompt(candidate, candidate_type, evidence)

    raw = call_llm_json(
        client,
        model=model,
        prompt=prompt,
        ledger=ledger,
        operation=f"postrag_validate_{candidate_type}_fallback",
    )
    verdict = normalize_verdict(raw, {ref["evidence_id"] for ref in refs})

    return {
        **candidate,
        "postrag": verdict,
        "postrag_evidence": refs,
    }


def attach_blockwise_verdicts(
    candidates: Sequence[JSONDict],
    candidate_type: CandidateType,
    evidence_store: JSONDict,
    verdicts_by_candidate: Dict[Tuple[CandidateType, str], List[JSONDict]],
) -> Tuple[List[JSONDict], List[JSONDict]]:
    out = []
    unresolved = []

    for candidate in candidates:
        cid = candidate_id(candidate, candidate_type)
        key = (candidate_type, cid)
        verdicts = verdicts_by_candidate.get(key, [])
        refs = pointer_refs(candidate, evidence_store)

        if verdicts:
            out.append({
                **candidate,
                "postrag": aggregate_candidate_verdict(verdicts),
                "postrag_evidence": refs,
            })
        else:
            unresolved.append(candidate)

    return out, unresolved


def validate_unresolved_candidates(
    client: OpenAI,
    evidence_store: JSONDict,
    unresolved: Sequence[Tuple[CandidateType, JSONDict]],
    *,
    model: str,
    embed_model: str,
    top_k: int,
    max_evidence_chars: int,
    fallback_candidate_validation: bool,
    ledger: Optional[Any] = None,
) -> Dict[Tuple[CandidateType, str], JSONDict]:
    if not unresolved:
        return {}

    if not fallback_candidate_validation:
        return {
            (candidate_type, candidate_id(candidate, candidate_type)): {
                **candidate,
                "postrag": fallback_verdict("Skipped candidate-level fallback validation."),
                "postrag_evidence": pointer_refs(candidate, evidence_store),
            }
            for candidate_type, candidate in unresolved
        }

    print(f"[postrag] fallback candidate validation for {len(unresolved)} candidate(s)")
    evidence_index = build_evidence_index(
        client,
        evidence_store,
        embed_model=embed_model,
        ledger=ledger,
    )

    resolved = {}
    for index, (candidate_type, candidate) in enumerate(unresolved, start=1):
        print(f"[postrag] fallback {candidate_type} {index}/{len(unresolved)}")
        resolved_candidate = validate_candidate(
            client,
            evidence_store,
            evidence_index,
            candidate,
            candidate_type,
            model=model,
            embed_model=embed_model,
            top_k=top_k,
            max_evidence_chars=max_evidence_chars,
            ledger=ledger,
        )
        resolved[(candidate_type, candidate_id(candidate, candidate_type))] = resolved_candidate

    return resolved


def validate_many_blockwise(
    client: OpenAI,
    evidence_store: JSONDict,
    entities: Sequence[JSONDict],
    relations: Sequence[JSONDict],
    *,
    model: str,
    embed_model: str,
    top_k: int,
    max_evidence_chars: int,
    max_block_candidates: int,
    fallback_candidate_validation: bool,
    ledger: Optional[Any] = None,
) -> Tuple[List[JSONDict], List[JSONDict]]:
    block_index, without_pointer = build_block_candidate_index(evidence_store, entities, relations)

    print(f"[postrag] evidence blocks with candidates: {len(block_index)}")
    print(f"[postrag] candidates without usable evidence pointer: {len(without_pointer)}")

    block_verdicts = validate_blocks(
        client,
        evidence_store,
        block_index,
        model=model,
        max_evidence_chars=max_evidence_chars,
        max_block_candidates=max_block_candidates,
        ledger=ledger,
    )

    validated_entities, unresolved_entities = attach_blockwise_verdicts(
        entities,
        "entity",
        evidence_store,
        block_verdicts,
    )
    validated_relations, unresolved_relations = attach_blockwise_verdicts(
        relations,
        "relation",
        evidence_store,
        block_verdicts,
    )

    unresolved: List[Tuple[CandidateType, JSONDict]] = [
        *(("entity", entity) for entity in unresolved_entities),
        *(("relation", relation) for relation in unresolved_relations),
    ]

    existing_unresolved = {(candidate_type, candidate_id(candidate, candidate_type)) for candidate_type, candidate in unresolved}
    for candidate_type, candidate in without_pointer:
        key = (candidate_type, candidate_id(candidate, candidate_type))
        if key not in existing_unresolved:
            unresolved.append((candidate_type, candidate))
            existing_unresolved.add(key)

    fallback_resolved = validate_unresolved_candidates(
        client,
        evidence_store,
        unresolved,
        model=model,
        embed_model=embed_model,
        top_k=top_k,
        max_evidence_chars=max_evidence_chars,
        fallback_candidate_validation=fallback_candidate_validation,
        ledger=ledger,
    )

    entity_by_id = {candidate_id(entity, "entity"): entity for entity in validated_entities}
    relation_by_id = {candidate_id(relation, "relation"): relation for relation in validated_relations}

    for (candidate_type, cid), candidate in fallback_resolved.items():
        if candidate_type == "entity":
            entity_by_id[cid] = candidate
        else:
            relation_by_id[cid] = candidate

    final_entities = [entity_by_id.get(candidate_id(entity, "entity"), entity) for entity in entities]
    final_relations = [relation_by_id.get(candidate_id(relation, "relation"), relation) for relation in relations]

    return final_entities, final_relations


# ============================================================
# Output
# ============================================================

def build_output(
    record: JSONDict,
    *,
    source_url: str,
    evidence_store: JSONDict,
    entities: Sequence[JSONDict],
    relations: Sequence[JSONDict],
    model: str,
    embed_model: str,
    top_k: int,
    pipeline_metadata: Optional[JSONDict] = None,
    cost_tracking: Optional[JSONDict] = None,
) -> JSONDict:
    return {
        **record,
        "source_url": source_url,
        "pipeline_metadata": clean_pipeline_metadata(pipeline_metadata),
        "evidence_store": evidence_store,
        "entities": list(entities),
        "relations": list(relations),
        "postrag": {
            "source_url": source_url,
            "model": model,
            "embed_model": embed_model,
            "top_k": top_k,
            "evidence_count": len(evidence_store),
            "cost_tracking": cost_tracking or {},
            "note": (
                "PostRAG validates candidates against EvidenceDocument blocks. "
                "Candidate evidence pointers are grouped by evidence block for blockwise validation; "
                "candidate-level embedding retrieval is used only as fallback."
            ),
        },
        "postrag_filtered": {
            "entities": [
                entity for entity in entities
                if entity.get("postrag", {}).get("decision") in {"accept", "coarsen"}
            ],
            "relations": [
                relation for relation in relations
                if relation.get("postrag", {}).get("decision") in {"accept", "coarsen"}
            ],
        },
    }


# ============================================================
# Pipeline
# ============================================================

def validate_record(args: argparse.Namespace) -> JSONDict:
    record = read_json(args.input)
    pipeline_metadata = merge_pipeline_metadata(record, args)
    ledger = ledger_from_args(args)

    source_url = get_source_url(
        record,
        args.source_url or args.input.resolve().as_uri(),
    )

    evidence_store = get_evidence_store(
        record,
        source_url=source_url,
        use_ollama_chunking=args.ollama_chunking,
    )

    client = OpenAI()

    print(f"[postrag] evidence blocks: {len(evidence_store)}")

    entities, relations = validate_many_blockwise(
        client,
        evidence_store,
        record.get("entities", []) or [],
        record.get("relations", []) or [],
        model=args.model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_evidence_chars=args.max_evidence_chars,
        max_block_candidates=args.max_block_candidates,
        fallback_candidate_validation=args.fallback_candidate_validation,
        ledger=ledger,
    )

    return build_output(
        record,
        source_url=source_url,
        evidence_store=evidence_store,
        entities=entities,
        relations=relations,
        model=args.model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        pipeline_metadata=pipeline_metadata,
        cost_tracking={
            "enabled": bool(args.cost_ledger),
            "cache_enabled": bool(args.cache_dir) and not args.disable_cache,
            "ledger_path": str(args.cost_ledger) if args.cost_ledger else None,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "prompt_version": PROMPT_VERSION,
            "validation_mode": "blockwise",
            "max_evidence_chars": args.max_evidence_chars,
            "max_block_candidates": args.max_block_candidates,
            "fallback_candidate_validation": bool(args.fallback_candidate_validation),
        },
    )


def main() -> int:
    args = parse_args()
    result = validate_record(args)
    write_json(args.output, result)
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())