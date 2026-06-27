#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import sys
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

try:
    from cost_ledger import CostLedger, load_pricing_config
except Exception:
    print("[extract] No cost ledger supplied, using None")
    CostLedger = None  # type: ignore
    load_pricing_config = None  # type: ignore


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

for path in (CURRENT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from .document import EvidenceBlock, HTMLWebpageDocument
    from .webpage import html_webpage_document_from_url
except ImportError:
    from document import EvidenceBlock, HTMLWebpageDocument
    from webpage import html_webpage_document_from_url


JSONDict = Dict[str, Any]

DEFAULT_URL = (
    "https://www.csis.org/analysis/"
    "consequences-chinas-new-rare-earths-export-restrictions"
)


# ============================================================
# CLI / I/O
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured KG candidates from an EvidenceDocument."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("structured_output.json"))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--ollama-chunking", action="store_true")
    parser.add_argument("--max-block-chars", type=int, default=2500)
    parser.add_argument("--max-total-chars", type=int, default=24000)
    parser.add_argument("--include-evidence-text", action="store_true")
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


def write_json(path: Path, data: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Loaded document JSON compatibility
# ============================================================

class LoadedEvidenceBlock:
    def __init__(self, raw: JSONDict, document_id: str, source_url: Optional[str]) -> None:
        self.raw = raw
        self.block_id = raw.get("block_id") or raw.get("evidence_id")
        self.document_id = raw.get("document_id") or document_id
        self.block_type = raw.get("block_type")
        self.source_url = raw.get("source_url") or source_url
        self.metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

    def to_text(self) -> str:
        text = self.raw.get("text")
        if isinstance(text, str):
            return text
        return ""

    def to_dict(self) -> JSONDict:
        data = dict(self.raw)
        data.setdefault("block_id", self.block_id)
        data.setdefault("document_id", self.document_id)
        data.setdefault("block_type", self.block_type)
        data.setdefault("source_url", self.source_url)
        data.setdefault("metadata", self.metadata)
        data.setdefault("text", self.to_text())
        return data


class LoadedEvidenceDocument:
    def __init__(self, raw: JSONDict) -> None:
        self.raw = raw

        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        credibility = raw.get("credibility") if isinstance(raw.get("credibility"), dict) else {}

        self.document_id = raw.get("document_id")
        self.source_type = raw.get("source_type")
        self.metadata = SimpleNamespace(
            title=metadata.get("title"),
            author=metadata.get("author"),
            publisher=metadata.get("publisher"),
            published_at=metadata.get("published_at"),
            source_url=metadata.get("source_url"),
            canonical_url=metadata.get("canonical_url"),
        )
        self.credibility = SimpleNamespace(
            score=credibility.get("score", 0.5),
            tier=credibility.get("tier", "unknown"),
            rationale=credibility.get("rationale", ""),
        )
        self.blocks = [
            LoadedEvidenceBlock(block, self.document_id, self.metadata.source_url)
            for block in raw.get("blocks", [])
            if isinstance(block, dict)
        ]

    def ensure_blocks(self) -> List[LoadedEvidenceBlock]:
        return self.blocks


def document_from_json(path: Path) -> LoadedEvidenceDocument:
    return LoadedEvidenceDocument(read_json(path))


# ============================================================
# Taxonomy
# ============================================================

def enum_value(x: Any) -> str:
    return getattr(x, "value", str(x))


def load_taxonomy() -> Any:
    try:
        return importlib.import_module("taxonomy")
    except ImportError as exc:
        raise ImportError("Expected taxonomy.py to be importable from src/.") from exc


def enum_values(module: Any, name: str) -> List[str]:
    cls = getattr(module, name, None)
    if cls is None:
        return []
    return [enum_value(x) for x in cls]


def relation_signature_map(module: Any, attr: str) -> Dict[str, List[str]]:
    raw = getattr(module, attr, None)
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for relation, entity_types in raw.items():
        out[enum_value(relation)] = sorted(enum_value(t) for t in entity_types)
    return out


def taxonomy_context() -> JSONDict:
    module = load_taxonomy()
    return {
        "entity_types": enum_values(module, "EntityType"),
        "relation_types": enum_values(module, "RelationType"),
        "allowed_subject_types_by_relation": relation_signature_map(
            module,
            "ALLOWED_RELATION_SIGNATURES",
        ),
        "allowed_object_types_by_relation": relation_signature_map(
            module,
            "ALLOWED_OBJECT_TYPES",
        ),
    }


# ============================================================
# Evidence/document serialization
# ============================================================

def pipeline_metadata_from_args(args: argparse.Namespace) -> JSONDict:
    return {
        "run_id": args.run_id,
        "source_id": args.source_id,
        "corpus_id": args.corpus_id,
        "branch_id": args.branch_id,
        "canonical_source": args.canonical_source,
    }


def clean_pipeline_metadata(metadata: Optional[JSONDict]) -> JSONDict:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ""
    }

def ledger_from_args(args: argparse.Namespace) -> Optional[Any]:
    if not args.cost_ledger and not args.cache_dir:
        return None

    if CostLedger is None:
        raise RuntimeError("cost_ledger.py could not be imported, but cost/cache options were provided.")

    pricing_config = {}
    if args.pricing_file and load_pricing_config is not None:
        pricing_config = load_pricing_config(args.pricing_file)

    return CostLedger(
        run_id=args.run_id or "extract_run",
        source_id=args.source_id,
        ledger_path=args.cost_ledger,
        cache_dir=args.cache_dir,
        pricing_config=pricing_config,
        enabled=bool(args.cost_ledger),
        cache_enabled=bool(args.cache_dir) and not args.disable_cache,
    )

def document_metadata(document: HTMLWebpageDocument) -> JSONDict:
    metadata = document.metadata
    return {
        "document_id": document.document_id,
        "source_type": enum_value(document.source_type),
        "title": metadata.title,
        "author": metadata.author,
        "publisher": metadata.publisher,
        "published_at": metadata.published_at,
        "source_url": metadata.source_url,
        "canonical_url": metadata.canonical_url,
        "credibility": {
            "score": document.credibility.score,
            "tier": enum_value(document.credibility.tier),
            "rationale": document.credibility.rationale,
        },
    }


def block_payload(block: EvidenceBlock, max_chars: int) -> JSONDict:
    text = block.to_text()
    return {
        "evidence_id": block.block_id,
        "block_type": enum_value(block.block_type),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "metadata": block.metadata,
    }


def evidence_payload(
    document: HTMLWebpageDocument,
    *,
    max_block_chars: int,
    max_total_chars: int,
) -> List[JSONDict]:
    out: List[JSONDict] = []
    used = 0

    for block in document.ensure_blocks():
        item = block_payload(block, max_block_chars)
        n = len(item["text"])

        if out and used + n > max_total_chars:
            break

        out.append(item)
        used += n

    return out


def evidence_manifest(document: HTMLWebpageDocument) -> JSONDict:
    return {
        block.block_id: {
            "evidence_id": block.block_id,
            "document_id": block.document_id,
            "block_type": enum_value(block.block_type),
            "source_url": block.source_url,
            "metadata": block.metadata,
        }
        for block in document.ensure_blocks()
    }


def evidence_store(document: HTMLWebpageDocument) -> JSONDict:
    return {
        block.block_id: block.to_dict()
        for block in document.ensure_blocks()
    }


# ============================================================
# LLM
# ============================================================

def call_llm_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    ledger: Optional[Any] = None,
    operation: str = "extract_document_candidates",
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
            stage="extract",
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

def build_prompt(
    *,
    document: HTMLWebpageDocument,
    taxonomy: JSONDict,
    evidence: Sequence[JSONDict],
) -> str:
    return f"""
You are extracting candidate knowledge-graph records from one source document.

Use only the evidence blocks provided.
Do not use outside knowledge.
Do not invent facts.
Do not perform final validation; a later PostRAG stage will validate candidates.
Return only valid JSON.

Your task:
1. Build one global entity table for the document.
2. Build candidate relations between those entities.
3. Attach evidence IDs and short supporting quotes.
4. Avoid duplicates.
5. Avoid vague generic entities unless they are explicitly named and useful as relation endpoints.

Entity rules:
- Extract only named or clearly defined entities that are useful claim endpoints.
- Keep distinct real-world objects distinct.
- Do not use a policy/event as a commodity.
- Do not use a country as a commodity.
- Do not use an agency as a regulation.
- If the correct taxonomy type is unclear, choose the broader valid type.
- Every entity must have at least one evidence item.

Relation rules:
- Relations must use subject_id and object_id from the entity table.
- Relations should be useful source-grounded candidates, not final truth.
- Do not substitute endpoints. If the evidence names one entity, do not output another.
- Do not infer a specific company relation from a generic industry statement.
- Every relation must have at least one evidence item.
- The evidence quote should support the subject, relation, and object.
- If a relation is plausible but uncertain, include it with lower extraction_confidence rather than over-filtering.
- If a relation is unsupported or endpoint mapping is unclear, omit it.
- Relations must be directional. For every relation, ensure the subject is the actor/source/holder and the object is the target/output/location. Do not invert triples to match a noun phrase.

Return exactly this JSON shape:
{{
  "entities": [
    {{
      "entity_id": "e1",
      "canonical_name": "string",
      "entity_type": "one value from taxonomy.entity_types",
      "aliases": [],
      "description": "brief source-grounded description",
      "attributes": {{}},
      "temporal": {{
        "event_date": null,
        "valid_from": null,
        "valid_to": null
      }},
      "evidence": [
        {{
          "evidence_id": "one provided evidence_id",
          "quote": "short quote"
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "relation_id": "r1",
      "subject_id": "entity_id",
      "relation_type": "one value from taxonomy.relation_types",
      "object_id": "entity_id",
      "description": "brief source-grounded description",
      "temporal": {{
        "event_date": null,
        "valid_from": null,
        "valid_to": null
      }},
      "extraction_confidence": 0.0,
      "attributes": {{}},
      "evidence": [
        {{
          "evidence_id": "one provided evidence_id",
          "quote": "short quote"
        }}
      ]
    }}
  ]
}}

Use sequential IDs: e1, e2, ... and r1, r2, ...

Taxonomy:
{json.dumps(taxonomy, indent=2, ensure_ascii=False)}

Document metadata:
{json.dumps(document_metadata(document), indent=2, ensure_ascii=False)}

Evidence blocks:
{json.dumps(evidence, indent=2, ensure_ascii=False)}
""".strip()


# ============================================================
# Minimal normalization
# ============================================================

def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text or None


def clean_dict(value: Any) -> JSONDict:
    return value if isinstance(value, dict) else {}


def clean_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def clean_temporal(value: Any) -> JSONDict:
    value = value if isinstance(value, dict) else {}
    return {
        "event_date": clean_text(value.get("event_date")),
        "valid_from": clean_text(value.get("valid_from")),
        "valid_to": clean_text(value.get("valid_to")),
    }


def evidence_refs(raw: Any, allowed_ids: set) -> List[JSONDict]:
    refs: List[JSONDict] = []

    for item in clean_list(raw):
        if isinstance(item, str):
            evidence_id = item
            quote = None
        elif isinstance(item, dict):
            evidence_id = clean_text(item.get("evidence_id"))
            quote = clean_text(item.get("quote"))
        else:
            continue

        if not evidence_id or evidence_id not in allowed_ids:
            continue

        ref = {"evidence_id": evidence_id, "quote": quote}
        if ref not in refs:
            refs.append(ref)

    return refs


def normalize_entities(
    raw_entities: Any,
    *,
    allowed_entity_types: set,
    allowed_evidence_ids: set,
    source_url: Optional[str],
    observed_at: Optional[str],
) -> Tuple[List[JSONDict], Dict[str, str]]:
    entities: List[JSONDict] = []
    old_to_new: Dict[str, str] = {}
    seen = set()

    for raw in clean_list(raw_entities):
        if not isinstance(raw, dict):
            continue

        old_id = clean_text(raw.get("entity_id"))
        name = clean_text(raw.get("canonical_name") or raw.get("name"))
        entity_type = clean_text(raw.get("entity_type") or raw.get("type"))

        if not old_id or not name or not entity_type:
            continue

        if allowed_entity_types and entity_type not in allowed_entity_types:
            continue

        refs = evidence_refs(raw.get("evidence"), allowed_evidence_ids)
        if not refs:
            continue

        key = (name.lower(), entity_type.lower())
        if key in seen:
            continue

        new_id = f"e{len(entities) + 1}"
        old_to_new[old_id] = new_id
        seen.add(key)

        entities.append({
            "entity_id": new_id,
            "canonical_name": name,
            "entity_type": entity_type,
            "aliases": [
                clean_text(x)
                for x in clean_list(raw.get("aliases"))
                if clean_text(x)
            ],
            "description": clean_text(raw.get("description")),
            "attributes": clean_dict(raw.get("attributes")),
            "temporal": clean_temporal(raw.get("temporal")),
            "source_url": source_url,
            "provenance": [
                {
                    "evidence_id": ref["evidence_id"],
                    "quote": ref.get("quote"),
                    "source_url": source_url,
                    "observed_at": observed_at,
                }
                for ref in refs
            ],
            "postrag_evidence": [
                {
                    "evidence_id": ref["evidence_id"],
                    "rank": None,
                    "retrieval_score": None,
                }
                for ref in refs
            ],
        })

    return entities, old_to_new


def normalize_relations(
    raw_relations: Any,
    *,
    entities: Sequence[JSONDict],
    old_to_new_entity_ids: Dict[str, str],
    allowed_relation_types: set,
    allowed_evidence_ids: set,
    source_url: Optional[str],
    observed_at: Optional[str],
) -> List[JSONDict]:
    entities_by_id = {entity["entity_id"]: entity for entity in entities}
    relations: List[JSONDict] = []
    seen = set()

    for raw in clean_list(raw_relations):
        if not isinstance(raw, dict):
            continue

        old_subject = clean_text(raw.get("subject_id"))
        old_object = clean_text(raw.get("object_id"))

        subject_id = old_to_new_entity_ids.get(old_subject or "", old_subject)
        object_id = old_to_new_entity_ids.get(old_object or "", old_object)

        relation_type = clean_text(raw.get("relation_type") or raw.get("type"))

        if not subject_id or not object_id or not relation_type:
            continue

        if subject_id not in entities_by_id or object_id not in entities_by_id:
            continue

        if allowed_relation_types and relation_type not in allowed_relation_types:
            continue

        refs = evidence_refs(raw.get("evidence"), allowed_evidence_ids)
        if not refs:
            continue

        temporal = clean_temporal(raw.get("temporal"))

        key = (
            subject_id,
            relation_type,
            object_id,
            json.dumps(temporal, sort_keys=True),
        )
        if key in seen:
            continue

        seen.add(key)

        subject = entities_by_id[subject_id]
        obj = entities_by_id[object_id]

        relations.append({
            "relation_id": f"r{len(relations) + 1}",
            "subject_id": subject_id,
            "subject": subject["canonical_name"],
            "relation_type": relation_type,
            "object_id": object_id,
            "object": obj["canonical_name"],
            "description": clean_text(raw.get("description")),
            "temporal": temporal,
            "confidence": raw.get("extraction_confidence") or raw.get("confidence"),
            "attributes": clean_dict(raw.get("attributes")),
            "source_url": source_url,
            "provenance": [
                {
                    "evidence_id": ref["evidence_id"],
                    "quote": ref.get("quote"),
                    "source_url": source_url,
                    "observed_at": observed_at,
                }
                for ref in refs
            ],
            "postrag_evidence": [
                {
                    "evidence_id": ref["evidence_id"],
                    "rank": None,
                    "retrieval_score": None,
                }
                for ref in refs
            ],
        })

    return relations


def normalize_output(
    raw: JSONDict,
    *,
    document: HTMLWebpageDocument,
    taxonomy: JSONDict,
    allowed_evidence_ids: set,
) -> Tuple[List[JSONDict], List[JSONDict]]:
    source_url = document.metadata.source_url
    observed_at = document.metadata.published_at

    entities, old_to_new = normalize_entities(
        raw.get("entities"),
        allowed_entity_types=set(taxonomy.get("entity_types") or []),
        allowed_evidence_ids=allowed_evidence_ids,
        source_url=source_url,
        observed_at=observed_at,
    )

    relations = normalize_relations(
        raw.get("relations"),
        entities=entities,
        old_to_new_entity_ids=old_to_new,
        allowed_relation_types=set(taxonomy.get("relation_types") or []),
        allowed_evidence_ids=allowed_evidence_ids,
        source_url=source_url,
        observed_at=observed_at,
    )

    return entities, relations


# ============================================================
# Pipeline
# ============================================================

def extract_structured_candidates_from_document(
    document: HTMLWebpageDocument,
    *,
    model: str = "gpt-4.1-mini",
    max_block_chars: int = 2500,
    max_total_chars: int = 24000,
    include_evidence_text: bool = False,
    pipeline_metadata: Optional[JSONDict] = None,
    ledger: Optional[Any] = None,
) -> JSONDict:
    client = OpenAI()
    taxonomy = taxonomy_context()

    evidence = evidence_payload(
        document,
        max_block_chars=max_block_chars,
        max_total_chars=max_total_chars,
    )

    allowed_evidence_ids = {item["evidence_id"] for item in evidence}

    print(f"[extract] document blocks available: {len(document.ensure_blocks())}")
    print(f"[extract] evidence blocks sent to LLM: {len(evidence)}")
    print(f"[extract] total evidence chars sent: {sum(len(item['text']) for item in evidence)}")

    prompt = build_prompt(
        document=document,
        taxonomy=taxonomy,
        evidence=evidence,
    )

    raw = call_llm_json(
        client,
        model=model,
        prompt=prompt,
        ledger=ledger,
        operation="extract_document_candidates",
    )

    raw_entity_count = len(raw.get("entities", []) or [])
    raw_relation_count = len(raw.get("relations", []) or [])

    print(f"[extract] raw entities: {raw_entity_count}")
    print(f"[extract] raw relations: {raw_relation_count}")

    entities, relations = normalize_output(
        raw,
        document=document,
        taxonomy=taxonomy,
        allowed_evidence_ids=allowed_evidence_ids,
    )

    print(f"[extract] normalized entities: {len(entities)}")
    print(f"[extract] normalized relations: {len(relations)}")

    output: JSONDict = {
        "document": document_metadata(document),
        "source_url": document.metadata.source_url,
        "source_id": document.document_id,
        "evidence_manifest": evidence_manifest(document),
        "entities": entities,
        "relations": relations,
        "extraction": {
            "model": model,
            "method": "global_single_pass_document_extraction",
            "llm_call_count": 1,
            "document_block_count": len(document.ensure_blocks()),
            "evidence_blocks_sent": len(evidence),
            "evidence_chars_sent": sum(len(item["text"]) for item in evidence),
            "raw_entity_count": raw_entity_count,
            "raw_relation_count": raw_relation_count,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "include_evidence_text": include_evidence_text,
            "pipeline_metadata": clean_pipeline_metadata(pipeline_metadata),
        },
    }

    if include_evidence_text:
        output["evidence_store"] = evidence_store(document)

    return output


def extract_from_url(
    url: str,
    *,
    model: str = "gpt-4.1-mini",
    use_ollama_chunking: bool = False,
    max_block_chars: int = 2500,
    max_total_chars: int = 24000,
    include_evidence_text: bool = False,
    pipeline_metadata: Optional[JSONDict] = None,
    ledger: Optional[Any] = None,
) -> JSONDict:
    document = html_webpage_document_from_url(
        url,
        use_ollama_chunking=use_ollama_chunking,
    )

    return extract_structured_candidates_from_document(
        document,
        model=model,
        max_block_chars=max_block_chars,
        max_total_chars=max_total_chars,
        include_evidence_text=include_evidence_text,
        pipeline_metadata=pipeline_metadata,
        ledger=ledger,
    )


def extract_from_input(
    path: Path,
    *,
    model: str = "gpt-4.1-mini",
    max_block_chars: int = 2500,
    max_total_chars: int = 24000,
    include_evidence_text: bool = False,
    pipeline_metadata: Optional[JSONDict] = None,
    ledger: Optional[Any] = None,
) -> JSONDict:
    document = document_from_json(path)

    return extract_structured_candidates_from_document(
        document,
        model=model,
        max_block_chars=max_block_chars,
        max_total_chars=max_total_chars,
        include_evidence_text=include_evidence_text,
        pipeline_metadata=pipeline_metadata,
        ledger=ledger,
    )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    arguments = parse_args()
    pipeline_metadata = pipeline_metadata_from_args(arguments)
    ledger = ledger_from_args(arguments)

    if arguments.input is not None:
        result = extract_from_input(
            arguments.input,
            model=arguments.model,
            max_block_chars=arguments.max_block_chars,
            max_total_chars=arguments.max_total_chars,
            include_evidence_text=arguments.include_evidence_text,
            pipeline_metadata=pipeline_metadata,
            ledger=ledger,
        )
    else:
        result = extract_from_url(
            arguments.url,
            model=arguments.model,
            use_ollama_chunking=arguments.ollama_chunking,
            max_block_chars=arguments.max_block_chars,
            max_total_chars=arguments.max_total_chars,
            include_evidence_text=arguments.include_evidence_text,
            pipeline_metadata=pipeline_metadata,
            ledger=ledger,
        )

    write_json(arguments.output, result)

    print(f"[done] wrote {arguments.output}")
    print(f"[done] entities: {len(result['entities'])}")
    print(f"[done] relations: {len(result['relations'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())