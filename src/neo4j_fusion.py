#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from neo4j import GraphDatabase
from openai import OpenAI
from cost_ledger import CostLedger, load_pricing_config, summarize_ledger


JSONDict = Dict[str, Any]


# ============================================================
# CLI / I/O
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse PostRAG output into Neo4j."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--clear", action="store_true")

    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--entity-neighbor-k", type=int, default=8)
    parser.add_argument("--run-id", default=os.getenv("LANTHIC_RUN_ID"))
    parser.add_argument("--source-id", default=os.getenv("LANTHIC_SOURCE_ID"))
    parser.add_argument("--corpus-id", default=os.getenv("LANTHIC_CORPUS_ID"))
    parser.add_argument("--branch-id", default=os.getenv("LANTHIC_BRANCH_ID"))
    parser.add_argument("--canonical-source", default=os.getenv("LANTHIC_CANONICAL_SOURCE"))

    parser.add_argument("--cost-ledger", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--pricing-file", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)

    return parser.parse_args()


def read_json(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def write_json(path: Path, data: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )   


# ============================================================
# Normalization
# ============================================================

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(value: Any, max_chars: int = 8000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip()


def json_prop(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def source_url(record: JSONDict) -> Optional[str]:
    return (
        record.get("source_url")
        or (record.get("document") or {}).get("source_url")
        or (record.get("document") or {}).get("canonical_url")
    )

def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def pipeline_metadata_from_record(
    record: JSONDict,
    overrides: Optional[JSONDict] = None,
) -> JSONDict:
    """
    Normalize run/corpus/branch/source metadata from PostRAG artifacts.

    Priority:
    1. CLI/import overrides
    2. record["pipeline_metadata"]
    3. record["mass_ingest"]
    4. record/document fields
    5. deterministic fallback
    """
    document = record.get("document") if isinstance(record.get("document"), dict) else {}

    raw = {}
    if isinstance(record.get("pipeline_metadata"), dict):
        raw.update(record["pipeline_metadata"])
    if isinstance(record.get("mass_ingest"), dict):
        raw.update(record["mass_ingest"])

    if overrides:
        for key, value in overrides.items():
            if value not in (None, ""):
                raw[key] = value

    url = source_url(record)
    canonical_source = first_nonempty(
        raw.get("canonical_source"),
        raw.get("canonical_url"),
        record.get("canonical_source"),
        document.get("canonical_url"),
        url,
    )

    document_id = first_nonempty(
        raw.get("document_id"),
        document.get("document_id"),
        record.get("document_id"),
        record.get("source_id"),
    )

    source_id = first_nonempty(
        raw.get("source_id"),
        record.get("source_id"),
        document.get("source_id"),
    )

    if not source_id:
        source_id = "src_" + stable_hash(canonical_source or url or document_id or record)

    run_id = first_nonempty(
        raw.get("run_id"),
        record.get("run_id"),
        os.getenv("LANTHIC_RUN_ID"),
    )

    if not run_id:
        run_id = "neo4j_import_" + stable_hash({
            "source_id": source_id,
            "document_id": document_id,
            "canonical_source": canonical_source,
        })

    corpus_id = first_nonempty(
        raw.get("corpus_id"),
        record.get("corpus_id"),
        os.getenv("LANTHIC_CORPUS_ID"),
        "default_corpus",
    )

    branch_id = first_nonempty(
        raw.get("branch_id"),
        record.get("branch_id"),
        os.getenv("LANTHIC_BRANCH_ID"),
        "staging",
    )

    source_kind = first_nonempty(
        raw.get("source_kind"),
        document.get("source_type"),
        record.get("source_type"),
    )

    return {
        "run_id": str(run_id),
        "corpus_id": str(corpus_id),
        "branch_id": str(branch_id),
        "branch_key": f"{corpus_id}::{branch_id}",
        "source_id": str(source_id),
        "document_id": document_id,
        "source_url": url,
        "canonical_source": canonical_source,
        "source_kind": source_kind,
        "metadata_json": json_prop(raw),
    }


def merge_pipeline_scope(tx, metadata: JSONDict) -> None:
    tx.run(
        """
        MERGE (co:Corpus {corpus_id: $corpus_id})
        SET
          co.updated_at = datetime()

        MERGE (br:Branch {key: $branch_key})
        SET
          br.branch_id = $branch_id,
          br.corpus_id = $corpus_id,
          br.updated_at = datetime()

        MERGE (br)-[:IN_CORPUS]->(co)

        MERGE (run:ImportRun {run_id: $run_id})
        SET
          run.corpus_id = $corpus_id,
          run.branch_id = $branch_id,
          run.branch_key = $branch_key,
          run.source_id = $source_id,
          run.document_id = $document_id,
          run.canonical_source = $canonical_source,
          run.source_url = $source_url,
          run.updated_at = datetime()

        MERGE (run)-[:IN_BRANCH]->(br)
        """,
        **metadata,
    )

def postrag_decision(item: JSONDict) -> Optional[str]:
    return (item.get("postrag") or {}).get("decision")


def grounding_score(item: JSONDict) -> Optional[float]:
    value = (item.get("postrag") or {}).get("grounding_score")
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def accepted_or_coarsened(item: JSONDict) -> bool:
    return postrag_decision(item) in {"accept", "coarsen"}


def entity_name(entity: JSONDict) -> str:
    return (
        entity.get("canonical_name")
        or entity.get("name")
        or entity.get("label")
        or ""
    )


def entity_type(entity: JSONDict) -> str:
    return (
        entity.get("entity_type")
        or entity.get("type")
        or "unknown"
    )


def deterministic_entity_key(entity: JSONDict) -> str:
    return f"{normalize_text(entity_type(entity))}::{normalize_text(entity_name(entity))}"


def relation_type(relation: JSONDict) -> str:
    return (
        relation.get("relation_type")
        or relation.get("type")
        or relation.get("predicate")
        or "related_to"
    )


def relation_temporal(relation: JSONDict) -> JSONDict:
    temporal = relation.get("temporal")
    if not isinstance(temporal, dict):
        temporal = {}

    return {
        "event_date": temporal.get("event_date") or relation.get("event_date"),
        "valid_from": temporal.get("valid_from") or relation.get("valid_from"),
        "valid_to": temporal.get("valid_to") or relation.get("valid_to"),
    }


def local_entity_id(entity: JSONDict) -> str:
    return str(entity.get("entity_id") or entity.get("id") or "")


def relation_subject_id(relation: JSONDict) -> str:
    return str(
        relation.get("subject_id")
        or relation.get("source_id")
        or relation.get("head_id")
        or relation.get("subject_entity_id")
        or ""
    )


def relation_object_id(relation: JSONDict) -> str:
    return str(
        relation.get("object_id")
        or relation.get("target_id")
        or relation.get("tail_id")
        or relation.get("object_entity_id")
        or ""
    )


def evidence_ids(item: JSONDict) -> List[str]:
    ids: List[str] = []

    for ref in item.get("postrag_evidence", []) or []:
        if isinstance(ref, dict) and ref.get("evidence_id"):
            ids.append(str(ref["evidence_id"]))

    for ref in item.get("provenance", []) or []:
        if isinstance(ref, dict) and ref.get("evidence_id"):
            ids.append(str(ref["evidence_id"]))

    for evidence_id in (item.get("postrag") or {}).get("used_evidence_ids", []) or []:
        if evidence_id:
            ids.append(str(evidence_id))

    out: List[str] = []
    seen = set()

    for evidence_id in ids:
        if evidence_id not in seen:
            out.append(evidence_id)
            seen.add(evidence_id)

    return out


def filtered_entities(record: JSONDict) -> List[JSONDict]:
    postrag_filtered = record.get("postrag_filtered")

    if isinstance(postrag_filtered, dict) and isinstance(postrag_filtered.get("entities"), list):
        return list(postrag_filtered["entities"])

    return [
        entity
        for entity in record.get("entities", []) or []
        if accepted_or_coarsened(entity)
    ]


def filtered_relations(record: JSONDict) -> List[JSONDict]:
    postrag_filtered = record.get("postrag_filtered")

    if isinstance(postrag_filtered, dict) and isinstance(postrag_filtered.get("relations"), list):
        return list(postrag_filtered["relations"])

    return [
        relation
        for relation in record.get("relations", []) or []
        if accepted_or_coarsened(relation)
    ]


# ============================================================
# Embedding / LLM helpers
# ============================================================

def embed_text(
    client: OpenAI,
    model: str,
    text: str,
    *,
    ledger: Optional[CostLedger] = None,
    operation: str = "embed_text",
) -> List[float]:
    bounded = compact_text(text)

    if ledger is not None:
        return ledger.embed_text(
            client,
            stage="neo4j_fusion",
            model=model,
            text=bounded,
            operation=operation,
        )

    response = client.embeddings.create(
        model=model,
        input=bounded,
    )
    return list(response.data[0].embedding)

def call_llm_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    ledger: Optional[CostLedger] = None,
    operation: str = "chat_json",
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
            stage="neo4j_fusion",
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


def entity_embedding_text(entity: JSONDict) -> str:
    return "\n".join([
        f"name: {entity_name(entity)}",
        f"type: {entity_type(entity)}",
        f"aliases: {json_prop(entity.get('aliases') or [])}",
        f"description: {entity.get('description') or ''}",
        f"attributes: {json_prop(entity.get('attributes') or {})}",
        f"temporal: {json_prop(entity.get('temporal') or {})}",
        f"provenance_quotes: {json_prop([ref.get('quote') for ref in entity.get('provenance', []) or [] if isinstance(ref, dict)])}",
    ])


def claim_embedding_text(
    relation: JSONDict,
    *,
    subject_name: str,
    object_name: str,
) -> str:
    return "\n".join([
        f"subject: {subject_name}",
        f"relation_type: {relation_type(relation)}",
        f"object: {object_name}",
        f"description: {relation.get('description') or ''}",
        f"attributes: {json_prop(relation.get('attributes') or {})}",
        f"temporal: {json_prop(relation_temporal(relation))}",
        f"provenance_quotes: {json_prop([ref.get('quote') for ref in relation.get('provenance', []) or [] if isinstance(ref, dict)])}",
    ])


def evidence_embedding_text(evidence: JSONDict) -> str:
    return "\n".join([
        f"block_type: {evidence.get('block_type') or ''}",
        f"metadata: {json_prop(evidence.get('metadata') or {})}",
        f"text: {evidence.get('text') or ''}",
    ])


# ============================================================
# Entity-resolution decision
# ============================================================

def entity_merge_prompt(
    *,
    incoming_entity: JSONDict,
    incoming_embedding_text: str,
    candidates: Sequence[JSONDict],
) -> str:
    return f"""
You are resolving whether an incoming extracted entity should be merged with an existing knowledge-graph Entity node.

Use only the incoming entity record and nearest-neighbour candidates provided below.
The nearest neighbours were retrieved from embedding space. Similarity alone is not sufficient for merging.
You must decide identity/equivalence.

Decision rules:
- Return "merge_with_existing" only if the incoming entity and candidate refer to the same real-world entity/concept.
- Return "create_new" if none of the candidates is clearly the same entity.
- Do not merge merely because entities are related, geographically near, causally connected, or in the same sector.
- Do not merge a country with a region, company, commodity, facility, policy, or event.
- Do not merge a company/organization with a region or commodity.
- Alias/spelling variants may be merged if they refer to the same entity.
- If uncertain, choose "create_new".

Return exactly this JSON shape:
{{
  "decision": "merge_with_existing|create_new",
  "merge_key": null,
  "matched_candidate_rank": null,
  "confidence": "high|moderate|low",
  "reason": "brief explanation"
}}

If decision is "merge_with_existing", merge_key must be one of the candidate keys.
If decision is "create_new", merge_key must be null.

Incoming entity:
{json.dumps(incoming_entity, indent=2, ensure_ascii=False, default=str)}

Incoming entity embedding text:
{incoming_embedding_text}

Nearest existing Entity candidates:
{json.dumps(list(candidates), indent=2, ensure_ascii=False, default=str)}
""".strip()


def merge_decision_key(
    *,
    incoming_key: str,
    record_source_url: Optional[str],
    entity: JSONDict,
) -> str:
    return "entity_merge_decision_" + stable_hash({
        "incoming_key": incoming_key,
        "source_url": record_source_url,
        "entity_id": entity.get("entity_id"),
        "name": entity_name(entity),
        "type": entity_type(entity),
    })


# ============================================================
# Neo4j setup
# ============================================================

def clear_database(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def create_constraints(tx) -> None:
    tx.run("""
    CREATE CONSTRAINT entity_key IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.key IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT claim_key IF NOT EXISTS
    FOR (c:Claim)
    REQUIRE c.key IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT evidence_id IF NOT EXISTS
    FOR (ev:Evidence)
    REQUIRE ev.evidence_id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT source_url IF NOT EXISTS
    FOR (s:Source)
    REQUIRE s.url IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT source_id IF NOT EXISTS
    FOR (s:Source)
    REQUIRE s.source_id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT corpus_id IF NOT EXISTS
    FOR (co:Corpus)
    REQUIRE co.corpus_id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT branch_key IF NOT EXISTS
    FOR (br:Branch)
    REQUIRE br.key IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT import_run_id IF NOT EXISTS
    FOR (run:ImportRun)
    REQUIRE run.run_id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT entity_merge_decision_key IF NOT EXISTS
    FOR (d:EntityMergeDecision)
    REQUIRE d.key IS UNIQUE
    """)


def create_vector_indexes(tx, embedding_dim: int) -> None:
    tx.run(
        """
        CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
        FOR (e:Entity)
        ON (e.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: $embedding_dim,
            `vector.similarity_function`: 'cosine'
          }
        }
        """,
        embedding_dim=embedding_dim,
    )

    tx.run(
        """
        CREATE VECTOR INDEX claim_embedding_index IF NOT EXISTS
        FOR (c:Claim)
        ON (c.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: $embedding_dim,
            `vector.similarity_function`: 'cosine'
          }
        }
        """,
        embedding_dim=embedding_dim,
    )

    tx.run(
        """
        CREATE VECTOR INDEX evidence_embedding_index IF NOT EXISTS
        FOR (ev:Evidence)
        ON (ev.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: $embedding_dim,
            `vector.similarity_function`: 'cosine'
          }
        }
        """,
        embedding_dim=embedding_dim,
    )


# ============================================================
# Neo4j reads for vector nearest neighbours
# ============================================================

def nearest_entity_candidates(
    tx,
    embedding: Sequence[float],
    *,
    k: int,
) -> List[JSONDict]:
    rows = tx.run(
        """
        CALL db.index.vector.queryNodes('entity_embedding_index', $k, $embedding)
        YIELD node, score
        RETURN
          node.key AS key,
          node.entity_id AS entity_id,
          node.canonical_name AS canonical_name,
          node.entity_type AS entity_type,
          node.description AS description,
          node.aliases_json AS aliases_json,
          node.attributes_json AS attributes_json,
          node.temporal_json AS temporal_json,
          node.created_from_source_url AS created_from_source_url,
          score
        ORDER BY score DESC
        """,
        k=k,
        embedding=list(embedding),
    ).data()

    out = []

    for i, row in enumerate(rows, start=1):
        out.append({
            "rank": i,
            "key": row.get("key"),
            "entity_id": row.get("entity_id"),
            "canonical_name": row.get("canonical_name"),
            "entity_type": row.get("entity_type"),
            "description": row.get("description"),
            "aliases_json": row.get("aliases_json"),
            "attributes_json": row.get("attributes_json"),
            "temporal_json": row.get("temporal_json"),
            "created_from_source_url": row.get("created_from_source_url"),
            "vector_score": row.get("score"),
        })

    return out


# ============================================================
# Neo4j writes
# ============================================================

def merge_source(tx, url: Optional[str], document: JSONDict, pipeline_metadata: JSONDict) -> None:
    merge_pipeline_scope(tx, pipeline_metadata)

    source_id = pipeline_metadata["source_id"]

    if url:
        tx.run(
            """
            MERGE (s:Source {url: $url})
            SET
              s.source_id = coalesce(s.source_id, $source_id),
              s.source_key = $source_id,
              s.title = $title,
              s.publisher = $publisher,
              s.published_at = $published_at,
              s.document_id = $document_id,
              s.source_kind = $source_kind,
              s.canonical_source = $canonical_source,
              s.latest_run_id = $run_id,
              s.latest_corpus_id = $corpus_id,
              s.latest_branch_id = $branch_id,
              s.document_json = $document_json,
              s.pipeline_metadata_json = $pipeline_metadata_json
            """,
            url=url,
            source_id=source_id,
            title=document.get("title"),
            publisher=document.get("publisher"),
            published_at=document.get("published_at"),
            document_id=pipeline_metadata.get("document_id"),
            source_kind=pipeline_metadata.get("source_kind"),
            canonical_source=pipeline_metadata.get("canonical_source"),
            run_id=pipeline_metadata.get("run_id"),
            corpus_id=pipeline_metadata.get("corpus_id"),
            branch_id=pipeline_metadata.get("branch_id"),
            document_json=json_prop(document),
            pipeline_metadata_json=json_prop(pipeline_metadata),
        )
    else:
        tx.run(
            """
            MERGE (s:Source {source_id: $source_id})
            SET
              s.source_key = $source_id,
              s.title = $title,
              s.publisher = $publisher,
              s.published_at = $published_at,
              s.document_id = $document_id,
              s.source_kind = $source_kind,
              s.canonical_source = $canonical_source,
              s.latest_run_id = $run_id,
              s.latest_corpus_id = $corpus_id,
              s.latest_branch_id = $branch_id,
              s.document_json = $document_json,
              s.pipeline_metadata_json = $pipeline_metadata_json
            """,
            source_id=source_id,
            title=document.get("title"),
            publisher=document.get("publisher"),
            published_at=document.get("published_at"),
            document_id=pipeline_metadata.get("document_id"),
            source_kind=pipeline_metadata.get("source_kind"),
            canonical_source=pipeline_metadata.get("canonical_source"),
            run_id=pipeline_metadata.get("run_id"),
            corpus_id=pipeline_metadata.get("corpus_id"),
            branch_id=pipeline_metadata.get("branch_id"),
            document_json=json_prop(document),
            pipeline_metadata_json=json_prop(pipeline_metadata),
        )

    tx.run(
        """
        MATCH (s:Source)
        WHERE s.source_id = $source_id OR ($url IS NOT NULL AND s.url = $url)
        WITH s LIMIT 1
        MATCH (co:Corpus {corpus_id: $corpus_id})
        MATCH (br:Branch {key: $branch_key})
        MATCH (run:ImportRun {run_id: $run_id})
        MERGE (s)-[:IN_CORPUS]->(co)
        MERGE (s)-[:IN_BRANCH]->(br)
        MERGE (run)-[:IMPORTED_SOURCE]->(s)
        """,
        source_id=source_id,
        url=url,
        corpus_id=pipeline_metadata["corpus_id"],
        branch_key=pipeline_metadata["branch_key"],
        run_id=pipeline_metadata["run_id"],
    )


def merge_evidence_store(
    tx,
    *,
    client: OpenAI,
    embed_model: str,
    evidence_store: JSONDict,
    fallback_source_url: Optional[str],
    pipeline_metadata: JSONDict,
    ledger: Optional[CostLedger] = None,
) -> None:
    for evidence_id, evidence in evidence_store.items():
        if not isinstance(evidence, dict):
            continue

        actual_id = evidence.get("evidence_id") or evidence_id
        source = evidence.get("source_url") or fallback_source_url
        embedding_text = evidence_embedding_text(evidence)
        embedding = embed_text(
            client,
            embed_model,
            embedding_text,
            ledger=ledger,
            operation="evidence_embedding",
        )

        tx.run(
            """
            MERGE (ev:Evidence {evidence_id: $evidence_id})
            SET
              ev.text = $text,
              ev.block_type = $block_type,
              ev.document_id = $document_id,
              ev.source_url = $source_url,
              ev.source_id = $source_id,
              ev.latest_run_id = $run_id,
              ev.latest_corpus_id = $corpus_id,
              ev.latest_branch_id = $branch_id,
              ev.run_ids = CASE
                WHEN $run_id IN coalesce(ev.run_ids, []) THEN coalesce(ev.run_ids, [])
                ELSE coalesce(ev.run_ids, []) + [$run_id]
              END,
              ev.corpus_ids = CASE
                WHEN $corpus_id IN coalesce(ev.corpus_ids, []) THEN coalesce(ev.corpus_ids, [])
                ELSE coalesce(ev.corpus_ids, []) + [$corpus_id]
              END,
              ev.branch_ids = CASE
                WHEN $branch_id IN coalesce(ev.branch_ids, []) THEN coalesce(ev.branch_ids, [])
                ELSE coalesce(ev.branch_ids, []) + [$branch_id]
              END,
              ev.source_ids = CASE
                WHEN $source_id IN coalesce(ev.source_ids, []) THEN coalesce(ev.source_ids, [])
                ELSE coalesce(ev.source_ids, []) + [$source_id]
              END,
              ev.start_char = $start_char,
              ev.end_char = $end_char,
              ev.metadata_json = $metadata_json,
              ev.pipeline_metadata_json = $pipeline_metadata_json,
              ev.embedding_text = $embedding_text,
              ev.embedding = $embedding

            WITH ev
            OPTIONAL MATCH (s:Source)
            WHERE s.source_id = $source_id OR ($source_url IS NOT NULL AND s.url = $source_url)
            FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
              MERGE (ev)-[:FROM_SOURCE]->(s)
              MERGE (s)-[:HAS_EVIDENCE]->(ev)
            )

            WITH ev
            OPTIONAL MATCH (br:Branch {key: $branch_key})
            FOREACH (_ IN CASE WHEN br IS NULL THEN [] ELSE [1] END |
              MERGE (ev)-[:IN_BRANCH]->(br)
            )

            WITH ev
            OPTIONAL MATCH (run:ImportRun {run_id: $run_id})
            FOREACH (_ IN CASE WHEN run IS NULL THEN [] ELSE [1] END |
              MERGE (run)-[:IMPORTED_EVIDENCE]->(ev)
            )
            """,
            evidence_id=actual_id,
            text=evidence.get("text"),
            block_type=evidence.get("block_type"),
            document_id=evidence.get("document_id") or pipeline_metadata.get("document_id"),
            source_url=source,
            source_id=pipeline_metadata["source_id"],
            run_id=pipeline_metadata["run_id"],
            corpus_id=pipeline_metadata["corpus_id"],
            branch_id=pipeline_metadata["branch_id"],
            branch_key=pipeline_metadata["branch_key"],
            start_char=evidence.get("start_char"),
            end_char=evidence.get("end_char"),
            metadata_json=json_prop(evidence.get("metadata") or {}),
            pipeline_metadata_json=json_prop(pipeline_metadata),
            embedding_text=embedding_text,
            embedding=embedding,
        )


def write_entity_merge_decision(
    tx,
    *,
    decision_key: str,
    incoming_key: str,
    resolved_key: str,
    entity: JSONDict,
    candidates: Sequence[JSONDict],
    decision: JSONDict,
    record_source_url: Optional[str],
    embedding_text: str,
    pipeline_metadata: JSONDict,
) -> None:
    tx.run(
        """
        MERGE (d:EntityMergeDecision {key: $key})
        SET
          d.incoming_key = $incoming_key,
          d.resolved_key = $resolved_key,
          d.incoming_entity_json = $incoming_entity_json,
          d.candidates_json = $candidates_json,
          d.decision_json = $decision_json,
          d.decision = $decision,
          d.merge_key = $merge_key,
          d.confidence = $confidence,
          d.reason = $reason,
          d.source_url = $source_url,
          d.embedding_text = $embedding_text,
          d.run_id = $run_id,
          d.corpus_id = $corpus_id,
          d.branch_id = $branch_id,
          d.source_id = $source_id,
          d.document_id = $document_id,
          d.pipeline_metadata_json = $pipeline_metadata_json

        WITH d
        MATCH (e:Entity {key: $resolved_key})
        MERGE (d)-[:RESOLVED_TO]->(e)
        WITH d
        OPTIONAL MATCH (run:ImportRun {run_id: $run_id})
        FOREACH (_ IN CASE WHEN run IS NULL THEN [] ELSE [1] END |
          MERGE (run)-[:MADE_ENTITY_DECISION]->(d)
        )
        """,
        key=decision_key,
        incoming_key=incoming_key,
        resolved_key=resolved_key,
        incoming_entity_json=json_prop(entity),
        candidates_json=json_prop(list(candidates)),
        decision_json=json_prop(decision),
        decision=decision.get("decision"),
        merge_key=decision.get("merge_key"),
        confidence=decision.get("confidence"),
        reason=decision.get("reason"),
        source_url=record_source_url,
        embedding_text=embedding_text,
        run_id=pipeline_metadata["run_id"],
        corpus_id=pipeline_metadata["corpus_id"],
        branch_id=pipeline_metadata["branch_id"],
        source_id=pipeline_metadata["source_id"],
        document_id=pipeline_metadata.get("document_id"),
        pipeline_metadata_json=json_prop(pipeline_metadata),
    )


def resolve_entity_key(
    tx,
    *,
    client: OpenAI,
    model: str,
    embed_model: str,
    entity: JSONDict,
    record_source_url: Optional[str],
    neighbor_k: int,
    ledger: Optional[CostLedger] = None,
) -> JSONDict:
    incoming_key = deterministic_entity_key(entity)
    embedding_text = entity_embedding_text(entity)
    embedding = embed_text(
        client,
        embed_model,
        embedding_text,
        ledger=ledger,
        operation="entity_embedding",
    )

    candidates = nearest_entity_candidates(
        tx,
        embedding,
        k=neighbor_k,
    )

    if not candidates:
        decision = {
            "decision": "create_new",
            "merge_key": None,
            "matched_candidate_rank": None,
            "confidence": "high",
            "reason": "No existing Entity candidates returned from vector index.",
        }
        return {
            "incoming_key": incoming_key,
            "resolved_key": incoming_key,
            "embedding_text": embedding_text,
            "embedding": embedding,
            "candidates": candidates,
            "decision": decision,
        }

    try:
        decision = call_llm_json(
            client,
            model=model,
            prompt=entity_merge_prompt(
                incoming_entity=entity,
                incoming_embedding_text=embedding_text,
                candidates=candidates,
            ),
            ledger=ledger,
            operation="entity_merge_decision",
        )
    except Exception as error:
        decision = {
            "decision": "create_new",
            "merge_key": None,
            "matched_candidate_rank": None,
            "confidence": "low",
            "reason": f"LLM merge adjudication failed; creating new entity: {error}",
        }

    if not isinstance(decision, dict):
        decision = {}

    raw_decision = decision.get("decision")
    merge_key = decision.get("merge_key")

    candidate_keys = {candidate.get("key") for candidate in candidates}

    if raw_decision == "merge_with_existing" and merge_key in candidate_keys:
        resolved_key = str(merge_key)
    else:
        decision = {
            **decision,
            "decision": "create_new",
            "merge_key": None,
            "matched_candidate_rank": None,
            "reason": decision.get("reason") or "No valid merge candidate selected.",
        }
        resolved_key = incoming_key

    return {
        "incoming_key": incoming_key,
        "resolved_key": resolved_key,
        "embedding_text": embedding_text,
        "embedding": embedding,
        "candidates": candidates,
        "decision": decision,
    }


def merge_entity(
    tx,
    *,
    client: OpenAI,
    model: str,
    embed_model: str,
    neighbor_k: int,
    entity: JSONDict,
    record_source_url: Optional[str],
    pipeline_metadata: JSONDict,
    ledger: Optional[CostLedger] = None,
) -> str:
    resolution = resolve_entity_key(
        tx,
        client=client,
        model=model,
        embed_model=embed_model,
        entity=entity,
        record_source_url=record_source_url,
        neighbor_k=neighbor_k,
        ledger=ledger,
    )

    key = resolution["resolved_key"]
    name = entity_name(entity)
    typ = entity_type(entity)
    incoming_key = resolution["incoming_key"]

    is_new = key == incoming_key

    tx.run(
        """
        MERGE (e:Entity {key: $key})
        ON CREATE SET
          e.entity_id = $entity_id,
          e.canonical_name = $canonical_name,
          e.entity_type = $entity_type,
          e.created_from_source_url = $source_url,
          e.embedding_text = $embedding_text,
          e.embedding = $embedding
        SET
          e.canonical_name = coalesce(e.canonical_name, $canonical_name),
          e.entity_type = coalesce(e.entity_type, $entity_type),
          e.description = coalesce(e.description, $description),
          e.aliases_json = $aliases_json,
          e.attributes_json = $attributes_json,
          e.temporal_json = $temporal_json,
          e.latest_postrag_decision = $postrag_decision,
          e.latest_grounding_score = $grounding_score,
          e.last_seen_source_url = $source_url,
          e.latest_run_id = $run_id,
          e.latest_corpus_id = $corpus_id,
          e.latest_branch_id = $branch_id,
          e.latest_source_id = $source_id,
          e.run_ids = CASE
            WHEN $run_id IN coalesce(e.run_ids, []) THEN coalesce(e.run_ids, [])
            ELSE coalesce(e.run_ids, []) + [$run_id]
          END,
          e.corpus_ids = CASE
            WHEN $corpus_id IN coalesce(e.corpus_ids, []) THEN coalesce(e.corpus_ids, [])
            ELSE coalesce(e.corpus_ids, []) + [$corpus_id]
          END,
          e.branch_ids = CASE
            WHEN $branch_id IN coalesce(e.branch_ids, []) THEN coalesce(e.branch_ids, [])
            ELSE coalesce(e.branch_ids, []) + [$branch_id]
          END,
          e.source_ids = CASE
            WHEN $source_id IN coalesce(e.source_ids, []) THEN coalesce(e.source_ids, [])
            ELSE coalesce(e.source_ids, []) + [$source_id]
          END,
          e.pipeline_metadata_json = $pipeline_metadata_json

        WITH e
        OPTIONAL MATCH (run:ImportRun {run_id: $run_id})
        FOREACH (_ IN CASE WHEN run IS NULL THEN [] ELSE [1] END |
          MERGE (run)-[:IMPORTED_ENTITY]->(e)
        )
        """,
        key=key,
        entity_id=entity.get("entity_id"),
        canonical_name=name,
        entity_type=typ,
        description=entity.get("description"),
        aliases_json=json_prop(entity.get("aliases") or []),
        attributes_json=json_prop(entity.get("attributes") or {}),
        temporal_json=json_prop(entity.get("temporal") or {}),
        postrag_decision=postrag_decision(entity),
        grounding_score=grounding_score(entity),
        source_url=record_source_url,
        embedding_text=resolution["embedding_text"],
        embedding=resolution["embedding"],
        run_id=pipeline_metadata["run_id"],
        corpus_id=pipeline_metadata["corpus_id"],
        branch_id=pipeline_metadata["branch_id"],
        source_id=pipeline_metadata["source_id"],
        pipeline_metadata_json=json_prop(pipeline_metadata),
    )

    if not is_new:
        tx.run(
            """
            MATCH (e:Entity {key: $key})
            SET
              e.merged_entity_count = coalesce(e.merged_entity_count, 0) + 1
            """,
            key=key,
            run_id=pipeline_metadata["run_id"],
            corpus_id=pipeline_metadata["corpus_id"],
            branch_id=pipeline_metadata["branch_id"],
            source_id=pipeline_metadata["source_id"],
            pipeline_metadata_json=json_prop(pipeline_metadata),
        )

    decision_key = merge_decision_key(
        incoming_key=incoming_key,
        record_source_url=record_source_url,
        entity=entity,
    )

    write_entity_merge_decision(
        tx,
        decision_key=decision_key,
        incoming_key=incoming_key,
        resolved_key=key,
        entity=entity,
        candidates=resolution["candidates"],
        decision=resolution["decision"],
        record_source_url=record_source_url,
        embedding_text=resolution["embedding_text"],
        pipeline_metadata=pipeline_metadata,
    )

    return key


def connect_entity_evidence(
    tx,
    entity_key_value: str,
    evidence_id_values: Sequence[str],
    *,
    pipeline_metadata: JSONDict,
) -> None:
    for evidence_id in evidence_id_values:
        tx.run(
            """
            MATCH (e:Entity {key: $entity_key})
            MATCH (ev:Evidence {evidence_id: $evidence_id})
            MERGE (e)-[r:SUPPORTED_BY]->(ev)
            SET
              r.latest_run_id = $run_id,
              r.latest_corpus_id = $corpus_id,
              r.latest_branch_id = $branch_id,
              r.latest_source_id = $source_id,
              r.run_ids = CASE
                WHEN $run_id IN coalesce(r.run_ids, []) THEN coalesce(r.run_ids, [])
                ELSE coalesce(r.run_ids, []) + [$run_id]
              END,
              r.branch_ids = CASE
                WHEN $branch_id IN coalesce(r.branch_ids, []) THEN coalesce(r.branch_ids, [])
                ELSE coalesce(r.branch_ids, []) + [$branch_id]
              END
            """,
            entity_key=entity_key_value,
            evidence_id=evidence_id,
            run_id=pipeline_metadata["run_id"],
            corpus_id=pipeline_metadata["corpus_id"],
            branch_id=pipeline_metadata["branch_id"],
            source_id=pipeline_metadata["source_id"],
        )

def claim_key(subject_key: str, relation: JSONDict, object_key: str) -> str:
    payload = {
        "subject_key": subject_key,
        "relation_type": relation_type(relation),
        "object_key": object_key,
        "temporal": relation_temporal(relation),
    }
    return json_prop(payload)


def merge_claim(
    tx,
    *,
    client: OpenAI,
    embed_model: str,
    relation: JSONDict,
    subject_key: str,
    object_key: str,
    subject_name: str,
    object_name: str,
    record_source_url: Optional[str],
    pipeline_metadata: JSONDict,
    ledger: Optional[CostLedger] = None,
) -> str:
    key = claim_key(subject_key, relation, object_key)
    embedding_text = claim_embedding_text(
        relation,
        subject_name=subject_name,
        object_name=object_name,
    )
    embedding = embed_text(
        client,
        embed_model,
        embedding_text,
        ledger=ledger,
        operation="claim_embedding",
    )

    tx.run(
        """
        MERGE (c:Claim {key: $key})
        ON CREATE SET
          c.claim_id = $claim_id,
          c.created_from_source_url = $source_url
        SET
          c.relation_type = $relation_type,
          c.description = $description,
          c.temporal_json = $temporal_json,
          c.attributes_json = $attributes_json,
          c.postrag_decision = $postrag_decision,
          c.grounding_score = $grounding_score,
          c.epistemic_status_json = $epistemic_status_json,
          c.corrected_candidate_json = $corrected_candidate_json,
          c.embedding_text = $embedding_text,
          c.embedding = $embedding,
          c.source_id = $source_id,
          c.latest_run_id = $run_id,
          c.latest_corpus_id = $corpus_id,
          c.latest_branch_id = $branch_id,
          c.run_ids = CASE
            WHEN $run_id IN coalesce(c.run_ids, []) THEN coalesce(c.run_ids, [])
            ELSE coalesce(c.run_ids, []) + [$run_id]
          END,
          c.corpus_ids = CASE
            WHEN $corpus_id IN coalesce(c.corpus_ids, []) THEN coalesce(c.corpus_ids, [])
            ELSE coalesce(c.corpus_ids, []) + [$corpus_id]
          END,
          c.branch_ids = CASE
            WHEN $branch_id IN coalesce(c.branch_ids, []) THEN coalesce(c.branch_ids, [])
            ELSE coalesce(c.branch_ids, []) + [$branch_id]
          END,
          c.source_ids = CASE
            WHEN $source_id IN coalesce(c.source_ids, []) THEN coalesce(c.source_ids, [])
            ELSE coalesce(c.source_ids, []) + [$source_id]
          END,
          c.pipeline_metadata_json = $pipeline_metadata_json

        WITH c
        MATCH (s:Entity {key: $subject_key})
        MATCH (o:Entity {key: $object_key})
        MERGE (s)-[:SUBJECT_OF]->(c)
        MERGE (c)-[:OBJECT_OF]->(o)
        MERGE (s)-[r:KG_REL {claim_key: $key}]->(o)
        SET
          r.relation_type = $relation_type,
          r.grounding_score = $grounding_score,
          r.postrag_decision = $postrag_decision,
          r.description = $description,
          r.temporal_json = $temporal_json,
          r.latest_run_id = $run_id,
          r.latest_corpus_id = $corpus_id,
          r.latest_branch_id = $branch_id,
          r.latest_source_id = $source_id,
          r.run_ids = CASE
            WHEN $run_id IN coalesce(r.run_ids, []) THEN coalesce(r.run_ids, [])
            ELSE coalesce(r.run_ids, []) + [$run_id]
          END,
          r.branch_ids = CASE
            WHEN $branch_id IN coalesce(r.branch_ids, []) THEN coalesce(r.branch_ids, [])
            ELSE coalesce(r.branch_ids, []) + [$branch_id]
          END

        WITH c
        OPTIONAL MATCH (run:ImportRun {run_id: $run_id})
        FOREACH (_ IN CASE WHEN run IS NULL THEN [] ELSE [1] END |
          MERGE (run)-[:IMPORTED_CLAIM]->(c)
        )
        """,
        key=key,
        claim_id=relation.get("relation_id"),
        relation_type=relation_type(relation),
        description=relation.get("description"),
        temporal_json=json_prop(relation_temporal(relation)),
        attributes_json=json_prop(relation.get("attributes") or {}),
        postrag_decision=postrag_decision(relation),
        grounding_score=grounding_score(relation),
        epistemic_status_json=json_prop((relation.get("postrag") or {}).get("epistemic_status") or {}),
        corrected_candidate_json=json_prop((relation.get("postrag") or {}).get("corrected_candidate")),
        subject_key=subject_key,
        object_key=object_key,
        source_url=record_source_url,
        embedding_text=embedding_text,
        embedding=embedding,
        run_id=pipeline_metadata["run_id"],
        corpus_id=pipeline_metadata["corpus_id"],
        branch_id=pipeline_metadata["branch_id"],
        source_id=pipeline_metadata["source_id"],
        pipeline_metadata_json=json_prop(pipeline_metadata),
    )

    return key


def connect_claim_evidence(
    tx,
    claim_key_value: str,
    evidence_id_values: Sequence[str],
    *,
    pipeline_metadata: JSONDict,
) -> None:
    for evidence_id in evidence_id_values:
        tx.run(
            """
            MATCH (c:Claim {key: $claim_key})
            MATCH (ev:Evidence {evidence_id: $evidence_id})
            MERGE (c)-[r:SUPPORTED_BY]->(ev)
            SET
              r.latest_run_id = $run_id,
              r.latest_corpus_id = $corpus_id,
              r.latest_branch_id = $branch_id,
              r.latest_source_id = $source_id,
              r.run_ids = CASE
                WHEN $run_id IN coalesce(r.run_ids, []) THEN coalesce(r.run_ids, [])
                ELSE coalesce(r.run_ids, []) + [$run_id]
              END,
              r.branch_ids = CASE
                WHEN $branch_id IN coalesce(r.branch_ids, []) THEN coalesce(r.branch_ids, [])
                ELSE coalesce(r.branch_ids, []) + [$branch_id]
              END
            """,
            claim_key=claim_key_value,
            evidence_id=evidence_id,
            run_id=pipeline_metadata["run_id"],
            corpus_id=pipeline_metadata["corpus_id"],
            branch_id=pipeline_metadata["branch_id"],
            source_id=pipeline_metadata["source_id"],
        )


# ============================================================
# Import pipeline
# ============================================================

def import_postrag_record(
    tx,
    *,
    client: OpenAI,
    model: str,
    embed_model: str,
    neighbor_k: int,
    record: JSONDict,
    pipeline_metadata: Optional[JSONDict] = None,
    ledger: Optional[CostLedger] = None,
) -> JSONDict:
    metadata = pipeline_metadata_from_record(record, pipeline_metadata)

    url = source_url(record) or metadata.get("source_url") or metadata.get("canonical_source")
    document = record.get("document") if isinstance(record.get("document"), dict) else {}

    merge_source(tx, url, document, metadata)

    evidence_store = record.get("evidence_store")
    if not isinstance(evidence_store, dict):
        evidence_store = {}

    merge_evidence_store(
        tx,
        client=client,
        embed_model=embed_model,
        evidence_store=evidence_store,
        fallback_source_url=url,
        pipeline_metadata=metadata,
        ledger=ledger,
    )

    entities = filtered_entities(record)
    relations = filtered_relations(record)

    local_entity_to_key: Dict[str, str] = {}
    local_entity_to_name: Dict[str, str] = {}

    for entity in entities:
        key = merge_entity(
            tx,
            client=client,
            model=model,
            embed_model=embed_model,
            neighbor_k=neighbor_k,
            entity=entity,
            record_source_url=url,
            pipeline_metadata=metadata,
            ledger=ledger,
        )

        name = entity_name(entity)

        local_id = local_entity_id(entity)
        if local_id:
            local_entity_to_key[local_id] = key
            local_entity_to_name[local_id] = name

        local_entity_to_key[name] = key
        local_entity_to_key[normalize_text(name)] = key
        local_entity_to_name[name] = name
        local_entity_to_name[normalize_text(name)] = name

        connect_entity_evidence(
            tx,
            key,
            evidence_ids(entity),
            pipeline_metadata=metadata,
        )

    imported_claims = 0
    skipped_relations: List[JSONDict] = []

    for relation in relations:
        subject_local = relation_subject_id(relation)
        object_local = relation_object_id(relation)

        subject_key = local_entity_to_key.get(subject_local)
        object_key = local_entity_to_key.get(object_local)

        subject_name = local_entity_to_name.get(subject_local) or relation.get("subject") or subject_local
        object_name = local_entity_to_name.get(object_local) or relation.get("object") or object_local

        if not subject_key:
            subject_key = local_entity_to_key.get(relation.get("subject"))
            subject_key = subject_key or local_entity_to_key.get(normalize_text(relation.get("subject")))

        if not object_key:
            object_key = local_entity_to_key.get(relation.get("object"))
            object_key = object_key or local_entity_to_key.get(normalize_text(relation.get("object")))

        if not subject_key or not object_key:
            skipped_relations.append({
                "relation_id": relation.get("relation_id"),
                "subject_id": subject_local,
                "object_id": object_local,
                "subject": relation.get("subject"),
                "object": relation.get("object"),
            })
            continue

        c_key = merge_claim(
            tx,
            client=client,
            embed_model=embed_model,
            relation=relation,
            subject_key=subject_key,
            object_key=object_key,
            subject_name=str(subject_name),
            object_name=str(object_name),
            record_source_url=url,
            pipeline_metadata=metadata,
            ledger=ledger,
        )

        connect_claim_evidence(
            tx,
            c_key,
            evidence_ids(relation),
            pipeline_metadata=metadata,
        )
        imported_claims += 1

    return {
        "source_url": url,
        "pipeline_metadata": metadata,
        "entities_in": len(entities),
        "relations_in": len(relations),
        "evidence_blocks_in": len(evidence_store),
        "claims_imported": imported_claims,
        "relations_skipped": skipped_relations,
    }


def count_graph(tx) -> JSONDict:
    result = tx.run(
        """
        RETURN
          count { MATCH (:Entity) } AS entities,
          count { MATCH (:Claim) } AS claims,
          count { MATCH (:Evidence) } AS evidence,
          count { MATCH (:Source) } AS sources,
          count { MATCH (:Corpus) } AS corpora,
          count { MATCH (:Branch) } AS branches,
          count { MATCH (:ImportRun) } AS import_runs,
          count { MATCH (:EntityMergeDecision) } AS entity_merge_decisions,
          count { MATCH (e:Entity) WHERE e.embedding IS NOT NULL } AS embedded_entities,
          count { MATCH (c:Claim) WHERE c.embedding IS NOT NULL } AS embedded_claims,
          count { MATCH (ev:Evidence) WHERE ev.embedding IS NOT NULL } AS embedded_evidence
        """
    ).single()

    return dict(result)


def main() -> int:
    args = parse_args()
    record = read_json(args.input)
    client = OpenAI()

    pricing_config = load_pricing_config(args.pricing_file) if args.pricing_file else {}

    ledger = CostLedger(
        run_id=args.run_id or "neo4j_import_" + stable_hash(str(args.input)),
        source_id=args.source_id,
        ledger_path=args.cost_ledger,
        cache_dir=args.cache_dir,
        pricing_config=pricing_config,
        enabled=bool(args.cost_ledger),
        cache_enabled=bool(args.cache_dir) and not args.disable_cache,
    )

    cli_pipeline_metadata = {
        "run_id": args.run_id,
        "source_id": args.source_id,
        "corpus_id": args.corpus_id,
        "branch_id": args.branch_id,
        "canonical_source": args.canonical_source,
    }

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    with driver.session(database=args.database) as session:
        session.execute_write(create_constraints)

        if args.clear:
            session.execute_write(clear_database)
            session.execute_write(create_constraints)

        session.execute_write(create_vector_indexes, args.embedding_dim)

        tx = session.begin_transaction()
        try:
            summary = import_postrag_record(
                tx,
                client=client,
                model=args.model,
                embed_model=args.embed_model,
                neighbor_k=args.entity_neighbor_k,
                record=record,
                pipeline_metadata=cli_pipeline_metadata,
                ledger=ledger,
            )
            tx.commit()
        except Exception:
            tx.rollback()
            raise

        counts = session.execute_read(count_graph)
        cost_summary = summarize_ledger(args.cost_ledger) if args.cost_ledger and args.cost_ledger.exists() else None

    driver.close()

    print("[neo4j] import summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("[neo4j] graph counts:")
    print(json.dumps(counts, indent=2, ensure_ascii=False))

    if cost_summary:
        print("[neo4j] cost ledger summary:")
        print(json.dumps(cost_summary, indent=2, ensure_ascii=False))

    if args.summary_output:
        write_json(
            args.summary_output,
            {
                "import_summary": summary,
                "graph_counts": counts,
                "cost_summary": cost_summary,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())