#!/usr/bin/env python3
"""
Simple RAG baseline for Lanthic evaluation.

Purpose:
  Retrieval-only baseline over the final ingested corpus.

What this does:
  question -> embedding -> Neo4j vector retrieval over evidence nodes -> LLM memo

What this deliberately does NOT do:
  - no SARG
  - no LangGraph
  - no KG path traversal
  - no graph expansion
  - no risk tool
  - no missing-evidence tool
  - no agent planning
  - no domain-specific query rewriting

Expected eval.py command:
  python src/simple_rag.py \
    --question "..." \
    --source-hint "..." \
    --corpus-id eval1 \
    --branch-id staging_eval1 \
    --output eval/runs/H1_simple_rag.simple_rag.raw.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def compact(text: str) -> str:
    return " ".join(str(text).split())


def truncate(text: str, max_chars: int) -> str:
    text = compact(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    return OpenAI()


def get_neo4j_driver(args: argparse.Namespace):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install neo4j") from exc

    uri = args.neo4j_uri or os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    user = args.neo4j_user or os.environ.get("NEO4J_USER") or "neo4j"
    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError("Missing Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")

    return GraphDatabase.driver(uri, auth=(user, password))


def embed_text(text: str, model: str) -> Tuple[List[float], Dict[str, Any]]:
    client = get_openai_client()

    response = client.embeddings.create(
        model=model,
        input=text,
    )

    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return response.data[0].embedding, usage


def list_vector_indexes(driver) -> List[Dict[str, Any]]:
    query = """
    SHOW VECTOR INDEXES
    YIELD name, type, entityType, labelsOrTypes, properties, state
    RETURN name, type, entityType, labelsOrTypes, properties, state
    ORDER BY name
    """

    with driver.session() as session:
        return [dict(record) for record in session.run(query)]


def choose_vector_index(driver, requested_index: Optional[str]) -> str:
    if requested_index:
        return requested_index

    indexes = list_vector_indexes(driver)
    online = [idx for idx in indexes if str(idx.get("state", "")).upper() == "ONLINE"]

    if len(online) == 1:
        return str(online[0]["name"])

    details = json.dumps(indexes, indent=2, ensure_ascii=False)

    raise RuntimeError(
        "Could not choose a Neo4j vector index automatically. "
        "Pass --vector-index explicitly.\n\nAvailable vector indexes:\n"
        f"{details}"
    )


def first_existing_string(props: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_evidence_text(props: Dict[str, Any], text_property: str) -> str:
    value = props.get(text_property)
    if isinstance(value, str) and value.strip():
        return value.strip()

    raise RuntimeError(
        f"Retrieved node does not contain configured text property '{text_property}'. "
        f"Available keys: {sorted(props.keys())}"
    )


def source_text(source_props: Dict[str, Any]) -> str:
    values = []
    for value in source_props.values():
        if isinstance(value, str):
            values.append(value)
    return " ".join(values).lower()


def source_matches_hint(source_props: Dict[str, Any], source_hint: str) -> bool:
    if not source_hint.strip():
        return True

    haystack = source_text(source_props)
    needle = source_hint.lower().strip()

    if needle in haystack:
        return True

    hint_parts = [part for part in needle.replace("'", " ").replace("-", " ").split() if part]
    if not hint_parts:
        return True

    matched = sum(1 for part in hint_parts if part in haystack)
    return matched >= max(1, min(3, len(hint_parts)))


def retrieve_evidence(
    *,
    driver,
    embedding: List[float],
    vector_index: str,
    top_k: int,
    fetch_k: int,
    text_property: str,
    title_property: str,
    url_property: str,
    corpus_id: str,
    branch_id: str,
    source_hint: str,
    apply_source_hint_filter: bool,
) -> List[Dict[str, Any]]:
    query = """
    CALL db.index.vector.queryNodes($vector_index, $fetch_k, $embedding)
    YIELD node, score

    OPTIONAL MATCH (node)-[*1..2]-(source_node)
    WHERE 'Source' IN labels(source_node)

    RETURN
      elementId(node) AS node_element_id,
      labels(node) AS node_labels,
      properties(node) AS node_props,
      score,
      collect(DISTINCT properties(source_node))[0..5] AS source_props
    ORDER BY score DESC
    LIMIT $fetch_k
    """

    with driver.session() as session:
        rows = [
            dict(record)
            for record in session.run(
                query,
                vector_index=vector_index,
                fetch_k=fetch_k,
                embedding=embedding,
            )
        ]

    evidence: List[Dict[str, Any]] = []

    for row in rows:
        node_props = row.get("node_props") or {}
        all_source_props = row.get("source_props") or []

        if corpus_id:
            node_corpus = node_props.get("corpus_id") or node_props.get("corpusId")
            if node_corpus and node_corpus != corpus_id:
                continue

        if branch_id:
            node_branch = node_props.get("branch_id") or node_props.get("branchId")
            if node_branch and node_branch != branch_id:
                continue

        if apply_source_hint_filter and source_hint:
            if all_source_props:
                if not any(source_matches_hint(sp, source_hint) for sp in all_source_props):
                    continue

        text = extract_evidence_text(node_props, text_property)

        source_props = all_source_props[0] if all_source_props else {}

        title = (
            first_existing_string(source_props, [title_property, "title", "source_title", "name"])
            or first_existing_string(node_props, ["source_title", "title"])
        )

        url = (
            first_existing_string(source_props, [url_property, "source_url", "canonical_url", "url"])
            or first_existing_string(node_props, ["source_url", "canonical_url", "url"])
        )

        publisher = (
            first_existing_string(source_props, ["publisher"])
            or first_existing_string(node_props, ["publisher"])
        )

        published_at = (
            first_existing_string(source_props, ["published_at", "publishedAt", "date"])
            or first_existing_string(node_props, ["published_at", "publishedAt", "date"])
        )

        evidence_id = (
            first_existing_string(node_props, ["evidence_id", "evidenceId", "id", "block_id", "blockId"])
            or row.get("node_element_id")
        )

        evidence.append(
            {
                "label": f"E{len(evidence) + 1}",
                "score": row.get("score"),
                "evidence_id": evidence_id,
                "text": truncate(text, 1800),
                "source_title": title,
                "source_url": url,
                "publisher": publisher,
                "published_at": published_at,
                "node_labels": row.get("node_labels") or [],
                "node_element_id": row.get("node_element_id"),
            }
        )

        if len(evidence) >= top_k:
            break

    return evidence


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    blocks = []

    for ev in evidence:
        blocks.append(
            f"[{ev['label']}]\n"
            f"evidence_id: {ev.get('evidence_id') or ''}\n"
            f"source_title: {ev.get('source_title') or ''}\n"
            f"publisher: {ev.get('publisher') or ''}\n"
            f"published_at: {ev.get('published_at') or ''}\n"
            f"source_url: {ev.get('source_url') or ''}\n"
            f"text: {ev.get('text') or ''}\n"
        )

    return "\n".join(blocks)


def generate_memo(
    *,
    question: str,
    task_type: str,
    source_hint: str,
    evidence: List[Dict[str, Any]],
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    client = get_openai_client()

    system_prompt = (
        "You are the simple RAG baseline in an evaluation. "
        "Use only the retrieved evidence snippets provided by the user. "
        "Cite factual claims using evidence labels such as [E1] or [E2]. "
        "Do not use knowledge graph traversal, path reasoning, agent planning, risk tools, "
        "or missing-evidence tools. "
        "If the retrieved evidence does not support an answer, say that the evidence is missing or weak."
    )

    user_prompt = (
        f"TASK_TYPE: {task_type}\n"
        f"SOURCE_HINT: {source_hint or 'none'}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE:\n{format_evidence(evidence)}\n\n"
        "Write a compact analyst memo with:\n"
        "1. Bottom line\n"
        "2. Evidence-supported points\n"
        "3. Missing or weak evidence\n"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content or ""

    usage: Dict[str, Any] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return answer, usage


def inspect(args: argparse.Namespace) -> None:
    driver = get_neo4j_driver(args)

    try:
        indexes = list_vector_indexes(driver)
        print(json.dumps({"vector_indexes": indexes}, indent=2, ensure_ascii=False))
    finally:
        driver.close()


def run(args: argparse.Namespace) -> None:
    started_at = now_utc()
    t0 = time.perf_counter()

    driver = get_neo4j_driver(args)

    try:
        vector_index = choose_vector_index(driver, args.vector_index)

        retrieval_query = args.question
        if args.source_hint:
            retrieval_query = f"{args.question}\n\nSource hint: {args.source_hint}"

        embedding, embedding_usage = embed_text(retrieval_query, args.embedding_model)

        evidence = retrieve_evidence(
            driver=driver,
            embedding=embedding,
            vector_index=vector_index,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            text_property=args.text_property,
            title_property=args.title_property,
            url_property=args.url_property,
            corpus_id=args.corpus_id,
            branch_id=args.branch_id,
            source_hint=args.source_hint,
            apply_source_hint_filter=not args.no_source_hint_filter,
        )

        answer, generation_usage = generate_memo(
            question=args.question,
            task_type=args.task_type,
            source_hint=args.source_hint,
            evidence=evidence,
            model=args.model,
        )

        ended_at = now_utc()
        latency_sec = round(time.perf_counter() - t0, 3)

        input_tokens = int(embedding_usage.get("input_tokens") or 0) + int(generation_usage.get("input_tokens") or 0)
        output_tokens = int(generation_usage.get("output_tokens") or 0)
        total_tokens = int(embedding_usage.get("total_tokens") or 0) + int(generation_usage.get("total_tokens") or 0)

        result = {
            "system": "simple_rag",
            "question": args.question,
            "task_type": args.task_type,
            "source_hint": args.source_hint,
            "answer": answer,
            "retrieved_evidence": evidence,
            "started_at": started_at,
            "ended_at": ended_at,
            "latency_sec": latency_sec,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "embedding_usage": embedding_usage,
                "generation_usage": generation_usage,
            },
            "config": {
                "model": args.model,
                "embedding_model": args.embedding_model,
                "vector_index": vector_index,
                "top_k": args.top_k,
                "fetch_k": args.fetch_k,
                "text_property": args.text_property,
                "title_property": args.title_property,
                "url_property": args.url_property,
                "corpus_id": args.corpus_id,
                "branch_id": args.branch_id,
                "source_hint_filter": not args.no_source_hint_filter,
            },
            "error": "",
        }

        if args.output:
            write_json(Path(args.output), result)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    finally:
        driver.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple RAG baseline over Lanthic Neo4j evidence store.")
    sub = parser.add_subparsers(dest="cmd")

    inspect_parser = sub.add_parser("inspect", help="List Neo4j vector indexes.")
    inspect_parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    inspect_parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    inspect_parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    inspect_parser.set_defaults(func=inspect)

    parser.add_argument("--question", required=False, default="")
    parser.add_argument("--task-type", default="")
    parser.add_argument("--source-hint", default="")
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--branch-id", default="")
    parser.add_argument("--output", default="")

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))

    parser.add_argument("--vector-index", default=os.environ.get("EVIDENCE_VECTOR_INDEX", ""))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fetch-k", type=int, default=40)

    parser.add_argument("--text-property", default=os.environ.get("EVIDENCE_TEXT_PROPERTY", "text"))
    parser.add_argument("--title-property", default=os.environ.get("SOURCE_TITLE_PROPERTY", "title"))
    parser.add_argument("--url-property", default=os.environ.get("SOURCE_URL_PROPERTY", "source_url"))

    parser.add_argument(
        "--no-source-hint-filter",
        action="store_true",
        help="Disable source-hint filtering for source-specific questions.",
    )

    parser.set_defaults(func=run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd is None and not args.question:
        parser.error("--question is required unless using `inspect`.")

    try:
        args.func(args)
    except Exception as exc:
        result = {
            "system": "simple_rag",
            "question": getattr(args, "question", ""),
            "task_type": getattr(args, "task_type", ""),
            "source_hint": getattr(args, "source_hint", ""),
            "answer": "",
            "retrieved_evidence": [],
            "started_at": "",
            "ended_at": now_utc(),
            "latency_sec": 0,
            "usage": {},
            "config": {},
            "error": repr(exc),
        }

        if getattr(args, "output", ""):
            write_json(Path(args.output), result)

        print(f"ERROR: {repr(exc)}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()