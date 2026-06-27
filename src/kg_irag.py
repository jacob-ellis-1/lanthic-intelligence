#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from kg_tools import CypherSafety, Neo4jKG, validate_readonly_cypher
from local_graph import LocalGraph

try:
    from cost_ledger import CostLedger, load_pricing_config
except Exception:
    CostLedger = None  # type: ignore
    load_pricing_config = None  # type: ignore


JSONDict = Dict[str, Any]


# ============================================================
# Config / state
# ============================================================

@dataclass
class KGIRAGBudget:
    max_iterations: int = 5
    max_rows_per_query: int = 50
    max_path_depth: int = 3

    kappa_max_nodes: int = 80
    kappa_max_edges: int = 120
    kappa_max_evidence: int = 35

    max_bad_queries: int = 1
    no_new_info_patience: int = 2

    evidence_lane_k: int = 12
    table_lane_k: int = 8

    @classmethod
    def large(cls) -> "KGIRAGBudget":
        return cls(
            max_iterations=8,
            max_rows_per_query=150,
            max_path_depth=4,
            kappa_max_nodes=300,
            kappa_max_edges=600,
            kappa_max_evidence=150,
            max_bad_queries=2,
            no_new_info_patience=2,
            evidence_lane_k=24,
            table_lane_k=16,
        )


@dataclass
class KGIRAGScope:
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)

    @property
    def branch_key(self) -> Optional[str]:
        if self.corpus_id and self.branch_id:
            return f"{self.corpus_id}::{self.branch_id}"
        return None

    def active(self) -> bool:
        return bool(self.corpus_id or self.branch_id or self.source_ids)

    def to_params(self) -> JSONDict:
        return {
            "corpus_id": self.corpus_id,
            "branch_id": self.branch_id,
            "branch_key": self.branch_key,
            "source_ids": list(self.source_ids),
        }

    def cypher_predicate(self, variable: str) -> str:
        clauses: List[str] = []

        if self.corpus_id:
            clauses.append(
                f"($corpus_id IN coalesce({variable}.corpus_ids, []) "
                f"OR {variable}.latest_corpus_id = $corpus_id "
                f"OR {variable}.corpus_id = $corpus_id)"
            )

        if self.branch_id:
            clauses.append(
                f"($branch_id IN coalesce({variable}.branch_ids, []) "
                f"OR {variable}.latest_branch_id = $branch_id "
                f"OR {variable}.branch_id = $branch_id)"
            )

        if self.source_ids:
            clauses.append(
                f"(any(source_id IN $source_ids WHERE source_id IN coalesce({variable}.source_ids, [])) "
                f"OR {variable}.source_id IN $source_ids "
                f"OR {variable}.latest_source_id IN $source_ids)"
            )

        if not clauses:
            return "true"

        return " AND ".join(clauses)

    def instruction(self) -> str:
        if not self.active():
            return "No corpus/branch/source scope is active."

        parts = [
            "Active retrieval scope is enabled. Every generated Cypher query must restrict retrieved Entity, Claim, Evidence, or Source nodes to this scope using properties, not unapproved relationship types.",
            f"corpus_id: {self.corpus_id}",
            f"branch_id: {self.branch_id}",
            f"source_ids: {self.source_ids}",
            "Use predicates of this form, adapting variable names as needed:",
            "- $corpus_id IN coalesce(e.corpus_ids, []) OR e.latest_corpus_id = $corpus_id OR e.corpus_id = $corpus_id",
            "- $branch_id IN coalesce(e.branch_ids, []) OR e.latest_branch_id = $branch_id OR e.branch_id = $branch_id",
            "- any(source_id IN $source_ids WHERE source_id IN coalesce(e.source_ids, [])) OR e.source_id IN $source_ids OR e.latest_source_id IN $source_ids",
            "Do not retrieve from the full graph when a scope is active.",
        ]
        return "\n".join(parts)

    def to_dict(self) -> JSONDict:
        return {
            "corpus_id": self.corpus_id,
            "branch_id": self.branch_id,
            "branch_key": self.branch_key,
            "source_ids": list(self.source_ids),
            "active": self.active(),
        }


@dataclass
class RetrievalBatch:
    iteration: int
    cypher: str
    validated_cypher: str
    row_count: int
    nodes_added: int
    edges_added: int
    evidence_added: int
    controller_reason: str = ""
    evidence_lane_rows: int = 0
    table_lane_rows: int = 0
    evidence_lane_added: int = 0
    table_lane_added: int = 0


class IRAGState(TypedDict):
    question: str
    anchors: List[str]
    reasoning_prompt: str
    pending_cypher: str

    graph: JSONDict
    batches: List[JSONDict]
    previous_queries: List[str]
    rejected_queries: List[JSONDict]
    acquisition_requests: List[JSONDict]

    iteration: int
    initial_rows: int
    previous_rows: int
    no_new_info_steps: int

    status: str
    stop_reason: str
    result: Optional[JSONDict]


# ============================================================
# Utilities
# ============================================================

def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def call_json(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    ledger: Optional[Any] = None,
    operation: str = "chat_json",
) -> JSONDict:
    messages = [
        {"role": "system", "content": "Return only valid JSON. No markdown."},
        {"role": "user", "content": prompt},
    ]

    if ledger is not None:
        return ledger.chat_json(
            client,
            stage="kg_irag",
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
        raise ValueError("Empty model response.")

    return json.loads(content)


def compact_text(value: Any, max_chars: int = 24000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip()


def cypher_string_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher_or_equals(variable: str, values: Sequence[str]) -> str:
    clean = [str(value) for value in values if str(value).strip()]

    if not clean:
        return "false"

    return " OR ".join(
        f"{variable} = {cypher_string_literal(value)}"
        for value in clean
    )


def dynamic_max_depth(iteration: int, previous_rows: int, initial_rows: int, cap: int) -> int:
    return max(1, cap)


def graph_from_json(data: JSONDict) -> LocalGraph:
    if hasattr(LocalGraph, "from_dict"):
        return LocalGraph.from_dict(data)

    graph = LocalGraph(focus_question=data.get("focus_question", ""))

    for node in (data.get("nodes") or {}).values():
        graph.add_node(
            key=node.get("key"),
            name=node.get("name"),
            entity_type=node.get("entity_type") or "unknown",
            labels=node.get("labels") or [],
            description=node.get("description"),
            properties=node.get("properties") or {},
            source=node.get("source") or "from_json",
        )

    for edge in (data.get("edges") or {}).values():
        graph.add_edge(
            subject=edge.get("subject"),
            relation_type=edge.get("relation_type"),
            obj=edge.get("object"),
            subject_key=edge.get("subject_key"),
            object_key=edge.get("object_key"),
            claim_key=edge.get("claim_key"),
            grounding_score=edge.get("grounding_score"),
            description=edge.get("description"),
            properties=edge.get("properties") or {},
            evidence_ids=edge.get("evidence_ids") or [],
            source=edge.get("source") or "from_json",
        )

    for item in (data.get("evidence") or {}).values():
        graph.add_evidence(
            evidence_id=item.get("evidence_id"),
            text=item.get("text"),
            source_url=item.get("source_url"),
            source_title=item.get("source_title"),
            claim_key=item.get("claim_key"),
            properties=item.get("properties") or {},
            source=item.get("source") or "from_json",
        )

    for diagnostic in data.get("diagnostics") or []:
        if isinstance(diagnostic, dict):
            graph.diagnostics.append(diagnostic)

    return graph


def compact_context(graph_json: JSONDict) -> JSONDict:
    graph = graph_from_json(graph_json)
    return graph.to_context(
        max_nodes=25,
        max_edges=30,
        max_evidence=8,
        max_evidence_chars=500,
    )


def source_ids_from_args(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


# ============================================================
# LLM modules
# ============================================================

class AnchoringModule:
    def __init__(self, client: OpenAI, model: str, ledger: Optional[Any] = None) -> None:
        self.client = client
        self.model = model
        self.ledger = ledger

    def initialize(
        self,
        question: str,
        schema_text: str,
        budget: KGIRAGBudget,
        *,
        scope_instruction: str,
    ) -> JSONDict:
        prompt = f"""
You are the anchoring module for KG-IRAG.

Initialize iterative local-subgraph retrieval from a Neo4j knowledge base.
Do not answer the question.

Return JSON:
{{
  "anchors": ["short anchor"],
  "reasoning_prompt": "persistent retrieval objective for later iterations",
  "cypher": "bounded read-only Cypher query"
}}

Schema:
{schema_text}

Scope requirements:
{scope_instruction}

Cypher requirements:
- Read-only only.
- Must RETURN paths, nodes, relationships, or claim/evidence rows.
- Prefer returning paths as `RETURN p LIMIT N` when retrieving graph structure.
- Use Entity.canonical_name, never Entity.name.
- Use KG_REL for entity-to-entity traversal.
- Use Claim/Evidence only when source support is needed.
- Include LIMIT <= {budget.max_rows_per_query}.
- Variable-length paths must be bounded to depth <= {budget.max_path_depth}.
- Use exact relationship syntax like [:KG_REL*1..{budget.max_path_depth}].
- Do not write [:KG_REL]*1..{budget.max_path_depth}; the * must be inside the square brackets.
- Do not invent relationship types. Use :KG_REL only, and filter r.relation_type as a property if needed.
- Use Entity.canonical_name, never Entity.name.
- Prefer returning p where p is a path if retrieving graph structure.

Question:
{question}
""".strip()

        try:
            raw = call_json(
                self.client,
                self.model,
                prompt,
                ledger=self.ledger,
                operation="kg_irag_anchor_initialize",
            )
        except Exception:
            raw = {}

        anchors = raw.get("anchors")
        if not isinstance(anchors, list):
            anchors = []

        anchors = [str(anchor).strip() for anchor in anchors if str(anchor).strip()]

        reasoning_prompt = str(raw.get("reasoning_prompt") or "").strip()
        if not reasoning_prompt:
            reasoning_prompt = "Retrieve a bounded local reasoning subgraph sufficient for downstream reasoning."

        cypher = str(raw.get("cypher") or "").strip()
        if not cypher:
            cypher = "MATCH p=(s:Entity)-[:KG_REL*1..2]-(o:Entity) RETURN p LIMIT 25"

        return {
            "anchors": anchors,
            "reasoning_prompt": reasoning_prompt,
            "cypher": cypher,
        }


class IterativeController:
    def __init__(self, client: OpenAI, model: str, ledger: Optional[Any] = None) -> None:
        self.client = client
        self.model = model
        self.ledger = ledger

    def decide(
        self,
        *,
        state: IRAGState,
        schema_text: str,
        budget: KGIRAGBudget,
        allowed_depth: int,
        scope_instruction: str,
        enable_web_acquire: bool,
    ) -> JSONDict:
        context = compact_context(state["graph"])

        acquisition_instruction = ""
        if enable_web_acquire:
            acquisition_instruction = """
If the graph is insufficient because external evidence appears missing, include acquisition_requests.
Each acquisition request should be a concise web/source acquisition query, not an answer.
Use this shape:
"acquisition_requests": [
  {"query": "search/source query", "reason": "why this is needed", "priority": "high|medium|low"}
]
""".strip()

        prompt = f"""
You are the iterative KG-IRAG controller.

You inspect the current local reasoning subgraph and decide the next retrieval action.
Do not answer the user's question.

Return JSON:
{{
  "stop": true/false,
  "status": "sufficient|partial|insufficient",
  "reason": "brief reason",
  "action": "stop|expand_frontier|retrieve_evidence|text2cypher_fallback",
  "cypher": "",
  "acquisition_requests": []
}}

Action meanings:
- stop: current graph is sufficient for downstream reasoning.
- expand_frontier: expand around current Entity frontier using deterministic graph tools.
- retrieve_evidence: retrieve more source support for current Entity/Claim frontier.
- text2cypher_fallback: only use if deterministic actions are clearly insufficient. If used, provide Cypher in "cypher".

Persistent retrieval objective:
{state["reasoning_prompt"]}

Original question:
{state["question"]}

Schema:
{schema_text}

Scope requirements:
{scope_instruction}

{acquisition_instruction}

Current local subgraph context:
{json.dumps(context, indent=2, ensure_ascii=False)}

Previous Cypher queries:
{json.dumps(state["previous_queries"], indent=2, ensure_ascii=False)}

Budgets:
- current iteration: {state["iteration"]}
- max iterations: {budget.max_iterations}
- max rows per query: {budget.max_rows_per_query}
- allowed path depth this iteration: {allowed_depth}
- max total nodes κ: {budget.kappa_max_nodes}
- max total edges κ: {budget.kappa_max_edges}
- max total evidence κ: {budget.kappa_max_evidence}

Rules:
- Prefer expand_frontier when relation chains or bridge entities are missing.
- Prefer retrieve_evidence when entities/claims exist but source support is weak.
- Use text2cypher_fallback only for unusual graph patterns not covered by the deterministic actions.
- Do not write Cypher unless action is text2cypher_fallback.
- Do not generate an answer.
""".strip()

        try:
            raw = call_json(
                self.client,
                self.model,
                prompt,
                ledger=self.ledger,
                operation="kg_irag_controller_decide",
            )
        except Exception as error:
            return {
                "stop": True,
                "status": "insufficient",
                "reason": f"Controller failed: {error}",
                "action": "stop",
                "cypher": "",
                "acquisition_requests": [],
            }

        stop = bool(raw.get("stop"))
        status = str(raw.get("status") or ("sufficient" if stop else "insufficient"))
        reason = str(raw.get("reason") or "").strip()
        action = str(raw.get("action") or ("stop" if stop else "expand_frontier")).strip()

        allowed_actions = {"stop", "expand_frontier", "retrieve_evidence", "text2cypher_fallback"}
        if action not in allowed_actions:
            action = "expand_frontier" if not stop else "stop"

        cypher = str(raw.get("cypher") or "").strip()

        if action != "text2cypher_fallback":
            cypher = ""

        acquisition_requests = raw.get("acquisition_requests")
        if not isinstance(acquisition_requests, list):
            acquisition_requests = []

        return {
            "stop": stop or action == "stop",
            "status": status,
            "reason": reason,
            "action": action,
            "cypher": cypher,
            "acquisition_requests": acquisition_requests,
        }

    def repair_query(
        self,
        *,
        question: str,
        schema_text: str,
        bad_cypher: str,
        error: str,
        budget: KGIRAGBudget,
        allowed_depth: int,
        scope_instruction: str,
    ) -> str:
        prompt = f"""
Repair this Cypher query for read-only KG-IRAG retrieval.

Question:
{question}

Schema:
{schema_text}

Scope requirements:
{scope_instruction}

Rejected Cypher:
{bad_cypher}

Validator/execution error:
{error}

Return JSON:
{{"cypher": "corrected bounded read-only Cypher"}}

Requirements:
- Read-only only.
- Include RETURN.
- Include LIMIT <= {budget.max_rows_per_query}.
- Variable-length paths must be bounded to depth <= {budget.max_path_depth}.
- Use Entity.canonical_name, never Entity.name.
- Prefer returning p where p is a path if retrieving graph structure.
- If scope is active, include explicit property predicates for corpus_id, branch_id, or source_ids.
""".strip()

        try:
            raw = call_json(
                self.client,
                self.model,
                prompt,
                ledger=self.ledger,
                operation="kg_irag_repair_query",
            )
            return str(raw.get("cypher") or "").strip()
        except Exception:
            return ""


# ============================================================
# LangGraph KG-IRAG workflow
# ============================================================

class KGIRAG:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        embed_model: str = "text-embedding-3-small",
        anchor_k: int = 8,
        budget: Optional[KGIRAGBudget] = None,
        scope: Optional[KGIRAGScope] = None,
        large: bool = False,
        continue_after_anchor: bool = False,
        enable_evidence_lane: bool = True,
        enable_table_lane: bool = True,
        table_lane_each_iteration: bool = False,
        web_acquire: bool = False,
        acquisition_requests_output: Optional[Path] = None,
        web_acquire_command: Optional[str] = None,
        max_acquisition_requests: int = 3,
        ledger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.embed_model = embed_model
        self.anchor_k = anchor_k
        self.budget = budget or (KGIRAGBudget.large() if large else KGIRAGBudget())
        self.scope = scope or KGIRAGScope()
        self.large = large
        self.continue_after_anchor = continue_after_anchor or large
        self.enable_evidence_lane = enable_evidence_lane
        self.enable_table_lane = enable_table_lane
        self.table_lane_each_iteration = table_lane_each_iteration
        self.web_acquire = web_acquire
        self.acquisition_requests_output = acquisition_requests_output
        self.web_acquire_command = web_acquire_command
        self.max_acquisition_requests = max(0, max_acquisition_requests)
        self.ledger = ledger

        self.client = OpenAI()
        self.kg = Neo4jKG.from_env()
        self.kg.ensure_projection_edges()
        self.schema_text = self.kg.get_schema_text()

        self.anchoring = AnchoringModule(self.client, model, ledger=ledger)
        self.controller = IterativeController(self.client, model, ledger=ledger)
        self.app = self._build_graph()

    def close(self) -> None:
        self.kg.close()

    def retrieve(self, question: str) -> JSONDict:
        initial_graph = LocalGraph(focus_question=question)

        state: IRAGState = {
            "question": question,
            "anchors": [],
            "reasoning_prompt": "",
            "pending_cypher": "",

            "graph": initial_graph.to_dict(),
            "batches": [],
            "previous_queries": [],
            "rejected_queries": [],
            "acquisition_requests": [],

            "iteration": 0,
            "initial_rows": 0,
            "previous_rows": 0,
            "no_new_info_steps": 0,

            "status": "running",
            "stop_reason": "",
            "result": None,
        }

        result = self.app.invoke(
            state,
            config={"recursion_limit": self.budget.max_iterations * 5 + 25},
        )

        final = result.get("result")
        if not final:
            final = self._export(result)

        return final

    # ------------------------------------------------------------
    # Scope / execution
    # ------------------------------------------------------------

    def _scope_instruction(self) -> str:
        return self.scope.instruction()

    def _scope_params(self, extra: Optional[Mapping[str, Any]] = None) -> JSONDict:
        params = self.scope.to_params()
        if extra:
            params.update(dict(extra))
        return params

    def _execute_scoped_read(self, cypher: str, **params: Any) -> List[JSONDict]:
        return self.kg.execute_read(cypher, **self._scope_params(params))

    def _scope_reference_error(self, cypher: str) -> Optional[str]:
        if not self.scope.active():
            return None

        lowered = cypher.lower()
        tokens = [
            "branch_id",
            "branch_ids",
            "corpus_id",
            "corpus_ids",
            "source_id",
            "source_ids",
            "latest_branch_id",
            "latest_corpus_id",
            "latest_source_id",
            "in_branch",
        ]

        if any(token in lowered for token in tokens):
            return None

        return (
            "Active KG-IRAG scope is set, but the Cypher query does not contain "
            "branch/corpus/source scope predicates. Add property filters using "
            "$branch_id, $corpus_id, or $source_ids."
        )

    def _scoped_node_predicate_any(self, variables: Sequence[str]) -> str:
        predicates = [self.scope.cypher_predicate(variable) for variable in variables]
        predicates = [p for p in predicates if p and p != "true"]
        if not predicates:
            return "true"
        return " OR ".join(f"({p})" for p in predicates)

    # ------------------------------------------------------------
    # Semantic anchoring
    # ------------------------------------------------------------

    def _embed_query(self, question: str, *, operation: str = "kg_irag_query_embedding") -> List[float]:
        text = compact_text(question)

        if self.ledger is not None:
            return self.ledger.embed_text(
                self.client,
                stage="kg_irag",
                model=self.embed_model,
                text=text,
                operation=operation,
            )

        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def _semantic_anchor_candidates(self, question: str) -> JSONDict:
        embedding = self._embed_query(question, operation="kg_irag_query_embedding")
        entity_scope = self.scope.cypher_predicate("node")
        claim_scope = self.scope.cypher_predicate("node")
        evidence_scope = self.scope.cypher_predicate("node")

        entity_rows = self._execute_scoped_read(
            f"""
            CALL db.index.vector.queryNodes('entity_embedding_index', $k, $embedding)
            YIELD node, score
            WHERE {entity_scope}
            RETURN
              'entity' AS kind,
              node.key AS entity_key,
              node.canonical_name AS label,
              node.entity_type AS entity_type,
              node.description AS description,
              score
            ORDER BY score DESC
            """,
            k=self.anchor_k,
            embedding=embedding,
        )

        claim_rows = self._execute_scoped_read(
            f"""
            CALL db.index.vector.queryNodes('claim_embedding_index', $k, $embedding)
            YIELD node, score
            MATCH (s:Entity)-[:SUBJECT_OF]->(node)-[:OBJECT_OF]->(o:Entity)
            WHERE {claim_scope}
            RETURN
              'claim' AS kind,
              node.key AS claim_key,
              s.key AS subject_key,
              o.key AS object_key,
              s.canonical_name AS subject,
              node.relation_type AS relation_type,
              o.canonical_name AS object,
              node.description AS description,
              score
            ORDER BY score DESC
            """,
            k=self.anchor_k,
            embedding=embedding,
        )

        evidence_rows = self._execute_scoped_read(
            f"""
            CALL db.index.vector.queryNodes('evidence_embedding_index', $k, $embedding)
            YIELD node, score
            WHERE {evidence_scope}
            OPTIONAL MATCH (c:Claim)-[:SUPPORTED_BY]->(node)
            OPTIONAL MATCH (s:Entity)-[:SUBJECT_OF]->(c)-[:OBJECT_OF]->(o:Entity)
            RETURN
              'evidence' AS kind,
              node.evidence_id AS evidence_id,
              left(node.text, 500) AS text,
              node.block_type AS block_type,
              c.key AS claim_key,
              s.key AS subject_key,
              o.key AS object_key,
              s.canonical_name AS subject,
              c.relation_type AS relation_type,
              o.canonical_name AS object,
              score
            ORDER BY score DESC
            """,
            k=self.anchor_k,
            embedding=embedding,
        )

        anchors: List[JSONDict] = []
        anchor_keys: List[str] = []
        seen_labels = set()

        def add_anchor_key(value: Any) -> None:
            key = str(value or "").strip()
            if key and key not in anchor_keys:
                anchor_keys.append(key)

        for row in entity_rows:
            label = row.get("label")
            if label and label not in seen_labels:
                anchors.append({
                    "kind": "entity",
                    "label": label,
                    "entity_key": row.get("entity_key"),
                    "entity_type": row.get("entity_type"),
                    "description": row.get("description"),
                    "score": row.get("score"),
                })
                seen_labels.add(label)

            add_anchor_key(row.get("entity_key"))

            if len(anchor_keys) >= self.anchor_k:
                break

        for row in claim_rows:
            label = f"{row.get('subject')} --{row.get('relation_type')}--> {row.get('object')}"
            if label and label not in seen_labels:
                anchors.append({
                    "kind": "claim",
                    "label": label,
                    "claim_key": row.get("claim_key"),
                    "subject_key": row.get("subject_key"),
                    "object_key": row.get("object_key"),
                    "description": row.get("description"),
                    "score": row.get("score"),
                })
                seen_labels.add(label)

            add_anchor_key(row.get("subject_key"))
            add_anchor_key(row.get("object_key"))

            if len(anchor_keys) >= self.anchor_k:
                break

        for row in evidence_rows:
            if row.get("subject") and row.get("object"):
                label = f"{row.get('subject')} --{row.get('relation_type')}--> {row.get('object')}"
            else:
                label = row.get("evidence_id")

            if label and label not in seen_labels:
                anchors.append({
                    "kind": "evidence",
                    "label": label,
                    "evidence_id": row.get("evidence_id"),
                    "claim_key": row.get("claim_key"),
                    "subject_key": row.get("subject_key"),
                    "object_key": row.get("object_key"),
                    "text": row.get("text"),
                    "block_type": row.get("block_type"),
                    "score": row.get("score"),
                })
                seen_labels.add(label)

            add_anchor_key(row.get("subject_key"))
            add_anchor_key(row.get("object_key"))

            if len(anchor_keys) >= self.anchor_k:
                break

        return {
            "anchors": anchors,
            "anchor_keys": anchor_keys[: self.anchor_k],
            "entity_rows": entity_rows,
            "claim_rows": claim_rows,
            "evidence_rows": evidence_rows,
            "scope": self.scope.to_dict(),
            "reason": "Neo4j vector-index semantic anchor retrieval over scoped Entity, Claim, and Evidence nodes.",
        }

    def _semantic_anchor_cypher(self, anchor_keys: Sequence[str]) -> str:
        where_clause = cypher_or_equals("s.key", anchor_keys)
        scope_clause = self._scoped_node_predicate_any(["s", "o", "c", "ev"])

        return f"""
MATCH (s:Entity)-[r]->(o:Entity)
MATCH (c:Claim {{key: r.claim_key}})
MATCH (c)--(ev:Evidence)
WHERE type(r) = 'KG_REL'
  AND ({where_clause})
  AND ({scope_clause})
RETURN s, c, o, ev
LIMIT {self.budget.max_rows_per_query}
""".strip()
    
    def _graph_value(self, item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _frontier_keys_from_graph(self, graph_obj: LocalGraph, max_keys: int = 16) -> List[str]:
        keys: List[str] = []

        for edge in getattr(graph_obj, "edges", {}).values():
            for value in [
                self._graph_value(edge, "subject_key"),
                self._graph_value(edge, "object_key"),
                self._graph_value(edge, "subject"),
                self._graph_value(edge, "object"),
            ]:
                value = str(value or "").strip()
                if value and value not in keys:
                    keys.append(value)
                if len(keys) >= max_keys:
                    return keys

        for key in getattr(graph_obj, "nodes", {}).keys():
            key = str(key or "").strip()
            if key and key not in keys:
                keys.append(key)
            if len(keys) >= max_keys:
                return keys

        return keys

    def _frontier_expansion_cypher(self, graph_obj: LocalGraph) -> str:
        keys = self._frontier_keys_from_graph(graph_obj)
        where_clause = cypher_or_equals("s.key", keys)
        scope_clause = self._scoped_node_predicate_any(["s", "o"])

        return f"""
MATCH (s:Entity)-[r:KG_REL]-(o:Entity)
WHERE ({where_clause})
  AND ({scope_clause})
MATCH (c:Claim {{key: r.claim_key}})
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(ev:Evidence)
RETURN s, c, o, ev
LIMIT {self.budget.max_rows_per_query}
""".strip()

    def _frontier_evidence_cypher(self, graph_obj: LocalGraph) -> str:
        keys = self._frontier_keys_from_graph(graph_obj)
        where_clause = cypher_or_equals("s.key", keys)
        scope_clause = self._scoped_node_predicate_any(["s", "c", "ev"])

        return f"""
MATCH (s:Entity)
WHERE ({where_clause})
OPTIONAL MATCH (s)-[:SUPPORTED_BY]->(ev1:Evidence)
OPTIONAL MATCH (s)-[:SUBJECT_OF]->(c:Claim)
OPTIONAL MATCH (c)-[:OBJECT_OF]->(o:Entity)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(ev2:Evidence)
WITH s, c, o, coalesce(ev1, ev2) AS ev
WHERE ({scope_clause})
RETURN s, c, o, ev
LIMIT {self.budget.max_rows_per_query}
""".strip()

    # ------------------------------------------------------------
    # Evidence and table lanes
    # ------------------------------------------------------------

    def _evidence_lane_rows(
        self,
        question: str,
        *,
        k: int,
        block_types: Optional[Sequence[str]],
        operation: str,
    ) -> List[JSONDict]:
        embedding = self._embed_query(question, operation=operation)
        evidence_scope = self.scope.cypher_predicate("node")
        claim_scope = self.scope.cypher_predicate("c")

        block_filter = "true"
        if block_types:
            block_filter = "node.block_type IN $block_types"

        return self._execute_scoped_read(
            f"""
            CALL db.index.vector.queryNodes('evidence_embedding_index', $k, $embedding)
            YIELD node, score
            WHERE ({evidence_scope})
              AND ({block_filter})
            OPTIONAL MATCH (c:Claim)-[:SUPPORTED_BY]->(node)
            OPTIONAL MATCH (s:Entity)-[:SUBJECT_OF]->(c)-[:OBJECT_OF]->(o:Entity)
            WHERE c IS NULL OR ({claim_scope})
            RETURN s, c, o, node AS ev, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            k=k,
            limit=max(k, self.budget.max_rows_per_query),
            embedding=embedding,
            block_types=list(block_types or []),
        )

    def _run_evidence_lanes(
        self,
        *,
        state: IRAGState,
        graph_obj: LocalGraph,
    ) -> Tuple[int, int, int, int]:
        run_lanes = state["iteration"] == 0 or self.table_lane_each_iteration
        if not run_lanes:
            return 0, 0, 0, 0

        evidence_rows: List[JSONDict] = []
        table_rows: List[JSONDict] = []
        evidence_added = 0
        table_added = 0

        if self.enable_evidence_lane:
            before = graph_obj.counts()
            try:
                evidence_rows = self._evidence_lane_rows(
                    state["question"],
                    k=self.budget.evidence_lane_k,
                    block_types=None,
                    operation="kg_irag_evidence_lane_embedding",
                )
                graph_obj.ingest_cypher_rows(
                    evidence_rows,
                    source=f"kg_irag_evidence_lane_iter_{state['iteration']}",
                )
                after = graph_obj.counts()
                evidence_added = after["evidence"] - before["evidence"]
                graph_obj.add_diagnostic(
                    "kg_irag_evidence_lane",
                    "Retrieved scoped evidence-nearest-neighbour rows.",
                    row_count=len(evidence_rows),
                    evidence_added=evidence_added,
                    scope=self.scope.to_dict(),
                )
            except Exception as error:
                graph_obj.add_diagnostic(
                    "kg_irag_evidence_lane_failed",
                    "Evidence lane retrieval failed.",
                    error=str(error),
                    scope=self.scope.to_dict(),
                )

        if self.enable_table_lane:
            before = graph_obj.counts()
            try:
                table_rows = self._evidence_lane_rows(
                    state["question"],
                    k=self.budget.table_lane_k,
                    block_types=["table", "time_series"],
                    operation="kg_irag_table_lane_embedding",
                )
                graph_obj.ingest_cypher_rows(
                    table_rows,
                    source=f"kg_irag_table_lane_iter_{state['iteration']}",
                )
                after = graph_obj.counts()
                table_added = after["evidence"] - before["evidence"]
                graph_obj.add_diagnostic(
                    "kg_irag_table_lane",
                    "Retrieved scoped table/time-series evidence rows.",
                    row_count=len(table_rows),
                    evidence_added=table_added,
                    scope=self.scope.to_dict(),
                )
            except Exception as error:
                graph_obj.add_diagnostic(
                    "kg_irag_table_lane_failed",
                    "Table/time-series evidence lane retrieval failed.",
                    error=str(error),
                    scope=self.scope.to_dict(),
                )

        return len(evidence_rows), len(table_rows), evidence_added, table_added

    # ------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(IRAGState)

        graph.add_node("anchor", self._anchor_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("export", self._export_node)

        graph.set_entry_point("anchor")
        graph.add_edge("anchor", "retrieve")
        graph.add_edge("retrieve", "decide")

        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "retrieve": "retrieve",
                "export": "export",
            },
        )

        graph.add_edge("export", END)

        return graph.compile()

    # ------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------

    def _anchor_node(self, state: IRAGState) -> IRAGState:
        init = self.anchoring.initialize(
            state["question"],
            self.schema_text,
            self.budget,
            scope_instruction=self._scope_instruction(),
        )

        semantic: JSONDict = {
            "anchors": [],
            "anchor_keys": [],
            "reason": "Semantic anchoring not attempted.",
        }

        anchors = init["anchors"]
        reasoning_prompt = init["reasoning_prompt"]
        cypher = init["cypher"]

        try:
            semantic = self._semantic_anchor_candidates(state["question"])
            anchor_keys = semantic.get("anchor_keys") or []

            if anchor_keys:
                cypher = self._semantic_anchor_cypher(anchor_keys)
                anchors = [
                    str(item.get("label"))
                    for item in semantic.get("anchors") or []
                    if item.get("label")
                ]
        except Exception as error:
            semantic = {
                "anchors": [],
                "anchor_keys": [],
                "reason": f"Semantic anchoring failed; falling back to LLM Cypher: {error}",
            }

        graph_obj = graph_from_json(state["graph"])
        graph_obj.add_diagnostic(
            "kg_irag_anchor",
            "Initialized KG-IRAG retrieval anchors.",
            anchors=anchors,
            reasoning_prompt=reasoning_prompt,
            cypher=cypher,
            semantic_anchors=semantic,
            llm_anchors=init["anchors"],
            llm_cypher=init["cypher"],
            scope=self.scope.to_dict(),
            large=self.large,
            continue_after_anchor=self.continue_after_anchor,
        )

        return {
            **state,
            "anchors": anchors,
            "reasoning_prompt": reasoning_prompt,
            "pending_cypher": cypher,
            "graph": graph_obj.to_dict(),
        }

    def _retrieve_node(self, state: IRAGState) -> IRAGState:
        if state["status"] != "running":
            return state

        allowed_depth = dynamic_max_depth(
            state["iteration"],
            previous_rows=state["previous_rows"],
            initial_rows=state["initial_rows"],
            cap=self.budget.max_path_depth,
        )

        query = state["pending_cypher"]
        graph_obj = graph_from_json(state["graph"])
        rejected_queries = list(state["rejected_queries"])

        current = query
        last_error = ""

        for attempt in range(self.budget.max_bad_queries + 1):
            safety = CypherSafety(
                max_rows=self.budget.max_rows_per_query,
                max_path_depth=allowed_depth,
            )
            validation = validate_readonly_cypher(current, safety)

            if validation.ok:
                scope_error = self._scope_reference_error(validation.cypher)
                if scope_error:
                    validation.ok = False
                    validation.reason = scope_error

            if not validation.ok:
                last_error = validation.reason or "Cypher validation failed."
                rejected_queries.append(
                    {
                        "iteration": state["iteration"],
                        "attempt": attempt,
                        "cypher": current,
                        "reason": last_error,
                    }
                )

                current = self.controller.repair_query(
                    question=state["question"],
                    schema_text=self.schema_text,
                    bad_cypher=current,
                    error=last_error,
                    budget=self.budget,
                    allowed_depth=allowed_depth,
                    scope_instruction=self._scope_instruction(),
                )

                if not current:
                    break

                continue

            try:
                before = graph_obj.counts()
                rows = self._execute_scoped_read(validation.cypher)
                graph_obj.ingest_cypher_rows(
                    rows,
                    source=f"kg_irag_iter_{state['iteration']}",
                )

                evidence_lane_rows, table_lane_rows, evidence_lane_added, table_lane_added = self._run_evidence_lanes(
                    state=state,
                    graph_obj=graph_obj,
                )

                after = graph_obj.counts()

                nodes_added = after["nodes"] - before["nodes"]
                edges_added = after["edges"] - before["edges"]
                evidence_added = after["evidence"] - before["evidence"]

                batch = RetrievalBatch(
                    iteration=state["iteration"],
                    cypher=current,
                    validated_cypher=validation.cypher,
                    row_count=len(rows),
                    nodes_added=nodes_added,
                    edges_added=edges_added,
                    evidence_added=evidence_added,
                    evidence_lane_rows=evidence_lane_rows,
                    table_lane_rows=table_lane_rows,
                    evidence_lane_added=evidence_lane_added,
                    table_lane_added=table_lane_added,
                )

                graph_obj.add_diagnostic(
                    "kg_irag_retrieval_batch",
                    "Executed KG-IRAG retrieval query.",
                    batch=asdict(batch),
                    allowed_depth=allowed_depth,
                    scope=self.scope.to_dict(),
                )

                initial_rows = state["initial_rows"]
                if state["iteration"] == 0:
                    initial_rows = len(rows)

                previous_rows = len(rows)

                no_new_info_steps = state["no_new_info_steps"]
                if nodes_added + edges_added + evidence_added == 0:
                    no_new_info_steps += 1
                else:
                    no_new_info_steps = 0

                status, stop_reason = self._budget_or_progress_status(
                    graph_obj,
                    no_new_info_steps,
                )

                return {
                    **state,
                    "graph": graph_obj.to_dict(),
                    "batches": state["batches"] + [asdict(batch)],
                    "previous_queries": state["previous_queries"] + [validation.cypher],
                    "rejected_queries": rejected_queries,
                    "initial_rows": initial_rows,
                    "previous_rows": previous_rows,
                    "no_new_info_steps": no_new_info_steps,
                    "status": status or state["status"],
                    "stop_reason": stop_reason or state["stop_reason"],
                }

            except Exception as error:
                last_error = str(error)
                rejected_queries.append(
                    {
                        "iteration": state["iteration"],
                        "attempt": attempt,
                        "cypher": validation.cypher,
                        "reason": last_error,
                    }
                )

                current = self.controller.repair_query(
                    question=state["question"],
                    schema_text=self.schema_text,
                    bad_cypher=validation.cypher,
                    error=last_error,
                    budget=self.budget,
                    allowed_depth=allowed_depth,
                    scope_instruction=self._scope_instruction(),
                )

                if not current:
                    break

        graph_obj.add_diagnostic(
            "kg_irag_retrieval_failed",
            "Could not produce or execute a valid retrieval query.",
            cypher=query,
            error=last_error,
            scope=self.scope.to_dict(),
        )

        return {
            **state,
            "graph": graph_obj.to_dict(),
            "rejected_queries": rejected_queries,
            "status": "insufficient",
            "stop_reason": "Could not produce or execute a valid retrieval query.",
        }

    def _decide_node(self, state: IRAGState) -> IRAGState:
        if state["status"] != "running":
            return state

        allowed_depth = dynamic_max_depth(
            state["iteration"],
            previous_rows=state["previous_rows"],
            initial_rows=state["initial_rows"],
            cap=self.budget.max_path_depth,
        )

        decision = self.controller.decide(
            state=state,
            schema_text=self.schema_text,
            budget=self.budget,
            allowed_depth=allowed_depth,
            scope_instruction=self._scope_instruction(),
            enable_web_acquire=self.web_acquire,
        )

        graph_obj = graph_from_json(state["graph"])
        graph_obj.add_diagnostic(
            "kg_irag_controller_decision",
            "Controller decided whether to stop KG-IRAG retrieval.",
            decision=decision,
            iteration=state["iteration"],
            scope=self.scope.to_dict(),
        )

        batches = list(state["batches"])
        if batches:
            batches[-1]["controller_reason"] = decision.get("reason", "")

        acquisition_requests = self._merge_acquisition_requests(
            state.get("acquisition_requests", []),
            decision.get("acquisition_requests") or [],
        )

        if decision.get("stop"):
            return {
                **state,
                "graph": graph_obj.to_dict(),
                "batches": batches,
                "acquisition_requests": acquisition_requests,
                "status": decision.get("status") or "sufficient",
                "stop_reason": decision.get("reason") or "Controller judged subgraph sufficient.",
            }

        next_iteration = state["iteration"] + 1

        if next_iteration >= self.budget.max_iterations:
            status = "acquisition_requested" if self.web_acquire and acquisition_requests else "insufficient"
            reason = "Reached maximum KG-IRAG iterations."
            if status == "acquisition_requested":
                reason += " External acquisition requests were generated."
            return {
                **state,
                "graph": graph_obj.to_dict(),
                "batches": batches,
                "iteration": next_iteration,
                "acquisition_requests": acquisition_requests,
                "status": status,
                "stop_reason": reason,
            }

        action = str(decision.get("action") or "expand_frontier")

        if action == "expand_frontier":
            next_cypher = self._frontier_expansion_cypher(graph_obj)

        elif action == "retrieve_evidence":
            next_cypher = self._frontier_evidence_cypher(graph_obj)

        elif action == "text2cypher_fallback":
            next_cypher = str(decision.get("cypher") or "").strip()

        else:
            return {
                **state,
                "graph": graph_obj.to_dict(),
                "batches": batches,
                "acquisition_requests": acquisition_requests,
                "status": "partial",
                "stop_reason": decision.get("reason") or f"Unsupported controller action: {action}",
            }

        if not next_cypher:
            status = "acquisition_requested" if self.web_acquire and acquisition_requests else "insufficient"
            reason = f"Controller action {action} produced no query."
            if status == "acquisition_requested":
                reason += " External acquisition requests were generated."
            return {
                **state,
                "graph": graph_obj.to_dict(),
                "batches": batches,
                "acquisition_requests": acquisition_requests,
                "status": status,
                "stop_reason": reason,
            }

        if (
            not self.continue_after_anchor
            and state["iteration"] == 0
            and graph_obj.counts()["edges"] > 0
        ):
            return {
                **state,
                "graph": graph_obj.to_dict(),
                "batches": batches,
                "acquisition_requests": acquisition_requests,
                "status": "partial",
                "stop_reason": (
                    "Semantic anchor retrieval produced a non-empty local graph; "
                    "stopping before controller expansion to preserve valid bounded retrieval. "
                    "Use --large or --continue-after-anchor to allow expansion."
                ),
            }

        return {
            **state,
            "graph": graph_obj.to_dict(),
            "batches": batches,
            "iteration": next_iteration,
            "pending_cypher": next_cypher,
            "acquisition_requests": acquisition_requests,
        }

    def _export_node(self, state: IRAGState) -> IRAGState:
        return {
            **state,
            "result": self._export(state),
        }

    # ------------------------------------------------------------
    # Routing / status
    # ------------------------------------------------------------

    def _route_after_decide(self, state: IRAGState) -> str:
        if state["status"] != "running":
            return "export"

        return "retrieve"

    def _budget_or_progress_status(
        self,
        graph: LocalGraph,
        no_new_info_steps: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        counts = graph.counts()

        if counts["nodes"] > self.budget.kappa_max_nodes:
            return (
                "budget_exceeded",
                f"Node retrieval threshold exceeded: {counts['nodes']} > {self.budget.kappa_max_nodes}",
            )

        if counts["edges"] > self.budget.kappa_max_edges:
            return (
                "budget_exceeded",
                f"Edge retrieval threshold exceeded: {counts['edges']} > {self.budget.kappa_max_edges}",
            )

        if counts["evidence"] > self.budget.kappa_max_evidence:
            return (
                "budget_exceeded",
                f"Evidence retrieval threshold exceeded: {counts['evidence']} > {self.budget.kappa_max_evidence}",
            )

        if no_new_info_steps >= self.budget.no_new_info_patience:
            counts = graph.counts()
            if counts["nodes"] > 0 or counts["edges"] > 0 or counts["evidence"] > 0:
                return (
                    "partial",
                    "Stopped because retrieval saturated: additional expansion produced no new subgraph facts.",
                )

            return (
                "insufficient",
                "Stopped because retrieval produced no new subgraph facts.",
            )

        return None, None

    # ------------------------------------------------------------
    # Acquisition requests
    # ------------------------------------------------------------

    def _merge_acquisition_requests(
        self,
        existing: Sequence[JSONDict],
        new_items: Sequence[Any],
    ) -> List[JSONDict]:
        out: List[JSONDict] = []
        seen = set()

        for item in list(existing or []) + list(new_items or []):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "query": query,
                "reason": str(item.get("reason") or "").strip(),
                "priority": str(item.get("priority") or "medium").strip() or "medium",
            })
            if len(out) >= self.max_acquisition_requests:
                break

        return out

    def _fallback_acquisition_requests(self, state: IRAGState) -> List[JSONDict]:
        if not self.web_acquire or self.max_acquisition_requests <= 0:
            return []

        prompt = f"""
The KG-IRAG graph retrieval was insufficient for this question.
Generate concise external source acquisition requests that would help fill the evidence gap.
Do not answer the question.

Return JSON:
{{
  "acquisition_requests": [
    {{"query": "search/source query", "reason": "why this evidence is needed", "priority": "high|medium|low"}}
  ]
}}

Question:
{state['question']}

Stop reason:
{state.get('stop_reason')}

Current anchors:
{json.dumps(state.get('anchors') or [], indent=2, ensure_ascii=False)}

Current compact graph context:
{json.dumps(compact_context(state['graph']), indent=2, ensure_ascii=False)}
""".strip()

        try:
            raw = call_json(
                self.client,
                self.model,
                prompt,
                ledger=self.ledger,
                operation="kg_irag_acquisition_request_generation",
            )
            requests = raw.get("acquisition_requests")
            if isinstance(requests, list):
                return self._merge_acquisition_requests([], requests)
        except Exception:
            pass

        return []

    def _handle_acquisition_requests(self, result: JSONDict) -> JSONDict:
        requests = result.get("acquisition_requests") or []
        if not self.web_acquire or not requests:
            result["acquisition_status"] = "not_requested"
            return result

        payload = {
            "question": result.get("question"),
            "status": result.get("status"),
            "stop_reason": result.get("stop_reason"),
            "scope": self.scope.to_dict(),
            "acquisition_requests": requests,
        }

        output_path = self.acquisition_requests_output
        if output_path is not None:
            write_json(output_path, payload)
            result["acquisition_requests_output"] = str(output_path)

        if self.web_acquire_command:
            env = dict(os.environ)
            if output_path is not None:
                env["LANTHIC_ACQUISITION_REQUESTS"] = str(output_path)
            env["LANTHIC_KG_IRAG_QUESTION"] = str(result.get("question") or "")
            completed = subprocess.run(
                self.web_acquire_command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            result["acquisition_command"] = self.web_acquire_command
            result["acquisition_command_exit_code"] = completed.returncode
            result["acquisition_command_output"] = compact_text(completed.stdout, 4000)
            result["acquisition_status"] = "command_succeeded" if completed.returncode == 0 else "command_failed"
        else:
            result["acquisition_status"] = "requests_written" if output_path is not None else "requests_generated"

        return result

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    def _context_for_export(self, graph_obj: LocalGraph) -> JSONDict:
        if self.large:
            return graph_obj.to_context(
                max_nodes=min(self.budget.kappa_max_nodes, 200),
                max_edges=min(self.budget.kappa_max_edges, 300),
                max_evidence=min(self.budget.kappa_max_evidence, 100),
                max_evidence_chars=900,
            )
        return graph_obj.to_context()

    def _export(self, state: IRAGState) -> JSONDict:
        graph_obj = graph_from_json(state["graph"])

        status = state["status"]
        stop_reason = state["stop_reason"]

        if status == "running":
            status = "insufficient"
            stop_reason = "Reached export while retrieval was still marked running."

        acquisition_requests = list(state.get("acquisition_requests") or [])
        if self.web_acquire and status in {"insufficient", "partial", "budget_exceeded"} and not acquisition_requests:
            acquisition_requests = self._fallback_acquisition_requests(state)
            if acquisition_requests and status == "insufficient":
                status = "acquisition_requested"
                stop_reason = f"{stop_reason} External acquisition requests were generated."

        result = {
            "question": state["question"],
            "status": status,
            "stop_reason": stop_reason,
            "anchors": state["anchors"],
            "reasoning_prompt": state["reasoning_prompt"],
            "scope": self.scope.to_dict(),
            "large": self.large,
            "continue_after_anchor": self.continue_after_anchor,
            "evidence_lane_enabled": self.enable_evidence_lane,
            "table_lane_enabled": self.enable_table_lane,
            "budgets": asdict(self.budget),
            "batches": state["batches"],
            "rejected_queries": state["rejected_queries"],
            "acquisition_requests": acquisition_requests,
            "local_reasoning_subgraph": graph_obj.to_dict(),
            "context": self._context_for_export(graph_obj),
        }

        return self._handle_acquisition_requests(result)


# ============================================================
# Output helpers
# ============================================================

def summarize(result: JSONDict) -> str:
    graph = result["local_reasoning_subgraph"]
    context = result["context"]

    lines = []
    lines.append("KG-IRAG LOCAL REASONING SUBGRAPH")
    lines.append("=" * 60)
    lines.append(f"Question: {result['question']}")
    lines.append(f"Status: {result['status']}")
    lines.append(f"Stop reason: {result['stop_reason']}")
    lines.append(f"Scope: {json.dumps(result.get('scope') or {}, ensure_ascii=False, sort_keys=True)}")
    lines.append(f"Large mode: {result.get('large')}")
    lines.append(f"Anchors: {', '.join(result['anchors'])}")
    lines.append("")
    lines.append(f"Nodes: {graph['counts']['nodes']}")
    lines.append(f"Edges: {graph['counts']['edges']}")
    lines.append(f"Evidence: {graph['counts']['evidence']}")
    lines.append("")

    lines.append("Batches:")
    for batch in result["batches"]:
        lines.append(
            f"  - iter {batch['iteration']}: rows={batch['row_count']} "
            f"+nodes={batch['nodes_added']} +edges={batch['edges_added']} "
            f"+evidence={batch['evidence_added']} "
            f"evidence_lane_rows={batch.get('evidence_lane_rows', 0)} "
            f"table_lane_rows={batch.get('table_lane_rows', 0)}"
        )
        if batch.get("controller_reason"):
            lines.append(f"    controller: {batch['controller_reason']}")
    lines.append("")

    if context.get("claims"):
        lines.append("Top claims:")
        for claim in context["claims"][:15]:
            lines.append(
                f"  - {claim.get('subject')} --{claim.get('relation_type')}--> "
                f"{claim.get('object')} ({claim.get('grounding_score')})"
            )

    if context.get("evidence"):
        table_like = [
            item for item in context.get("evidence") or []
            if (item.get("properties") or {}).get("block_type") in {"table", "time_series"}
        ]
        if table_like:
            lines.append("")
            lines.append("Table/time-series evidence:")
            for item in table_like[:8]:
                lines.append(
                    f"  - {item.get('evidence_id')} "
                    f"{(item.get('properties') or {}).get('block_type')}: "
                    f"{compact_text(item.get('text'), 180)}"
                )

    if result.get("acquisition_requests"):
        lines.append("")
        lines.append("Acquisition requests:")
        for item in result["acquisition_requests"][:8]:
            lines.append(f"  - [{item.get('priority')}] {item.get('query')} — {item.get('reason')}")

    if result.get("rejected_queries"):
        lines.append("")
        lines.append("Rejected queries:")
        for item in result["rejected_queries"][:8]:
            lines.append(
                f"  - iter {item['iteration']} attempt {item['attempt']}: {item['reason']}"
            )

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph KG-IRAG iterative local subgraph retriever")

    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--anchor-k", type=int, default=8)

    parser.add_argument("--output", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--summary", action="store_true")

    parser.add_argument("--corpus-id", default=os.getenv("LANTHIC_CORPUS_ID"))
    parser.add_argument("--branch-id", default=os.getenv("LANTHIC_BRANCH_ID"))
    parser.add_argument("--source-id", action="append", default=[])

    parser.add_argument("--large", action="store_true")
    parser.add_argument("--continue-after-anchor", action="store_true")

    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--max-evidence", type=int, default=None)
    parser.add_argument("--evidence-lane-k", type=int, default=None)
    parser.add_argument("--table-lane-k", type=int, default=None)

    parser.add_argument("--enable-evidence-lane", dest="enable_evidence_lane", action="store_true", default=True)
    parser.add_argument("--disable-evidence-lane", dest="enable_evidence_lane", action="store_false")
    parser.add_argument("--enable-table-lane", dest="enable_table_lane", action="store_true", default=True)
    parser.add_argument("--disable-table-lane", dest="enable_table_lane", action="store_false")
    parser.add_argument("--table-lane-each-iteration", action="store_true")

    parser.add_argument("--web-acquire", action="store_true")
    parser.add_argument("--web-acquire-command", default=None)
    parser.add_argument("--max-acquisition-requests", type=int, default=3)
    parser.add_argument("--acquisition-requests-output", type=Path, default=None)

    parser.add_argument("--run-id", default=os.getenv("LANTHIC_RUN_ID"))
    parser.add_argument("--cost-ledger", type=Path, default=Path(os.getenv("LANTHIC_COST_LEDGER")) if os.getenv("LANTHIC_COST_LEDGER") else None)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("LANTHIC_CACHE_DIR")) if os.getenv("LANTHIC_CACHE_DIR") else None)
    parser.add_argument("--disable-cache", action="store_true", default=os.getenv("LANTHIC_DISABLE_CACHE", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--pricing-file", type=Path, default=Path(os.getenv("LANTHIC_PRICING_FILE")) if os.getenv("LANTHIC_PRICING_FILE") else None)

    return parser.parse_args()


def budget_from_args(args: argparse.Namespace) -> KGIRAGBudget:
    budget = KGIRAGBudget.large() if args.large else KGIRAGBudget()

    if args.max_iterations is not None:
        budget.max_iterations = args.max_iterations
    if args.max_rows is not None:
        budget.max_rows_per_query = args.max_rows
    if args.max_depth is not None:
        budget.max_path_depth = args.max_depth
    if args.max_nodes is not None:
        budget.kappa_max_nodes = args.max_nodes
    if args.max_edges is not None:
        budget.kappa_max_edges = args.max_edges
    if args.max_evidence is not None:
        budget.kappa_max_evidence = args.max_evidence
    if args.evidence_lane_k is not None:
        budget.evidence_lane_k = args.evidence_lane_k
    if args.table_lane_k is not None:
        budget.table_lane_k = args.table_lane_k

    return budget


def ledger_from_args(args: argparse.Namespace) -> Optional[Any]:
    if CostLedger is None:
        if args.cost_ledger or args.cache_dir:
            raise RuntimeError("cost_ledger.py could not be imported, but cost/cache options were provided.")
        return None

    if not args.cost_ledger and not args.cache_dir:
        return None

    pricing_config = {}
    if args.pricing_file and load_pricing_config is not None:
        pricing_config = load_pricing_config(args.pricing_file)

    return CostLedger(
        run_id=args.run_id or "kg_irag_run",
        source_id=None,
        ledger_path=args.cost_ledger,
        cache_dir=args.cache_dir,
        pricing_config=pricing_config,
        enabled=bool(args.cost_ledger),
        cache_enabled=bool(args.cache_dir) and not args.disable_cache,
    )


def main() -> int:
    args = parse_args()

    budget = budget_from_args(args)
    source_ids = source_ids_from_args(args.source_id)
    scope = KGIRAGScope(
        corpus_id=args.corpus_id,
        branch_id=args.branch_id,
        source_ids=source_ids,
    )

    anchor_k = args.anchor_k
    if args.large and args.anchor_k == 8:
        anchor_k = 20

    ledger = ledger_from_args(args)

    retriever = KGIRAG(
        model=args.model,
        embed_model=args.embed_model,
        anchor_k=anchor_k,
        budget=budget,
        scope=scope,
        large=args.large,
        continue_after_anchor=args.continue_after_anchor,
        enable_evidence_lane=args.enable_evidence_lane,
        enable_table_lane=args.enable_table_lane,
        table_lane_each_iteration=args.table_lane_each_iteration,
        web_acquire=args.web_acquire,
        acquisition_requests_output=args.acquisition_requests_output,
        web_acquire_command=args.web_acquire_command,
        max_acquisition_requests=args.max_acquisition_requests,
        ledger=ledger,
    )

    try:
        result = retriever.retrieve(args.question)
    finally:
        retriever.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[done] wrote {args.output}")

    graph_obj = None

    if args.html or args.dot:
        graph_obj = graph_from_json(result["local_reasoning_subgraph"])

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        graph_obj.write_html(args.html)
        print(f"[done] wrote {args.html}")

    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        graph_obj.write_dot(args.dot)
        print(f"[done] wrote {args.dot}")

    if args.summary or not any([args.output, args.html, args.dot]):
        print(summarize(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())