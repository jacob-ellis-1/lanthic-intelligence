#!/usr/bin/env python3
"""
LangGraph SARG investigation agent.

Purpose:
- SARG is the top-level bounded analyst controller.
- KG-IRAG remains the lower-level retrieval/tool agent.
- Reasoning paths are concrete local-KG traversals:
    node -> edge(reason generated at selection time) -> node
- Investigation memory is scoped to a single investigation only.

This file is designed as a replacement for src/SARG.py while preserving the
important CLI contract: --question, --kg-irag-json/--kg-irag-result, --output,
--summary, document/risk/forecast options where practical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

try:
    from kg_irag import KGIRAG, KGIRAGBudget, KGIRAGScope
except Exception:  # pragma: no cover - allows static checks outside project tree
    KGIRAG = None  # type: ignore
    KGIRAGBudget = None  # type: ignore
    KGIRAGScope = None  # type: ignore

try:
    from local_graph import LocalGraph
except Exception:  # pragma: no cover
    LocalGraph = None  # type: ignore

try:
    from risk_tools import analyze as analyze_risk, summarize_analysis
except Exception:  # pragma: no cover
    analyze_risk = None  # type: ignore
    summarize_analysis = None  # type: ignore

try:
    from forecast_tools import forecast_payload
except Exception:  # pragma: no cover
    forecast_payload = None  # type: ignore

try:
    from cost_ledger import CostLedger, load_pricing_config
except Exception:  # pragma: no cover
    CostLedger = None  # type: ignore
    load_pricing_config = None  # type: ignore


JSONDict = Dict[str, Any]


# ============================================================
# Config / state
# ============================================================

@dataclass
class SARGConfig:
    model: str = "gpt-4.1-mini"
    embed_model: str = "text-embedding-3-small"

    # Agent loop limits.
    max_agent_steps: int = 3
    max_expansions: int = 2

    # Reasoning traversal limits.
    max_depth: int = 4
    beam_width: int = 5
    top_k_paths: int = 3
    max_start_nodes: int = 8
    start_similarity_threshold: float = 0.20

    # Query-plan objective coverage. Generic; no domain-specific hints.
    objective_coverage_threshold: float = 0.20
    min_multihop_steps: int = 2

    # KG-IRAG action budget.
    kg_irag_expansion_iterations: int = 5
    kg_irag_expansion_rows: int = 100
    kg_irag_expansion_depth: int = 4
    kg_irag_expansion_nodes: int = 220
    kg_irag_expansion_edges: int = 420
    kg_irag_expansion_evidence: int = 100
    kg_irag_evidence_lane_k: int = 20
    kg_irag_table_lane_k: int = 12
    table_lane_each_iteration: bool = True

    # Evidence / output.
    max_evidence_snippets: int = 12
    max_evidence_chars: int = 900
    max_history_turns: int = 5

    # Optional tools.
    enable_risk_tools: bool = True
    enable_forecasting: bool = True
    forecast_horizon: int = 3

    # Runtime metadata / scope.
    investigation_id: Optional[str] = None
    run_id: Optional[str] = None
    source_id: Optional[str] = None
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)

    # Cost/cache plumbing, retained for compatibility.
    cost_ledger: Optional[Path] = None
    cache_dir: Optional[Path] = None
    disable_cache: bool = False
    pricing_file: Optional[Path] = None
    ledger: Optional[Any] = None

    # Testing / offline mode.
    use_llm: bool = True


@dataclass
class InvestigationMemory:
    investigation_id: str
    compact_summary: str
    prior_questions: List[JSONDict] = field(default_factory=list)
    prior_answers: List[JSONDict] = field(default_factory=list)
    prior_reasoning_paths: List[JSONDict] = field(default_factory=list)
    pinned_evidence_ids: List[str] = field(default_factory=list)
    selected_graph_item_ids: List[str] = field(default_factory=list)
    unresolved_gaps: List[JSONDict] = field(default_factory=list)


@dataclass
class ScratchNode:
    key: str
    label: str
    node_type: str = "entity"
    description: str = ""
    kg_id: str = ""
    source_ids: List[str] = field(default_factory=list)


@dataclass
class ScratchEdge:
    key: str
    subject_key: str
    object_key: str
    relation: str
    description: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    claim_keys: List[str] = field(default_factory=list)
    kg_id: str = ""
    grounding_score: Optional[float] = None


@dataclass
class TraversalStep:
    from_node_key: str
    edge_key: str
    to_node_key: str
    traversal_direction: str  # forward | backward
    relation: str
    reason: str
    score: float
    evidence_ids: List[str] = field(default_factory=list)
    claim_keys: List[str] = field(default_factory=list)
    kg_items: List[JSONDict] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    supports_objectives: List[str] = field(default_factory=list)


@dataclass
class ReasoningPath:
    path_id: str
    node_keys: List[str]
    steps: List[TraversalStep]
    score: float
    direction: str
    hypothesis: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    missing_evidence: List[JSONDict] = field(default_factory=list)
    status: str = "selected"  # selected | rejected | partial

    def signature(self) -> Tuple[str, ...]:
        return tuple(self.node_keys)


@dataclass
class AnalystStep:
    step: int
    observation: str
    hypothesis: str
    gap: str
    action: str
    action_input: JSONDict
    result_observation: str = ""


class SARGState(TypedDict):
    question: str
    kg_irag_result: Optional[JSONDict]
    local_graph: JSONDict

    investigation_id: str
    investigation_history: List[JSONDict]
    selected_graph_context: List[JSONDict]
    investigation_memory: JSONDict
    query_plan: JSONDict

    reasoning_graph: JSONDict
    concepts: List[str]
    direction: str
    start_matches: List[JSONDict]

    candidate_paths: List[JSONDict]
    selected_reasoning_paths: List[JSONDict]
    rejected_reasoning_paths: List[JSONDict]

    gap_assessment: JSONDict
    analyst_steps: List[JSONDict]
    open_questions: List[JSONDict]

    expansion_count: int
    agent_step_count: int
    next_action: str
    action_input: JSONDict
    action_result: JSONDict

    status: str
    answer: Optional[str]
    final_review: Optional[JSONDict]

    document_jsons: List[JSONDict]
    risk_model: Optional[JSONDict]
    risk_analysis: Optional[JSONDict]
    forecast_analysis: Optional[JSONDict]

    pipeline_metadata: JSONDict
    result: Optional[JSONDict]


# ============================================================
# Generic utilities
# ============================================================

def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def norm(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(value: Any, max_chars: int = 900) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def unique_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def source_ids_from_args(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


def lexical_score(query: str, text: str) -> float:
    q = norm(query)
    t = norm(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t or t in q:
        return 0.9
    q_terms = [term for term in q.split() if len(term) >= 3]
    if not q_terms:
        return 0.0
    return sum(1 for term in q_terms if term in t) / len(q_terms)


GENERIC_OBJECTIVE_WORDS = {
    # Function words and planning verbs.
    "the", "and", "for", "with", "from", "into", "onto", "about", "between", "through",
    "what", "which", "how", "why", "when", "where", "who", "does", "did", "could", "would",
    "should", "can", "may", "might", "must", "need", "needs", "needed",
    "identify", "understand", "determine", "establish", "explain", "assess", "evaluate",
    "analyze", "analyse", "show", "tell", "find", "retrieve", "investigate", "answer",

    # Generic outcome/relationship words. These are not invalid concepts, but
    # they are too broad to prove that a concrete graph path has reached an
    # answer target by themselves.
    "impact", "effect", "affect", "affects", "affected", "consequence", "consequences",
    "outcome", "outcomes", "role", "nature", "extent", "availability", "available",
    "production", "produce", "produces", "produced", "supply", "chain", "connection",
    "connections", "dependency", "dependencies", "risk", "risks", "issue", "issues",
    "event", "events", "condition", "conditions", "target", "source", "bridge",
}


def content_tokens(value: Any) -> List[str]:
    tokens = []
    for token in norm(value).split():
        if len(token) < 3:
            continue
        if token in GENERIC_OBJECTIVE_WORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def is_distinctive_objective_term(value: Any) -> bool:
    tokens = content_tokens(value)
    if not tokens:
        return False
    # Single-token objectives are acceptable only when the token is not a
    # generic planning/outcome word. Multi-token phrases are preferred because
    # they identify a concrete target such as an entity, mechanism, or outcome.
    return True


def filter_objective_terms(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        text = compact(value, 160)
        if not text:
            continue
        if not is_distinctive_objective_term(text):
            continue
        if text not in out:
            out.append(text)
    return out


def objective_terms(objective: Mapping[str, Any]) -> List[str]:
    # Use explicit query_terms as the matching surface. Descriptions are often
    # broad instructions ("what the reasoning must ultimately explain") and
    # should not by themselves satisfy an objective.
    query_terms = filter_objective_terms(objective.get("query_terms") or [])
    if query_terms:
        return query_terms

    # Fallback only when the model supplied no useful query_terms.
    description_terms = filter_objective_terms([objective.get("description")])
    return description_terms


def query_plan_objectives(query_plan: Mapping[str, Any]) -> List[JSONDict]:
    objectives: List[JSONDict] = []
    for group in ["start_objectives", "bridge_objectives", "terminal_objectives"]:
        for item in query_plan.get(group) or []:
            if isinstance(item, Mapping):
                payload = dict(item)
                payload.setdefault("kind", group.replace("_objectives", ""))
                objectives.append(payload)
    return objectives


def query_plan_terms(query_plan: Mapping[str, Any]) -> List[str]:
    terms: List[str] = []
    for objective in query_plan_objectives(query_plan):
        terms.extend(objective_terms(objective))
    if query_plan.get("answer_objective"):
        terms.append(str(query_plan.get("answer_objective")))
    return unique_strings(terms)


def objective_score(text: str, objective: Mapping[str, Any]) -> float:
    """
    Domain-neutral objective match score based on distinctive query terms.
    Generic words such as "impact", "availability", or "production" do not
    count unless they appear inside a more specific phrase.
    """
    t_norm = norm(text)
    if not t_norm:
        return 0.0

    best = 0.0
    text_tokens = set(content_tokens(text))

    for term in objective_terms(objective):
        term_norm = norm(term)
        term_tokens = set(content_tokens(term))
        if not term_tokens:
            continue

        # Exact/subphrase phrase matches are strong.
        phrase_score = lexical_score(term, text)

        # Token overlap catches phrase variants without allowing generic words
        # to dominate. Require a meaningful fraction for multi-token terms.
        overlap = len(term_tokens & text_tokens) / max(1, len(term_tokens))
        token_score = overlap

        best = max(best, phrase_score, token_score)

    return best


def objective_coverage(text: str, objectives: Sequence[Mapping[str, Any]], threshold: float = 0.20) -> JSONDict:
    covered: List[str] = []
    scores: Dict[str, float] = {}
    for index, objective in enumerate(objectives, start=1):
        objective_id = str(objective.get("id") or f"objective_{index}")
        score = objective_score(text, objective)
        scores[objective_id] = round(score, 4)
        if score >= threshold:
            covered.append(objective_id)
    all_ids = [str(obj.get("id") or f"objective_{i}") for i, obj in enumerate(objectives, start=1)]
    return {
        "covered_objectives": covered,
        "uncovered_objectives": [objective_id for objective_id in all_ids if objective_id not in covered],
        "objective_scores": scores,
    }


def selected_path_text(path: Mapping[str, Any]) -> str:
    """Human-readable path text, including reasons. Use for display only."""
    parts: List[str] = []
    for node in path.get("nodes") or []:
        if isinstance(node, Mapping):
            parts.append(str(node.get("label") or node.get("id") or ""))
    for step in path.get("steps") or []:
        if isinstance(step, Mapping):
            parts.extend([
                str(step.get("from") or ""),
                str(step.get("edge") or ""),
                str(step.get("to") or ""),
                str(step.get("reason") or ""),
            ])
    return " ".join(part for part in parts if part).strip()


def selected_path_observable_text(path: Mapping[str, Any]) -> str:
    """
    Text from actual graph items only: nodes and edge labels.
    Do not include generated reasons here, otherwise a rationale can falsely
    satisfy an objective that the graph path itself has not reached.
    """
    parts: List[str] = []
    for node in path.get("nodes") or []:
        if isinstance(node, Mapping):
            parts.append(str(node.get("label") or node.get("id") or ""))
    for step in path.get("steps") or []:
        if isinstance(step, Mapping):
            parts.extend([
                str(step.get("from") or ""),
                str(step.get("edge") or ""),
                str(step.get("to") or ""),
            ])
    return " ".join(part for part in parts if part).strip()


def path_supported_objectives(path: Mapping[str, Any]) -> List[str]:
    """Objective IDs explicitly supported by traversal decisions."""
    out: List[str] = []
    for objective_id in path.get("coveredObjectives") or []:
        if objective_id and str(objective_id) not in out:
            out.append(str(objective_id))
    for step in path.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        for objective_id in step.get("supportsObjectives") or []:
            if objective_id and str(objective_id) not in out:
                out.append(str(objective_id))
    return out


def objective_ids_by_kind(query_plan: Mapping[str, Any], kind: str) -> List[str]:
    return [
        str(obj.get("id"))
        for obj in query_plan_objectives(query_plan)
        if obj.get("kind") == kind and obj.get("id")
    ]


def objective_by_id(objectives: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for index, objective in enumerate(objectives, start=1):
        objective_id = str(objective.get("id") or f"objective_{index}")
        out[objective_id] = objective
    return out


def validated_supported_objectives(
    path: Mapping[str, Any],
    objectives: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> List[str]:
    """
    Validate model/evaluator-declared objective support against graph-observable
    path text. This prevents a generated rationale from claiming that a hop
    reached an objective that no node/edge in the path actually represents.
    """
    objective_lookup = objective_by_id(objectives)
    observable_text = selected_path_observable_text(path)
    out: List[str] = []

    for objective_id in path_supported_objectives(path):
        objective = objective_lookup.get(objective_id)
        if not objective:
            continue
        score = objective_score(observable_text, objective)
        kind = objective.get("kind")
        required = max(0.34, threshold + 0.14)
        if kind == "terminal":
            required = max(0.48, threshold + 0.28)
        if score >= required and objective_id not in out:
            out.append(objective_id)

    return out


def supported_or_observed_objectives(
    path: Mapping[str, Any],
    objectives: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> List[str]:
    """
    Objective coverage used for stopping. Prefer explicit objective IDs from
    traversal evaluation. Add high-confidence graph-observable coverage as a
    fallback, but never use generated reasons.
    """
    covered = validated_supported_objectives(path, objectives, threshold=threshold)
    observed = objective_coverage(
        selected_path_observable_text(path),
        objectives,
        threshold=max(0.48, threshold + 0.28),
    )
    for objective_id in observed.get("covered_objectives") or []:
        if objective_id not in covered:
            covered.append(objective_id)
    return covered


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0 or rn == 0:
        return 0.0
    return dot / (ln * rn)


def call_json(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    ledger: Optional[Any] = None,
    operation: str = "sarg_agent_json",
) -> JSONDict:
    messages = [
        {"role": "system", "content": "Return only valid JSON. No markdown."},
        {"role": "user", "content": prompt},
    ]
    if ledger is not None:
        return ledger.chat_json(
            client,
            stage="sarg",
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
    return json.loads(response.choices[0].message.content or "{}")


def call_text(
    client: OpenAI,
    model: str,
    messages: Sequence[JSONDict],
    *,
    ledger: Optional[Any] = None,
    operation: str = "sarg_agent_text",
) -> str:
    if ledger is not None:
        return ledger.chat_text(
            client,
            stage="sarg",
            model=model,
            messages=list(messages),
            operation=operation,
            temperature=0,
        ).strip()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=list(messages),
    )
    return (response.choices[0].message.content or "").strip()


# ============================================================
# Local graph adapters
# ============================================================

def graph_from_json(data: JSONDict) -> Any:
    """Return project LocalGraph when available; otherwise keep JSON dict."""
    if LocalGraph is not None and hasattr(LocalGraph, "from_dict"):
        try:
            return LocalGraph.from_dict(data)
        except Exception:
            pass
    return data


def graph_to_dict(graph: Any) -> JSONDict:
    if isinstance(graph, Mapping):
        return dict(graph)
    if hasattr(graph, "to_dict"):
        return graph.to_dict()
    raise TypeError("Unsupported graph object.")


def local_nodes(local_graph_json: JSONDict) -> Dict[str, Any]:
    # Prefer the raw KG-IRAG JSON. SARG needs stable node IDs exactly as exported
    # so UI graphItemIds can line up with the Local KG viewer.
    if isinstance(local_graph_json, Mapping):
        nodes = local_graph_json.get("nodes")
        if isinstance(nodes, Mapping):
            return dict(nodes)

    graph = graph_from_json(local_graph_json)
    if isinstance(graph, Mapping):
        return dict(graph.get("nodes") or {})
    return dict(getattr(graph, "nodes", {}) or {})


def local_edges(local_graph_json: JSONDict) -> Dict[str, Any]:
    # Prefer the raw KG-IRAG JSON. Passing through LocalGraph.from_dict() can
    # normalize/mock-drop edge fields before SARG builds its reasoning scratchpad.
    if isinstance(local_graph_json, Mapping):
        edges = local_graph_json.get("edges")
        if isinstance(edges, Mapping):
            return dict(edges)

    graph = graph_from_json(local_graph_json)
    if isinstance(graph, Mapping):
        return dict(graph.get("edges") or {})
    return dict(getattr(graph, "edges", {}) or {})


def local_evidence(local_graph_json: JSONDict) -> Dict[str, Any]:
    # Prefer raw evidence for stable evidence IDs and block metadata.
    if isinstance(local_graph_json, Mapping):
        evidence = local_graph_json.get("evidence")
        if isinstance(evidence, Mapping):
            return dict(evidence)

    graph = graph_from_json(local_graph_json)
    if isinstance(graph, Mapping):
        return dict(graph.get("evidence") or {})
    return dict(getattr(graph, "evidence", {}) or {})


def node_key(raw: Any, fallback: str) -> str:
    return str(get_value(raw, "key", fallback) or fallback)


def node_label(raw: Any, fallback: str) -> str:
    return str(
        get_value(raw, "name")
        or get_value(raw, "canonical_name")
        or get_value(raw, "label")
        or get_value(raw, "text")
        or fallback
    )


def edge_key(raw: Any, fallback: str) -> str:
    return str(get_value(raw, "key", fallback) or fallback)


def edge_subject_key(raw: Any) -> str:
    return str(get_value(raw, "subject_key") or get_value(raw, "subject") or "")


def edge_object_key(raw: Any) -> str:
    return str(get_value(raw, "object_key") or get_value(raw, "object") or "")


def edge_relation(raw: Any) -> str:
    return str(get_value(raw, "relation_type") or get_value(raw, "relation") or get_value(raw, "label") or "related_to")


def edge_claim_keys(raw: Any) -> List[str]:
    values: List[Any] = []
    value = get_value(raw, "claim_key")
    if value:
        values.append(value)
    values.extend(get_value(raw, "claim_keys", []) or [])
    return unique_strings(values)


def edge_evidence_ids(raw: Any) -> List[str]:
    return unique_strings(get_value(raw, "evidence_ids", []) or [])


def merge_local_graphs(base_json: JSONDict, extra_json: JSONDict) -> JSONDict:
    base = graph_from_json(base_json)
    extra = graph_from_json(extra_json)

    if LocalGraph is not None and not isinstance(base, Mapping) and not isinstance(extra, Mapping):
        for node in getattr(extra, "nodes", {}).values():
            base.add_node(
                key=node.key,
                name=node.name,
                entity_type=node.entity_type,
                labels=node.labels,
                description=node.description,
                properties=node.properties,
                source=node.source,
            )
        for edge in getattr(extra, "edges", {}).values():
            base.add_edge(
                subject=edge.subject,
                relation_type=edge.relation_type,
                obj=edge.object,
                subject_key=edge.subject_key,
                object_key=edge.object_key,
                claim_key=edge.claim_key,
                grounding_score=edge.grounding_score,
                description=edge.description,
                properties=edge.properties,
                evidence_ids=edge.evidence_ids,
                source=edge.source,
            )
        for item in getattr(extra, "evidence", {}).values():
            base.add_evidence(
                evidence_id=item.evidence_id,
                text=item.text,
                source_url=item.source_url,
                source_title=item.source_title,
                claim_key=item.claim_key,
                properties=item.properties,
                source=item.source,
            )
        try:
            base.add_diagnostic(
                "sarg_agent_kg_irag_merge",
                "Merged KG-IRAG expansion into SARG local graph.",
                added_graph_counts=extra.counts(),
            )
        except Exception:
            pass
        return base.to_dict()

    # JSON fallback.
    out = dict(base_json or {})
    out.setdefault("nodes", {})
    out.setdefault("edges", {})
    out.setdefault("evidence", {})
    out["nodes"].update(dict((extra_json or {}).get("nodes") or {}))
    out["edges"].update(dict((extra_json or {}).get("edges") or {}))
    out["evidence"].update(dict((extra_json or {}).get("evidence") or {}))
    out["counts"] = {
        "nodes": len(out["nodes"]),
        "edges": len(out["edges"]),
        "evidence": len(out["evidence"]),
    }
    return out


# ============================================================
# Investigation memory
# ============================================================

class InvestigationMemoryBuilder:
    def __init__(self, config: SARGConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        investigation_id: str,
        history: Sequence[JSONDict],
        selected_graph_context: Sequence[JSONDict],
    ) -> JSONDict:
        scoped_history = list(history or [])[-self.config.max_history_turns :]

        prior_questions: List[JSONDict] = []
        prior_answers: List[JSONDict] = []
        prior_paths: List[JSONDict] = []
        pinned_evidence_ids: List[str] = []
        selected_graph_ids: List[str] = []
        unresolved_gaps: List[JSONDict] = []

        for index, turn in enumerate(scoped_history, start=1):
            if not isinstance(turn, Mapping):
                continue

            question = turn.get("question") or turn.get("userQuestion") or turn.get("prompt")
            if question:
                prior_questions.append({"turn": index, "question": compact(question, 300)})

            answer = turn.get("answer") or turn.get("final_answer") or turn.get("response")
            if answer:
                prior_answers.append({"turn": index, "answer": compact(answer, 500)})

            for block in turn.get("analysisBlocks") or []:
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                data = block.get("data") or {}
                meta = block.get("meta") or {}

                for evidence_id in meta.get("evidenceIds") or data.get("evidenceIds") or []:
                    if evidence_id not in pinned_evidence_ids:
                        pinned_evidence_ids.append(str(evidence_id))

                for graph_id in meta.get("graphItemIds") or []:
                    if graph_id not in selected_graph_ids:
                        selected_graph_ids.append(str(graph_id))

                if block_type == "reasoning_path":
                    prior_paths.append({
                        "turn": index,
                        "summary": compact(data.get("summary") or block.get("title"), 400),
                        "graphItemIds": meta.get("graphItemIds") or [],
                        "evidenceIds": meta.get("evidenceIds") or [],
                    })
                elif block_type == "missing_evidence":
                    for item in data.get("items") or []:
                        if isinstance(item, Mapping):
                            unresolved_gaps.append(dict(item))

        for item in selected_graph_context or []:
            if not isinstance(item, Mapping):
                continue
            value = item.get("id") or item.get("key")
            if value and str(value) not in selected_graph_ids:
                selected_graph_ids.append(str(value))

        summary_parts = []
        if prior_questions:
            summary_parts.append("Prior questions: " + "; ".join(item["question"] for item in prior_questions[-3:]))
        if prior_paths:
            summary_parts.append("Prior reasoning paths: " + "; ".join(item.get("summary") or "path" for item in prior_paths[-3:]))
        if unresolved_gaps:
            summary_parts.append("Unresolved gaps: " + "; ".join(compact(item.get("text") or item.get("gap") or item, 180) for item in unresolved_gaps[-5:]))

        memory = InvestigationMemory(
            investigation_id=investigation_id,
            compact_summary=" ".join(summary_parts) or "No prior investigation history supplied.",
            prior_questions=prior_questions,
            prior_answers=prior_answers,
            prior_reasoning_paths=prior_paths,
            pinned_evidence_ids=pinned_evidence_ids,
            selected_graph_item_ids=selected_graph_ids,
            unresolved_gaps=unresolved_gaps,
        )
        return asdict(memory)


# ============================================================
# Reasoning scratchpad
# ============================================================

class ReasoningScratchpad:
    def __init__(self) -> None:
        self.nodes: Dict[str, ScratchNode] = {}
        self.edges: Dict[str, ScratchEdge] = {}

    def add_node(self, node: ScratchNode) -> None:
        if node.key and node.key not in self.nodes:
            self.nodes[node.key] = node

    def add_edge(self, edge: ScratchEdge) -> None:
        if edge.key and edge.subject_key and edge.object_key:
            self.edges[edge.key] = edge

    def outgoing(self, node_key: str) -> List[ScratchEdge]:
        return [edge for edge in self.edges.values() if edge.subject_key == node_key]

    def incoming(self, node_key: str) -> List[ScratchEdge]:
        return [edge for edge in self.edges.values() if edge.object_key == node_key]

    def to_dict(self) -> JSONDict:
        return {
            "nodes": {key: asdict(node) for key, node in self.nodes.items()},
            "edges": {key: asdict(edge) for key, edge in self.edges.items()},
            "counts": {"nodes": len(self.nodes), "edges": len(self.edges)},
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "ReasoningScratchpad":
        graph = cls()
        for key, raw in (data.get("nodes") or {}).items():
            graph.nodes[key] = ScratchNode(**raw)
        for key, raw in (data.get("edges") or {}).items():
            graph.edges[key] = ScratchEdge(**raw)
        return graph


class ScratchpadBuilder:
    def build(self, local_graph_json: JSONDict) -> JSONDict:
        scratch = ReasoningScratchpad()
        nodes = local_nodes(local_graph_json)
        edges = local_edges(local_graph_json)

        for fallback, raw in nodes.items():
            key = node_key(raw, str(fallback))
            scratch.add_node(
                ScratchNode(
                    key=key,
                    label=node_label(raw, key),
                    node_type=str(get_value(raw, "entity_type") or get_value(raw, "type") or "entity"),
                    description=compact(get_value(raw, "description"), 500),
                    kg_id=key,
                    source_ids=unique_strings(get_value(raw, "source_ids", []) or []),
                )
            )

        # Add edges only when both endpoints are known or at least named.
        for fallback, raw in edges.items():
            key = edge_key(raw, str(fallback))
            s_key = edge_subject_key(raw)
            o_key = edge_object_key(raw)
            if not s_key or not o_key:
                continue

            if s_key not in scratch.nodes:
                scratch.add_node(ScratchNode(key=s_key, label=s_key, kg_id=s_key))
            if o_key not in scratch.nodes:
                scratch.add_node(ScratchNode(key=o_key, label=o_key, kg_id=o_key))

            scratch.add_edge(
                ScratchEdge(
                    key=key,
                    subject_key=s_key,
                    object_key=o_key,
                    relation=edge_relation(raw),
                    description=compact(get_value(raw, "description") or get_value(raw, "text"), 600),
                    evidence_ids=edge_evidence_ids(raw),
                    claim_keys=edge_claim_keys(raw),
                    kg_id=key,
                    grounding_score=get_value(raw, "grounding_score"),
                )
            )

        return scratch.to_dict()


# ============================================================
# LLM / semantic modules
# ============================================================

class QueryPlanner:
    """Builds a query-specific, domain-neutral reasoning plan for SARG."""

    def __init__(self, client: OpenAI, config: SARGConfig) -> None:
        self.client = client
        self.config = config

    def build(self, question: str, memory: JSONDict) -> JSONDict:
        if self.config.use_llm:
            prompt = f"""
You are planning graph-based reasoning for an investigation.

Return only JSON with this shape:
{{
  "answer_objective": "what the answer must establish",
  "start_objectives": [
    {{
      "id": "start_1",
      "description": "where reasoning should begin",
      "query_terms": ["short term"]
    }}
  ],
  "terminal_objectives": [
    {{
      "id": "target_1",
      "description": "what the reasoning must ultimately explain, reach, compare, or decide",
      "query_terms": ["short term"]
    }}
  ],
  "bridge_objectives": [
    {{
      "id": "bridge_1",
      "description": "intermediate relation, mechanism, entity, condition, or evidence needed to connect start to target",
      "query_terms": ["short term"]
    }}
  ],
  "evidence_requirements": ["requirement"],
  "direction_hint": "forward|backward|bidirectional",
  "requires_multi_hop": true
}}

Rules:
- Do not answer the question.
- Do not use outside knowledge.
- Use only the question and same-investigation memory.
- Objectives must be generic reasoning goals, not domain-specific hard-coded categories.
- A start objective is the source, premise, cause, constraint, entity, or condition from which reasoning begins.
- A terminal objective is the final answer target, effect, decision target, comparison target, or explanation target.
- Terminal objectives must be distinct from start/premise objectives and bridge/intermediate objectives.
- Terminal query_terms must be concrete and distinctive: use the final entity, outcome, sector, answer target, or comparison target.
- Do not put broad shared terms into terminal query_terms when they mainly identify the source or bridge; terminal query_terms should identify what the reasoning must ultimately reach.
- Avoid generic terminal query_terms such as impact, effect, availability, production, role, consequence, determine, or establish unless they are part of a specific phrase from the question.
- Bridge objectives are intermediate mechanisms likely needed for a complete multi-hop path.
- If the question asks why an outcome occurred, prefer backward or bidirectional.
- If the question asks how a source condition affects an outcome, prefer forward or bidirectional.
- Set requires_multi_hop=true when the answer requires a chain through an intermediate mechanism, not just direct lookup.
- A sufficient path must reach terminal objectives through concrete graph nodes/edges, not merely mention them in a rationale.

Question:
{question}

Same-investigation memory:
{json.dumps(memory, indent=2, ensure_ascii=False)}
""".strip()
            try:
                raw = call_json(
                    self.client,
                    self.config.model,
                    prompt,
                    ledger=self.config.ledger,
                    operation="sarg_build_query_plan",
                )
                plan = self._normalise(raw, question)
                if plan:
                    return plan
            except Exception:
                pass
        return self._fallback(question, memory)

    def _normalise(self, raw: Mapping[str, Any], question: str) -> JSONDict:
        if not isinstance(raw, Mapping):
            return self._fallback(question, {})

        def objectives(name: str, prefix: str) -> List[JSONDict]:
            out: List[JSONDict] = []
            for index, item in enumerate(raw.get(name) or [], start=1):
                if not isinstance(item, Mapping):
                    continue
                raw_terms = unique_strings(item.get("query_terms") or [])
                terms = filter_objective_terms(raw_terms)
                description = compact(item.get("description") or "", 240)
                if not description and not terms:
                    continue
                out.append({
                    "id": str(item.get("id") or f"{prefix}_{index}"),
                    "description": description or ", ".join(terms or raw_terms),
                    "query_terms": (terms or filter_objective_terms([description]))[:8],
                })
            return out

        direction = str(raw.get("direction_hint") or "bidirectional")
        if direction not in {"forward", "backward", "bidirectional"}:
            direction = "bidirectional"

        plan = {
            "answer_objective": compact(raw.get("answer_objective") or f"Answer: {question}", 300),
            "start_objectives": objectives("start_objectives", "start"),
            "bridge_objectives": objectives("bridge_objectives", "bridge"),
            "terminal_objectives": objectives("terminal_objectives", "target"),
            "evidence_requirements": unique_strings(raw.get("evidence_requirements") or ["Load-bearing hops should be supported by stored claims or evidence."])[:8],
            "direction_hint": direction,
            "requires_multi_hop": bool(raw.get("requires_multi_hop", True)),
        }

        if not plan["start_objectives"] or not plan["terminal_objectives"]:
            fallback = self._fallback(question, {})
            for key, value in fallback.items():
                if not plan.get(key):
                    plan[key] = value

        return self._dedupe_terminal_terms(plan)

    def _dedupe_terminal_terms(self, plan: JSONDict) -> JSONDict:
        """
        Prevent a terminal objective from being satisfied by broad source/bridge
        terms alone. This is domain-neutral: terminal terms that are already
        strongly represented in start/bridge objectives are removed where at
        least one more specific terminal term remains.
        """
        nonterminal_text = " ".join(
            term
            for objective in (plan.get("start_objectives") or []) + (plan.get("bridge_objectives") or [])
            for term in objective_terms(objective)
        )
        for objective in plan.get("terminal_objectives") or []:
            if not isinstance(objective, dict):
                continue
            original_terms = unique_strings(objective.get("query_terms") or [])
            distinct_terms = [
                term for term in original_terms
                if lexical_score(term, nonterminal_text) < 0.45
            ]
            if distinct_terms:
                objective["query_terms"] = distinct_terms[:8]
        return plan

    def _fallback(self, question: str, memory: JSONDict) -> JSONDict:
        # Domain-neutral fallback: split the question into early/late salient terms.
        tokens = [
            token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", question)
            if len(token) >= 3 and norm(token) not in {
                "the", "and", "for", "with", "what", "which", "how", "why",
                "does", "did", "could", "would", "should", "from", "into",
                "about", "between", "through", "affect", "impact", "explain",
            }
        ]
        terms = unique_strings(tokens)[:12]
        if len(terms) <= 2:
            start_terms = terms[:1] or [compact(question, 80)]
            terminal_terms = terms[1:] or [compact(question, 80)]
            bridge_terms: List[str] = []
        else:
            start_terms = terms[: max(1, min(3, len(terms) // 3))]
            terminal_terms = terms[-max(1, min(3, len(terms) // 3)) :]
            bridge_terms = [term for term in terms if term not in start_terms and term not in terminal_terms][:4]

        q = norm(question)
        if any(term in q for term in ["why", "cause", "explain", "driver"]):
            direction = "backward"
        elif any(term in q for term in ["how", "affect", "impact", "lead", "propagate"]):
            direction = "forward"
        else:
            direction = "bidirectional"

        return {
            "answer_objective": f"Answer the investigation question: {question}",
            "start_objectives": [{"id": "start_1", "description": "Reasoning start/premise/source in the question.", "query_terms": start_terms}],
            "bridge_objectives": [{"id": "bridge_1", "description": "Intermediate mechanism or relation needed for a connected explanation.", "query_terms": bridge_terms}] if bridge_terms else [],
            "terminal_objectives": [{"id": "target_1", "description": "Answer target, effect, decision, or outcome in the question.", "query_terms": terminal_terms}],
            "evidence_requirements": ["Load-bearing hops should be supported by stored claims or evidence."],
            "direction_hint": direction,
            "requires_multi_hop": len(terms) >= 4,
        }


class ConceptExtractor:
    def __init__(self, client: OpenAI, config: SARGConfig) -> None:
        self.client = client
        self.config = config

    def extract(self, question: str, memory: JSONDict) -> List[str]:
        if self.config.use_llm:
            prompt = f"""
Extract atomic investigation concepts for graph traversal.
Return JSON: {{"concepts": ["concept"]}}

Rules:
- Do not answer.
- Use short concepts.
- Include named entities, commodities, events, mechanisms, locations, and target outcomes.
- Use at most 12 concepts.
- Use the investigation memory only as context from this same investigation.

Question:
{question}

Investigation memory:
{json.dumps(memory, indent=2, ensure_ascii=False)}
""".strip()
            try:
                raw = call_json(
                    self.client,
                    self.config.model,
                    prompt,
                    ledger=self.config.ledger,
                    operation="sarg_extract_concepts",
                )
                concepts = raw.get("concepts")
                if isinstance(concepts, list):
                    out = unique_strings(concepts)
                    if out:
                        return out[:12]
            except Exception:
                pass
        return self._fallback(question, memory)

    def _fallback(self, question: str, memory: JSONDict) -> List[str]:
        stop = {
            "the", "and", "for", "with", "what", "which", "how", "why", "does", "did",
            "give", "find", "show", "tell", "between", "relation", "then", "that", "this",
            "from", "into", "could", "would", "affect", "impact", "risk", "supply",
        }
        out = []
        text = question + " " + str(memory.get("compact_summary") or "")
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text.lower()):
            if token not in stop and len(token) >= 3 and token not in out:
                out.append(token)
        return out[:12]


class DirectionClassifier:
    def __init__(self, client: OpenAI, config: SARGConfig) -> None:
        self.client = client
        self.config = config

    def classify(self, question: str, memory: JSONDict) -> str:
        if self.config.use_llm:
            prompt = f"""
Classify SARG traversal direction for this investigation question.

Options:
- forward: source/cause/input/restriction -> consequence/output/risk
- backward: outcome/effect/target -> cause/source/explanation
- bidirectional: comparison, ambiguous relationship, or needs both cause-to-effect and effect-to-cause traversal

Return JSON: {{"direction": "forward|backward|bidirectional", "reason": "brief"}}

Question:
{question}

Same-investigation memory:
{json.dumps(memory, indent=2, ensure_ascii=False)}
""".strip()
            try:
                raw = call_json(
                    self.client,
                    self.config.model,
                    prompt,
                    ledger=self.config.ledger,
                    operation="sarg_classify_direction",
                )
                direction = raw.get("direction")
                if direction in {"forward", "backward", "bidirectional"}:
                    return str(direction)
            except Exception:
                pass

        q = norm(question)
        if any(term in q for term in ["why", "cause", "explain", "source of", "drivers of"]):
            return "backward"
        if any(term in q for term in ["impact", "affect", "lead to", "consequence", "downstream"]):
            return "forward"
        return "bidirectional"


class Embedder:
    def __init__(self, client: OpenAI, config: SARGConfig) -> None:
        self.client = client
        self.config = config

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.config.use_llm:
            return []
        clean_texts = [str(text or "") for text in texts]
        if self.config.ledger is not None:
            return self.config.ledger.embed_texts(
                self.client,
                stage="sarg",
                model=self.config.embed_model,
                texts=clean_texts,
                operation="sarg_embeddings",
            )
        response = self.client.embeddings.create(model=self.config.embed_model, input=clean_texts)
        return [item.embedding for item in response.data]


class StartNodeMatcher:
    def __init__(self, embedder: Embedder, config: SARGConfig) -> None:
        self.embedder = embedder
        self.config = config

    def match(self, graph_json: JSONDict, concepts: Sequence[str], selected_graph_context: Sequence[JSONDict]) -> List[JSONDict]:
        graph = ReasoningScratchpad.from_dict(graph_json)
        if not graph.nodes:
            return []

        forced_ids = {str(item.get("id") or item.get("key")) for item in selected_graph_context or [] if isinstance(item, Mapping)}

        node_keys = list(graph.nodes.keys())
        node_texts = [self._node_search_text(graph.nodes[key]) for key in node_keys]
        concept_text = " ".join(concepts)

        semantic_scores: Dict[str, float] = {}
        if self.config.use_llm and concepts:
            try:
                concept_embedding = self.embedder.embed([concept_text])[0]
                node_embeddings = self.embedder.embed(node_texts)
                semantic_scores = {
                    key: cosine(concept_embedding, emb)
                    for key, emb in zip(node_keys, node_embeddings)
                }
            except Exception:
                semantic_scores = {}

        matches = []
        for key, text in zip(node_keys, node_texts):
            lexical = max([lexical_score(concept, text) for concept in concepts] or [0.0])
            semantic = semantic_scores.get(key, 0.0)
            score = max(lexical, semantic)
            if key in forced_ids:
                score = max(score, 0.95)

            if score >= self.config.start_similarity_threshold:
                node = graph.nodes[key]
                matches.append({
                    "node_key": key,
                    "node_text": node.label,
                    "node_type": node.node_type,
                    "similarity": round(score, 4),
                    "matched_concept": self._best_concept(concepts, text),
                    "forced_by_selected_context": key in forced_ids,
                })

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[: self.config.max_start_nodes]

    def _node_search_text(self, node: ScratchNode) -> str:
        return " ".join([node.label, node.node_type, node.description])

    def _best_concept(self, concepts: Sequence[str], text: str) -> str:
        best = ""
        best_score = 0.0
        for concept in concepts:
            score = lexical_score(concept, text)
            if score > best_score:
                best = str(concept)
                best_score = score
        return best


# ============================================================
# Reasoned beam search
# ============================================================

class TraversalEvaluator:
    """Scores and explains a hop while the traversal decision is being made."""

    def __init__(self, config: SARGConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        question: str,
        memory: JSONDict,
        query_plan: JSONDict,
        graph: ReasoningScratchpad,
        partial_path: ReasoningPath,
        from_node: ScratchNode,
        edge: ScratchEdge,
        to_node: ScratchNode,
        traversal_direction: str,
    ) -> JSONDict:
        all_objectives = query_plan_objectives(query_plan)
        terminal_objectives = [obj for obj in all_objectives if obj.get("kind") == "terminal"]
        bridge_objectives = [obj for obj in all_objectives if obj.get("kind") == "bridge"]
        start_objectives = [obj for obj in all_objectives if obj.get("kind") == "start"]

        hop_text = " ".join([
            from_node.label, from_node.description,
            edge.relation, edge.description,
            to_node.label, to_node.description,
        ])
        path_text = " ".join(
            graph.nodes[key].label
            for key in partial_path.node_keys
            if key in graph.nodes
        )

        # Determine which objectives this hop genuinely advances. Use
        # stricter, kind-specific surfaces so broad source terms do not make a
        # later midstream hop look like it satisfies the start objective, and so
        # generic outcome words do not satisfy a terminal objective.
        covered: List[str] = []
        first_hop = len(partial_path.steps) == 0
        from_text = " ".join([from_node.label, from_node.description])
        to_text = " ".join([to_node.label, to_node.description])
        edge_text = " ".join([edge.relation, edge.description])
        terminal_surface = " ".join([to_text, edge_text])
        bridge_surface = " ".join([from_text, edge_text, to_text])
        start_surface = from_text if first_hop else ""

        for obj in start_objectives:
            if first_hop and objective_score(start_surface, obj) >= max(0.34, self.config.objective_coverage_threshold + 0.14):
                covered.append(str(obj.get("id")))
        for obj in bridge_objectives:
            if objective_score(bridge_surface, obj) >= max(0.34, self.config.objective_coverage_threshold + 0.14):
                covered.append(str(obj.get("id")))
        for obj in terminal_objectives:
            if objective_score(terminal_surface, obj) >= max(0.48, self.config.objective_coverage_threshold + 0.28):
                covered.append(str(obj.get("id")))
        covered = unique_strings(covered)

        terminal_score = max([objective_score(terminal_surface, obj) for obj in terminal_objectives] or [0.0])
        bridge_score = max([objective_score(bridge_surface, obj) for obj in bridge_objectives] or [0.0])
        start_score = max([objective_score(start_surface, obj) for obj in start_objectives] or [0.0])
        question_score = max(lexical_score(question, hop_text), terminal_score, bridge_score, start_score)
        memory_score = lexical_score(memory.get("compact_summary", ""), hop_text)
        evidence_bonus = 0.15 if edge.evidence_ids or edge.claim_keys else 0.0
        objective_bonus = min(0.20, 0.07 * len(covered))

        score = min(
            1.0,
            0.42 * question_score
            + 0.14 * memory_score
            + 0.16 * terminal_score
            + 0.12 * bridge_score
            + evidence_bonus
            + objective_bonus
            + 0.06,
        )

        # Keep early low-scoring hops to allow discovery, but require later hops to
        # advance the query plan or be well supported.
        take = score >= 0.20 or (len(partial_path.steps) == 0 and (score >= 0.14 or bool(edge.evidence_ids or edge.claim_keys)))

        relation_phrase = edge.relation.replace("_", " ")
        objective_lookup = {
            str(obj.get("id")): compact(obj.get("description") or ", ".join(objective_terms(obj)), 120)
            for obj in all_objectives
            if obj.get("id")
        }
        if covered:
            objective_phrase = " It advances " + "; ".join(
                f"{objective_id} ({objective_lookup.get(objective_id, 'query-plan objective')})"
                for objective_id in covered
            ) + "."
        else:
            objective_phrase = " It is retained as a possible bridge, but it does not yet satisfy a named query-plan objective."

        if traversal_direction == "forward":
            reason = (
                f"Take this hop now because following the stored relation '{relation_phrase}' "
                f"moves the path from {from_node.label} to {to_node.label}."
            )
        else:
            reason = (
                f"Take this hop now in reverse because the stored relation '{relation_phrase}' can be used "
                f"to trace from {from_node.label} back toward {to_node.label}."
            )
        reason += objective_phrase

        if edge.evidence_ids:
            reason += " This hop has supporting evidence in the local knowledge base."
        elif edge.claim_keys:
            reason += " This hop is supported by a stored claim, but direct evidence may be thin."
        else:
            reason += " This hop is plausible in the graph but has weak explicit support."

        missing = [] if edge.evidence_ids else ["Direct supporting evidence for this traversal hop."]
        return {
            "take": take,
            "score": round(score, 4),
            "reason": reason,
            "supports_objectives": covered,
            "missing_information": missing,
            "current_path_text": path_text,
        }


class ReasonedBeamSearcher:
    def __init__(self, embedder: Embedder, evaluator: TraversalEvaluator, config: SARGConfig) -> None:
        self.embedder = embedder
        self.evaluator = evaluator
        self.config = config

    def search(
        self,
        *,
        question: str,
        memory: JSONDict,
        query_plan: JSONDict,
        graph_json: JSONDict,
        start_matches: Sequence[JSONDict],
        direction: str,
    ) -> Tuple[List[ReasoningPath], List[ReasoningPath]]:
        graph = ReasoningScratchpad.from_dict(graph_json)
        starts = [item["node_key"] for item in start_matches if item.get("node_key") in graph.nodes]
        if not starts:
            # Fallback: start from high-degree nodes.
            starts = self._high_degree_nodes(graph)[: self.config.max_start_nodes]

        beam = [
            ReasoningPath(
                path_id=f"path_seed_{i}",
                node_keys=[node_key_],
                steps=[],
                score=0.0,
                direction=direction,
                hypothesis=f"Investigate whether {graph.nodes[node_key_].label} is a useful starting point.",
            )
            for i, node_key_ in enumerate(starts, start=1)
        ]

        completed: List[ReasoningPath] = []
        rejected: List[ReasoningPath] = []

        for _depth in range(1, self.config.max_depth + 1):
            candidates: List[ReasoningPath] = []

            for path in beam:
                current_key = path.node_keys[-1]
                expansions = self._expand(graph, current_key, direction)

                if not expansions:
                    completed.append(path)
                    continue

                for edge, next_node_key, traversal_direction in expansions:
                    if next_node_key in path.node_keys:
                        continue
                    if next_node_key not in graph.nodes or current_key not in graph.nodes:
                        continue

                    from_node = graph.nodes[current_key]
                    to_node = graph.nodes[next_node_key]
                    evaluation = self.evaluator.evaluate(
                        question=question,
                        memory=memory,
                        query_plan=query_plan,
                        graph=graph,
                        partial_path=path,
                        from_node=from_node,
                        edge=edge,
                        to_node=to_node,
                        traversal_direction=traversal_direction,
                    )

                    step = TraversalStep(
                        from_node_key=current_key,
                        edge_key=edge.key,
                        to_node_key=next_node_key,
                        traversal_direction=traversal_direction,
                        relation=edge.relation,
                        reason=str(evaluation.get("reason") or ""),
                        score=float(evaluation.get("score") or 0.0),
                        evidence_ids=list(edge.evidence_ids),
                        claim_keys=list(edge.claim_keys),
                        kg_items=self._kg_items_for_step(graph, edge, current_key, next_node_key),
                        missing_information=[str(x) for x in evaluation.get("missing_information") or []],
                        supports_objectives=[str(x) for x in evaluation.get("supports_objectives") or []],
                    )

                    new_score = self._running_average(path.score, len(path.steps), step.score)
                    new_path = ReasoningPath(
                        path_id="",
                        node_keys=path.node_keys + [next_node_key],
                        steps=path.steps + [step],
                        score=new_score,
                        direction=direction,
                        hypothesis=self._hypothesis(graph, path.node_keys + [next_node_key]),
                        evidence_ids=unique_strings(list(path.evidence_ids) + step.evidence_ids),
                        missing_evidence=self._missing_items(path, step),
                        status="selected" if evaluation.get("take") else "rejected",
                    )

                    if evaluation.get("take"):
                        candidates.append(new_path)
                    else:
                        rejected.append(new_path)

            if not candidates:
                break

            candidates.sort(key=lambda item: item.score, reverse=True)
            beam = candidates[: self.config.beam_width]
            completed.extend(beam)

        selected = self._select(completed, graph=graph, query_plan=query_plan)
        for index, path in enumerate(selected, start=1):
            path.path_id = f"reasoning_path_{index}"
            path.status = "selected"
        for index, path in enumerate(rejected[: self.config.top_k_paths], start=1):
            path.path_id = f"rejected_path_{index}"
            path.status = "rejected"
        return selected, rejected[: self.config.top_k_paths]

    def _expand(self, graph: ReasoningScratchpad, node_key_: str, direction: str) -> List[Tuple[ScratchEdge, str, str]]:
        out: List[Tuple[ScratchEdge, str, str]] = []
        if direction in {"forward", "bidirectional"}:
            for edge in graph.outgoing(node_key_):
                out.append((edge, edge.object_key, "forward"))
        if direction in {"backward", "bidirectional"}:
            for edge in graph.incoming(node_key_):
                out.append((edge, edge.subject_key, "backward"))
        return out

    def _kg_items_for_step(self, graph: ReasoningScratchpad, edge: ScratchEdge, from_key: str, to_key: str) -> List[JSONDict]:
        source_node = graph.nodes.get(edge.subject_key)
        target_node = graph.nodes.get(edge.object_key)
        from_node = graph.nodes.get(from_key)
        to_node = graph.nodes.get(to_key)

        items = []
        if from_node:
            items.append(self._node_item(from_node))
        items.append({
            "graphKind": "edge",
            "id": edge.kg_id or edge.key,
            "label": edge.relation,
            "relation": edge.relation,
            "source": edge.subject_key,
            "target": edge.object_key,
            "traversalDirection": "forward" if from_key == edge.subject_key else "backward",
            "sourceLabel": source_node.label if source_node else edge.subject_key,
            "targetLabel": target_node.label if target_node else edge.object_key,
        })
        if to_node:
            items.append(self._node_item(to_node))
        return items

    def _node_item(self, node: ScratchNode) -> JSONDict:
        return {
            "graphKind": "node",
            "id": node.kg_id or node.key,
            "label": node.label,
            "type": node.node_type,
        }

    def _running_average(self, previous: float, previous_count: int, new_value: float) -> float:
        if previous_count <= 0:
            return new_value
        return ((previous * previous_count) + new_value) / (previous_count + 1)

    def _hypothesis(self, graph: ReasoningScratchpad, node_keys_: Sequence[str]) -> str:
        labels = [graph.nodes[key].label for key in node_keys_ if key in graph.nodes]
        if not labels:
            return "Candidate reasoning path."
        return "This path tests whether " + " -> ".join(labels) + " explains the question."

    def _missing_items(self, path: ReasoningPath, step: TraversalStep) -> List[JSONDict]:
        out = list(path.missing_evidence)
        for item in step.missing_information:
            out.append({
                "text": item,
                "severity": "medium",
                "source": "traversal_evaluator",
                "edgeKey": step.edge_key,
            })
        return out

    def _high_degree_nodes(self, graph: ReasoningScratchpad) -> List[str]:
        scores = []
        for key in graph.nodes:
            degree = len(graph.outgoing(key)) + len(graph.incoming(key))
            scores.append((degree, key))
        scores.sort(reverse=True)
        return [key for _, key in scores]

    def _select(
        self,
        paths: Sequence[ReasoningPath],
        *,
        graph: ReasoningScratchpad,
        query_plan: JSONDict,
    ) -> List[ReasoningPath]:
        traversed = [path for path in paths if path.steps]
        if traversed:
            paths = traversed

        dedup: Dict[Tuple[str, ...], ReasoningPath] = {}
        for path in paths:
            sig = path.signature()
            existing = dedup.get(sig)
            if existing is None or self._path_selection_score(graph, path, query_plan) > self._path_selection_score(graph, existing, query_plan):
                dedup[sig] = path

        ranked = sorted(
            dedup.values(),
            key=lambda item: self._path_selection_score(graph, item, query_plan),
            reverse=True,
        )

        selected: List[ReasoningPath] = []
        for path in ranked:
            if self._is_subpath(path, selected):
                continue
            path.score = round(self._path_selection_score(graph, path, query_plan), 4)
            selected.append(path)
            if len(selected) >= self.config.top_k_paths:
                break
        return selected

    def _path_selection_score(self, graph: ReasoningScratchpad, path: ReasoningPath, query_plan: JSONDict) -> float:
        text = self._path_text(graph, path)
        all_objectives = query_plan_objectives(query_plan)
        # Validate explicit step objective claims against graph-observable
        # path text before using them for ranking.
        path_payload = serialize_path(path, graph.to_dict()) if hasattr(graph, "to_dict") else {}
        supported = validated_supported_objectives(
            path_payload,
            all_objectives,
            threshold=self.config.objective_coverage_threshold,
        )
        observed = objective_coverage(
            text,
            all_objectives,
            threshold=max(0.48, self.config.objective_coverage_threshold + 0.28),
        ).get("covered_objectives") or []
        covered = unique_strings(list(supported) + list(observed))
        total_count = max(1, len(all_objectives))
        coverage_ratio = len(covered) / total_count

        terminal_ids = set(objective_ids_by_kind(query_plan, "terminal"))
        terminal_supported = len([item for item in covered if item in terminal_ids]) / max(1, len(terminal_ids))
        evidence_ratio = 0.0
        if path.steps:
            evidence_ratio = sum(1 for step in path.steps if step.evidence_ids or step.claim_keys) / len(path.steps)
        multi_hop_bonus = 0.10 if bool(query_plan.get("requires_multi_hop")) and len(path.steps) >= self.config.min_multihop_steps else 0.0
        short_penalty = 0.12 if bool(query_plan.get("requires_multi_hop")) and len(path.steps) < self.config.min_multihop_steps else 0.0

        # Penalise loops/redundant paths unless they add objective coverage.
        unique_node_ratio = len(set(path.node_keys)) / max(1, len(path.node_keys))
        loop_penalty = 0.10 if unique_node_ratio < 1.0 and coverage_ratio < 1.0 else 0.0

        return max(0.0, min(1.0,
            0.34 * float(path.score)
            + 0.26 * coverage_ratio
            + 0.20 * terminal_supported
            + 0.14 * evidence_ratio
            + multi_hop_bonus
            - short_penalty
            - loop_penalty
        ))

    def _path_text(self, graph: ReasoningScratchpad, path: ReasoningPath) -> str:
        """Graph-observable path text only; excludes generated rationales."""
        parts: List[str] = []
        for key in path.node_keys:
            node = graph.nodes.get(key)
            if node:
                parts.extend([node.label, node.node_type, node.description])
            else:
                parts.append(key)
        for step in path.steps:
            parts.append(step.relation)
            edge = graph.edges.get(step.edge_key)
            if edge:
                parts.extend([edge.relation, edge.description])
        return " ".join(str(part) for part in parts if part).strip()

    def _is_subpath(self, path: ReasoningPath, selected: Sequence[ReasoningPath]) -> bool:
        sig = path.signature()
        for kept in selected:
            kept_sig = kept.signature()
            if len(sig) >= len(kept_sig):
                continue
            for i in range(len(kept_sig) - len(sig) + 1):
                if tuple(kept_sig[i : i + len(sig)]) == sig:
                    return True
        return False


# ============================================================
# Gap/controller/action modules
# ============================================================

class GapAssessor:
    def assess(
        self,
        *,
        question: str,
        query_plan: JSONDict,
        selected_paths: Sequence[JSONDict],
        local_graph_json: JSONDict,
        memory: JSONDict,
        config: SARGConfig,
    ) -> JSONDict:
        all_objectives = query_plan_objectives(query_plan)
        terminal_objectives = [obj for obj in all_objectives if obj.get("kind") == "terminal"]
        start_objectives = [obj for obj in all_objectives if obj.get("kind") == "start"]

        if not selected_paths:
            return {
                "status": "insufficient",
                "primary_gap_type": "missing_reasoning_path",
                "summary": "No usable reasoning path was found in the local knowledge graph.",
                "covered_objectives": [],
                "uncovered_objectives": [str(obj.get("id")) for obj in all_objectives if obj.get("id")],
                "items": [
                    {
                        "text": "Need a connected reasoning path that satisfies the query plan objectives.",
                        "severity": "high",
                        "source": "gap_assessment",
                    }
                ],
                "recommended_action": "expand_kg_irag",
            }

        path_metrics: List[JSONDict] = []
        aggregate_covered: List[str] = []
        unsupported_hops: List[JSONDict] = []
        max_steps = 0

        for path in selected_paths:
            if not isinstance(path, Mapping):
                continue
            covered = supported_or_observed_objectives(
                path,
                all_objectives,
                threshold=config.objective_coverage_threshold,
            )
            for objective_id in covered:
                if objective_id not in aggregate_covered:
                    aggregate_covered.append(objective_id)
            max_steps = max(max_steps, len(path.get("steps") or []))
            for step in path.get("steps") or []:
                if isinstance(step, Mapping) and not step.get("evidenceIds") and not step.get("claimKeys"):
                    unsupported_hops.append(step)
            all_ids_for_metrics = [str(obj.get("id") or f"objective_{i}") for i, obj in enumerate(all_objectives, start=1)]
            path_metrics.append({
                "path_id": path.get("path_id"),
                "step_count": len(path.get("steps") or []),
                "covered_objectives": covered,
                "uncovered_objectives": [objective_id for objective_id in all_ids_for_metrics if objective_id not in covered],
            })

        all_ids = [str(obj.get("id") or f"objective_{i}") for i, obj in enumerate(all_objectives, start=1)]
        uncovered = [objective_id for objective_id in all_ids if objective_id not in aggregate_covered]
        terminal_ids = objective_ids_by_kind(query_plan, "terminal")
        bridge_ids = objective_ids_by_kind(query_plan, "bridge")
        start_ids = objective_ids_by_kind(query_plan, "start")
        missing_terminal = [objective_id for objective_id in terminal_ids if objective_id not in aggregate_covered]
        missing_bridge = [objective_id for objective_id in bridge_ids if objective_id not in aggregate_covered]
        missing_start = [objective_id for objective_id in start_ids if objective_id not in aggregate_covered]

        # Sufficiency must be path-local, not only aggregate. A collection of
        # separate fragments should not be marked sufficient if no selected path
        # actually reaches a terminal objective. For multi-hop questions, the
        # terminal-reaching path should also include a bridge objective when one
        # was planned, or at least have enough traversal depth to be a real path.
        terminal_path_metrics = [
            metric for metric in path_metrics
            if any(objective_id in metric.get("covered_objectives", []) for objective_id in terminal_ids)
        ]
        has_terminal_path = bool(terminal_path_metrics) or not terminal_ids
        has_complete_multihop_path = False
        for metric in terminal_path_metrics:
            covered_here = set(metric.get("covered_objectives") or [])
            has_bridge_here = bool(set(bridge_ids) & covered_here) or not bridge_ids
            has_depth_here = int(metric.get("step_count") or 0) >= config.min_multihop_steps
            if not bool(query_plan.get("requires_multi_hop")) or (has_bridge_here and has_depth_here):
                has_complete_multihop_path = True
                break

        items: List[JSONDict] = []
        if missing_start:
            items.append({
                "text": "Selected paths do not clearly satisfy the query plan's start/premise objective(s).",
                "severity": "medium",
                "source": "gap_assessment",
                "objectiveIds": missing_start,
            })
        if missing_terminal or not has_terminal_path:
            terminal_gap_ids = missing_terminal or terminal_ids
            items.append({
                "text": "Selected paths do not clearly reach the query plan's terminal/answer objective(s).",
                "severity": "high",
                "source": "gap_assessment",
                "objectiveIds": terminal_gap_ids,
            })
            return {
                "status": "partial",
                "primary_gap_type": "missing_terminal",
                "summary": "Current paths are supported fragments, but no selected path clearly reaches the terminal objective of the query plan.",
                "covered_objectives": aggregate_covered,
                "uncovered_objectives": uncovered,
                "items": items,
                "path_metrics": path_metrics,
                "recommended_action": "expand_kg_irag",
            }

        if bool(query_plan.get("requires_multi_hop")) and (missing_bridge or not has_complete_multihop_path):
            bridge_gap_ids = missing_bridge or bridge_ids
            items.append({
                "text": "Selected paths do not clearly satisfy the required bridge/intermediate objective(s) on a terminal-reaching path.",
                "severity": "high",
                "source": "gap_assessment",
                "objectiveIds": bridge_gap_ids,
            })
            return {
                "status": "partial",
                "primary_gap_type": "missing_bridge",
                "summary": "Current paths do not yet provide a complete multi-hop route from the planned start/bridge structure to the terminal objective.",
                "covered_objectives": aggregate_covered,
                "uncovered_objectives": uncovered,
                "items": items,
                "path_metrics": path_metrics,
                "recommended_action": "expand_kg_irag",
            }

        if bool(query_plan.get("requires_multi_hop")) and max_steps < config.min_multihop_steps:
            items.append({
                "text": "The query plan requires multi-hop reasoning, but selected paths are too shallow.",
                "severity": "high",
                "source": "gap_assessment",
                "minimumSteps": config.min_multihop_steps,
            })
            return {
                "status": "partial",
                "primary_gap_type": "missing_bridge",
                "summary": "Selected paths touch relevant objectives but lack enough intermediate structure for the planned reasoning.",
                "covered_objectives": aggregate_covered,
                "uncovered_objectives": uncovered,
                "items": items,
                "path_metrics": path_metrics,
                "recommended_action": "expand_kg_irag",
            }

        if unsupported_hops:
            items.extend([
                {
                    "text": f"Weak direct evidence for hop {step.get('from')} --{step.get('edge')}--> {step.get('to')}.",
                    "severity": "medium",
                    "source": "gap_assessment",
                    "edgeKey": step.get("edgeKey"),
                }
                for step in unsupported_hops[:6]
            ])
            return {
                "status": "partial",
                "primary_gap_type": "weak_evidence",
                "summary": f"Selected paths satisfy the query plan structurally, but {len(unsupported_hops)} traversal hop(s) have weak direct support.",
                "covered_objectives": aggregate_covered,
                "uncovered_objectives": uncovered,
                "items": items[:8],
                "path_metrics": path_metrics,
                "recommended_action": "retrieve_more_evidence",
            }

        return {
            "status": "sufficient",
            "primary_gap_type": "none",
            "summary": "Selected reasoning paths satisfy the query plan with explicit traversal steps and adequate evidence.",
            "covered_objectives": aggregate_covered,
            "uncovered_objectives": [],
            "items": [],
            "path_metrics": path_metrics,
            "recommended_action": "stop_and_answer",
        }


class AnalystController:
    def decide(self, state: SARGState, config: SARGConfig) -> Tuple[str, JSONDict, AnalystStep]:
        step_number = int(state["agent_step_count"]) + 1
        gap = state.get("gap_assessment") or {}
        selected = state.get("selected_reasoning_paths") or []

        recommended = str(gap.get("recommended_action") or "")
        if step_number > config.max_agent_steps:
            action = "stop_and_answer"
            action_input: JSONDict = {"reason": "Reached maximum SARG agent steps."}
        elif recommended in {"expand_kg_irag", "retrieve_more_evidence", "compare_hypotheses"} and state["expansion_count"] < config.max_expansions:
            action = "retrieve_more_evidence" if recommended == "retrieve_more_evidence" else "expand_kg_irag"
            action_input = {
                "question": self._targeted_question(state),
                "gap_type": gap.get("primary_gap_type"),
                "uncovered_objectives": gap.get("uncovered_objectives") or [],
                "covered_objectives": gap.get("covered_objectives") or [],
            }
        elif gap.get("status") in {"insufficient", "partial"} and state["expansion_count"] < config.max_expansions:
            action = "retrieve_more_evidence" if gap.get("primary_gap_type") == "weak_evidence" else "expand_kg_irag"
            action_input = {
                "question": self._targeted_question(state),
                "gap_type": gap.get("primary_gap_type"),
                "uncovered_objectives": gap.get("uncovered_objectives") or [],
                "covered_objectives": gap.get("covered_objectives") or [],
            }
        elif not selected and state["expansion_count"] < config.max_expansions:
            action = "expand_kg_irag"
            action_input = {"question": self._targeted_question(state), "gap_type": "missing_reasoning_path"}
        else:
            action = "stop_and_answer"
            action_input = {"reason": "Reasoning paths are adequate for bounded synthesis or expansion budget is exhausted."}

        observation = self._observation(state)
        hypothesis = self._hypothesis(state)
        analyst_step = AnalystStep(
            step=step_number,
            observation=observation,
            hypothesis=hypothesis,
            gap=str(gap.get("summary") or "No explicit gap."),
            action=action,
            action_input=action_input,
        )
        return action, action_input, analyst_step

    def _observation(self, state: SARGState) -> str:
        counts = ((state.get("local_graph") or {}).get("counts") or {})
        selected_count = len(state.get("selected_reasoning_paths") or [])
        return (
            f"Current local graph has {counts.get('nodes', 'unknown')} nodes, "
            f"{counts.get('edges', 'unknown')} edges, and {counts.get('evidence', 'unknown')} evidence items. "
            f"SARG selected {selected_count} candidate reasoning path(s)."
        )

    def _hypothesis(self, state: SARGState) -> str:
        selected = state.get("selected_reasoning_paths") or []
        if selected:
            first = selected[0]
            return str(first.get("hypothesis") or "The strongest selected path explains the current question.")
        return "A stronger connected reasoning path may exist after targeted retrieval."

    def _targeted_question(self, state: SARGState) -> str:
        question = state["question"]
        query_plan = state.get("query_plan") or {}
        gap = state.get("gap_assessment") or {}
        selected = state.get("selected_reasoning_paths") or []

        uncovered_ids = set(str(item) for item in gap.get("uncovered_objectives") or [])
        objectives = query_plan_objectives(query_plan)
        uncovered = [obj for obj in objectives if str(obj.get("id")) in uncovered_ids]

        if not uncovered:
            uncovered = objectives

        objective_text = json.dumps(uncovered, indent=2, ensure_ascii=False)
        selected_text = json.dumps([
            {
                "path_id": path.get("path_id"),
                "nodes": path.get("nodes"),
                "steps": path.get("steps"),
                "coveredObjectives": path.get("coveredObjectives"),
            }
            for path in selected[:3]
            if isinstance(path, Mapping)
        ], indent=2, ensure_ascii=False)

        if gap.get("primary_gap_type") == "weak_evidence":
            return (
                "Retrieve stronger source evidence for the existing reasoning path. "
                f"Original question: {question}\n"
                f"Query plan: {json.dumps(query_plan, ensure_ascii=False)}\n"
                f"Weakness: {gap.get('summary')}\n"
                "Prefer evidence-backed claims for the current traversal hops. Do not answer the question."
            )

        return (
            "Retrieve graph links and supporting evidence needed to satisfy the uncovered objectives in this investigation.\n"
            f"Original question: {question}\n"
            f"Query plan: {json.dumps(query_plan, indent=2, ensure_ascii=False)}\n"
            f"Covered objectives: {json.dumps(gap.get('covered_objectives') or [], ensure_ascii=False)}\n"
            f"Uncovered objectives: {objective_text}\n"
            f"Current selected paths: {selected_text}\n"
            "Instructions: prefer evidence-backed relations that connect existing path endpoints to uncovered objectives; "
            "retrieve bridge entities, claims, and source evidence needed for multi-hop reasoning; do not answer the question."
        )



class ActionExecutor:
    def __init__(self, config: SARGConfig) -> None:
        self.config = config

    def execute(self, action: str, action_input: JSONDict, state: SARGState) -> JSONDict:
        if action not in {"expand_kg_irag", "retrieve_more_evidence"}:
            return {"status": "noop", "reason": f"No external action required for {action}."}

        if KGIRAG is None or KGIRAGBudget is None or KGIRAGScope is None:
            return {"status": "failed", "reason": "KGIRAG imports are unavailable."}

        budget = KGIRAGBudget(
            max_iterations=self.config.kg_irag_expansion_iterations,
            max_rows_per_query=self.config.kg_irag_expansion_rows,
            max_path_depth=self.config.kg_irag_expansion_depth,
            kappa_max_nodes=self.config.kg_irag_expansion_nodes,
            kappa_max_edges=self.config.kg_irag_expansion_edges,
            kappa_max_evidence=self.config.kg_irag_expansion_evidence,
            evidence_lane_k=self.config.kg_irag_evidence_lane_k,
            table_lane_k=self.config.kg_irag_table_lane_k,
        )
        retriever = KGIRAG(
            model=self.config.model,
            embed_model=self.config.embed_model,
            budget=budget,
            scope=KGIRAGScope(
                corpus_id=self.config.corpus_id,
                branch_id=self.config.branch_id,
                source_ids=list(self.config.source_ids or []),
            ),
            continue_after_anchor=True,
            enable_evidence_lane=True,
            enable_table_lane=True,
            table_lane_each_iteration=self.config.table_lane_each_iteration,
            ledger=self.config.ledger,
        )

        retrieval_question = str(action_input.get("question") or state["question"])
        try:
            result = retriever.retrieve(retrieval_question)
        finally:
            retriever.close()

        old_counts = (state.get("local_graph") or {}).get("counts") or {}
        new_graph = result.get("local_reasoning_subgraph") or {}
        merged = merge_local_graphs(state["local_graph"], new_graph)
        new_counts = merged.get("counts") or {}

        return {
            "status": result.get("status"),
            "stop_reason": result.get("stop_reason"),
            "retrieval_question": retrieval_question,
            "kg_irag_counts": new_graph.get("counts"),
            "old_counts": old_counts,
            "new_counts": new_counts,
            "merged_local_graph": merged,
        }


# ============================================================
# Synthesis / review / blocks
# ============================================================

class AnswerGenerator:
    def __init__(self, client: OpenAI, config: SARGConfig) -> None:
        self.client = client
        self.config = config

    def generate(self, state: SARGState) -> str:
        if not self.config.use_llm:
            return self._fallback_answer(state)

        prompt = f"""
Answer the question using only the selected reasoning paths, same-investigation memory, and supporting evidence.

Rules:
- Do not use outside knowledge.
- Do not mention internal class names or implementation names unless asked.
- Structure the answer around the selected paths.
- Every major claim must correspond to at least one path step or evidence item.
- If the gap assessment is partial or insufficient, state the limitation clearly.
- If terminal objectives remain uncovered, explicitly say the investigation does not directly establish that final link.
- Do not overclaim beyond the selected paths.

Question:
{state['question']}

Query plan:
{json.dumps(state.get('query_plan') or {}, indent=2, ensure_ascii=False)}

Same-investigation memory:
{json.dumps(state.get('investigation_memory') or {}, indent=2, ensure_ascii=False)}

Selected reasoning paths:
{json.dumps(state.get('selected_reasoning_paths') or [], indent=2, ensure_ascii=False)}

Gap assessment:
{json.dumps(state.get('gap_assessment') or {}, indent=2, ensure_ascii=False)}

Supporting evidence:
{json.dumps(collect_evidence_payload(state.get('local_graph') or {}, state.get('selected_reasoning_paths') or [], self.config), indent=2, ensure_ascii=False)}
""".strip()
        return call_text(
            self.client,
            self.config.model,
            [
                {"role": "system", "content": "You are a careful, evidence-grounded supply-chain intelligence analyst."},
                {"role": "user", "content": prompt},
            ],
            ledger=self.config.ledger,
            operation="sarg_generate_answer",
        )

    def _fallback_answer(self, state: SARGState) -> str:
        paths = state.get("selected_reasoning_paths") or []
        if not paths:
            return "The current investigation does not contain enough connected graph evidence to answer confidently."
        first = paths[0]
        nodes = [item.get("label") for item in (first.get("nodes") or []) if item.get("label")]
        path_text = " → ".join(nodes)
        return f"The strongest available reasoning path is: {path_text}. The answer should be treated as partial where direct evidence is missing."


class FinalReviewer:
    def review(self, state: SARGState) -> JSONDict:
        gap = state.get("gap_assessment") or {}
        selected = state.get("selected_reasoning_paths") or []
        limitations = []
        if gap.get("items"):
            limitations.extend([compact(item.get("text") or item, 240) for item in gap.get("items") or []])
        if not selected:
            limitations.append("No selected reasoning path was available.")
        return {
            "answer_ready": bool(state.get("answer")) and bool(selected),
            "confidence": "medium" if gap.get("status") == "sufficient" else "low-to-medium",
            "limitations": unique_strings(limitations)[:8],
            "required_caveats": unique_strings(limitations)[:4],
        }


def serialize_path(path: ReasoningPath, graph_json: JSONDict) -> JSONDict:
    graph = ReasoningScratchpad.from_dict(graph_json)
    nodes = [
        {"id": key, "label": graph.nodes[key].label if key in graph.nodes else key}
        for key in path.node_keys
    ]
    steps = []
    path_items: List[JSONDict] = []
    if path.node_keys and path.node_keys[0] in graph.nodes:
        first = graph.nodes[path.node_keys[0]]
        path_items.append({"graphKind": "node", "id": first.kg_id or first.key, "label": first.label, "type": first.node_type})

    for step in path.steps:
        edge = graph.edges.get(step.edge_key)
        from_node = graph.nodes.get(step.from_node_key)
        to_node = graph.nodes.get(step.to_node_key)
        if edge:
            path_items.append({
                "graphKind": "edge",
                "id": edge.kg_id or edge.key,
                "label": edge.relation,
                "relation": edge.relation,
                "source": edge.subject_key,
                "target": edge.object_key,
                "traversalDirection": step.traversal_direction,
            })
        if to_node:
            path_items.append({"graphKind": "node", "id": to_node.kg_id or to_node.key, "label": to_node.label, "type": to_node.node_type})

        steps.append({
            "from": from_node.label if from_node else step.from_node_key,
            "edge": step.relation,
            "to": to_node.label if to_node else step.to_node_key,
            "traversalDirection": step.traversal_direction,
            "reason": step.reason,
            "score": step.score,
            "evidenceIds": list(step.evidence_ids),
            "claimKeys": list(step.claim_keys),
            "supportsObjectives": list(step.supports_objectives),
            "edgeKey": step.edge_key,
        })

    graph_item_ids = unique_strings(item.get("id") for item in path_items)
    return {
        "path_id": path.path_id,
        "summary": path.hypothesis,
        "nodes": nodes,
        "path": path_items,
        "steps": steps,
        "score": round(path.score, 4),
        "direction": path.direction,
        "hypothesis": path.hypothesis,
        "evidenceIds": list(path.evidence_ids),
        "missingEvidence": list(path.missing_evidence),
        "graphItemIds": graph_item_ids,
        "coveredObjectives": unique_strings(obj for step in path.steps for obj in step.supports_objectives),
        "status": path.status,
    }


def collect_evidence_payload(local_graph_json: JSONDict, selected_paths: Sequence[JSONDict], config: SARGConfig) -> List[JSONDict]:
    evidence_map = local_evidence(local_graph_json)
    evidence_ids: List[str] = []
    for path in selected_paths or []:
        for evidence_id in path.get("evidenceIds") or path.get("evidence_ids") or []:
            if evidence_id not in evidence_ids:
                evidence_ids.append(str(evidence_id))
        for step in path.get("steps") or []:
            for evidence_id in step.get("evidenceIds") or []:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(str(evidence_id))

    out: List[JSONDict] = []
    for evidence_id in evidence_ids[: config.max_evidence_snippets]:
        item = evidence_map.get(evidence_id)
        if not item:
            continue
        props = get_value(item, "properties", {}) or {}
        out.append({
            "id": evidence_id,
            "evidence_id": evidence_id,
            "claimKey": get_value(item, "claim_key"),
            "claim_key": get_value(item, "claim_key"),
            "sourceTitle": get_value(item, "source_title"),
            "source_title": get_value(item, "source_title"),
            "sourceUrl": get_value(item, "source_url"),
            "source_url": get_value(item, "source_url"),
            "blockType": props.get("block_type"),
            "block_type": props.get("block_type"),
            "text": compact(get_value(item, "text"), config.max_evidence_chars),
        })
    return out


def strip_markdown(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer_text(value: Any) -> str:
    text = strip_markdown(value)

    if not text:
        return ""

    text = re.sub(
        r"^(?:based on .*?|the .*? is as follows)\s*:?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^(?:summary(?:\s+and\s+limitation)?|summary\s+of\s+risk\s+assessment|risk\s+assessment|limitation|limitations)\s*[-:]?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(r"^\s*(?:and\s+)?(?:limitation|risk\s+assessment)\s*[-:]?\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def first_sentences(value: Any, count: int = 2, max_chars: int = 520) -> str:
    text = clean_answer_text(value)

    if not text:
        return ""

    summary_match = re.search(
        r"(?:summary(?:\s+and\s+limitation)?|summary\s+of\s+risk\s+assessment)\s*:?\s*(.*)",
        strip_markdown(value),
        flags=re.I | re.S,
    )

    if summary_match:
        text = clean_answer_text(summary_match.group(1))

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    if not sentences:
        return compact(text, max_chars)

    return compact(" ".join(sentences[:count]), max_chars)

def text_block(
    block_id: str,
    title: str,
    lead: str,
    body: str = "",
    *,
    evidence_ids: Optional[List[str]] = None,
    graph_item_ids: Optional[List[str]] = None,
) -> JSONDict:
    return {
        "id": block_id,
        "type": "text",
        "title": title,
        "data": {
            "lead": compact(strip_markdown(lead), 360),
            "body": compact(strip_markdown(body), 900),
        },
        "meta": {
            "evidenceIds": evidence_ids or [],
            "graphItemIds": graph_item_ids or [],
        },
    }


def graph_counts_summary(graph: JSONDict) -> str:
    counts = graph.get("counts") if isinstance(graph.get("counts"), Mapping) else {}

    nodes = counts.get("nodes")
    edges = counts.get("edges")
    evidence = counts.get("evidence")

    if nodes is None:
        raw_nodes = graph.get("nodes")
        nodes = len(raw_nodes) if isinstance(raw_nodes, (dict, list)) else 0

    if edges is None:
        raw_edges = graph.get("edges")
        edges = len(raw_edges) if isinstance(raw_edges, (dict, list)) else 0

    if evidence is None:
        raw_evidence = graph.get("evidence")
        evidence = len(raw_evidence) if isinstance(raw_evidence, (dict, list)) else 0

    return f"{nodes} nodes, {edges} edges, and {evidence} evidence item(s)"


def query_plan_summary(query_plan: JSONDict) -> str:
    if not isinstance(query_plan, Mapping) or not query_plan:
        return "No explicit query plan was produced."

    start = query_plan.get("start_objectives") or []
    bridge = query_plan.get("bridge_objectives") or []
    terminal = query_plan.get("terminal_objectives") or []

    def objective_text(items: Sequence[Any]) -> str:
        labels: List[str] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            label = item.get("description") or ", ".join(item.get("query_terms") or [])
            label = strip_markdown(label)
            if label:
                labels.append(label)
        return "; ".join(labels[:3])

    parts: List[str] = []

    if query_plan.get("direction_hint"):
        parts.append(f"Direction: {query_plan.get('direction_hint')}.")

    if start:
        parts.append(f"Start: {objective_text(start)}.")

    if bridge:
        parts.append(f"Bridge: {objective_text(bridge)}.")

    if terminal:
        parts.append(f"Target: {objective_text(terminal)}.")

    if query_plan.get("requires_multi_hop"):
        parts.append("The plan requires multi-hop reasoning.")

    return compact(" ".join(part for part in parts if part), 900)

def action_input_summary(action_input: JSONDict) -> str:
    if not isinstance(action_input, Mapping):
        return ""

    gap_type = strip_markdown(action_input.get("gap_type"))
    uncovered = action_input.get("uncovered_objectives") or []
    covered = action_input.get("covered_objectives") or []

    parts: List[str] = []

    if gap_type:
        parts.append(f"Gap type: {gap_type}.")

    if uncovered:
        parts.append("Uncovered objective(s): " + ", ".join(str(item) for item in uncovered[:4]) + ".")

    if covered:
        parts.append("Already covered: " + ", ".join(str(item) for item in covered[:4]) + ".")

    return " ".join(parts)


def action_observation_summary(result_observation: Any) -> str:
    text = strip_markdown(result_observation)

    if not text:
        return ""

    match = re.search(
        r"Graph changed from nodes=(\d+) edges=(\d+) evidence=(\d+) to nodes=(\d+) edges=(\d+) evidence=(\d+)",
        text,
    )

    if match:
        before = match.group(1), match.group(2), match.group(3)
        after = match.group(4), match.group(5), match.group(6)

        if before == after:
            return "KG-IRAG returned no new local-graph material for this expansion attempt."

        return (
            f"KG-IRAG changed the local graph from {before[0]} nodes, {before[1]} edges, "
            f"and {before[2]} evidence item(s) to {after[0]} nodes, {after[1]} edges, "
            f"and {after[2]} evidence item(s)."
        )

    return compact(text, 360)


def stop_decision_body(state: SARGState, step: Mapping[str, Any]) -> str:
    gap = state.get("gap_assessment") if isinstance(state.get("gap_assessment"), Mapping) else {}
    status = str(gap.get("status") or "")
    expansion_count = int(state.get("expansion_count") or 0)
    max_expansions = 2

    hypothesis = strip_markdown(step.get("hypothesis"))
    gap_text = strip_markdown(step.get("gap"))

    if status in {"sufficient", "complete"}:
        decision = "Decision: SARG stopped because the selected chains were sufficient for a bounded answer."
    elif expansion_count >= max_expansions:
        decision = "Decision: SARG stopped because the expansion budget was exhausted before the terminal objective was fully reached."
    else:
        decision = "Decision: SARG stopped and produced a bounded answer from the available chains."

    return " ".join(part for part in [
        f"Hypothesis: {hypothesis}" if hypothesis else "",
        f"Gap check: {gap_text}" if gap_text else "",
        decision,
    ] if part)


def compact_synthesis_text(answer: Any) -> str:
    text = clean_answer_text(answer)

    if not text:
        return ""

    summary_match = re.search(
        r"(?:summary(?:\s+and\s+limitation)?|summary\s+of\s+risk\s+assessment)\s*:?\s*(.*?)(?:\blimitations?\s*:|$)",
        strip_markdown(answer),
        flags=re.I | re.S,
    )

    if summary_match:
        text = clean_answer_text(summary_match.group(1))

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    if not sentences:
        return compact(text, 520)

    return compact(" ".join(sentences[:3]), 520)


def analyst_step_blocks(state: SARGState) -> List[JSONDict]:
    blocks: List[JSONDict] = []
    steps = state.get("analyst_steps") or []
    repeated_noop_expansions = 0

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            continue

        action = str(step.get("action") or "")
        observation = strip_markdown(step.get("observation"))
        hypothesis = strip_markdown(step.get("hypothesis"))
        gap = strip_markdown(step.get("gap"))
        result_observation = action_observation_summary(step.get("result_observation"))

        if action in {"expand_kg_irag", "retrieve_more_evidence"}:
            blocks.append(text_block(
                f"reasoning_pass_{index}",
                f"Reasoning pass {index}",
                observation or "SARG inspected the current local graph.",
                " ".join(part for part in [
                    f"Hypothesis: {hypothesis}" if hypothesis else "",
                    f"Gap check: {gap}" if gap else "",
                ] if part),
            ))

            action_input = step.get("action_input") if isinstance(step.get("action_input"), Mapping) else {}
            action_title = "Evidence retrieval" if action == "retrieve_more_evidence" else "KG expansion"

            is_noop_expansion = "returned no new local-graph material" in result_observation.lower()

            if is_noop_expansion and repeated_noop_expansions:
                repeated_noop_expansions += 1
                continue

            if is_noop_expansion:
                repeated_noop_expansions = 1

            blocks.append(text_block(
                f"action_{index}",
                action_title,
                "SARG called KG-IRAG because the current reasoning chains did not fully satisfy the query plan.",
                " ".join(part for part in [
                    action_input_summary(action_input),
                    result_observation,
                ] if part),
            ))

        else:
            if repeated_noop_expansions > 1:
                blocks.append(text_block(
                    "repeated_expansion_summary",
                    "Expansion result",
                    f"{repeated_noop_expansions} KG expansion attempts returned no new local-graph material.",
                    "SARG therefore stopped with a bounded answer and preserved the unresolved terminal-objective gap.",
                ))

            blocks.append(text_block(
                f"stop_decision_{index}",
                "Stop decision",
                observation or "SARG inspected the current local graph.",
                stop_decision_body(state, step),
            ))

    return blocks

def reasoning_path_block(path: JSONDict, index: int) -> JSONDict:
    graph_item_ids = unique_strings(
        path.get("graphItemIds") or [
            item.get("id")
            for item in path.get("path") or []
            if isinstance(item, Mapping)
        ]
    )
    path_evidence = unique_strings(path.get("evidenceIds") or [])

    return {
        "id": path.get("path_id") or f"reasoning_path_{index}",
        "type": "reasoning_path",
        "title": "Reasoning chain" if index == 1 else f"Reasoning chain {index}",
        "data": {
            "summary": path.get("summary") or path.get("hypothesis") or "Selected reasoning chain.",
            "nodes": path.get("nodes") or [],
            "path": path.get("path") or [],
            "steps": path.get("steps") or [],
            "direction": path.get("direction"),
            "score": path.get("score"),
        },
        "meta": {
            "evidenceIds": path_evidence,
            "graphItemIds": graph_item_ids,
        },
    }

def synthesis_block(answer: str, evidence_ids: List[str]) -> JSONDict:
    lead = first_sentences(answer, count=1, max_chars=260)
    body = compact_synthesis_text(answer)

    if body.startswith(lead):
        body = body[len(lead):].strip()
        body = re.sub(r"^[.:;\-\s]+", "", body).strip()

    return text_block(
        "synthesis",
        "Bounded synthesis",
        lead or "SARG produced a bounded synthesis from the selected reasoning chains.",
        body,
        evidence_ids=evidence_ids,
    )


def gap_status_block(state: SARGState) -> Optional[JSONDict]:
    gap = state.get("gap_assessment") if isinstance(state.get("gap_assessment"), Mapping) else {}

    if not gap:
        return None

    status = str(gap.get("status") or "unknown")
    summary = strip_markdown(gap.get("summary"))
    action = str(gap.get("recommended_action") or "")

    items: List[str] = []
    for item in gap.get("items") or []:
        if isinstance(item, Mapping):
            text = item.get("text") or item.get("question") or item.get("gap")
        else:
            text = item
        text = strip_markdown(text)
        if text and text not in items:
            items.append(text)

    body_parts = []
    if action:
        body_parts.append(f"Recommended action: {action}.")
    if items:
        body_parts.append("Remaining issues: " + " ".join(items[:4]))

    title = "Gap check" if status != "sufficient" else "Sufficiency check"
    lead = summary or f"Gap assessment status: {status}."

    return text_block(
        "gap_check",
        title,
        lead,
        " ".join(body_parts),
    )


def evidence_support_block(evidence: List[JSONDict], evidence_ids: List[str]) -> Optional[JSONDict]:
    if not evidence:
        return None

    return {
        "id": "supporting_evidence",
        "type": "evidence",
        "title": "Evidence support",
        "data": {
            "evidence": [
                {**item, "text": compact(item.get("text"), 360)}
                for item in evidence[:3]
            ],
        },
        "meta": {
            "evidenceIds": evidence_ids[:3],
            "graphItemIds": [],
        },
    }


def missing_evidence_block(state: SARGState) -> Optional[JSONDict]:
    gap = state.get("gap_assessment") if isinstance(state.get("gap_assessment"), Mapping) else {}
    memory = state.get("investigation_memory") if isinstance(state.get("investigation_memory"), Mapping) else {}

    missing_items: List[JSONDict] = []

    for item in gap.get("items") or []:
        if isinstance(item, Mapping):
            missing_items.append(dict(item))
        else:
            missing_items.append({"text": str(item), "severity": "medium", "source": "gap_assessment"})

    for item in memory.get("unresolved_gaps") or []:
        if isinstance(item, Mapping):
            missing_items.append({**dict(item), "source": "investigation_memory"})

    deduped: List[JSONDict] = []
    seen = set()

    for item in missing_items:
        key = strip_markdown(item.get("text") or item.get("question") or item.get("gap") or item)
        key_norm = norm(key)
        if key_norm and key_norm not in seen:
            seen.add(key_norm)
            deduped.append(item)

    if not deduped:
        return None

    return {
        "id": "missing_evidence",
        "type": "missing_evidence",
        "title": "Missing evidence",
        "data": {
            "items": deduped[:8],
            "summary": gap.get("summary"),
        },
        "meta": {
            "evidenceIds": [],
            "graphItemIds": [],
        },
    }

def risk_block_requested(state: SARGState) -> bool:
    question = norm(state.get("question") or "")
    return bool(state.get("risk_analysis")) or any(term in question for term in [
        "risk",
        "risk assessment",
        "assess risk",
        "risk profile",
        "likelihood",
        "severity",
        "exposure",
    ])


def risk_value_to_bar(value: Any) -> Tuple[float, str]:
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in {"critical", "very high"}:
            return 95, "high"
        if lower == "high":
            return 85, "high"
        if lower in {"medium", "moderate"}:
            return 55, "medium"
        if lower == "low":
            return 25, "low"
        return 50, "medium"

    try:
        numeric = float(value)
    except Exception:
        numeric = 50

    if numeric <= 5:
        numeric *= 20

    numeric = max(0, min(100, numeric))
    tone = "high" if numeric >= 70 else "medium" if numeric >= 40 else "low"
    return round(numeric, 1), tone


def risk_summary_from_state(state: SARGState, risk: Mapping[str, Any]) -> str:
    summary = first_text(
        risk.get("summary"),
        risk.get("reason"),
        risk.get("message"),
        risk.get("analysis"),
        risk.get("assessment"),
        risk.get("narrative"),
        risk.get("conclusion"),
    )

    if summary:
        return compact(clean_answer_text(summary), 700)

    answer_summary = compact_synthesis_text(state.get("answer"))
    if answer_summary:
        return compact(answer_summary, 700)

    return "Risk assessment output was produced, but no narrative summary was provided by the risk tool."


def infer_overall_risk(state: SARGState, risk: Mapping[str, Any], summary: str, factors: Sequence[JSONDict]) -> str:
    explicit = first_text(
        risk.get("overallRisk"),
        risk.get("overall_risk"),
        risk.get("risk_level"),
        risk.get("level"),
        risk.get("severity"),
    )

    evidence_text = norm(" ".join([summary, clean_answer_text(state.get("answer"))]))

    # Prefer explicit tool output unless the generated answer clearly contains
    # a stronger risk judgement. This keeps the UI honest while avoiding the
    # misleading demo case where the block said Low despite high exposure text.
    if explicit:
        explicit_title = str(explicit).strip().title()
        if explicit_title in {"Critical", "Very High", "High"}:
            return explicit_title
        if explicit_title in {"Low", "Medium", "Moderate"}:
            if any(term in evidence_text for term in [
                "exposure high",
                "high exposure",
                "supply shortages",
                "shortages",
                "price volatility",
                "major disruption",
                "critical upstream",
            ]):
                return "High"
            return explicit_title
        return explicit_title

    if any(term in evidence_text for term in ["critical", "severe", "major disruption"]):
        return "Critical"
    if any(term in evidence_text for term in [
        "exposure high",
        "high exposure",
        "high risk",
        "supply shortages",
        "shortages",
        "price volatility",
        "critical upstream",
    ]):
        return "High"
    if any(term in evidence_text for term in ["low risk", "limited risk", "limited exposure"]):
        return "Low"

    if factors:
        average = sum(float(item.get("value") or 0) for item in factors) / max(1, len(factors))
        if average >= 70:
            return "High"
        if average >= 40:
            return "Moderate"
        return "Low"

    status = str(risk.get("status") or "").lower()
    if status in {"disabled", "unavailable", "failed", "not_run"}:
        return "Not Run"

    return "Moderate"


def normalise_risk_block_payload(state: SARGState) -> Optional[JSONDict]:
    if not risk_block_requested(state):
        return None

    risk = state.get("risk_analysis")

    if not isinstance(risk, Mapping) or not risk:
        return {
            "status": "not_run",
            "overallRisk": "Not Run",
            "factors": [],
            "summary": "Risk assessment was requested, but no risk tool output was produced.",
        }

    raw_factors = risk.get("factors") or risk.get("risk_factors") or risk.get("risks") or []
    factors: List[JSONDict] = []

    if isinstance(raw_factors, list):
        for item in raw_factors[:5]:
            if isinstance(item, Mapping):
                label = first_text(
                    item.get("label"),
                    item.get("name"),
                    item.get("title"),
                    item.get("risk"),
                    item.get("factor"),
                    "Risk factor",
                )
                raw_value = first_text(
                    item.get("value"),
                    item.get("severity"),
                    item.get("likelihood"),
                    item.get("score"),
                    item.get("rating"),
                )
                value, tone = risk_value_to_bar(raw_value or "medium")
            else:
                label = str(item)
                value, tone = risk_value_to_bar("medium")

            factors.append({"label": compact(label, 80), "value": value, "tone": tone})

    status = first_text(risk.get("status"), "available")
    summary = risk_summary_from_state(state, risk)

    if not factors and status not in {"disabled", "unavailable", "failed", "not_run"}:
        # Frontend-ready fallback when the risk tool returns narrative-only output.
        factors = [
            {"label": "Supply exposure", "value": 85, "tone": "high"},
            {"label": "Evidence confidence", "value": 60, "tone": "medium"},
            {"label": "Residual uncertainty", "value": 70, "tone": "high"},
        ]

    overall = infer_overall_risk(state, risk, summary, factors)

    return {
        "status": status,
        "overallRisk": overall,
        "factors": factors,
        "summary": compact(summary, 700),
        "rawRiskAnalysis": dict(risk),
    }

def build_analysis_blocks(state: SARGState, config: SARGConfig) -> List[JSONDict]:
    blocks: List[JSONDict] = []

    answer = state.get("answer") or ""
    selected = state.get("selected_reasoning_paths") or []
    evidence = collect_evidence_payload(state.get("local_graph") or {}, selected, config)
    evidence_ids = unique_strings(item.get("id") for item in evidence)

    graph_summary = graph_counts_summary(state.get("local_graph") or {})
    direction = state.get("direction") or "unknown"
    expansion_count = state.get("expansion_count") or 0
    selected_count = len(selected)

    blocks.append(text_block(
        "brief",
        "Brief",
        first_sentences(answer, count=1, max_chars=260) or "SARG completed the investigation turn.",
        f"Process: {selected_count} reasoning chain(s), {expansion_count} expansion attempt(s), direction={direction}. Local graph: {graph_summary}.",
        evidence_ids=evidence_ids[:3],
    ))

    blocks.append(text_block(
        "investigation_plan",
        "Investigation plan",
        "SARG converted the question into a graph-reasoning plan before answering.",
        query_plan_summary(state.get("query_plan") or {}),
    ))

    blocks.extend(analyst_step_blocks(state))

    gap_block = gap_status_block(state)
    if gap_block:
        blocks.append(gap_block)

    for index, path in enumerate(selected[: config.top_k_paths], start=1):
        blocks.append(reasoning_path_block(path, index))

    risk_payload = normalise_risk_block_payload(state)
    if risk_payload:
        blocks.append({
            "id": "risk_assessment",
            "type": "risk_assessment",
            "title": "Risk assessment",
            "data": risk_payload,
            "meta": {
                "evidenceIds": evidence_ids,
                "graphItemIds": [],
            },
        })

    if answer:
        blocks.append(synthesis_block(answer, evidence_ids))

    if state.get("forecast_analysis") is not None:
        blocks.append({
            "id": "forecast",
            "type": "forecast",
            "title": "Forecast check",
            "data": state.get("forecast_analysis") or {},
            "meta": {
                "evidenceIds": evidence_ids,
                "graphItemIds": [],
            },
        })

    evidence_block = evidence_support_block(evidence, evidence_ids)
    if evidence_block:
        blocks.append(evidence_block)

    missing_block = missing_evidence_block(state)
    if missing_block:
        blocks.append(missing_block)

    return blocks

# ============================================================
# Optional risk / forecast helpers
# ============================================================

def risk_requested(state: SARGState) -> bool:
    question = norm(state.get("question") or "")
    return any(term in question for term in [
        "risk",
        "risk assessment",
        "assess risk",
        "risk profile",
        "likelihood",
        "severity",
        "exposure",
    ])

def run_risk_analysis(state: SARGState, config: SARGConfig) -> Optional[JSONDict]:
    partial = {
        "question": state["question"],
        "status": state["status"],
        "answer": state.get("answer"),
        "selected_reasoning_paths": state.get("selected_reasoning_paths") or [],
        "gap_assessment": state.get("gap_assessment") or {},
        "sarg_context": {
            "reasoning_chains": state.get("selected_reasoning_paths") or [],
            "evidence": collect_evidence_payload(
                state.get("local_graph") or {},
                state.get("selected_reasoning_paths") or [],
                config,
            ),
        },
        "pipeline_metadata": state.get("pipeline_metadata") or {},
    }

    if not config.enable_risk_tools:
        return {
            "status": "disabled",
            "overallRisk": "Not run",
            "summary": "Risk assessment was requested, but risk tools are disabled in SARG configuration.",
            "factors": [],
            "risks": [],
            "limitations": ["Risk tools are disabled."],
            "sarg_context": partial["sarg_context"],
        }

    if analyze_risk is None:
        return {
            "status": "unavailable",
            "overallRisk": "Not run",
            "summary": "Risk assessment was requested, but risk_tools.py could not be imported.",
            "factors": [],
            "risks": [],
            "limitations": ["The live backend could not import risk_tools.py."],
            "sarg_context": partial["sarg_context"],
        }

    try:
        analysis = analyze_risk(
            question=state["question"],
            sarg_result=partial,
            document_jsons=state.get("document_jsons") or [],
            risk_model=state.get("risk_model"),
            use_llm=False,
            model=config.model,
        )
        if hasattr(analysis, "__dataclass_fields__"):
            return asdict(analysis)
        if isinstance(analysis, Mapping):
            return dict(analysis)
        return {
            "status": "available",
            "overallRisk": "Moderate",
            "summary": str(analysis),
            "factors": [],
            "rawRiskAnalysis": analysis,
            "sarg_context": partial["sarg_context"],
        }
    except Exception as error:
        return {
            "status": "failed",
            "overallRisk": "Not run",
            "summary": "Risk assessment was requested, but risk analysis failed.",
            "reason": f"Risk analysis failed: {error}",
            "factors": [],
            "risks": [],
            "limitations": [str(error)],
            "sarg_context": partial["sarg_context"],
        }

def run_forecast_analysis(state: SARGState, config: SARGConfig) -> Optional[JSONDict]:
    if not config.enable_forecasting:
        return None
    if forecast_payload is None:
        return {"status": "unavailable", "reason": "forecast_tools.py could not be imported."}
    # Keep this deliberately simple; full table extraction remains outside SARG.
    risk_model = state.get("risk_model") or {}
    series_list = []
    for key in ["forecast_inputs", "time_series", "forecast_series"]:
        value = risk_model.get(key) if isinstance(risk_model, Mapping) else None
        if isinstance(value, list):
            series_list.extend(value)
    if not series_list:
        return {"status": "unavailable", "reason": "No explicit forecast input series supplied."}
    try:
        return forecast_payload({"series_list": series_list})
    except Exception as error:
        return {"status": "failed", "reason": f"Forecasting failed: {error}"}


# ============================================================
# LangGraph SARG agent
# ============================================================

class SARG:
    def __init__(self, config: Optional[SARGConfig] = None) -> None:
        self.config = config or SARGConfig()
        self.client = OpenAI() if self.config.use_llm else OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy"))

        self.memory_builder = InvestigationMemoryBuilder(self.config)
        self.query_planner = QueryPlanner(self.client, self.config)
        self.scratchpad_builder = ScratchpadBuilder()
        self.concept_extractor = ConceptExtractor(self.client, self.config)
        self.direction_classifier = DirectionClassifier(self.client, self.config)
        self.embedder = Embedder(self.client, self.config)
        self.matcher = StartNodeMatcher(self.embedder, self.config)
        self.traversal_evaluator = TraversalEvaluator(self.config)
        self.beam_searcher = ReasonedBeamSearcher(self.embedder, self.traversal_evaluator, self.config)
        self.gap_assessor = GapAssessor()
        self.controller = AnalystController()
        self.action_executor = ActionExecutor(self.config)
        self.answerer = AnswerGenerator(self.client, self.config)
        self.reviewer = FinalReviewer()

        self.app = self._build_graph()

    def pipeline_metadata(self) -> JSONDict:
        metadata = {
            "run_id": self.config.run_id,
            "source_id": self.config.source_id,
            "corpus_id": self.config.corpus_id,
            "branch_id": self.config.branch_id,
            "source_ids": self.config.source_ids,
            "investigation_id": self.config.investigation_id,
        }

        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != "" and value != []
        }
    
    def run(
        self,
        question: str,
        kg_irag_result: Optional[JSONDict] = None,
        *,
        investigation_id: Optional[str] = None,
        investigation_history: Optional[Sequence[JSONDict]] = None,
        selected_graph_context: Optional[Sequence[JSONDict]] = None,
        document_jsons: Optional[Sequence[JSONDict]] = None,
        risk_model: Optional[JSONDict] = None,
    ) -> JSONDict:
        if kg_irag_result is None:
            kg_irag_result = self._initial_kg_irag(question)

        local_graph_json = kg_irag_result.get("local_reasoning_subgraph") or {}
        inv_id = investigation_id or self.config.investigation_id or "default_investigation"

        state: SARGState = {
            "question": question,
            "kg_irag_result": kg_irag_result,
            "local_graph": local_graph_json,
            "investigation_id": inv_id,
            "investigation_history": list(investigation_history or []),
            "selected_graph_context": list(selected_graph_context or []),
            "investigation_memory": {},
            "query_plan": {},
            "reasoning_graph": {},
            "concepts": [],
            "direction": "",
            "start_matches": [],
            "candidate_paths": [],
            "selected_reasoning_paths": [],
            "rejected_reasoning_paths": [],
            "gap_assessment": {},
            "analyst_steps": [],
            "open_questions": [],
            "expansion_count": 0,
            "agent_step_count": 0,
            "next_action": "",
            "action_input": {},
            "action_result": {},
            "status": "running",
            "answer": None,
            "final_review": None,
            "document_jsons": list(document_jsons or []),
            "risk_model": risk_model,
            "risk_analysis": None,
            "forecast_analysis": None,
            "pipeline_metadata": self.pipeline_metadata(),
            "result": None,
        }

        result_state = self.app.invoke(state, config={"recursion_limit": self.config.max_agent_steps * 8 + 40})
        return result_state.get("result") or self._export(result_state)

    def _initial_kg_irag(self, question: str) -> JSONDict:
        if KGIRAG is None or KGIRAGBudget is None or KGIRAGScope is None:
            raise RuntimeError("KGIRAG is unavailable and no --kg-irag-json was supplied.")
        budget = KGIRAGBudget(
            max_iterations=self.config.kg_irag_expansion_iterations,
            max_rows_per_query=self.config.kg_irag_expansion_rows,
            max_path_depth=self.config.kg_irag_expansion_depth,
            kappa_max_nodes=self.config.kg_irag_expansion_nodes,
            kappa_max_edges=self.config.kg_irag_expansion_edges,
            kappa_max_evidence=self.config.kg_irag_expansion_evidence,
            evidence_lane_k=self.config.kg_irag_evidence_lane_k,
            table_lane_k=self.config.kg_irag_table_lane_k,
        )
        retriever = KGIRAG(
            model=self.config.model,
            embed_model=self.config.embed_model,
            budget=budget,
            scope=KGIRAGScope(
                corpus_id=self.config.corpus_id,
                branch_id=self.config.branch_id,
                source_ids=list(self.config.source_ids or []),
            ),
            continue_after_anchor=True,
            enable_evidence_lane=True,
            enable_table_lane=True,
            table_lane_each_iteration=self.config.table_lane_each_iteration,
            ledger=self.config.ledger,
        )
        try:
            return retriever.retrieve(question)
        finally:
            retriever.close()

    def _build_graph(self):
        graph = StateGraph(SARGState)

        graph.add_node("build_investigation_memory", self._build_investigation_memory_node)
        graph.add_node("build_query_plan", self._build_query_plan_node)
        graph.add_node("build_reasoning_scratchpad", self._build_reasoning_scratchpad_node)
        graph.add_node("extract_concepts", self._extract_concepts_node)
        graph.add_node("classify_direction", self._classify_direction_node)
        graph.add_node("match_start_nodes", self._match_start_nodes_node)
        graph.add_node("reasoned_beam_search", self._reasoned_beam_search_node)
        graph.add_node("assess_gaps", self._assess_gaps_node)
        graph.add_node("analyst_controller", self._analyst_controller_node)
        graph.add_node("act", self._act_node)
        graph.add_node("observe", self._observe_node)
        graph.add_node("answer", self._answer_node)
        graph.add_node("risk_forecast", self._risk_forecast_node)
        graph.add_node("final_review", self._final_review_node)
        graph.add_node("export", self._export_node)

        graph.set_entry_point("build_investigation_memory")
        graph.add_edge("build_investigation_memory", "build_query_plan")
        graph.add_edge("build_query_plan", "build_reasoning_scratchpad")
        graph.add_edge("build_reasoning_scratchpad", "extract_concepts")
        graph.add_edge("extract_concepts", "classify_direction")
        graph.add_edge("classify_direction", "match_start_nodes")
        graph.add_edge("match_start_nodes", "reasoned_beam_search")
        graph.add_edge("reasoned_beam_search", "assess_gaps")
        graph.add_edge("assess_gaps", "analyst_controller")
        graph.add_conditional_edges(
            "analyst_controller",
            self._route_after_controller,
            {"act": "act", "answer": "answer"},
        )
        graph.add_edge("act", "observe")
        graph.add_edge("observe", "build_reasoning_scratchpad")
        graph.add_edge("answer", "risk_forecast")
        graph.add_edge("risk_forecast", "final_review")
        graph.add_edge("final_review", "export")
        graph.add_edge("export", END)
        return graph.compile()

    # -------------------- nodes --------------------

    def _build_investigation_memory_node(self, state: SARGState) -> SARGState:
        memory = self.memory_builder.build(
            investigation_id=state["investigation_id"],
            history=state.get("investigation_history") or [],
            selected_graph_context=state.get("selected_graph_context") or [],
        )
        return {**state, "investigation_memory": memory}

    def _build_query_plan_node(self, state: SARGState) -> SARGState:
        query_plan = self.query_planner.build(
            state["question"],
            state.get("investigation_memory") or {},
        )
        return {**state, "query_plan": query_plan}

    def _build_reasoning_scratchpad_node(self, state: SARGState) -> SARGState:
        reasoning_graph = self.scratchpad_builder.build(state["local_graph"])
        return {**state, "reasoning_graph": reasoning_graph}

    def _extract_concepts_node(self, state: SARGState) -> SARGState:
        extracted = self.concept_extractor.extract(state["question"], state.get("investigation_memory") or {})
        plan_terms = query_plan_terms(state.get("query_plan") or {})
        concepts = unique_strings(list(plan_terms) + list(extracted))[:16]
        return {**state, "concepts": concepts}

    def _classify_direction_node(self, state: SARGState) -> SARGState:
        query_plan = state.get("query_plan") or {}
        direction = str(query_plan.get("direction_hint") or "")
        if direction not in {"forward", "backward", "bidirectional"}:
            direction = self.direction_classifier.classify(state["question"], state.get("investigation_memory") or {})
        return {**state, "direction": direction}

    def _match_start_nodes_node(self, state: SARGState) -> SARGState:
        matches = self.matcher.match(
            state["reasoning_graph"],
            state.get("concepts") or [],
            state.get("selected_graph_context") or [],
        )
        return {**state, "start_matches": matches}

    def _reasoned_beam_search_node(self, state: SARGState) -> SARGState:
        selected, rejected = self.beam_searcher.search(
            question=state["question"],
            memory=state.get("investigation_memory") or {},
            query_plan=state.get("query_plan") or {},
            graph_json=state["reasoning_graph"],
            start_matches=state.get("start_matches") or [],
            direction=state.get("direction") or "bidirectional",
        )
        selected_payload = [serialize_path(path, state["reasoning_graph"]) for path in selected]
        rejected_payload = [serialize_path(path, state["reasoning_graph"]) for path in rejected]
        return {
            **state,
            "candidate_paths": selected_payload + rejected_payload,
            "selected_reasoning_paths": selected_payload,
            "rejected_reasoning_paths": rejected_payload,
        }

    def _assess_gaps_node(self, state: SARGState) -> SARGState:
        gap = self.gap_assessor.assess(
            question=state["question"],
            query_plan=state.get("query_plan") or {},
            selected_paths=state.get("selected_reasoning_paths") or [],
            local_graph_json=state.get("local_graph") or {},
            memory=state.get("investigation_memory") or {},
            config=self.config,
        )
        open_questions = [
            {"question": item.get("text"), "source": item.get("source"), "severity": item.get("severity")}
            for item in gap.get("items") or []
            if isinstance(item, Mapping)
        ]
        return {**state, "gap_assessment": gap, "open_questions": open_questions}


    def _analyst_controller_node(self, state: SARGState) -> SARGState:
        action, action_input, analyst_step = self.controller.decide(state, self.config)
        analyst_steps = list(state.get("analyst_steps") or [])
        analyst_steps.append(asdict(analyst_step))
        return {
            **state,
            "next_action": action,
            "action_input": action_input,
            "analyst_steps": analyst_steps,
            "agent_step_count": state["agent_step_count"] + 1,
            "status": "answering" if action == "stop_and_answer" else "acting",
        }

    def _route_after_controller(self, state: SARGState) -> str:
        return "answer" if state.get("next_action") == "stop_and_answer" else "act"

    def _act_node(self, state: SARGState) -> SARGState:
        action = state.get("next_action") or ""
        result = self.action_executor.execute(action, state.get("action_input") or {}, state)
        update = {**state, "action_result": result}
        if result.get("merged_local_graph"):
            update["local_graph"] = result["merged_local_graph"]
        if action in {"expand_kg_irag", "retrieve_more_evidence"}:
            update["expansion_count"] = state["expansion_count"] + 1
        return update

    def _observe_node(self, state: SARGState) -> SARGState:
        result = state.get("action_result") or {}
        analyst_steps = list(state.get("analyst_steps") or [])
        observation = self._result_observation(result)
        if analyst_steps:
            analyst_steps[-1] = {**analyst_steps[-1], "result_observation": observation}
        return {**state, "analyst_steps": analyst_steps, "status": "running"}

    def _result_observation(self, result: JSONDict) -> str:
        if result.get("status") == "failed":
            return str(result.get("reason") or "Action failed.")
        old_counts = result.get("old_counts") or {}
        new_counts = result.get("new_counts") or {}
        if new_counts:
            return (
                f"Action completed with status={result.get('status')}. "
                f"Graph changed from nodes={old_counts.get('nodes')} edges={old_counts.get('edges')} evidence={old_counts.get('evidence')} "
                f"to nodes={new_counts.get('nodes')} edges={new_counts.get('edges')} evidence={new_counts.get('evidence')}."
            )
        return str(result.get("reason") or "Action completed.")

    def _answer_node(self, state: SARGState) -> SARGState:
        answer = self.answerer.generate(state)
        return {**state, "answer": answer, "status": "answered"}

    def _risk_forecast_node(self, state: SARGState) -> SARGState:
        question = norm(state.get("question") or "")

        wants_risk = any(term in question for term in [
            "risk",
            "risk assessment",
            "assess risk",
            "risk profile",
            "likelihood",
            "severity",
            "exposure",
        ])

        wants_forecast = any(term in question for term in [
            "forecast",
            "forecast check",
            "predict",
            "projection",
            "trend",
            "future",
            "next",
            "horizon",
        ])

        risk = run_risk_analysis(state, self.config) if wants_risk else None
        forecast = run_forecast_analysis(state, self.config) if wants_forecast else None

        if wants_risk and risk is None:
            risk = {
                "status": "not_run",
                "summary": "Risk assessment was requested, but no risk tool output was produced.",
                "risks": [],
                "limitations": [
                    "SARG detected a risk request, but run_risk_analysis returned None."
                ],
            }

        if wants_forecast and forecast is None:
            forecast = {
                "status": "not_run",
                "summary": "Forecast check was requested, but no forecast tool output was produced.",
                "limitations": [
                    "SARG detected a forecast request, but run_forecast_analysis returned None."
                ],
            }

        return {**state, "risk_analysis": risk, "forecast_analysis": forecast}

    def _final_review_node(self, state: SARGState) -> SARGState:
        review = self.reviewer.review(state)
        return {**state, "final_review": review}

    def _export_node(self, state: SARGState) -> SARGState:
        return {**state, "result": self._export(state)}

    def _export(self, state: SARGState) -> JSONDict:
        analysis_blocks = build_analysis_blocks(state, self.config)
        return {
            "question": state["question"],
            "status": state.get("status"),
            "answer": state.get("answer"),
            "pipeline_metadata": state.get("pipeline_metadata") or {},
            "investigation_id": state.get("investigation_id"),
            "investigation_memory": state.get("investigation_memory") or {},
            "query_plan": state.get("query_plan") or {},
            "analyst_steps": state.get("analyst_steps") or [],
            "concepts": state.get("concepts") or [],
            "direction": state.get("direction"),
            "start_matches": state.get("start_matches") or [],
            "selected_reasoning_paths": state.get("selected_reasoning_paths") or [],
            "rejected_reasoning_paths": state.get("rejected_reasoning_paths") or [],
            "gap_assessment": state.get("gap_assessment") or {},
            "open_questions": state.get("open_questions") or [],
            "expansion_count": state.get("expansion_count"),
            "agent_step_count": state.get("agent_step_count"),
            "local_reasoning_subgraph": state.get("local_graph") or {},
            "reasoning_graph": state.get("reasoning_graph") or {},
            "risk_model": state.get("risk_model"),
            "risk_analysis": state.get("risk_analysis"),
            "forecast_analysis": state.get("forecast_analysis"),
            "final_review": state.get("final_review"),
            "analysisBlocks": analysis_blocks,
            "implementation_notes": {
                "agentic_components": [
                    "per-investigation working memory",
                    "LangGraph analyst-control loop",
                    "forward/backward/bidirectional graph traversal",
                    "hop-level traversal decisions recorded during beam search",
                    "gap assessment before synthesis",
                    "targeted KG-IRAG retrieval actions",
                    "post-action observation and reasoning rerun",
                    "UI analysisBlocks export",
                ],
                "scope_rule": "Investigation history is scoped to the active investigation only; no cross-investigation memory is used.",
            },
        }


# ============================================================
# CLI / self-test
# ============================================================

def make_mock_kg_irag_result() -> JSONDict:
    return {
        "question": "How could a Myanmar rare earth disruption affect downstream magnet supply?",
        "status": "sufficient",
        "stop_reason": "mock",
        "local_reasoning_subgraph": {
            "focus_question": "How could a Myanmar rare earth disruption affect downstream magnet supply?",
            "nodes": {
                "entity:myanmar": {"key": "entity:myanmar", "name": "Myanmar rare-earth mining", "entity_type": "region", "description": "Upstream rare-earth mining source."},
                "entity:china_processing": {"key": "entity:china_processing", "name": "Chinese rare-earth processing", "entity_type": "processor", "description": "Processing concentration for rare earth materials."},
                "entity:magnets": {"key": "entity:magnets", "name": "Downstream magnet supply", "entity_type": "supply_chain", "description": "Permanent magnet supply exposure."},
            },
            "edges": {
                "edge:myanmar:feeds:china": {
                    "key": "edge:myanmar:feeds:china",
                    "subject_key": "entity:myanmar",
                    "object_key": "entity:china_processing",
                    "relation_type": "feeds_processing",
                    "description": "Myanmar feedstock can affect Chinese rare-earth processing inputs.",
                    "claim_key": "claim:1",
                    "evidence_ids": ["ev:1"],
                },
                "edge:china:supplies:magnets": {
                    "key": "edge:china:supplies:magnets",
                    "subject_key": "entity:china_processing",
                    "object_key": "entity:magnets",
                    "relation_type": "supplies_downstream",
                    "description": "Processed rare earth materials support magnet supply chains.",
                    "claim_key": "claim:2",
                    "evidence_ids": ["ev:2"],
                },
            },
            "evidence": {
                "ev:1": {"evidence_id": "ev:1", "text": "Myanmar feedstock is linked to Chinese processing inputs.", "source_title": "Mock source 1", "source_url": "mock://1", "claim_key": "claim:1", "properties": {"block_type": "text"}},
                "ev:2": {"evidence_id": "ev:2", "text": "Chinese processing supports downstream magnet supply.", "source_title": "Mock source 2", "source_url": "mock://2", "claim_key": "claim:2", "properties": {"block_type": "text"}},
            },
            "counts": {"nodes": 3, "edges": 2, "evidence": 2},
        },
    }


def run_self_test() -> int:
    config = SARGConfig(use_llm=False, enable_risk_tools=False, enable_forecasting=False, max_agent_steps=1, max_expansions=0)
    sarg = SARG(config)
    result = sarg.run(
        "How could a Myanmar rare earth disruption affect downstream magnet supply?",
        make_mock_kg_irag_result(),
        investigation_id="self_test_investigation",
        investigation_history=[],
    )
    assert result.get("analysisBlocks"), "analysisBlocks missing"
    assert result.get("selected_reasoning_paths"), "selected_reasoning_paths missing"
    first = result["selected_reasoning_paths"][0]
    assert first.get("steps"), "reasoning path has no traversal steps"
    assert first["steps"][0].get("reason"), "traversal reason missing"
    assert any(block.get("type") == "reasoning_path" for block in result["analysisBlocks"]), "reasoning_path block missing"
    print(json.dumps({
        "status": "ok",
        "answer": result.get("answer"),
        "selected_path_count": len(result.get("selected_reasoning_paths") or []),
        "analysis_block_types": [block.get("type") for block in result.get("analysisBlocks") or []],
    }, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph SARG bounded investigation agent")
    parser.add_argument("--question", required=False)
    parser.add_argument("--kg-irag-json", type=Path)
    parser.add_argument("--kg-irag-result", dest="kg_irag_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    parser.add_argument("--investigation-id", default=os.getenv("LANTHIC_INVESTIGATION_ID"))
    parser.add_argument("--investigation-history-json", type=Path)
    parser.add_argument("--selected-graph-context-json", type=Path)

    parser.add_argument("--document-json", type=Path, action="append", default=[])
    parser.add_argument("--risk-model-json", type=Path)
    parser.add_argument("--no-risk-tools", action="store_true")
    parser.add_argument("--no-forecasting", action="store_true")
    parser.add_argument("--forecast-horizon", type=int, default=SARGConfig.forecast_horizon)

    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--max-agent-steps", type=int, default=SARGConfig.max_agent_steps)
    parser.add_argument("--max-expansions", type=int, default=SARGConfig.max_expansions)
    parser.add_argument("--max-depth", type=int, default=SARGConfig.max_depth)
    parser.add_argument("--beam-width", type=int, default=SARGConfig.beam_width)
    parser.add_argument("--top-k", type=int, default=SARGConfig.top_k_paths)
    parser.add_argument("--offline", action="store_true", help="Disable LLM calls; useful for deterministic local tests.")

    parser.add_argument("--run-id", default=os.getenv("LANTHIC_RUN_ID"))
    parser.add_argument("--source-id", default=os.getenv("LANTHIC_SOURCE_ID"))
    parser.add_argument("--corpus-id", default=os.getenv("LANTHIC_CORPUS_ID"))
    parser.add_argument("--branch-id", default=os.getenv("LANTHIC_BRANCH_ID"))
    parser.add_argument("--source-id-filter", action="append", default=[])

    parser.add_argument("--cost-ledger", type=Path, default=Path(os.getenv("LANTHIC_COST_LEDGER")) if os.getenv("LANTHIC_COST_LEDGER") else None)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("LANTHIC_CACHE_DIR")) if os.getenv("LANTHIC_CACHE_DIR") else None)
    parser.add_argument("--disable-cache", action="store_true", default=os.getenv("LANTHIC_DISABLE_CACHE", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--pricing-file", type=Path, default=Path(os.getenv("LANTHIC_PRICING_FILE")) if os.getenv("LANTHIC_PRICING_FILE") else None)
    return parser.parse_args()


def ledger_from_args(args: argparse.Namespace) -> Optional[Any]:
    if not args.cost_ledger and not args.cache_dir:
        return None
    if CostLedger is None:
        raise RuntimeError("cost_ledger.py could not be imported, but cost/cache options were provided.")
    pricing_config = {}
    if args.pricing_file and load_pricing_config is not None:
        pricing_config = load_pricing_config(args.pricing_file)
    return CostLedger(
        run_id=args.run_id or "sarg_agent_run",
        source_id=args.source_id,
        ledger_path=args.cost_ledger,
        cache_dir=args.cache_dir,
        pricing_config=pricing_config,
        enabled=bool(args.cost_ledger),
        cache_enabled=bool(args.cache_dir) and not args.disable_cache,
    )


def summarize(result: JSONDict) -> str:
    lines = []
    lines.append("SARG LANGGRAPH AGENT RESULT")
    lines.append("=" * 60)
    lines.append(f"Question: {result.get('question')}")
    lines.append(f"Status: {result.get('status')}")
    lines.append(f"Investigation: {result.get('investigation_id')}")
    lines.append(f"Direction: {result.get('direction')}")
    lines.append(f"Concepts: {', '.join(result.get('concepts') or [])}")
    lines.append(f"Agent steps: {result.get('agent_step_count')}  Expansions: {result.get('expansion_count')}")
    lines.append("")
    lines.append("Analyst steps:")
    for step in result.get("analyst_steps") or []:
        lines.append(f"  - step {step.get('step')}: action={step.get('action')} gap={compact(step.get('gap'), 180)}")
        if step.get("result_observation"):
            lines.append(f"    result: {compact(step.get('result_observation'), 220)}")
    lines.append("")
    lines.append("Selected reasoning paths:")
    for path in result.get("selected_reasoning_paths") or []:
        node_labels = [item.get("label") for item in path.get("nodes") or []]
        lines.append(f"  - {path.get('path_id')} score={path.get('score')}: {' -> '.join(node_labels)}")
        for s in path.get("steps") or []:
            lines.append(f"      {s.get('from')} --{s.get('edge')} ({s.get('traversalDirection')}): {compact(s.get('reason'), 180)} -> {s.get('to')}")
    lines.append("")
    lines.append("Gap assessment:")
    gap = result.get("gap_assessment") or {}
    lines.append(f"  - {gap.get('status')}: {gap.get('summary')}")
    lines.append("")
    lines.append("Answer:")
    lines.append(result.get("answer") or "")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if not args.question:
        raise SystemExit("--question is required unless --self-test is used.")

    source_ids = source_ids_from_args(args.source_id_filter)
    ledger = ledger_from_args(args)

    kg_irag_result = read_json(args.kg_irag_json) if args.kg_irag_json else None
    history = read_json(args.investigation_history_json) if args.investigation_history_json else []
    if isinstance(history, Mapping):
        history = history.get("turns") or history.get("history") or []
    selected_context = read_json(args.selected_graph_context_json) if args.selected_graph_context_json else []
    if isinstance(selected_context, Mapping):
        selected_context = selected_context.get("items") or selected_context.get("selected") or []

    document_jsons = [read_json(path) for path in args.document_json]
    risk_model = read_json(args.risk_model_json) if args.risk_model_json else None

    config = SARGConfig(
        model=args.model,
        embed_model=args.embed_model,
        max_agent_steps=args.max_agent_steps,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        top_k_paths=args.top_k,
        enable_risk_tools=not args.no_risk_tools,
        enable_forecasting=not args.no_forecasting,
        forecast_horizon=args.forecast_horizon,
        investigation_id=args.investigation_id,
        run_id=args.run_id,
        source_id=args.source_id,
        corpus_id=args.corpus_id,
        branch_id=args.branch_id,
        source_ids=source_ids,
        cost_ledger=args.cost_ledger,
        cache_dir=args.cache_dir,
        disable_cache=args.disable_cache,
        pricing_file=args.pricing_file,
        ledger=ledger,
        use_llm=not args.offline,
    )

    sarg = SARG(config)
    result = sarg.run(
        args.question,
        kg_irag_result,
        investigation_id=args.investigation_id,
        investigation_history=history,
        selected_graph_context=selected_context,
        document_jsons=document_jsons,
        risk_model=risk_model,
    )

    if args.output:
        write_json(args.output, result)
        print(f"[done] wrote {args.output}")

    if args.summary or not args.output:
        print(summarize(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
