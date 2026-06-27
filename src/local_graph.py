#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JSONDict = Dict[str, Any]


# ============================================================
# Utilities
# ============================================================

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compact_text(value: Any, max_chars: int = 800) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def stable_node_key(name: str, entity_type: str = "unknown", key: Optional[str] = None) -> str:
    if key:
        return str(key)

    return f"{normalize_text(entity_type or 'unknown')}::{normalize_text(name)}"


def stable_edge_key(
    subject_key: str,
    relation_type: str,
    object_key: str,
    claim_key: Optional[str] = None,
) -> str:
    if claim_key:
        return str(claim_key)

    return json.dumps(
        {
            "subject_key": subject_key,
            "relation_type": relation_type,
            "object_key": object_key,
        },
        sort_keys=True,
    )


def has_label(obj: JSONDict, label: str) -> bool:
    return label in (obj.get("labels") or [])


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


# ============================================================
# Data classes
# ============================================================

@dataclass
class LocalNode:
    key: str
    name: str
    entity_type: str = "unknown"
    labels: List[str] = field(default_factory=list)
    description: Optional[str] = None
    properties: JSONDict = field(default_factory=dict)
    source: str = "unknown"


@dataclass
class LocalEdge:
    key: str
    subject_key: str
    subject: str
    relation_type: str
    object_key: str
    object: str
    claim_key: Optional[str] = None
    grounding_score: float = 0.0
    description: Optional[str] = None
    properties: JSONDict = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    source: str = "unknown"


@dataclass
class EvidenceSnippet:
    evidence_id: str
    text: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    claim_key: Optional[str] = None
    properties: JSONDict = field(default_factory=dict)
    source: str = "unknown"


# ============================================================
# Local graph
# ============================================================

class LocalGraph:
    """
    Neutral query-local subgraph container.

    It does not decide what to retrieve.
    It only ingests normalized Neo4j rows from kg_tools.py and exports
    a compact local reasoning subgraph for KG-IRAG/SARG/risk tools.
    """

    def __init__(self, focus_question: str = "") -> None:
        self.focus_question = focus_question
        self.nodes: Dict[str, LocalNode] = {}
        self.edges: Dict[str, LocalEdge] = {}
        self.evidence: Dict[str, EvidenceSnippet] = {}
        self.diagnostics: List[JSONDict] = []

    # ------------------------------------------------------------
    # Add / merge primitives
    # ------------------------------------------------------------

    def add_node(
        self,
        *,
        name: str,
        entity_type: str = "unknown",
        key: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        properties: Optional[JSONDict] = None,
        source: str = "unknown",
    ) -> str:
        if not name:
            return ""

        props = dict(properties or {})
        node_id = stable_node_key(name, entity_type, key)

        if node_id not in self.nodes:
            self.nodes[node_id] = LocalNode(
                key=node_id,
                name=str(name),
                entity_type=str(entity_type or "unknown"),
                labels=list(labels or []),
                description=description,
                properties=props,
                source=source,
            )
            return node_id

        existing = self.nodes[node_id]

        if existing.entity_type == "unknown" and entity_type:
            existing.entity_type = str(entity_type)

        if not existing.description and description:
            existing.description = description

        for label in labels or []:
            if label not in existing.labels:
                existing.labels.append(label)

        existing.properties.update(props)

        return node_id

    def add_edge(
        self,
        *,
        subject: str,
        relation_type: str,
        obj: str,
        subject_key: Optional[str] = None,
        object_key: Optional[str] = None,
        subject_type: str = "unknown",
        object_type: str = "unknown",
        claim_key: Optional[str] = None,
        grounding_score: Any = None,
        description: Optional[str] = None,
        properties: Optional[JSONDict] = None,
        evidence_ids: Optional[Sequence[str]] = None,
        source: str = "unknown",
    ) -> str:
        if not subject or not relation_type or not obj:
            return ""

        s_key = self.add_node(
            key=subject_key,
            name=subject,
            entity_type=subject_type,
            source=source,
        )

        o_key = self.add_node(
            key=object_key,
            name=obj,
            entity_type=object_type,
            source=source,
        )

        if not s_key or not o_key:
            return ""

        edge_id = stable_edge_key(s_key, relation_type, o_key, claim_key)

        if edge_id not in self.edges:
            self.edges[edge_id] = LocalEdge(
                key=edge_id,
                subject_key=s_key,
                subject=str(subject),
                relation_type=str(relation_type),
                object_key=o_key,
                object=str(obj),
                claim_key=claim_key,
                grounding_score=safe_float(grounding_score),
                description=description,
                properties=dict(properties or {}),
                evidence_ids=list(dict.fromkeys(evidence_ids or [])),
                source=source,
            )
        else:
            existing = self.edges[edge_id]
            existing.grounding_score = max(existing.grounding_score, safe_float(grounding_score))

            if not existing.description and description:
                existing.description = description

            existing.properties.update(properties or {})

            for evidence_id in evidence_ids or []:
                if evidence_id not in existing.evidence_ids:
                    existing.evidence_ids.append(evidence_id)

        return edge_id

    def add_evidence(
        self,
        *,
        evidence_id: str,
        text: str,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        claim_key: Optional[str] = None,
        properties: Optional[JSONDict] = None,
        source: str = "unknown",
    ) -> str:
        if not evidence_id or not text:
            return ""

        if evidence_id not in self.evidence:
            self.evidence[evidence_id] = EvidenceSnippet(
                evidence_id=str(evidence_id),
                text=str(text),
                source_url=source_url,
                source_title=source_title,
                claim_key=claim_key,
                properties=dict(properties or {}),
                source=source,
            )
        else:
            existing = self.evidence[evidence_id]

            if not existing.source_url and source_url:
                existing.source_url = source_url

            if not existing.source_title and source_title:
                existing.source_title = source_title

            if not existing.claim_key and claim_key:
                existing.claim_key = claim_key

            existing.properties.update(properties or {})

        if claim_key:
            self.link_evidence_to_claim(claim_key, evidence_id)

        return evidence_id

    def link_evidence_to_claim(self, claim_key: str, evidence_id: str) -> None:
        for edge in self.edges.values():
            if edge.claim_key == claim_key and evidence_id not in edge.evidence_ids:
                edge.evidence_ids.append(evidence_id)

    def add_diagnostic(self, kind: str, message: str, **extra: Any) -> None:
        self.diagnostics.append(
            {
                "kind": kind,
                "message": message,
                **extra,
            }
        )

    # ------------------------------------------------------------
    # Ingestion from kg_tools.py normalized rows
    # ------------------------------------------------------------

    def ingest_cypher_rows(self, rows: Sequence[JSONDict], source: str = "cypher") -> None:
        before = self.counts()

        for row in rows:
            self._ingest_claim_evidence_pattern(row, source=source)

            for value in row.values():
                self.ingest_value(value, source=source)

        after = self.counts()

        self.add_diagnostic(
            "ingest_cypher_rows",
            "Ingested normalized Cypher rows into local graph.",
            source=source,
            rows=len(rows),
            nodes_added=after["nodes"] - before["nodes"],
            edges_added=after["edges"] - before["edges"],
            evidence_added=after["evidence"] - before["evidence"],
        )

    def ingest_value(self, value: Any, source: str = "cypher") -> None:
        if isinstance(value, list):
            for item in value:
                self.ingest_value(item, source=source)
            return

        if not isinstance(value, dict):
            return

        kind = value.get("kind")

        if kind == "node":
            self.ingest_node_object(value, source=source)
            return

        if kind == "relationship":
            self.ingest_relationship_object(value, source=source)
            return

        if kind == "path":
            self.ingest_path_object(value, source=source)
            return

        if self._looks_like_triple(value):
            self.ingest_triple_object(value, source=source)
            return

        for item in value.values():
            self.ingest_value(item, source=source)

    def ingest_node_object(self, obj: JSONDict, source: str = "cypher") -> str:
        props = obj.get("properties") or {}
        labels = obj.get("labels") or []

        if has_label(obj, "Evidence"):
            return self._ingest_evidence_node(obj, source=source)

        if not has_label(obj, "Entity"):
            return ""

        return self.add_node(
            key=props.get("key"),
            name=first_nonempty(props.get("canonical_name"), props.get("name"), props.get("key")),
            entity_type=props.get("entity_type") or "unknown",
            labels=labels,
            description=props.get("description"),
            properties=props,
            source=source,
        )

    def ingest_relationship_object(self, obj: JSONDict, source: str = "cypher") -> str:
        rel_type = obj.get("type")
        props = obj.get("properties") or {}

        if rel_type != "KG_REL":
            return ""

        start = obj.get("start_node") or {}
        end = obj.get("end_node") or {}

        return self._add_edge_from_refs(
            subject_ref=start,
            relation={
                "type": rel_type,
                "properties": props,
                "element_id": obj.get("element_id"),
            },
            object_ref=end,
            source=source,
        )

    def ingest_path_object(self, obj: JSONDict, source: str = "cypher") -> None:
        for node in obj.get("nodes") or []:
            self.ingest_node_object(node, source=source)

        triples = obj.get("triples") or []

        if triples:
            for triple in triples:
                self.ingest_triple_object(triple, source=source)
            return

        for rel in obj.get("relationships") or []:
            self.ingest_relationship_object(rel, source=source)

    def ingest_triple_object(self, triple: JSONDict, source: str = "cypher") -> str:
        subject_ref = triple.get("subject") or {}
        relation = triple.get("relation") or {}
        object_ref = triple.get("object") or {}

        return self._add_edge_from_refs(
            subject_ref=subject_ref,
            relation=relation,
            object_ref=object_ref,
            source=source,
        )

    def _ingest_evidence_node(self, obj: JSONDict, source: str = "cypher") -> str:
        props = obj.get("properties") or {}

        return self.add_evidence(
            evidence_id=props.get("evidence_id") or props.get("key") or obj.get("element_id"),
            text=props.get("text"),
            source_url=props.get("source_url"),
            source_title=props.get("source_title"),
            claim_key=props.get("claim_key"),
            properties=props,
            source=source,
        )

    def _ingest_claim_evidence_pattern(self, row: JSONDict, source: str = "cypher") -> None:
        """
        Handles rows like:
          RETURN s, c, o, ev
        where s/o are Entity nodes, c is Claim node, ev is Evidence node.

        This is separate from KG_REL path ingestion because provenance queries
        often return Claim/Evidence nodes without returning KG_REL.
        """
        entities = []
        claim = None
        evidence_nodes = []

        for value in row.values():
            if not isinstance(value, dict) or value.get("kind") != "node":
                continue

            if has_label(value, "Entity"):
                entities.append(value)
            elif has_label(value, "Claim"):
                claim = value
            elif has_label(value, "Evidence"):
                evidence_nodes.append(value)

        if claim is None or len(entities) < 2:
            return

        subject = self._preferred_entity_from_row(row, "s") or entities[0]
        obj = self._preferred_entity_from_row(row, "o") or entities[1]

        claim_props = claim.get("properties") or {}
        claim_key = claim_props.get("key")

        self._add_edge_from_refs(
            subject_ref=self._node_ref_from_node_object(subject),
            relation={
                "type": "KG_REL",
                "properties": {
                    "claim_key": claim_key,
                    "relation_type": claim_props.get("relation_type"),
                    "grounding_score": claim_props.get("grounding_score"),
                    "description": claim_props.get("description"),
                    "temporal_json": claim_props.get("temporal_json"),
                },
            },
            object_ref=self._node_ref_from_node_object(obj),
            source=source,
        )

        for evidence in evidence_nodes:
            evidence_props = evidence.get("properties") or {}
            evidence_id = self.add_evidence(
                evidence_id=evidence_props.get("evidence_id") or evidence.get("element_id"),
                text=evidence_props.get("text"),
                source_url=evidence_props.get("source_url"),
                source_title=evidence_props.get("source_title"),
                claim_key=claim_key,
                properties=evidence_props,
                source=source,
            )
            if claim_key and evidence_id:
                self.link_evidence_to_claim(claim_key, evidence_id)

    def _preferred_entity_from_row(self, row: JSONDict, key: str) -> Optional[JSONDict]:
        value = row.get(key)
        if isinstance(value, dict) and value.get("kind") == "node" and has_label(value, "Entity"):
            return value
        return None

    def _node_ref_from_node_object(self, obj: JSONDict) -> JSONDict:
        props = obj.get("properties") or {}
        return {
            "key": props.get("key"),
            "canonical_name": props.get("canonical_name"),
            "entity_type": props.get("entity_type"),
            "labels": obj.get("labels") or [],
            "element_id": obj.get("element_id"),
        }

    def _looks_like_triple(self, value: JSONDict) -> bool:
        return (
            isinstance(value.get("subject"), dict)
            and isinstance(value.get("relation"), dict)
            and isinstance(value.get("object"), dict)
        )

    def _add_edge_from_refs(
        self,
        *,
        subject_ref: JSONDict,
        relation: JSONDict,
        object_ref: JSONDict,
        source: str,
    ) -> str:
        rel_props = relation.get("properties") or {}
        rel_type = rel_props.get("relation_type") or relation.get("type") or "related_to"

        subject_name = first_nonempty(
            subject_ref.get("canonical_name"),
            subject_ref.get("name"),
            subject_ref.get("key"),
        )

        object_name = first_nonempty(
            object_ref.get("canonical_name"),
            object_ref.get("name"),
            object_ref.get("key"),
        )

        if not subject_name or not object_name:
            return ""

        return self.add_edge(
            subject=subject_name,
            relation_type=rel_type,
            obj=object_name,
            subject_key=subject_ref.get("key"),
            object_key=object_ref.get("key"),
            subject_type=subject_ref.get("entity_type") or "unknown",
            object_type=object_ref.get("entity_type") or "unknown",
            claim_key=rel_props.get("claim_key"),
            grounding_score=rel_props.get("grounding_score"),
            description=rel_props.get("description"),
            properties=rel_props,
            source=source,
        )

    # ------------------------------------------------------------
    # Ranking / context
    # ------------------------------------------------------------

    def focus_terms(self) -> List[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
            "with", "what", "which", "how", "why", "is", "are", "was", "were",
            "does", "do", "did", "give", "find", "show", "tell", "me", "about",
            "then", "if", "so", "that", "this", "between", "relation",
        }

        out = []

        for token in normalize_text(self.focus_question).split():
            if token not in stopwords and len(token) >= 3 and token not in out:
                out.append(token)

        return out

    def edge_relevance_score(self, edge: LocalEdge) -> float:
        text = normalize_text(
            " ".join([
                edge.subject,
                edge.relation_type,
                edge.object,
                edge.description or "",
            ])
        )

        hits = sum(1 for term in self.focus_terms() if term in text)
        return edge.grounding_score + 0.15 * hits

    def ranked_edges(self) -> List[LocalEdge]:
        return sorted(
            self.edges.values(),
            key=self.edge_relevance_score,
            reverse=True,
        )

    def ranked_nodes(self) -> List[LocalNode]:
        incidence = {key: 0 for key in self.nodes}

        for edge in self.edges.values():
            incidence[edge.subject_key] = incidence.get(edge.subject_key, 0) + 1
            incidence[edge.object_key] = incidence.get(edge.object_key, 0) + 1

        focus = self.focus_terms()

        def score(node: LocalNode) -> Tuple[int, int]:
            text = normalize_text(" ".join([node.name, node.entity_type, node.description or ""]))
            hits = sum(1 for term in focus if term in text)
            return hits, incidence.get(node.key, 0)

        return sorted(self.nodes.values(), key=score, reverse=True)

    def counts(self) -> JSONDict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "evidence": len(self.evidence),
            "diagnostics": len(self.diagnostics),
        }

    def to_dict(self) -> JSONDict:
        return {
            "focus_question": self.focus_question,
            "nodes": {
                key: asdict(node)
                for key, node in self.nodes.items()
            },
            "edges": {
                key: asdict(edge)
                for key, edge in self.edges.items()
            },
            "evidence": {
                key: asdict(item)
                for key, item in self.evidence.items()
            },
            "diagnostics": self.diagnostics,
            "counts": self.counts(),
        }

    def to_context(
        self,
        *,
        max_nodes: int = 25,
        max_edges: int = 80,
        max_evidence: int = 12,
        max_evidence_chars: int = 900,
    ) -> JSONDict:
        nodes = self.ranked_nodes()[:max_nodes]
        edges = self.ranked_edges()[:max_edges]
        evidence = list(self.evidence.values())[:max_evidence]

        evidence_payload = []

        for item in evidence:
            props = item.properties or {}

            evidence_payload.append({
                "evidence_id": item.evidence_id,
                "claim_key": item.claim_key,
                "source_title": item.source_title,
                "source_url": item.source_url,

                # Preserve evidence type through KG-IRAG/SARG context.
                "block_type": props.get("block_type"),
                "document_id": props.get("document_id"),
                "metadata_json": props.get("metadata_json"),

                # These may be absent for Neo4j-ingested evidence today,
                # but keeping the fields makes table-aware context explicit.
                "row_count": props.get("row_count"),
                "columns": props.get("columns"),
                "caption": props.get("caption"),
                "data_ref": props.get("data_ref"),
                "quality_flags": props.get("quality_flags"),

                "text": compact_text(item.text, max_evidence_chars),
            })

        return {
            "focus_question": self.focus_question,
            "nodes": [
                {
                    "key": node.key,
                    "name": node.name,
                    "entity_type": node.entity_type,
                    "description": node.description,
                }
                for node in nodes
            ],
            "claims": [
                {
                    "claim_key": edge.claim_key,
                    "subject": edge.subject,
                    "relation_type": edge.relation_type,
                    "object": edge.object,
                    "grounding_score": edge.grounding_score,
                    "description": edge.description,
                    "evidence_ids": edge.evidence_ids,
                }
                for edge in edges
            ],
            "evidence": evidence_payload,
            "diagnostics": self.diagnostics,
            "counts": {
                "nodes_total": len(self.nodes),
                "edges_total": len(self.edges),
                "evidence_total": len(self.evidence),
                "nodes_shown": len(nodes),
                "edges_shown": len(edges),
                "evidence_shown": len(evidence),
                "table_evidence_shown": sum(
                    1 for item in evidence_payload
                    if item.get("block_type") == "table"
                ),
                "time_series_evidence_shown": sum(
                    1 for item in evidence_payload
                    if item.get("block_type") == "time_series"
                ),
            },
        }
    
    @classmethod
    def from_dict(cls, data: JSONDict) -> "LocalGraph":
        graph = cls(focus_question=data.get("focus_question", ""))

        for node in (data.get("nodes") or {}).values():
            graph.add_node(
                key=node.get("key"),
                name=node.get("name"),
                entity_type=node.get("entity_type") or "unknown",
                labels=node.get("labels") or [],
                description=node.get("description"),
                properties=node.get("properties") or {},
                source=node.get("source") or "from_dict",
            )

        for edge in (data.get("edges") or {}).values():
            graph.add_edge(
                subject=edge.get("subject"),
                relation_type=edge.get("relation_type"),
                obj=edge.get("object"),
                subject_key=edge.get("subject_key"),
                object_key=edge.get("object_key"),
                subject_type=(
                    graph.nodes[edge.get("subject_key")].entity_type
                    if edge.get("subject_key") in graph.nodes
                    else "unknown"
                ),
                object_type=(
                    graph.nodes[edge.get("object_key")].entity_type
                    if edge.get("object_key") in graph.nodes
                    else "unknown"
                ),
                claim_key=edge.get("claim_key"),
                grounding_score=edge.get("grounding_score"),
                description=edge.get("description"),
                properties=edge.get("properties") or {},
                evidence_ids=edge.get("evidence_ids") or [],
                source=edge.get("source") or "from_dict",
            )

        for item in (data.get("evidence") or {}).values():
            graph.add_evidence(
                evidence_id=item.get("evidence_id"),
                text=item.get("text"),
                source_url=item.get("source_url"),
                source_title=item.get("source_title"),
                claim_key=item.get("claim_key"),
                properties=item.get("properties") or {},
                source=item.get("source") or "from_dict",
            )

        for diagnostic in data.get("diagnostics") or []:
            if isinstance(diagnostic, dict):
                graph.diagnostics.append(diagnostic)

        return graph

    def summary(self, max_nodes: int = 10, max_edges: int = 15, max_evidence: int = 5) -> str:
        lines = []
        lines.append("LOCAL REASONING SUBGRAPH")
        lines.append("=" * 60)
        lines.append(f"Question: {self.focus_question}")
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {len(self.edges)}")
        lines.append(f"Evidence: {len(self.evidence)}")
        lines.append("")

        if self.nodes:
            lines.append("Top nodes:")
            for node in self.ranked_nodes()[:max_nodes]:
                lines.append(f"  - {node.name} [{node.entity_type}]")
            lines.append("")

        if self.edges:
            lines.append("Top edges:")
            for edge in self.ranked_edges()[:max_edges]:
                lines.append(
                    f"  - {edge.subject} --{edge.relation_type}--> {edge.object} "
                    f"(score={edge.grounding_score})"
                )
            lines.append("")

        if self.evidence:
            lines.append("Evidence:")
            for item in list(self.evidence.values())[:max_evidence]:
                lines.append(f"  - {item.evidence_id}: {compact_text(item.text, 180)}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # File export
    # ------------------------------------------------------------

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def write_dot(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "digraph LocalGraph {",
            '  graph [rankdir=LR, overlap=false, splines=true];',
            '  node [shape=box, style="rounded,filled", fillcolor="#eef5ff"];',
            '  edge [fontsize=10];',
        ]

        for key, node in self.nodes.items():
            label = f"{node.name}\\n[{node.entity_type}]"
            lines.append(f'  "{escape_dot(key)}" [label="{escape_dot(label)}"];')

        for edge in self.edges.values():
            label = edge.relation_type
            if edge.grounding_score:
                label += f"\\n{edge.grounding_score:.2f}"

            lines.append(
                f'  "{escape_dot(edge.subject_key)}" -> "{escape_dot(edge.object_key)}" '
                f'[label="{escape_dot(label)}"];'
            )

        lines.append("}")
        path.write_text("\n".join(lines), encoding="utf-8")

    def write_html(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = self.ranked_nodes()
        edges = self.ranked_edges()
        positions = circular_layout(nodes)

        edge_svg = []
        for edge in edges:
            start = positions.get(edge.subject_key)
            end = positions.get(edge.object_key)

            if not start or not end:
                continue

            mx = (start["x"] + end["x"]) / 2
            my = (start["y"] + end["y"]) / 2

            edge_svg.append(
                f'''
                <line x1="{start["x"]}" y1="{start["y"]}" x2="{end["x"]}" y2="{end["y"]}"
                      stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
                <text x="{mx}" y="{my}" class="edge-label">{html.escape(edge.relation_type)}</text>
                '''
            )

        node_svg = []
        for node in nodes:
            pos = positions[node.key]
            node_svg.append(
                f'''
                <g>
                  <circle cx="{pos["x"]}" cy="{pos["y"]}" r="38"
                          fill="#eaf2ff" stroke="#3366aa" stroke-width="1.5"/>
                  <text x="{pos["x"]}" y="{pos["y"] - 4}" class="node-label">
                    {html.escape(compact_text(node.name, 22))}
                  </text>
                  <text x="{pos["x"]}" y="{pos["y"] + 12}" class="node-type">
                    {html.escape(compact_text(node.entity_type, 18))}
                  </text>
                </g>
                '''
            )

        claims_json = json.dumps(
            self.to_context(max_edges=30).get("claims", []),
            indent=2,
            ensure_ascii=False,
        )

        evidence_json = json.dumps(
            self.to_context(max_evidence=10).get("evidence", []),
            indent=2,
            ensure_ascii=False,
        )

        document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Local Reasoning Subgraph</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; }}
svg {{ border: 1px solid #ddd; background: #fff; }}
.node-label {{ text-anchor: middle; font-size: 10px; font-weight: bold; }}
.node-type {{ text-anchor: middle; font-size: 9px; fill: #555; }}
.edge-label {{ text-anchor: middle; font-size: 9px; fill: #333; }}
pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Local Reasoning Subgraph</h1>
<p><b>Question:</b> {html.escape(self.focus_question)}</p>
<p><b>Nodes:</b> {len(self.nodes)} &nbsp; <b>Edges:</b> {len(self.edges)} &nbsp; <b>Evidence:</b> {len(self.evidence)}</p>

<svg width="960" height="780">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#888"/>
  </marker>
</defs>
{''.join(edge_svg)}
{''.join(node_svg)}
</svg>

<h2>Claims</h2>
<pre>{html.escape(claims_json)}</pre>

<h2>Evidence</h2>
<pre>{html.escape(evidence_json)}</pre>
</body>
</html>
"""
        path.write_text(document, encoding="utf-8")


# ============================================================
# Export helpers
# ============================================================

def circular_layout(nodes: Sequence[LocalNode]) -> Dict[str, JSONDict]:
    n = max(len(nodes), 1)
    radius = 300
    cx = 480
    cy = 390

    out = {}

    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        out[node.key] = {
            "x": cx + radius * math.cos(angle),
            "y": cy + radius * math.sin(angle),
        }

    return out


def escape_dot(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


# ============================================================
# Rigorous tests
# ============================================================

def fake_entity(key: str, name: str, entity_type: str) -> JSONDict:
    return {
        "kind": "node",
        "labels": ["Entity"],
        "properties": {
            "key": key,
            "canonical_name": name,
            "entity_type": entity_type,
            "description": f"{name} description",
        },
        "element_id": f"node:{key}",
    }


def fake_relationship(
    subject: JSONDict,
    relation_type: str,
    obj: JSONDict,
    claim_key: str,
    score: float = 0.9,
) -> JSONDict:
    return {
        "kind": "relationship",
        "type": "KG_REL",
        "properties": {
            "claim_key": claim_key,
            "relation_type": relation_type,
            "grounding_score": score,
            "description": f"{subject['properties']['canonical_name']} {relation_type} {obj['properties']['canonical_name']}",
        },
        "element_id": f"rel:{claim_key}",
        "start_node": {
            "key": subject["properties"]["key"],
            "canonical_name": subject["properties"]["canonical_name"],
            "entity_type": subject["properties"]["entity_type"],
            "labels": ["Entity"],
            "element_id": subject["element_id"],
        },
        "end_node": {
            "key": obj["properties"]["key"],
            "canonical_name": obj["properties"]["canonical_name"],
            "entity_type": obj["properties"]["entity_type"],
            "labels": ["Entity"],
            "element_id": obj["element_id"],
        },
    }


def fake_path(nodes: List[JSONDict], relationships: List[JSONDict]) -> JSONDict:
    return {
        "kind": "path",
        "nodes": nodes,
        "relationships": relationships,
        "triples": [
            {
                "subject": rel["start_node"],
                "relation": {
                    "type": rel["type"],
                    "properties": rel["properties"],
                    "element_id": rel["element_id"],
                },
                "object": rel["end_node"],
            }
            for rel in relationships
        ],
    }


def fake_claim(key: str, relation_type: str, score: float = 0.95) -> JSONDict:
    return {
        "kind": "node",
        "labels": ["Claim"],
        "properties": {
            "key": key,
            "relation_type": relation_type,
            "grounding_score": score,
            "description": "Claim description",
        },
        "element_id": f"claim:{key}",
    }


def fake_evidence(evidence_id: str, text: str) -> JSONDict:
    return {
        "kind": "node",
        "labels": ["Evidence"],
        "properties": {
            "evidence_id": evidence_id,
            "text": text,
            "source_url": "https://example.com/source",
            "source_title": "Example Source",
        },
        "element_id": f"evidence:{evidence_id}",
    }


def test_directed_path_ingestion() -> None:
    graph = LocalGraph("What elements is China restricting?")

    china = fake_entity("country::china", "China", "country")
    dysprosium = fake_entity("commodity::dysprosium", "dysprosium", "commodity")
    rel = fake_relationship(china, "restricts", dysprosium, "claim_restricts_dysprosium", 0.96)

    # Path node order is deliberately reversed to test that relationship direction wins.
    path = fake_path([dysprosium, china], [rel])

    graph.ingest_cypher_rows([{"p": path}], source="test")

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    edge = next(iter(graph.edges.values()))
    assert edge.subject == "China"
    assert edge.relation_type == "restricts"
    assert edge.object == "dysprosium"
    assert edge.claim_key == "claim_restricts_dysprosium"


def test_claim_evidence_pattern() -> None:
    graph = LocalGraph("What supports the claim?")

    usa = fake_entity("company::usa rare earths", "USA Rare Earths", "company")
    dysprosium = fake_entity("commodity::dysprosium", "dysprosium", "commodity")
    claim = fake_claim("claim_produces_dysprosium", "produces", 0.97)
    evidence = fake_evidence("ev_1", "USA Rare Earths produced dysprosium oxide samples.")

    graph.ingest_cypher_rows(
        [
            {
                "s": usa,
                "c": claim,
                "o": dysprosium,
                "ev": evidence,
            }
        ],
        source="test",
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert len(graph.evidence) == 1

    edge = next(iter(graph.edges.values()))
    assert edge.subject == "USA Rare Earths"
    assert edge.relation_type == "produces"
    assert edge.object == "dysprosium"
    assert edge.evidence_ids == ["ev_1"]


def test_duplicate_merge() -> None:
    graph = LocalGraph("duplicate test")

    china = fake_entity("country::china", "China", "country")
    yttrium = fake_entity("commodity::yttrium", "yttrium", "commodity")
    rel = fake_relationship(china, "restricts", yttrium, "claim_restricts_yttrium", 0.90)

    graph.ingest_cypher_rows([{"r": rel}, {"r": rel}], source="test")

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    edge = next(iter(graph.edges.values()))
    assert edge.grounding_score == 0.90


def test_exports() -> None:
    graph = LocalGraph("export test")

    china = fake_entity("country::china", "China", "country")
    samarium = fake_entity("commodity::samarium", "samarium", "commodity")
    rel = fake_relationship(china, "restricts", samarium, "claim_restricts_samarium", 0.91)

    graph.ingest_cypher_rows([{"r": rel}], source="test")

    context = graph.to_context()
    assert context["counts"]["nodes_total"] == 2
    assert context["counts"]["edges_total"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "graph.json"
        dot_path = tmp_path / "graph.dot"
        html_path = tmp_path / "graph.html"

        graph.save_json(json_path)
        graph.write_dot(dot_path)
        graph.write_html(html_path)

        assert json_path.exists()
        assert dot_path.exists()
        assert html_path.exists()
        assert "China" in html_path.read_text(encoding="utf-8")


def test_all() -> None:
    test_directed_path_ingestion()
    test_claim_evidence_pattern()
    test_duplicate_merge()
    test_exports()
    print("[ok] local_graph.py tests passed")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local reasoning subgraph container")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--input", type=Path, default=None, help="JSON file containing a list of normalized Cypher rows")
    parser.add_argument("--question", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dot", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test or not args.input:
        test_all()
        return 0

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--input must contain a JSON list of normalized Cypher rows")

    graph = LocalGraph(focus_question=args.question)
    graph.ingest_cypher_rows(rows)

    if args.output:
        graph.save_json(args.output)
        print(f"[done] wrote {args.output}")

    if args.dot:
        graph.write_dot(args.dot)
        print(f"[done] wrote {args.dot}")

    if args.html:
        graph.write_html(args.html)
        print(f"[done] wrote {args.html}")

    if args.summary or not any([args.output, args.dot, args.html]):
        print(graph.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())