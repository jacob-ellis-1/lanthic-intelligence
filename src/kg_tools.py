#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.graph import Node, Path as Neo4jPath, Relationship


JSONDict = Dict[str, Any]


# ============================================================
# Config
# ============================================================

ALLOWED_REL_TYPES = {
    "KG_REL",
    "SUBJECT_OF",
    "OBJECT_OF",
    "SUPPORTED_BY",
    "FROM_SOURCE",
}

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


@dataclass
class CypherSafety:
    max_rows: int = 100
    max_path_depth: int = 3
    max_probe_rows: int = 200


@dataclass
class CypherValidation:
    ok: bool
    cypher: str
    reason: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


# ============================================================
# Neo4j value normalization
# ============================================================

def neo4j_value_to_plain(value: Any) -> Any:
    if isinstance(value, Node):
        return {
            "kind": "node",
            "labels": list(value.labels),
            "properties": dict(value),
            "element_id": value.element_id,
        }

    if isinstance(value, Relationship):
        return {
            "kind": "relationship",
            "type": value.type,
            "properties": dict(value),
            "element_id": value.element_id,
            "start_node": node_ref(value.start_node),
            "end_node": node_ref(value.end_node),
        }

    if isinstance(value, Neo4jPath):
        return {
            "kind": "path",
            "nodes": [neo4j_value_to_plain(node) for node in value.nodes],
            "relationships": [
                neo4j_value_to_plain(rel)
                for rel in value.relationships
            ],
            "triples": [
                {
                    "subject": node_ref(rel.start_node),
                    "relation": {
                        "type": rel.type,
                        "properties": dict(rel),
                        "element_id": rel.element_id,
                    },
                    "object": node_ref(rel.end_node),
                }
                for rel in value.relationships
            ],
        }

    if isinstance(value, list):
        return [neo4j_value_to_plain(item) for item in value]

    if isinstance(value, tuple):
        return [neo4j_value_to_plain(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): neo4j_value_to_plain(item)
            for key, item in value.items()
        }

    return value


def node_ref(node: Node) -> JSONDict:
    props = dict(node)

    return {
        "key": props.get("key"),
        "canonical_name": props.get("canonical_name"),
        "entity_type": props.get("entity_type"),
        "labels": list(node.labels),
        "element_id": node.element_id,
    }


def record_to_plain(record: Any) -> JSONDict:
    return {
        key: neo4j_value_to_plain(record[key])
        for key in record.keys()
    }


# ============================================================
# Cypher validation
# ============================================================

def strip_comments(cypher: str) -> str:
    cypher = re.sub(r"/\*.*?\*/", " ", cypher, flags=re.DOTALL)
    cypher = re.sub(r"//.*?$", " ", cypher, flags=re.MULTILINE)
    return cypher


def normalize_schema_aliases(cypher: str) -> str:
    """
    Compatibility shim for LLM-generated Cypher.

    Our Entity label uses canonical_name, not name.
    """
    cypher = re.sub(r"\.name\b", ".canonical_name", cypher)
    cypher = re.sub(r"\{\s*name\s*:", "{canonical_name:", cypher)
    return cypher


def enforce_limit(cypher: str, max_rows: int) -> str:
    cypher = cypher.strip().rstrip(";")
    matches = list(re.finditer(r"\bLIMIT\s+(\d+)\b", cypher, flags=re.IGNORECASE))

    if not matches:
        return f"{cypher}\nLIMIT {max_rows}"

    last = matches[-1]
    existing = int(last.group(1))

    if existing <= max_rows:
        return cypher

    start, end = last.span(1)
    return cypher[:start] + str(max_rows) + cypher[end:]


def validate_path_depth(cypher: str, max_depth: int) -> Optional[str]:
    """
    Reject unbounded or over-budget variable-length relationship patterns.
    Examples:
      [:KG_REL*1..3] OK if max_depth >= 3
      [:KG_REL*]     rejected
      [:KG_REL*1..]  rejected
      [:KG_REL*1..8] rejected if max_depth < 8
    """
    patterns = re.findall(r"\[[^\]]*\*[^\]]*\]", cypher)

    for pattern in patterns:
        after_star = pattern.split("*", 1)[1]
        after_star = after_star.split("]", 1)[0].strip()

        if ".." in after_star:
            upper = after_star.split("..", 1)[1]
            upper_match = re.match(r"\s*(\d+)", upper)

            if not upper_match:
                return f"Rejected unbounded variable-length path: {pattern}"

            depth = int(upper_match.group(1))

            if depth > max_depth:
                return f"Rejected path depth {depth}; max allowed is {max_depth}: {pattern}"

            continue

        exact_match = re.match(r"\s*(\d+)", after_star)

        if not exact_match:
            return f"Rejected unbounded variable-length path: {pattern}"

        depth = int(exact_match.group(1))

        if depth > max_depth:
            return f"Rejected path depth {depth}; max allowed is {max_depth}: {pattern}"

    return None

def validate_relationship_types(cypher: str) -> Optional[str]:
    """
    Reject relationship types not present in the actual Neo4j schema.

    Handles valid Cypher relationship patterns such as:
      [r:KG_REL]
      [:SUBJECT_OF]
      [:KG_REL*1..3]
      [r:KG_REL {claim_key: c.key}]

    Prevents LLM errors like:
      [:KG_REL|restricts*1..3]

    because `restricts` is a KG_REL.relation_type property, not a relationship type.
    """
    relationship_patterns = re.findall(r"\[[^\]]*\]", cypher)

    for pattern in relationship_patterns:
        inner = pattern[1:-1].strip()  # remove [ and ]

        if ":" not in inner:
            continue

        type_section = inner.split(":", 1)[1]

        # Remove property map if present.
        type_section = type_section.split("{", 1)[0]

        # Remove variable-length suffix, e.g. KG_REL*1..3
        type_section = type_section.split("*", 1)[0]

        rel_types = [
            item.strip().strip("`")
            for item in type_section.split("|")
            if item.strip()
        ]

        for rel_type in rel_types:
            if rel_type not in ALLOWED_REL_TYPES:
                return (
                    f"Rejected unknown relationship type `{rel_type}`. "
                    f"Use :KG_REL and filter r.relation_type instead."
                )

    return None

def validate_readonly_cypher(cypher: str, safety: CypherSafety) -> CypherValidation:
    if not isinstance(cypher, str) or not cypher.strip():
        return CypherValidation(False, "", "Empty Cypher query.")

    cypher = normalize_schema_aliases(cypher)
    cypher = strip_comments(cypher).strip()

    if ";" in cypher.rstrip(";"):
        return CypherValidation(False, cypher, "Multiple Cypher statements are not allowed.")

    cypher = enforce_limit(cypher, safety.max_rows)
    lowered = cypher.lower()

    banned_patterns = [
        r"\bcreate\b",
        r"\bmerge\b",
        r"\bdelete\b",
        r"\bdetach\b",
        r"\bset\b",
        r"\bremove\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\bgrant\b",
        r"\bdeny\b",
        r"\brevoke\b",
        r"\bload\s+csv\b",
        r"\bcall\b",
        r"\bapoc\b",
        r"\bdbms\b",
        r"\bindex\b",
        r"\bconstraint\b",
    ]

    for pattern in banned_patterns:
        if re.search(pattern, lowered):
            return CypherValidation(False, cypher, f"Rejected non-read-only or unsafe token: {pattern}")

    if not re.match(r"^\s*(match|optional\s+match|with|unwind|return)\b", lowered):
        return CypherValidation(
            False,
            cypher,
            "Query must start with MATCH, OPTIONAL MATCH, WITH, UNWIND, or RETURN.",
        )

    if not re.search(r"\breturn\b", lowered):
        return CypherValidation(False, cypher, "Query must contain RETURN.")

    depth_error = validate_path_depth(cypher, safety.max_path_depth)
    if depth_error:
        return CypherValidation(False, cypher, depth_error)
    
    rel_type_error = validate_relationship_types(cypher)
    if rel_type_error:
        return CypherValidation(False, cypher, rel_type_error)

    warnings = []

    if " limit " not in f" {lowered} ":
        warnings.append(f"LIMIT {safety.max_rows} was added.")

    return CypherValidation(True, cypher, warnings=warnings)


# ============================================================
# Neo4j access
# ============================================================

class Neo4jKG:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    @classmethod
    def from_env(cls) -> "Neo4jKG":
        return cls(
            Neo4jConfig(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password"),
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
            )
        )

    def close(self) -> None:
        self.driver.close()

    def execute_read(self, cypher: str, **params: Any) -> List[JSONDict]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher, parameters=params)
            return [record_to_plain(record) for record in result]

    def execute_write(self, cypher: str, **params: Any) -> List[JSONDict]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher, parameters=params)
            return [record_to_plain(record) for record in result]

    def ensure_projection_edges(self) -> None:
        self.execute_write(
            """
            MATCH (s:Entity)-[:SUBJECT_OF]->(c:Claim)-[:OBJECT_OF]->(o:Entity)
            MERGE (s)-[r:KG_REL {claim_key: c.key}]->(o)
            SET
              r.relation_type = c.relation_type,
              r.grounding_score = c.grounding_score,
              r.postrag_decision = c.postrag_decision,
              r.description = c.description,
              r.temporal_json = c.temporal_json
            """
        )

    def run_readonly_cypher(
        self,
        cypher: str,
        *,
        safety: Optional[CypherSafety] = None,
    ) -> List[JSONDict]:
        safety = safety or CypherSafety()
        validation = validate_readonly_cypher(cypher, safety)

        if not validation.ok:
            raise ValueError(validation.reason)

        return self.execute_read(validation.cypher)

    def read_only_cypher(self, cypher: str, limit: int = 100) -> List[JSONDict]:
        """
        Backward-compatible alias.
        """
        return self.run_readonly_cypher(
            cypher,
            safety=CypherSafety(max_rows=limit),
        )

    def run_count_probe(
        self,
        cypher: str,
        *,
        safety: Optional[CypherSafety] = None,
    ) -> JSONDict:
        """
        Bounded cardinality probe.

        This intentionally does not attempt an exact COUNT over arbitrary Cypher.
        It runs the validated query with a small LIMIT and reports whether the
        result hit the probe cap.
        """
        safety = safety or CypherSafety()
        probe_safety = CypherSafety(
            max_rows=safety.max_probe_rows,
            max_path_depth=safety.max_path_depth,
            max_probe_rows=safety.max_probe_rows,
        )

        validation = validate_readonly_cypher(cypher, probe_safety)

        if not validation.ok:
            return {
                "ok": False,
                "reason": validation.reason,
                "rows_returned": 0,
                "capped": False,
            }

        rows = self.execute_read(validation.cypher)

        return {
            "ok": True,
            "rows_returned": len(rows),
            "capped": len(rows) >= probe_safety.max_probe_rows,
            "max_probe_rows": probe_safety.max_probe_rows,
            "validated_cypher": validation.cypher,
        }

    def graph_counts(self) -> JSONDict:
        rows = self.execute_read(
            """
            RETURN
              count { MATCH (:Entity) } AS entities,
              count { MATCH (:Claim) } AS claims,
              count { MATCH (:Evidence) } AS evidence,
              count { MATCH (:Source) } AS sources,
              count { MATCH ()-[:KG_REL]->() } AS kg_rel_edges
            """
        )
        return rows[0] if rows else {}

    def get_schema(self) -> JSONDict:
        entity_types = self.execute_read(
            """
            MATCH (e:Entity)
            RETURN e.entity_type AS entity_type, count(*) AS count
            ORDER BY count DESC
            LIMIT 50
            """
        )

        relation_types = self.execute_read(
            """
            MATCH ()-[r:KG_REL]->()
            RETURN r.relation_type AS relation_type, count(*) AS count
            ORDER BY count DESC
            LIMIT 50
            """
        )

        return {
            "nodes": {
                "Entity": {
                    "properties": [
                        "key",
                        "canonical_name",
                        "entity_type",
                        "description",
                        "aliases_json",
                        "attributes_json",
                        "temporal_json",
                        "latest_postrag_decision",
                        "latest_grounding_score",
                    ],
                    "important_note": "Entity nodes do not have a name property. Use canonical_name.",
                },
                "Claim": {
                    "properties": [
                        "key",
                        "claim_id",
                        "relation_type",
                        "description",
                        "grounding_score",
                        "temporal_json",
                        "postrag_decision",
                        "epistemic_status_json",
                        "corrected_candidate_json",
                    ],
                },
                "Evidence": {
                    "properties": [
                        "evidence_id",
                        "text",
                        "source_url",
                        "block_type",
                        "document_id",
                        "start_char",
                        "end_char",
                        "metadata_json",
                    ],
                },
                "Source": {
                    "properties": [
                        "url",
                        "title",
                        "publisher",
                        "published_at",
                        "document_json",
                    ],
                },
            },
            "relationships": {
                "KG_REL": {
                    "pattern": "(:Entity)-[:KG_REL]->(:Entity)",
                    "properties": [
                        "claim_key",
                        "relation_type",
                        "grounding_score",
                        "postrag_decision",
                        "description",
                        "temporal_json",
                    ],
                    "use": "Primary entity-to-entity traversal relation.",
                },
                "SUBJECT_OF": {
                    "pattern": "(:Entity)-[:SUBJECT_OF]->(:Claim)",
                },
                "OBJECT_OF": {
                    "pattern": "(:Claim)-[:OBJECT_OF]->(:Entity)",
                },
                "SUPPORTED_BY": {
                    "pattern": "(:Claim)-[:SUPPORTED_BY]->(:Evidence)",
                },
                "FROM_SOURCE": {
                    "pattern": "(:Evidence)-[:FROM_SOURCE]->(:Source)",
                },
            },
            "entity_type_counts": entity_types,
            "relation_type_counts": relation_types,
            "graph_counts": self.graph_counts(),
        }

    def get_schema_text(self) -> str:
        schema = self.get_schema()

        return f"""
Neo4j knowledge-base schema:

Nodes:
  (:Entity {{key, canonical_name, entity_type, description, aliases_json, attributes_json, temporal_json}})
    IMPORTANT: Entity does NOT have a `name` property. Use `canonical_name`.

  (:Claim {{key, claim_id, relation_type, description, grounding_score, temporal_json, postrag_decision}})
  (:Evidence {{evidence_id, text, source_url, block_type}})
  (:Source {{url, title, publisher, published_at}})

Relationships:
  (:Entity)-[:KG_REL {{claim_key, relation_type, grounding_score, description, temporal_json}}]->(:Entity)
    Use this for entity-to-entity traversal.

  (:Entity)-[:SUBJECT_OF]->(:Claim)
  (:Claim)-[:OBJECT_OF]->(:Entity)
  (:Claim)-[:SUPPORTED_BY]->(:Evidence)
  (:Evidence)-[:FROM_SOURCE]->(:Source)

Useful patterns:

  MATCH p=(a:Entity)-[:KG_REL*1..3]-(b:Entity)
  WHERE toLower(a.canonical_name) CONTAINS "china"
  RETURN p
  LIMIT 25

  MATCH (s:Entity)-[r:KG_REL]->(o:Entity)
  WHERE toLower(r.relation_type) CONTAINS "restrict"
  RETURN s, r, o
  LIMIT 25

  MATCH (s:Entity)-[:SUBJECT_OF]->(c:Claim)-[:OBJECT_OF]->(o:Entity)
  MATCH (c)-[:SUPPORTED_BY]->(ev:Evidence)
  RETURN s, c, o, ev
  LIMIT 25

Entity types present:
{json.dumps(schema["entity_type_counts"], indent=2, ensure_ascii=False)}

Relation types present:
{json.dumps(schema["relation_type_counts"], indent=2, ensure_ascii=False)}
""".strip()


# ============================================================
# CLI
# ============================================================

def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-level Neo4j KG access tools")

    parser.add_argument("--ensure-projection", action="store_true")
    parser.add_argument("--counts", action="store_true")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--schema-text", action="store_true")

    parser.add_argument("--cypher", default=None)
    parser.add_argument("--cypher-file", type=Path, default=None)
    parser.add_argument("--count-probe", action="store_true")

    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kg = Neo4jKG.from_env()

    outputs: JSONDict = {}

    try:
        if args.ensure_projection:
            kg.ensure_projection_edges()
            outputs["ensure_projection"] = "ok"

        if args.counts:
            outputs["counts"] = kg.graph_counts()

        if args.schema:
            outputs["schema"] = kg.get_schema()

        if args.schema_text:
            outputs["schema_text"] = kg.get_schema_text()

        cypher = args.cypher

        if args.cypher_file:
            cypher = args.cypher_file.read_text(encoding="utf-8")

        if cypher:
            safety = CypherSafety(
                max_rows=args.limit,
                max_path_depth=args.max_depth,
            )

            if args.count_probe:
                outputs["count_probe"] = kg.run_count_probe(cypher, safety=safety)
            else:
                validation = validate_readonly_cypher(cypher, safety)

                if not validation.ok:
                    outputs["cypher"] = {
                        "ok": False,
                        "reason": validation.reason,
                        "validated_cypher": validation.cypher,
                    }
                else:
                    outputs["cypher"] = {
                        "ok": True,
                        "validated_cypher": validation.cypher,
                        "warnings": validation.warnings,
                        "rows": kg.execute_read(validation.cypher),
                    }

    finally:
        kg.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[done] wrote {args.output}")
    else:
        print_json(outputs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())