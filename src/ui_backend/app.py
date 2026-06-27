from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

try:
    from sarg import SARG, SARGConfig
except Exception:  # pragma: no cover - keeps static demo fallback importable outside project env
    SARG = None  # type: ignore
    SARGConfig = None  # type: ignore

JSONDict = Dict[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
STATE_DIR = RUNS_DIR / "ui_state" / "local_user"
INVESTIGATIONS_DIR = STATE_DIR / "investigations"

DEFAULT_USER = {"name": "Demo Analyst", "email": "analyst@lanthic.local"}
DEFAULT_QUESTION = "How could disruption in Kachin rare-earth mining affect downstream rare-earth supply chains?"
DEMO_INVESTIGATION_ID = "inv_demo_001"

DEFAULT_RUN_ID = "e2e_postrag_blockwise_004"
DEFAULT_CORPUS_ID = "eval1"
DEFAULT_BRANCH_ID = "staging_eval1"
PLACEHOLDER_SCOPE_VALUES = {
    "",
    "none",
    "null",
    "undefined",
    "pending",
    "corpus-pending",
    "workspace",
    "demo",
    "unknown",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css",
    ".yaml", ".yml", ".xml", ".rst", ".log",
}
INTERNAL_ARTIFACT_NAMES = {
    "extract.json", "postrag.json", "sarg_demo.json", "risk_demo.json",
    "forecast_demo.json", "kg_irag_demo.json", "run_summary.json",
    "cost_ledger.jsonl", "manifest.jsonl",
}

app = FastAPI(title="Lanthic Intelligence UI Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreateRequest(BaseModel):
    workspace: str = "demo"
    run_id: Optional[str] = None
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None


class SignInRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    workspace: Optional[str] = "demo"


class CreateInvestigationRequest(BaseModel):
    question: Optional[str] = None
    title: Optional[str] = None
    run_id: Optional[str] = None
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None


class TurnRequest(BaseModel):
    question: str
    corpus_id: Optional[str] = None
    branch_id: Optional[str] = None
    run_id: Optional[str] = None
    selected_graph_context: Optional[List[JSONDict]] = None
    selectedGraphContext: Optional[List[JSONDict]] = None

class WorkspaceStateRequest(BaseModel):
    bookmarks: Optional[JSONDict] = None
    pins: Optional[JSONDict] = None
    selectedDrawerItem: Optional[JSONDict] = None
    activeTab: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str, fallback: str = "file") -> str:
    raw = Path(value or fallback).name
    stem = Path(raw).stem or fallback
    suffix = Path(raw).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or fallback
    suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix)[:16]
    return f"{stem[:90]}{suffix}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def first_text(value: Any, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def short_text(value: Any, max_chars: int = 420) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "No text available."
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def human_date(iso_value: Optional[str]) -> str:
    if not iso_value:
        return "Just now"
    try:
        return datetime.fromisoformat(iso_value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_placeholder_scope_value(value: Any) -> bool:
    return str(value or "").strip().lower() in PLACEHOLDER_SCOPE_VALUES


def normalize_run_id(value: Any) -> str:
    text = str(value or "").strip()
    if is_placeholder_scope_value(text):
        return newest_existing_run_dir(DEFAULT_RUN_ID).name
    return newest_existing_run_dir(text).name


def normalize_corpus_id(value: Any) -> str:
    text = str(value or "").strip()
    return DEFAULT_CORPUS_ID if is_placeholder_scope_value(text) else text


def normalize_branch_id(value: Any) -> str:
    text = str(value or "").strip()
    return DEFAULT_BRANCH_ID if is_placeholder_scope_value(text) else text


def normalize_investigation_scope(meta: JSONDict) -> JSONDict:
    meta["run_id"] = normalize_run_id(meta.get("run_id"))
    meta["corpus_id"] = normalize_corpus_id(meta.get("corpus_id"))
    meta["branch_id"] = normalize_branch_id(meta.get("branch_id"))
    return meta


def empty_local_kg(investigation_id: str) -> JSONDict:
    timestamp = now_iso()
    return {
        "version": 1,
        "graphKind": "sarg_local_graph",
        "investigationId": investigation_id,
        "updatedAt": timestamp,
        "truncated": False,
        "nodes": [],
        "edges": [],
        "summary": {
            "nodeCount": 0,
            "edgeCount": 0,
            "evidenceCount": 0,
        },
    }


def inv_dir(investigation_id: str) -> Path:
    return INVESTIGATIONS_DIR / safe_filename(investigation_id, "investigation")


def meta_path(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "metadata.json"


def turns_dir(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "turns"


def uploads_dir(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "uploads"


def subgraph_path(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "subgraph" / "local_reasoning_subgraph.json"


def sarg_graph_path(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "sarg" / "local_reasoning_subgraph.json"


def sarg_result_path(investigation_id: str, turn_id: str) -> Path:
    return inv_dir(investigation_id) / "sarg" / "turns" / f"{safe_filename(turn_id)}.json"


def workspace_state_path(investigation_id: str) -> Path:
    return inv_dir(investigation_id) / "workspace_state.json"


def read_meta(investigation_id: str) -> Optional[JSONDict]:
    meta = read_json(meta_path(investigation_id), None)
    return meta if isinstance(meta, dict) else None


def save_meta(meta: JSONDict) -> JSONDict:
    meta["updatedAt"] = now_iso()
    meta["lastModifiedAt"] = meta["updatedAt"]
    write_json(meta_path(meta["investigationId"]), meta)
    return meta


def read_subgraph(investigation_id: str) -> JSONDict:
    graph = read_json(subgraph_path(investigation_id), None)
    if isinstance(graph, dict):
        graph.setdefault("nodes", [])
        graph.setdefault("edges", [])
        return graph
    created = now_iso()
    return {
        "version": 1,
        "investigationId": investigation_id,
        "createdAt": created,
        "updatedAt": created,
        "nodes": [],
        "edges": [],
    }


def save_subgraph(investigation_id: str, graph: JSONDict) -> None:
    graph["updatedAt"] = now_iso()
    write_json(subgraph_path(investigation_id), graph)


def read_sarg_graph(investigation_id: str) -> JSONDict:
    graph = read_json(sarg_graph_path(investigation_id), None)
    return graph if isinstance(graph, dict) else {}


def save_sarg_graph(investigation_id: str, graph: JSONDict) -> None:
    if not isinstance(graph, dict):
        return
    graph["updatedAt"] = now_iso()
    write_json(sarg_graph_path(investigation_id), graph)


def save_sarg_result(investigation_id: str, turn_id: str, result: JSONDict) -> None:
    if isinstance(result, dict):
        write_json(sarg_result_path(investigation_id, turn_id), result)


def has_sarg_graph(graph: Any) -> bool:
    return (
        isinstance(graph, dict)
        and (bool(graph.get("nodes")) or bool(graph.get("edges")) or bool(graph.get("evidence")))
    )

WORKSPACE_STATE_BUCKETS = ("sources", "evidence", "assumptions", "graphItems")
WORKSPACE_STATE_TABS = {"Sources", "Evidence", "Assumptions"}


def default_workspace_state(investigation_id: Optional[str] = None) -> JSONDict:
    created = now_iso()

    return {
        "schemaVersion": 1,
        "investigationId": investigation_id,
        "createdAt": created,
        "updatedAt": created,
        "activeTab": "Sources",
        "bookmarks": {
            "sources": [],
            "evidence": [],
            "assumptions": [],
            "graphItems": [],
        },
        "pins": {
            "sources": [],
            "evidence": [],
            "assumptions": [],
            "graphItems": [],
        },
        "selectedDrawerItem": None,
    }


def normalise_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    seen = set()
    result: List[str] = []

    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def normalise_workspace_bucket(value: Any) -> JSONDict:
    source = value if isinstance(value, dict) else {}

    return {
        bucket: normalise_string_list(source.get(bucket))
        for bucket in WORKSPACE_STATE_BUCKETS
    }


def normalise_workspace_state(
    value: Any,
    investigation_id: Optional[str] = None,
) -> JSONDict:
    base = default_workspace_state(investigation_id)
    source = value if isinstance(value, dict) else {}

    active_tab = source.get("activeTab")
    if active_tab not in WORKSPACE_STATE_TABS:
        active_tab = base["activeTab"]

    selected = source.get("selectedDrawerItem")
    if not isinstance(selected, dict):
        selected = None

    state = {
        **base,
        **{
            key: source.get(key)
            for key in ("schemaVersion", "createdAt", "updatedAt")
            if source.get(key) is not None
        },
        "investigationId": investigation_id or source.get("investigationId"),
        "activeTab": active_tab,
        "bookmarks": normalise_workspace_bucket(source.get("bookmarks")),
        "pins": normalise_workspace_bucket(source.get("pins")),
        "selectedDrawerItem": selected,
    }

    return state


def read_workspace_state(investigation_id: str) -> JSONDict:
    state = read_json(workspace_state_path(investigation_id), None)
    return normalise_workspace_state(state, investigation_id)


def save_workspace_state(investigation_id: str, state: Any) -> JSONDict:
    current = read_workspace_state(investigation_id)
    incoming = normalise_workspace_state(state, investigation_id)

    merged = {
        **current,
        **incoming,
        "investigationId": investigation_id,
        "createdAt": current.get("createdAt") or incoming.get("createdAt") or now_iso(),
        "updatedAt": now_iso(),
    }

    write_json(workspace_state_path(investigation_id), merged)
    return merged


def upsert_node(graph: JSONDict, node: JSONDict) -> None:
    node_id = node.get("id")
    if not node_id:
        return
    nodes = graph.setdefault("nodes", [])
    for existing in nodes:
        if existing.get("id") == node_id:
            existing.update({k: v for k, v in node.items() if v not in (None, "")})
            return
    nodes.append(node)


def upsert_edge(graph: JSONDict, source: str, target: str, relation: str, **metadata: Any) -> None:
    if not source or not target or not relation:
        return
    edges = graph.setdefault("edges", [])
    for edge in edges:
        if edge.get("source") == source and edge.get("target") == target and edge.get("relation") == relation:
            edge.update(metadata)
            return
    edges.append({"source": source, "target": target, "relation": relation, **metadata})


def newest_existing_run_dir(preferred_run_id: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if preferred_run_id:
        candidates.extend([RUNS_DIR / preferred_run_id, RUNS_DIR / safe_filename(preferred_run_id, preferred_run_id)])
    candidates.extend([
        RUNS_DIR / "e2e_postrag_blockwise_004",
        RUNS_DIR / "e2e_postrag_blockwise_003",
        RUNS_DIR / "e2e_postrag_blockwise_002",
        RUNS_DIR / "e2e_postrag_blockwise_001",
        RUNS_DIR / "e2e_neo4j_001",
    ])
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return RUNS_DIR / (preferred_run_id or "e2e_postrag_blockwise_004")


def create_investigation_record(
    *,
    investigation_id: Optional[str] = None,
    title: Optional[str] = None,
    question: Optional[str] = None,
    run_id: Optional[str] = None,
    corpus_id: Optional[str] = None,
    branch_id: Optional[str] = None,
) -> JSONDict:
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    created = now_iso()
    resolved_id = investigation_id or f"inv_{uuid.uuid4().hex[:10]}"
    resolved_title = first_text(title, first_text(question, DEFAULT_QUESTION))
    run_dir = newest_existing_run_dir(run_id or DEFAULT_RUN_ID)
    meta = {
        "schemaVersion": 1,
        "investigationId": resolved_id,
        "chatName": resolved_title,
        "title": resolved_title,
        "status": "Ready",
        "createdAt": created,
        "updatedAt": created,
        "lastModifiedAt": created,
        "lastRunAt": None,
        "run_id": normalize_run_id(run_dir.name),
        "corpus_id": normalize_corpus_id(corpus_id),
        "branch_id": normalize_branch_id(branch_id),
        "turnIds": [],
        "documents": [],
        "localSubgraphPath": str(subgraph_path(resolved_id).relative_to(PROJECT_ROOT)),
    }
    turns_dir(resolved_id).mkdir(parents=True, exist_ok=True)
    uploads_dir(resolved_id).mkdir(parents=True, exist_ok=True)
    save_subgraph(resolved_id, empty_local_kg(resolved_id))
    save_workspace_state(resolved_id, default_workspace_state(resolved_id))
    write_json(meta_path(resolved_id), meta)
    return meta


def ensure_default_investigation() -> JSONDict:
    meta = read_meta(DEMO_INVESTIGATION_ID)
    if meta:
        before = json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str)
        normalize_investigation_scope(meta)
        after = json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str)
        if before != after:
            save_meta(meta)
        return meta
    return create_investigation_record(
        investigation_id=DEMO_INVESTIGATION_ID,
        title=DEFAULT_QUESTION,
        question=DEFAULT_QUESTION,
        run_id="e2e_postrag_blockwise_004",
        corpus_id="eval1",
        branch_id="staging_eval1",
    )


def get_meta_or_404(investigation_id: str) -> JSONDict:
    if investigation_id == DEMO_INVESTIGATION_ID:
        return ensure_default_investigation()
    meta = read_meta(investigation_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return meta


def read_turn(investigation_id: str, turn_id: str) -> Optional[JSONDict]:
    turn = read_json(turns_dir(investigation_id) / f"{safe_filename(turn_id)}.json", None)
    return turn if isinstance(turn, dict) else None


def write_turn(investigation_id: str, turn: JSONDict) -> None:
    write_json(turns_dir(investigation_id) / f"{safe_filename(turn['turnId'])}.json", turn)


def load_turns(meta: JSONDict) -> List[JSONDict]:
    turns: List[JSONDict] = []
    for turn_id in meta.get("turnIds", []):
        turn = read_turn(meta["investigationId"], turn_id)
        if isinstance(turn, dict):
            turns.append(turn)
    return turns


def investigation_summary(meta: JSONDict) -> JSONDict:
    return {
        "investigationId": meta["investigationId"],
        "title": meta.get("title") or meta.get("chatName") or "Untitled investigation",
        "chatName": meta.get("chatName") or meta.get("title") or "Untitled investigation",
        "status": meta.get("status", "Ready"),
        "createdAt": meta.get("createdAt"),
        "updatedAt": human_date(meta.get("updatedAt")),
        "lastModifiedAt": meta.get("lastModifiedAt") or meta.get("updatedAt"),
        "turnCount": len(meta.get("turnIds", [])),
        "documentCount": len(meta.get("documents", [])),
    }


def list_metas() -> List[JSONDict]:
    ensure_default_investigation()
    metas: List[JSONDict] = []
    for path in sorted(INVESTIGATIONS_DIR.glob("*/metadata.json")):
        meta = read_json(path, None)
        if isinstance(meta, dict) and meta.get("investigationId"):
            metas.append(meta)
    return sorted(metas, key=lambda item: item.get("updatedAt") or item.get("createdAt") or "", reverse=True)


def public_document(doc: JSONDict) -> JSONDict:
    return {
        "documentId": doc.get("documentId"),
        "filename": doc.get("filename"),
        "sourceTitle": doc.get("sourceTitle") or doc.get("filename"),
        "sizeBytes": doc.get("sizeBytes"),
        "contentType": doc.get("contentType"),
        "uploadedAt": doc.get("uploadedAt"),
        "textExtractAvailable": doc.get("textExtractAvailable", False),
        "excerpt": doc.get("excerpt"),
        "sha256": doc.get("sha256"),
    }


def subgraph_summary(investigation_id: str) -> JSONDict:
    sarg_graph = read_sarg_graph(investigation_id)
    if has_sarg_graph(sarg_graph):
        ui_graph = sarg_graph_to_ui_subgraph(investigation_id, sarg_graph)
        counts: Dict[str, int] = {}
        for node in ui_graph.get("nodes", []):
            node_type = str(node.get("type") or "node")
            counts[node_type] = counts.get(node_type, 0) + 1
        return {
            "nodeCount": len(ui_graph.get("nodes", [])),
            "edgeCount": len(ui_graph.get("edges", [])),
            "nodeCounts": counts,
            "updatedAt": ui_graph.get("updatedAt"),
            "graphKind": "sarg_local_graph",
        }

    empty = empty_local_kg(investigation_id)
    return {
        "nodeCount": 0,
        "edgeCount": 0,
        "nodeCounts": {},
        "updatedAt": empty.get("updatedAt"),
        "graphKind": "empty_local_graph",
    }


def investigation_response(meta: JSONDict, include_turns: bool = True) -> JSONDict:
    payload = {
        **investigation_summary(meta),
        "createdAt": meta.get("createdAt"),
        "lastModifiedAt": meta.get("lastModifiedAt") or meta.get("updatedAt"),
        "run_id": meta.get("run_id"),
        "corpus_id": meta.get("corpus_id"),
        "branch_id": meta.get("branch_id"),
        "documents": [public_document(doc) for doc in meta.get("documents", [])],
        "localSubgraph": subgraph_summary(meta["investigationId"]),
    }
    if include_turns:
        payload["turns"] = load_turns(meta)
    return payload


def is_internal_ref(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip().lower()
    name = Path(text).name
    return (
        name in INTERNAL_ARTIFACT_NAMES
        or "/runs/" in text
        or "\\runs\\" in text
        or "/ui_state/" in text
        or "\\ui_state\\" in text
        or text.startswith(str(PROJECT_ROOT).lower())
        or (text.startswith("file://") and ("/runs/" in text or name in INTERNAL_ARTIFACT_NAMES))
    )


def clean_title(value: Any, fallback: str = "Source") -> str:
    if not isinstance(value, str) or not value.strip() or is_internal_ref(value):
        return fallback
    text = value.strip().replace("file://", "", 1)
    if "/" in text or "\\" in text:
        text = Path(text).name
    text = re.sub(r"[_-]+", " ", text).strip()
    return text or fallback


def find_artifact(run_dir: Path, filename: str) -> Optional[Path]:
    direct = run_dir / filename
    if direct.exists():
        return direct
    matches = list(run_dir.rglob(filename)) if run_dir.exists() else []
    if matches:
        return matches[0]
    fallback = RUNS_DIR / "e2e_neo4j_001" / filename
    return fallback if fallback.exists() else None


def load_artifacts(run_dir: Path) -> JSONDict:
    artifacts = {
        "sarg": read_json(find_artifact(run_dir, "sarg_demo.json") or Path("__missing__"), {}),
        "risk": read_json(find_artifact(run_dir, "risk_demo.json") or Path("__missing__"), {}),
        "forecast": read_json(find_artifact(run_dir, "forecast_demo.json") or Path("__missing__"), {}),
        "postrag_records": [],
    }
    for postrag_path in sorted(run_dir.rglob("postrag.json")) if run_dir.exists() else []:
        record = read_json(postrag_path, {})
        if isinstance(record, dict):
            artifacts["postrag_records"].append(record)
    if not artifacts["postrag_records"]:
        fallback_dir = RUNS_DIR / "e2e_postrag_blockwise_004"
        if fallback_dir.exists() and fallback_dir != run_dir:
            for postrag_path in sorted(fallback_dir.rglob("postrag.json")):
                record = read_json(postrag_path, {})
                if isinstance(record, dict):
                    artifacts["postrag_records"].append(record)
    return artifacts


def source_from_record(record: JSONDict, index: int) -> JSONDict:
    document = record.get("document") if isinstance(record.get("document"), dict) else {}
    extraction = record.get("extraction") if isinstance(record.get("extraction"), dict) else {}
    metadata = extraction.get("pipeline_metadata") if isinstance(extraction.get("pipeline_metadata"), dict) else {}
    credibility = document.get("credibility") if isinstance(document.get("credibility"), dict) else {}
    title = ""
    for candidate in [
        document.get("title"), document.get("name"), document.get("filename"),
        document.get("publisher"), metadata.get("title"), metadata.get("source_name"),
        metadata.get("canonical_source"), record.get("source_url"),
        document.get("source_url"), document.get("canonical_url"),
    ]:
        title = clean_title(candidate, "")
        if title:
            break
    title = title or f"Source {index}"
    date = first_text(document.get("published_at"), first_text(document.get("date"), first_text(metadata.get("published_at"), "Date unavailable")))
    tag = first_text(credibility.get("tier"), first_text(document.get("source_type"), first_text(metadata.get("source_type"), "Source")))
    return {"id": f"artifact_source_{index}", "logo": "S" if index > 9 else str(index), "title": title, "date": date, "tag": tag}


def evidence_lookup(postrag_records: List[JSONDict]) -> Dict[str, JSONDict]:
    lookup: Dict[str, JSONDict] = {}
    for index, record in enumerate(postrag_records, start=1):
        source = source_from_record(record, index)
        store = record.get("evidence_store")
        if not isinstance(store, dict):
            continue
        for evidence_id, evidence in store.items():
            if isinstance(evidence, dict):
                item = dict(evidence)
                item["__source_title"] = source["title"]
                item["__source_date"] = source["date"]
                lookup[str(evidence_id)] = item
    return lookup


def collect_evidence_ids(*objects: Any) -> List[str]:
    ids: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_ids" and isinstance(child, list):
                    for item in child:
                        if isinstance(item, str) and item not in ids:
                            ids.append(item)
                elif key == "evidence_id" and isinstance(child, str) and child not in ids:
                    ids.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for obj in objects:
        visit(obj)
    return ids


def artifact_sources(postrag_records: List[JSONDict]) -> List[JSONDict]:
    sources: List[JSONDict] = []
    for index, record in enumerate(postrag_records, start=1):
        source = source_from_record(record, index)
        store = record.get("evidence_store") if isinstance(record.get("evidence_store"), dict) else {}
        first_block = next((block for block in store.values() if isinstance(block, dict)), {})
        sources.append({**source, "excerpt": short_text(first_block.get("text") if isinstance(first_block, dict) else "No excerpt available.", 300)})
    return sources


def artifact_evidence_cards(evidence_ids: List[str], lookup: Dict[str, JSONDict]) -> List[JSONDict]:
    cards: List[JSONDict] = []
    for evidence_id in evidence_ids:
        evidence = lookup.get(evidence_id)
        if not isinstance(evidence, dict):
            continue
        page = evidence.get("page_number") or evidence.get("page")
        cards.append({
            "id": evidence_id,
            "title": f"Page {page}" if page else "Evidence excerpt",
            "source": first_text(evidence.get("__source_title"), "Attached source"),
            "date": first_text(evidence.get("__source_date"), "Evidence block"),
            "text": short_text(evidence.get("text"), 360),
            "support": "Supporting evidence",
        })
    return cards


def extract_text(path: Path, content_type: Optional[str]) -> Tuple[str, bool, str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS or (content_type or "").startswith("text/"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:12000], True, "plain_text"
        except Exception:
            return "", False, "plain_text_failed"
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            chunks: List[str] = []
            for page in reader.pages[:8]:
                chunks.append(page.extract_text() or "")
                if sum(len(chunk) for chunk in chunks) >= 12000:
                    break
            text = "\n".join(chunks).strip()[:12000]
            return text, bool(text), "pdf_text"
        except Exception:
            return "", False, "pdf_text_unavailable"
    return "", False, "unsupported_binary"


def upload_sources(docs: List[JSONDict]) -> List[JSONDict]:
    return [{
        "id": doc.get("documentId") or f"uploaded_source_{index}",
        "logo": "U",
        "title": doc.get("sourceTitle") or doc.get("filename") or f"Uploaded document {index}",
        "date": human_date(doc.get("uploadedAt")),
        "excerpt": doc.get("excerpt") or "Uploaded file is attached to this investigation.",
        "tag": "Uploaded file" if doc.get("textExtractAvailable") else "Uploaded file · extraction pending",
    } for index, doc in enumerate(docs, start=1)]


def upload_evidence(docs: List[JSONDict]) -> List[JSONDict]:
    return [{
        "id": f"upload_evidence_{doc.get('documentId') or index}",
        "title": "Uploaded document excerpt" if doc.get("textExtractAvailable") else "Uploaded document attached",
        "source": doc.get("sourceTitle") or doc.get("filename") or f"Uploaded document {index}",
        "date": human_date(doc.get("uploadedAt")),
        "text": doc.get("excerpt") or "The document is attached locally but no text excerpt could be extracted automatically.",
        "support": "User-provided evidence",
    } for index, doc in enumerate(docs, start=1)]


def reasoning_path(sarg: JSONDict, graph: JSONDict) -> List[JSONDict]:
    chains = (((sarg.get("sarg_context") or {}).get("reasoning_chains") or []) if isinstance(sarg, dict) else [])
    if chains and isinstance(chains[0], dict) and isinstance(chains[0].get("path"), list) and chains[0]["path"]:
        return [{"label": str(item)} for item in chains[0]["path"][:5]]
    if graph.get("nodes"):
        return [
            {"label": "Retrieve preserved investigation context"},
            {"label": "Check uploaded and indexed sources"},
            {"label": "Ground answer in available evidence"},
            {"label": "Assess risk and missing evidence"},
        ]
    return [
        {"label": "Evidence retrieval"},
        {"label": "Source-grounded reasoning"},
        {"label": "Risk assessment"},
        {"label": "Missing-evidence review"},
    ]


def risk_assessment(risk: JSONDict) -> JSONDict:
    factors: List[JSONDict] = []
    raw = risk.get("risk_factors") if isinstance(risk, dict) else []
    if isinstance(raw, list):
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            score = item.get("score") if isinstance(item.get("score"), dict) else {}
            value = score.get("adjusted_score") or score.get("raw_score") or item.get("severity") or 0
            try:
                value = float(value)
            except Exception:
                value = 0
            if value <= 5:
                value *= 20
            factors.append({"label": first_text(item.get("name"), "Risk factor"), "value": max(0, min(100, round(value, 1))), "tone": "high" if value >= 70 else "medium"})
    if not factors:
        factors = [
            {"label": "Evidence coverage", "value": 55, "tone": "medium"},
            {"label": "Supply-chain exposure", "value": 65, "tone": "medium"},
            {"label": "Missing data", "value": 70, "tone": "high"},
        ]
    level = first_text(risk.get("risk_level") if isinstance(risk, dict) else None, "Moderate").title()
    risk_score = risk.get("risk_score") if isinstance(risk, dict) else None
    confidence = first_text(risk.get("confidence_level") if isinstance(risk, dict) else None, "unknown")
    summary = f"Overall risk is {level.lower()}"
    if isinstance(risk_score, (int, float)):
        summary += f" with a score of {risk_score:.1f}"
    summary += f". Confidence is {confidence}."
    return {"overallRisk": level, "factors": factors, "summary": summary}


def forecast_check(forecast: JSONDict) -> JSONDict:
    status = first_text(forecast.get("status") if isinstance(forecast, dict) else None, "unavailable")
    reason = first_text(
        forecast.get("reason") if isinstance(forecast, dict) else None,
        "Forecast unavailable: no reliable time-series evidence was returned for this turn.",
    )
    unavailable = status.lower().replace(" ", "_") in {"unavailable", "failed", "error"}
    return {"status": status.replace("_", " ").title(), "summary": reason, "showChart": not unavailable}


def missing_evidence(sarg: JSONDict, risk: JSONDict, forecast: JSONDict, docs: List[JSONDict]) -> JSONDict:
    items: List[str] = []
    for item in risk.get("missing_variables", []) if isinstance(risk, dict) else []:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    for request in sarg.get("expansion_requests", []) if isinstance(sarg, dict) else []:
        if isinstance(request, dict) and isinstance(request.get("missing_information"), list):
            for item in request["missing_information"]:
                text = str(item).strip()
                if text and text not in items:
                    items.append(text)
    if isinstance(forecast, dict) and forecast.get("status") == "unavailable":
        items.append("Usable time-series evidence for forecasting")
    if any(not doc.get("textExtractAvailable") for doc in docs):
        items.append("Automatic text extraction for one or more uploaded documents")
    if not items:
        items = ["More direct evidence on downstream industry exposure"]
    return {"totalCount": len(items), "items": items[:8]}


def assumptions_from_missing(missing: JSONDict, forecast: JSONDict) -> List[JSONDict]:
    assumptions = [{"title": "Missing evidence assumption", "text": f"The answer should be treated cautiously because this evidence is missing: {item}."} for item in missing.get("items", [])[:6]]
    if isinstance(forecast, dict) and forecast.get("status") == "unavailable":
        assumptions.append({"title": "Forecast limitation", "text": "The system did not produce a forecast because the available evidence did not contain enough reliable time-series observations."})
    return assumptions


def brief(question: str, sarg: JSONDict, risk: JSONDict, docs: List[JSONDict]) -> JSONDict:
    answer = first_text(sarg.get("answer") if isinstance(sarg, dict) else None, "")
    if not answer:
        level = first_text(risk.get("risk_level") if isinstance(risk, dict) else None, "material")
        upload_clause = " User-uploaded documents are attached to this investigation and preserved locally." if docs else ""
        answer = f"The available evidence indicates a {level} supply-chain risk, but the system needs more evidence to fully resolve downstream impacts.{upload_clause}"
    answer = answer.replace("\n\n", " ").replace("\n", " ").strip()
    if len(answer) > 900:
        answer = answer[:899].rstrip() + "…"
    lead = "Partial." if isinstance(sarg, dict) and sarg.get("status") in {"partial", "insufficient"} else "Yes."
    return {"answerLead": lead, "summary": answer}




def sarg_graph_counts(graph: JSONDict) -> JSONDict:
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    edges = graph.get("edges") if isinstance(graph, dict) else None
    evidence = graph.get("evidence") if isinstance(graph, dict) else None
    return {
        "nodes": len(nodes) if isinstance(nodes, dict) else len(nodes or []),
        "edges": len(edges) if isinstance(edges, dict) else len(edges or []),
        "evidence": len(evidence) if isinstance(evidence, dict) else len(evidence or []),
    }


def sarg_evidence_card(evidence_id: str, evidence: Any) -> JSONDict:
    item = evidence if isinstance(evidence, dict) else {}
    props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
    title = first_text(item.get("source_title"), first_text(item.get("sourceTitle"), "Evidence excerpt"))
    return {
        "id": evidence_id,
        "title": title if title != "Evidence excerpt" else "Evidence excerpt",
        "source": title,
        "date": first_text(item.get("published_at"), first_text(item.get("date"), "Evidence block")),
        "text": short_text(item.get("text"), 420),
        "support": "Supporting evidence",
        "sourceUrl": item.get("source_url") or item.get("sourceUrl"),
        "blockType": props.get("block_type") or item.get("block_type") or item.get("blockType"),
    }


def sarg_graph_to_ui_subgraph(investigation_id: str, graph: JSONDict) -> JSONDict:
    nodes_raw = graph.get("nodes") if isinstance(graph.get("nodes"), dict) else {}
    edges_raw = graph.get("edges") if isinstance(graph.get("edges"), dict) else {}
    evidence_raw = graph.get("evidence") if isinstance(graph.get("evidence"), dict) else {}

    nodes: List[JSONDict] = []
    edges: List[JSONDict] = []

    for fallback_id, raw in nodes_raw.items():
        item = raw if isinstance(raw, dict) else {}
        node_id = str(item.get("key") or item.get("id") or fallback_id)
        label = first_text(
            item.get("name"),
            first_text(item.get("canonical_name"), first_text(item.get("label"), node_id)),
        )
        node_type = first_text(item.get("entity_type"), first_text(item.get("type"), "entity")).lower()
        evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        nodes.append({
            "id": node_id,
            "type": node_type,
            "taxonomyType": node_type,
            "label": label,
            "text": first_text(item.get("description"), first_text(item.get("text"), "")),
            "evidence": [sarg_evidence_card(str(eid), evidence_raw.get(str(eid))) for eid in evidence_ids if str(eid) in evidence_raw],
            "metadata": {
                "kind": "sarg_node",
                "source": "sarg_local_graph",
                "rawKey": node_id,
                "sourceIds": item.get("source_ids") or [],
            },
        })

    for fallback_id, raw in edges_raw.items():
        item = raw if isinstance(raw, dict) else {}
        edge_id = str(item.get("key") or item.get("id") or fallback_id)
        source = str(item.get("subject_key") or item.get("source") or item.get("subject") or "")
        target = str(item.get("object_key") or item.get("target") or item.get("object") or "")
        relation = first_text(item.get("relation_type"), first_text(item.get("relation"), first_text(item.get("label"), "related_to")))
        if not source or not target:
            continue
        evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "label": relation.replace("_", " "),
            "type": "relation",
            "text": first_text(item.get("description"), first_text(item.get("text"), "")),
            "evidence": [sarg_evidence_card(str(eid), evidence_raw.get(str(eid))) for eid in evidence_ids if str(eid) in evidence_raw],
            "metadata": {
                "kind": "sarg_relation",
                "source": "sarg_local_graph",
                "rawKey": edge_id,
                "claimKeys": item.get("claim_keys") or ([item.get("claim_key")] if item.get("claim_key") else []),
                "groundingScore": item.get("grounding_score"),
            },
        })

    return {
        "version": 1,
        "graphKind": "sarg_local_graph",
        "investigationId": investigation_id,
        "updatedAt": graph.get("updatedAt") or now_iso(),
        "truncated": False,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "evidenceCount": len(evidence_raw),
        },
    }


def document_jsons_from_meta(meta: JSONDict) -> List[JSONDict]:
    docs: List[JSONDict] = []
    for doc in meta.get("documents", []) if isinstance(meta.get("documents"), list) else []:
        if not isinstance(doc, dict):
            continue
        item = dict(doc)
        relative = item.get("relativePath")
        if isinstance(relative, str):
            upload_path = inv_dir(meta["investigationId"]) / relative
            text_record = upload_path.with_suffix(upload_path.suffix + ".text.json")
            extracted = read_json(text_record, {})
            if isinstance(extracted, dict):
                item["text"] = extracted.get("text") or item.get("excerpt") or ""
                item["textExtractAvailable"] = bool(extracted.get("textExtractAvailable"))
                item["extractionMethod"] = extracted.get("extractionMethod") or item.get("extractionMethod")
        docs.append(item)
    return docs


def sarg_history_from_turns(turns: List[JSONDict]) -> List[JSONDict]:
    history: List[JSONDict] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        result = turn.get("result") if isinstance(turn.get("result"), dict) else {}
        history.append({
            "turnId": turn.get("turnId"),
            "question": turn.get("question") or result.get("question"),
            "answer": result.get("answer") or (result.get("brief") or {}).get("summary"),
            "analysisBlocks": result.get("analysisBlocks") or [],
            "selected_reasoning_paths": result.get("selected_reasoning_paths") or [],
            "gap_assessment": result.get("gap_assessment") or {},
            "open_questions": result.get("open_questions") or [],
        })
    return history


def selected_graph_context_from_request(investigation_id: str, request: TurnRequest) -> List[JSONDict]:
    selected: List[JSONDict] = []
    for item in (request.selected_graph_context or request.selectedGraphContext or []):
        if isinstance(item, dict):
            selected.append(dict(item))

    workspace = read_workspace_state(investigation_id)
    selected_drawer = workspace.get("selectedDrawerItem")
    if isinstance(selected_drawer, dict):
        selected.append(dict(selected_drawer))

    pins = workspace.get("pins") if isinstance(workspace.get("pins"), dict) else {}
    bookmarks = workspace.get("bookmarks") if isinstance(workspace.get("bookmarks"), dict) else {}
    for source in [pins, bookmarks]:
        for graph_id in source.get("graphItems", []) if isinstance(source.get("graphItems"), list) else []:
            selected.append({"id": str(graph_id), "source": "workspace_state"})

    deduped: List[JSONDict] = []
    seen = set()
    for item in selected:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def sarg_seed_from_saved_graph(investigation_id: str, question: str) -> Optional[JSONDict]:
    graph = read_sarg_graph(investigation_id)
    if not has_sarg_graph(graph):
        return None
    return {
        "question": question,
        "status": "seeded_from_investigation",
        "stop_reason": "backend_supplied_existing_sarg_local_graph",
        "local_reasoning_subgraph": graph,
    }


def sarg_drawer_from_result(sarg_result: JSONDict, meta: JSONDict) -> JSONDict:
    evidence_items: List[JSONDict] = []
    source_by_key: Dict[str, JSONDict] = {}

    for block in sarg_result.get("analysisBlocks") or []:
        if not isinstance(block, dict) or block.get("type") != "evidence":
            continue
        data = block.get("data") if isinstance(block.get("data"), dict) else {}
        for item in data.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or item.get("evidence_id") or f"evidence_{len(evidence_items)+1}")
            source_title = first_text(item.get("sourceTitle"), first_text(item.get("source_title"), "Evidence source"))
            source_url = item.get("sourceUrl") or item.get("source_url")
            text = short_text(item.get("text"), 420)
            evidence_items.append({
                "id": evidence_id,
                "title": source_title if source_title != "Evidence source" else "Evidence excerpt",
                "source": source_title,
                "date": "Evidence block",
                "text": text,
                "support": "Supporting evidence",
                "sourceUrl": source_url,
            })
            source_key = str(source_url or source_title or evidence_id)
            source_by_key.setdefault(source_key, {
                "id": f"source_{safe_filename(source_key, 'sarg_source')}",
                "logo": "S",
                "title": source_title,
                "date": "Evidence source",
                "excerpt": text,
                "tag": first_text(item.get("blockType"), first_text(item.get("block_type"), "SARG evidence")),
                "sourceUrl": source_url,
            })

    if not evidence_items:
        evidence_items = upload_evidence(meta.get("documents", []) if isinstance(meta.get("documents"), list) else [])
    sources = upload_sources(meta.get("documents", []) if isinstance(meta.get("documents"), list) else []) + list(source_by_key.values())

    missing_items: List[JSONDict] = []
    gap = sarg_result.get("gap_assessment") if isinstance(sarg_result.get("gap_assessment"), dict) else {}
    for item in gap.get("items") or []:
        if isinstance(item, dict):
            missing_items.append(dict(item))
        else:
            missing_items.append({"text": str(item), "severity": "medium", "source": "gap_assessment"})
    for item in sarg_result.get("open_questions") or []:
        if isinstance(item, dict) and item not in missing_items:
            missing_items.append(dict(item))

    assumptions = [
        {"title": "Missing evidence", "text": short_text(item.get("text") or item.get("question") or item, 320)}
        for item in missing_items[:8]
        if isinstance(item, dict)
    ]

    return {
        "sources": sources[:16],
        "evidence": evidence_items[:12],
        "assumptions": assumptions,
    }


def sarg_missing_evidence_payload(sarg_result: JSONDict) -> JSONDict:
    gap = sarg_result.get("gap_assessment") if isinstance(sarg_result.get("gap_assessment"), dict) else {}
    items: List[str] = []
    for item in gap.get("items") or []:
        text = item.get("text") if isinstance(item, dict) else str(item)
        if text and text not in items:
            items.append(str(text))
    for item in sarg_result.get("open_questions") or []:
        text = item.get("question") if isinstance(item, dict) else str(item)
        if text and text not in items:
            items.append(str(text))
    if not items and gap.get("summary") and gap.get("status") != "sufficient":
        items.append(str(gap.get("summary")))
    return {"totalCount": len(items), "items": items[:8], "summary": gap.get("summary")}


def clean_story_text(value: Any, max_chars: int = 700) -> str:
    text = str(value or "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\d+\.\s+", "", text).strip()

    if not text:
        return ""

    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def story_sentences(value: Any) -> List[str]:
    text = clean_story_text(value, 5000)

    if not text:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    boilerplate_prefixes = (
        "based on the selected reasoning paths",
        "based on the available evidence",
        "the current investigation",
        "the disruption in kachin rare-earth mining can affect",
    )

    filtered = [
        sentence
        for sentence in sentences
        if not sentence.lower().startswith(boilerplate_prefixes)
    ]

    return filtered or sentences


def answer_section(answer: Any, heading: str, max_chars: int = 900) -> str:
    text = str(answer or "")

    if not text.strip():
        return ""

    pattern = (
        rf"(?:\*\*)?{re.escape(heading)}(?:\*\*)?"
        rf"\s*:?\s*(.*?)"
        rf"(?=(?:\*\*)?[A-Z][A-Za-z0-9 /&,\-]{{2,90}}(?:\*\*)?\s*:|\Z)"
    )

    match = re.search(pattern, text, flags=re.S | re.I)

    if not match:
        return ""

    return clean_story_text(match.group(1), max_chars)


def concise_answer_takeaway(answer: Any) -> str:
    summary = answer_section(answer, "Summary", 420)

    if summary:
        sentences = story_sentences(summary)
        if sentences:
            return clean_story_text(" ".join(sentences[:2]), 360)

    sentences = story_sentences(answer)

    if not sentences:
        return "No answer was produced."

    return clean_story_text(" ".join(sentences[:2]), 360)


def selected_context_summary(selected_graph_context: List[JSONDict]) -> str:
    if not selected_graph_context:
        return ""

    labels: List[str] = []
    relations: List[str] = []

    for item in selected_graph_context:
        if not isinstance(item, dict):
            continue

        label = first_text(
            item.get("label"),
            first_text(
                item.get("name"),
                first_text(item.get("title"), first_text(item.get("id"), "")),
            ),
        )

        item_type = first_text(
            item.get("type"),
            first_text(item.get("taxonomyType"), first_text(item.get("entity_type"), "")),
        )

        relation = first_text(item.get("relation"), "")

        if label and label not in labels:
            labels.append(f"{item_type}: {label}" if item_type else label)

        if relation and relation not in relations:
            relations.append(relation)

    parts: List[str] = []

    if labels:
        parts.append("; ".join(labels[:5]))

    if relations:
        parts.append("Relations: " + ", ".join(relations[:4]))

    return clean_story_text(" | ".join(parts), 420)


def selected_path_summary(sarg_result: JSONDict) -> str:
    paths = sarg_result.get("selected_reasoning_paths")
    paths = paths if isinstance(paths, list) else []

    if not paths:
        return ""

    first = paths[0] if isinstance(paths[0], dict) else {}
    nodes = first.get("nodes") if isinstance(first.get("nodes"), list) else []
    labels = [
        str(item.get("label"))
        for item in nodes
        if isinstance(item, dict) and item.get("label")
    ]

    if labels:
        path_text = " → ".join(labels[:5])
    else:
        path_text = clean_story_text(first.get("summary") or first.get("description"), 260)

    if not path_text:
        return ""

    extra_count = max(0, len(paths) - 1)
    suffix = f" There are {extra_count} additional selected path(s)." if extra_count else ""

    return clean_story_text(f"The main selected path is: {path_text}.{suffix}", 420)


def evidence_basis_summary(sarg_result: JSONDict) -> str:
    evidence_count = 0

    for block in sarg_result.get("analysisBlocks") or []:
        if not isinstance(block, dict):
            continue

        if block.get("type") != "evidence":
            continue

        data = block.get("data") if isinstance(block.get("data"), dict) else {}
        evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
        evidence_count += len(evidence)

    if evidence_count:
        return f"The answer is grounded in {evidence_count} retrieved evidence item(s), with source details preserved in the evidence drawer."

    graph = sarg_result.get("local_reasoning_subgraph") if isinstance(sarg_result.get("local_reasoning_subgraph"), dict) else {}
    evidence = graph.get("evidence")

    if isinstance(evidence, dict) and evidence:
        return f"The answer is grounded in {len(evidence)} evidence item(s) attached to the local reasoning graph."

    if isinstance(evidence, list) and evidence:
        return f"The answer is grounded in {len(evidence)} evidence item(s) attached to the local reasoning graph."

    return ""


def limitation_summary(sarg_result: JSONDict) -> str:
    answer = sarg_result.get("answer") or ""
    limitations = answer_section(answer, "Limitations", 700)

    if limitations:
        return limitations

    gap = sarg_result.get("gap_assessment") if isinstance(sarg_result.get("gap_assessment"), dict) else {}

    items: List[str] = []

    for item in gap.get("items") or []:
        if isinstance(item, dict):
            text = item.get("text") or item.get("question") or item.get("gap")
        else:
            text = str(item)

        text = clean_story_text(text, 220)

        if text and text not in items:
            items.append(text)

    for item in sarg_result.get("open_questions") or []:
        if isinstance(item, dict):
            text = item.get("question") or item.get("text") or item.get("gap")
        else:
            text = str(item)

        text = clean_story_text(text, 220)

        if text and text not in items:
            items.append(text)

    if items:
        return " ".join(items[:3])

    if gap.get("status") and gap.get("status") != "sufficient":
        return clean_story_text(gap.get("summary"), 500)

    return ""


def story_block(
    block_id: str,
    title: str,
    lead: str,
    body: str,
    *,
    evidence_ids: Optional[List[str]] = None,
    graph_item_ids: Optional[List[str]] = None,
) -> JSONDict:
    return {
        "id": block_id,
        "type": "text",
        "title": title,
        "data": {
            "lead": lead,
            "body": clean_story_text(body, 900),
        },
        "meta": {
            "evidenceIds": evidence_ids or [],
            "graphItemIds": graph_item_ids or [],
        },
    }


def sarg_story_payload(
    question: str,
    sarg_result: JSONDict,
    selected_graph_context: List[JSONDict],
) -> JSONDict:
    answer = first_text(sarg_result.get("answer"), "No answer was produced.")
    gap = sarg_result.get("gap_assessment") if isinstance(sarg_result.get("gap_assessment"), dict) else {}

    brief_summary = concise_answer_takeaway(answer)
    lead = "Partial." if gap.get("status") in {"partial", "insufficient"} else "Takeaway."

    blocks: List[JSONDict] = []

    context = selected_context_summary(selected_graph_context)
    if context:
        blocks.append(story_block(
            "selected_context_story",
            "KG context used",
            "This turn is anchored to the selected graph context.",
            context,
        ))

    answer_sentences = story_sentences(answer)
    answer_body = " ".join(answer_sentences[:3]) if answer_sentences else brief_summary

    blocks.append(story_block(
        "answer_story",
        "Answer",
        "The main analytical conclusion.",
        answer_body,
    ))

    mechanism = (
        answer_section(answer, "Mechanisms Linking Mining Disruption to Supply Chain Changes", 800)
        or answer_section(answer, "Impact on Downstream Rare-Earth Supply Chains", 800)
        or answer_section(answer, "Effects on Availability and Flow within Downstream Rare-Earth Supply Chains", 800)
    )

    if mechanism:
        blocks.append(story_block(
            "mechanism_story",
            "Mechanism",
            "How the disruption propagates through the supply chain.",
            mechanism,
        ))

    path_summary = selected_path_summary(sarg_result)
    if path_summary:
        blocks.append(story_block(
            "reasoning_basis_story",
            "Reasoning basis",
            "The selected path explains why this answer follows from the graph.",
            path_summary,
        ))

    evidence_summary = evidence_basis_summary(sarg_result)
    if evidence_summary:
        blocks.append(story_block(
            "evidence_basis_story",
            "Evidence basis",
            "Source grounding is available without repeating the full evidence excerpt here.",
            evidence_summary,
        ))

    limits = limitation_summary(sarg_result)
    if limits:
        blocks.append(story_block(
            "limits_story",
            "What remains unresolved",
            "The current evidence does not fully settle this part.",
            limits,
        ))

    return {
        "brief": {
            "answerLead": lead,
            "summary": brief_summary,
        },
        "analysisBlocks": blocks[:6],
    }


def sarg_brief_payload(question: str, sarg_result: JSONDict) -> JSONDict:
    blocks = sarg_result.get("analysisBlocks") if isinstance(sarg_result.get("analysisBlocks"), list) else []
    brief_block = next(
        (
            block for block in blocks
            if isinstance(block, dict)
            and block.get("id") == "brief"
            and isinstance(block.get("data"), dict)
        ),
        {},
    )

    data = brief_block.get("data") if isinstance(brief_block.get("data"), dict) else {}
    answer = first_text(data.get("lead"), first_text(sarg_result.get("answer"), "No answer was produced."))

    gap = sarg_result.get("gap_assessment") if isinstance(sarg_result.get("gap_assessment"), dict) else {}
    lead = "Partial." if gap.get("status") in {"partial", "insufficient"} else "Brief."

    return {
        "answerLead": lead,
        "summary": clean_story_text(answer, 320),
    }


def sarg_reasoning_path_payload(sarg_result: JSONDict) -> List[JSONDict]:
    paths = sarg_result.get("selected_reasoning_paths") if isinstance(sarg_result.get("selected_reasoning_paths"), list) else []

    if not paths:
        return []

    first = paths[0] if isinstance(paths[0], dict) else {}
    nodes = first.get("nodes") if isinstance(first.get("nodes"), list) else []
    labels = [
        str(item.get("label"))
        for item in nodes
        if isinstance(item, dict) and item.get("label")
    ]

    return [{"label": label} for label in labels[:6]]


def build_live_workspace_result(question: str, meta: JSONDict, request: TurnRequest) -> JSONDict:
    if SARG is None or SARGConfig is None:
        raise RuntimeError("sarg.py could not be imported by ui_backend.app")

    prior_turns = load_turns(meta)
    investigation_history = sarg_history_from_turns(prior_turns)
    selected_graph_context = selected_graph_context_from_request(meta["investigationId"], request)
    kg_seed = sarg_seed_from_saved_graph(meta["investigationId"], question)
    document_jsons = document_jsons_from_meta(meta)

    normalize_investigation_scope(meta)

    config = SARGConfig(
        model="gpt-4.1-mini",
        embed_model="text-embedding-3-small",
        investigation_id=meta["investigationId"],
        run_id=meta.get("run_id"),
        corpus_id=meta.get("corpus_id"),
        branch_id=meta.get("branch_id"),
        enable_risk_tools=True,
        enable_forecasting=True,
    )
    sarg = SARG(config)
    sarg_result = sarg.run(
        question=question,
        kg_irag_result=kg_seed,
        investigation_id=meta["investigationId"],
        investigation_history=investigation_history,
        selected_graph_context=selected_graph_context,
        document_jsons=document_jsons,
        risk_model=None,
    )

    updated_graph = sarg_result.get("local_reasoning_subgraph") or (kg_seed or {}).get("local_reasoning_subgraph") or {}
    if has_sarg_graph(updated_graph):
        save_sarg_graph(meta["investigationId"], updated_graph)

    drawer = sarg_drawer_from_result(sarg_result, meta)
    evidence_cards = drawer.get("evidence") or []
    risk_payload = risk_assessment(sarg_result.get("risk_analysis") or {})
    forecast_payload_ = forecast_check(sarg_result.get("forecast_analysis") or {})
    missing_payload = sarg_missing_evidence_payload(sarg_result)
    story_payload = sarg_story_payload(question, sarg_result, selected_graph_context)

    return {
        **sarg_result,
        "question": question,
        "status": sarg_result.get("status") or "answered",
        "currentInvestigation": investigation_summary(meta),
        "progress": {"stages": [
            {"id": "search", "label": "Searching relevant sources", "status": "complete"},
            {"id": "support", "label": "Checking support", "status": "complete"},
            {"id": "risk", "label": "Assessing risk", "status": "complete"},
            {"id": "gaps", "label": "Reviewing gaps", "status": "complete"},
        ], "loopMessage": ""},
        "brief": sarg_brief_payload(question, sarg_result),
        "analysisBlocks": sarg_result.get("analysisBlocks") or [],
        "evidenceSupport": {
            "totalCount": len(evidence_cards),
            "evidence": [
                {"id": item.get("id"), "text": item.get("text"), "source": item.get("source"), "date": item.get("date") or "Evidence block"}
                for item in evidence_cards[:5]
            ],
        },
        "reasoningPath": sarg_reasoning_path_payload(sarg_result),
        "riskAssessment": risk_payload,
        "forecastCheck": forecast_payload_,
        "missingEvidence": missing_payload,
        "drawer": drawer,
        "selectedGraphContext": selected_graph_context,
        "sargLocalGraph": sarg_graph_counts(updated_graph) if isinstance(updated_graph, dict) else {},
    }

def build_workspace_result(question: str, meta: JSONDict) -> JSONDict:
    artifacts = load_artifacts(newest_existing_run_dir(meta.get("run_id")))
    sarg = artifacts.get("sarg") if isinstance(artifacts.get("sarg"), dict) else {}
    risk = artifacts.get("risk") if isinstance(artifacts.get("risk"), dict) else {}
    forecast = artifacts.get("forecast") if isinstance(artifacts.get("forecast"), dict) else {}
    postrag_records = artifacts.get("postrag_records") if isinstance(artifacts.get("postrag_records"), list) else []
    docs = meta.get("documents") if isinstance(meta.get("documents"), list) else []
    graph = read_subgraph(meta["investigationId"])

    lookup = evidence_lookup(postrag_records)
    evidence_ids = collect_evidence_ids(sarg, risk) or list(lookup.keys())[:8]
    evidence_cards = (upload_evidence(docs) + artifact_evidence_cards(evidence_ids, lookup))[:10]
    if not evidence_cards:
        evidence_cards = [{"id": "evidence_pending", "title": "Evidence pending", "source": "Lanthic workspace", "date": "Not yet available", "text": "No evidence excerpts were available. Add documents or run ingestion before relying on this answer.", "support": "Missing evidence"}]

    sources = (upload_sources(docs) + artifact_sources(postrag_records))[:16]
    missing = missing_evidence(sarg, risk, forecast, docs)
    return {
        "question": question,
        "status": "complete",
        "currentInvestigation": investigation_summary(meta),
        "progress": {"stages": [
            {"id": "search", "label": "Searching relevant sources", "status": "complete"},
            {"id": "support", "label": "Checking support", "status": "complete"},
            {"id": "risk", "label": "Assessing risk", "status": "complete"},
            {"id": "gaps", "label": "Reviewing gaps", "status": "complete"},
        ], "loopMessage": ""},
        "brief": brief(question, sarg, risk, docs),
        "evidenceSupport": {"totalCount": len(evidence_cards), "evidence": [{"id": item["id"], "text": item["text"], "source": item["source"], "date": item.get("date") or "Evidence block"} for item in evidence_cards[:5]]},
        "reasoningPath": reasoning_path(sarg, graph),
        "riskAssessment": risk_assessment(risk),
        "forecastCheck": forecast_check(forecast),
        "missingEvidence": missing,
        "drawer": {"sources": sources, "evidence": evidence_cards, "assumptions": assumptions_from_missing(missing, forecast)},
    }


def expand_subgraph_with_documents(meta: JSONDict, docs: List[JSONDict]) -> None:
    graph = read_subgraph(meta["investigationId"])
    inv_node = f"investigation:{meta['investigationId']}"
    for doc in docs:
        doc_id = doc.get("documentId")
        if not doc_id:
            continue
        source_node = f"source:{doc_id}"
        upsert_node(graph, {"id": source_node, "type": "source", "label": doc.get("sourceTitle") or doc.get("filename"), "createdAt": doc.get("uploadedAt"), "metadata": {"filename": doc.get("filename"), "sizeBytes": doc.get("sizeBytes"), "contentType": doc.get("contentType"), "textExtractAvailable": doc.get("textExtractAvailable")}})
        upsert_edge(graph, inv_node, source_node, "has_uploaded_source", createdAt=doc.get("uploadedAt"))
    save_subgraph(meta["investigationId"], graph)


def expand_subgraph_with_turn(meta: JSONDict, turn: JSONDict, result: JSONDict) -> None:
    graph = read_subgraph(meta["investigationId"])
    inv_node = f"investigation:{meta['investigationId']}"
    turn_node = f"turn:{turn['turnId']}"
    upsert_node(graph, {"id": turn_node, "type": "turn", "label": turn.get("question"), "createdAt": turn.get("createdAt"), "metadata": {"status": turn.get("status")}})
    upsert_edge(graph, inv_node, turn_node, "has_turn", createdAt=turn.get("createdAt"))
    for item in result.get("drawer", {}).get("sources", []):
        source_id = item.get("id")
        if source_id:
            node = f"source:{source_id}"
            upsert_node(graph, {"id": node, "type": "source", "label": item.get("title"), "metadata": {"date": item.get("date"), "tag": item.get("tag")}})
            upsert_edge(graph, turn_node, node, "uses_source", createdAt=turn.get("createdAt"))
    for item in result.get("drawer", {}).get("evidence", []):
        evidence_id = item.get("id")
        if evidence_id:
            node = f"evidence:{evidence_id}"
            upsert_node(graph, {"id": node, "type": "evidence", "label": item.get("title"), "text": item.get("text"), "metadata": {"source": item.get("source"), "support": item.get("support")}})
            upsert_edge(graph, turn_node, node, "supported_by", createdAt=turn.get("createdAt"))
    risk = result.get("riskAssessment", {})
    risk_node = f"risk:{turn['turnId']}"
    upsert_node(graph, {"id": risk_node, "type": "risk_assessment", "label": risk.get("overallRisk"), "text": risk.get("summary"), "metadata": {"factors": risk.get("factors", [])}})
    upsert_edge(graph, turn_node, risk_node, "produces_risk_assessment", createdAt=turn.get("createdAt"))
    for index, item in enumerate(result.get("missingEvidence", {}).get("items", []), start=1):
        gap = f"gap:{turn['turnId']}:{index}"
        upsert_node(graph, {"id": gap, "type": "missing_evidence", "label": str(item)})
        upsert_edge(graph, turn_node, gap, "identifies_gap", createdAt=turn.get("createdAt"))
    save_subgraph(meta["investigationId"], graph)


@app.on_event("startup")
def startup() -> None:
    ensure_default_investigation()


@app.get("/api/health")
def health() -> JSONDict:
    ensure_default_investigation()
    return {"ok": True, "service": "lanthic-ui-backend", "stateDir": str(STATE_DIR.relative_to(PROJECT_ROOT))}


@app.post("/api/session/create")
def create_session(request: SessionCreateRequest) -> JSONDict:
    meta = ensure_default_investigation()
    if request.run_id is not None:
        meta["run_id"] = normalize_run_id(request.run_id)
    else:
        meta["run_id"] = normalize_run_id(meta.get("run_id"))

    if request.corpus_id is not None:
        meta["corpus_id"] = normalize_corpus_id(request.corpus_id)
    else:
        meta["corpus_id"] = normalize_corpus_id(meta.get("corpus_id"))

    if request.branch_id is not None:
        meta["branch_id"] = normalize_branch_id(request.branch_id)
    else:
        meta["branch_id"] = normalize_branch_id(meta.get("branch_id"))

    save_meta(meta)
    return {"session_id": f"sess_{uuid.uuid4().hex[:12]}", "workspace": request.workspace, "user": DEFAULT_USER, "investigationId": meta["investigationId"], "run_id": meta.get("run_id"), "corpus_id": meta.get("corpus_id"), "branch_id": meta.get("branch_id"), "mode": "local"}


@app.post("/api/auth/sign-in")
def sign_in(request: SignInRequest) -> JSONDict:
    meta = ensure_default_investigation()
    email = request.email or DEFAULT_USER["email"]
    return {"session_id": f"sess_{uuid.uuid4().hex[:12]}", "workspace": request.workspace or "demo", "user": {"name": email.split("@")[0], "email": email}, "investigationId": meta["investigationId"], "run_id": meta.get("run_id"), "corpus_id": meta.get("corpus_id"), "branch_id": meta.get("branch_id"), "mode": "local"}


@app.get("/api/investigations")
def list_investigations() -> JSONDict:
    return {"investigations": [investigation_summary(meta) for meta in list_metas()]}


@app.post("/api/investigations")
def create_investigation(request: CreateInvestigationRequest) -> JSONDict:
    meta = create_investigation_record(title=request.title, question=request.question, run_id=request.run_id, corpus_id=request.corpus_id, branch_id=request.branch_id)
    return investigation_response(meta, include_turns=True)


@app.get("/api/investigations/{investigation_id}")
def get_investigation(investigation_id: str) -> JSONDict:
    return investigation_response(get_meta_or_404(investigation_id), include_turns=True)

@app.get("/api/investigations/{investigation_id}/workspace-state")
def get_workspace_state(investigation_id: str) -> JSONDict:
    get_meta_or_404(investigation_id)
    state = read_workspace_state(investigation_id)
    save_workspace_state(investigation_id, state)
    return state


@app.post("/api/investigations/{investigation_id}/workspace-state")
def update_workspace_state(
    investigation_id: str,
    request: WorkspaceStateRequest,
) -> JSONDict:
    get_meta_or_404(investigation_id)
    return save_workspace_state(
        investigation_id,
        request.dict(exclude_none=True),
    )


@app.post("/api/investigations/{investigation_id}/turns")
def run_turn(investigation_id: str, request: TurnRequest) -> JSONDict:
    meta = get_meta_or_404(investigation_id)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.run_id is not None:
        meta["run_id"] = normalize_run_id(request.run_id)
    else:
        meta["run_id"] = normalize_run_id(meta.get("run_id"))

    if request.corpus_id is not None:
        meta["corpus_id"] = normalize_corpus_id(request.corpus_id)
    else:
        meta["corpus_id"] = normalize_corpus_id(meta.get("corpus_id"))

    if request.branch_id is not None:
        meta["branch_id"] = normalize_branch_id(request.branch_id)
    else:
        meta["branch_id"] = normalize_branch_id(meta.get("branch_id"))

    if not meta.get("turnIds"):
        meta["title"] = question
        meta["chatName"] = question

    meta["status"] = "Running"
    save_meta(meta)

    created = now_iso()
    turn_id = f"turn_{uuid.uuid4().hex[:10]}"

    try:
        result = build_live_workspace_result(question, meta, request)
        turn_status = "complete"
        meta_status = "Answer ready"
    except Exception as error:
        # Keep a saved turn so the UI can display the failure and the investigation
        # history remains auditable. Do not silently fall back to stale demo artifacts.
        result = {
            "question": question,
            "status": "failed",
            "error": str(error),
            "currentInvestigation": investigation_summary(meta),
            "progress": {"stages": [
                {"id": "search", "label": "Searching relevant sources", "status": "failed"},
                {"id": "support", "label": "Checking support", "status": "pending"},
                {"id": "risk", "label": "Assessing risk", "status": "pending"},
                {"id": "gaps", "label": "Reviewing gaps", "status": "pending"},
            ], "loopMessage": str(error)},
            "brief": {"answerLead": "Error.", "summary": f"Live SARG execution failed: {error}"},
            "evidenceSupport": {"totalCount": 0, "evidence": []},
            "reasoningPath": [{"label": "Live SARG execution failed"}],
            "riskAssessment": risk_assessment({}),
            "forecastCheck": forecast_check({"status": "unavailable", "reason": "SARG execution failed before forecasting."}),
            "missingEvidence": {"totalCount": 1, "items": ["Successful live SARG execution"], "summary": str(error)},
            "drawer": {"sources": upload_sources(meta.get("documents", []) if isinstance(meta.get("documents"), list) else []), "evidence": [], "assumptions": [{"title": "Execution failure", "text": str(error)}]},
            "analysisBlocks": [{
                "id": "backend_error",
                "type": "text",
                "title": "Backend error",
                "data": {"lead": "Live SARG execution failed.", "body": str(error)},
                "meta": {"evidenceIds": [], "graphItemIds": []},
            }],
        }
        turn_status = "failed"
        meta_status = "SARG failed"

    turn = {
        "turnId": turn_id,
        "question": question,
        "createdAt": created,
        "status": turn_status,
        "result": result,
    }
    write_turn(investigation_id, turn)
    save_sarg_result(investigation_id, turn_id, result)

    meta.setdefault("turnIds", []).append(turn_id)
    meta["status"] = meta_status
    meta["lastRunAt"] = created
    save_meta(meta)

    expand_subgraph_with_turn(meta, turn, result)

    return {
        "investigationId": investigation_id,
        "turnId": turn_id,
        **result,
        "turns": load_turns(meta),
        "recentInvestigations": [investigation_summary(item) for item in list_metas()],
    }

@app.post("/api/investigations/{investigation_id}/documents")
async def add_documents(investigation_id: str, files: List[UploadFile] = File(...)) -> JSONDict:
    meta = get_meta_or_404(investigation_id)
    target_dir = uploads_dir(investigation_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[JSONDict] = []

    for upload in files:
        original = safe_filename(upload.filename or f"upload_{uuid.uuid4().hex[:8]}")
        document_id = f"doc_{uuid.uuid4().hex[:10]}"
        target = target_dir / f"{document_id}_{original}"
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        text, text_available, extraction_method = extract_text(target, upload.content_type)
        text_record = target.with_suffix(target.suffix + ".text.json")
        write_json(text_record, {"documentId": document_id, "filename": original, "contentType": upload.content_type, "textExtractAvailable": text_available, "extractionMethod": extraction_method, "text": text})
        saved.append({
            "documentId": document_id,
            "filename": original,
            "storedFilename": target.name,
            "relativePath": str(target.relative_to(inv_dir(investigation_id))),
            "sizeBytes": target.stat().st_size,
            "contentType": upload.content_type or "application/octet-stream",
            "uploadedAt": now_iso(),
            "sha256": sha256_file(target),
            "sourceTitle": clean_title(original, original),
            "textExtractAvailable": text_available,
            "extractionMethod": extraction_method,
            "excerpt": short_text(text, 420) if text else "Uploaded file is preserved locally. Automatic text extraction is pending or unavailable.",
        })

    meta.setdefault("documents", []).extend(saved)
    meta["status"] = "Documents attached"
    save_meta(meta)
    expand_subgraph_with_documents(meta, saved)
    return {"ok": True, "message": f"Attached {len(saved)} document{'s' if len(saved) != 1 else ''}. Re-run the investigation to include them in the evidence drawer.", "documents": [public_document(doc) for doc in saved], "investigation": investigation_summary(meta), "localSubgraph": subgraph_summary(investigation_id)}


@app.get("/api/investigations/{investigation_id}/export", response_class=PlainTextResponse)
def export_investigation(investigation_id: str) -> str:
    meta = get_meta_or_404(investigation_id)
    turns = load_turns(meta)
    latest = turns[-1] if turns else None
    result = latest.get("result") if isinstance(latest, dict) else build_workspace_result(meta.get("title", DEFAULT_QUESTION), meta)
    lines = [
        f"# {meta.get('title', 'Lanthic investigation')}",
        "",
        f"Created: {human_date(meta.get('createdAt'))}",
        f"Last modified: {human_date(meta.get('updatedAt'))}",
        f"Uploaded documents: {len(meta.get('documents', []))}",
        f"Turns: {len(turns)}",
        "",
        "## Brief",
        result.get("brief", {}).get("summary", "No brief available."),
        "",
        "## Reasoning path",
    ]
    for item in result.get("reasoningPath", []):
        lines.append(f"- {item.get('label')}")
    lines.extend(["", "## Risk assessment", result.get("riskAssessment", {}).get("summary", "No risk assessment available."), "", "## Evidence"])
    for item in result.get("drawer", {}).get("evidence", []):
        lines.append(f"- **{item.get('title')}** ({item.get('source') or 'Source'}): {item.get('text')}")
    lines.extend(["", "## Sources"])
    for item in result.get("drawer", {}).get("sources", []):
        lines.append(f"- **{item.get('title')}** — {item.get('date')} — {item.get('tag')}")
    lines.extend(["", "## Missing evidence"])
    for item in result.get("missingEvidence", {}).get("items", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Uploaded documents"])
    for doc in meta.get("documents", []):
        lines.append(f"- {doc.get('filename')} ({doc.get('sizeBytes')} bytes, uploaded {human_date(doc.get('uploadedAt'))})")
    return "\n".join(lines) + "\n"

def candidate_evidence_ids(candidate: JSONDict) -> List[str]:
    ids: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in ids:
            ids.append(value)

    postrag = candidate.get("postrag") if isinstance(candidate.get("postrag"), dict) else {}

    for evidence_id in postrag.get("used_evidence_ids", []):
        add(evidence_id)

    for item in candidate.get("provenance", []):
        if isinstance(item, dict):
            add(item.get("evidence_id"))

    for item in candidate.get("postrag_evidence", []):
        if isinstance(item, dict):
            add(item.get("evidence_id"))

    return ids


def candidate_evidence_cards(candidate: JSONDict, lookup: Dict[str, JSONDict]) -> List[JSONDict]:
    evidence_ids = candidate_evidence_ids(candidate)
    cards = artifact_evidence_cards(evidence_ids, lookup)

    if not cards:
        return []

    return cards[:6]


def candidate_decision(candidate: JSONDict) -> str:
    postrag = candidate.get("postrag") if isinstance(candidate.get("postrag"), dict) else {}
    return first_text(postrag.get("decision"), "unvalidated")


def candidate_grounding_score(candidate: JSONDict) -> Optional[float]:
    postrag = candidate.get("postrag") if isinstance(candidate.get("postrag"), dict) else {}
    value = postrag.get("grounding_score")

    try:
        return float(value)
    except Exception:
        return None


def accepted_or_useful_candidate(candidate: JSONDict) -> bool:
    decision = candidate_decision(candidate).lower()
    return decision in {"accept", "coarsen", "accepted"}


def build_taxonomy_subgraph(meta: JSONDict) -> JSONDict:
    artifacts = load_artifacts(newest_existing_run_dir(meta.get("run_id")))
    postrag_records = artifacts.get("postrag_records") if isinstance(artifacts.get("postrag_records"), list) else []
    lookup = evidence_lookup(postrag_records)

    nodes_by_id: Dict[str, JSONDict] = {}
    edges: List[JSONDict] = []

    max_nodes = 140
    max_edges = 220
    truncated = False

    for record_index, record in enumerate(postrag_records, start=1):
        if not isinstance(record, dict):
            continue

        source_key = first_text(record.get("source_id"), f"source_{record_index}")
        filtered = record.get("postrag_filtered") if isinstance(record.get("postrag_filtered"), dict) else {}
        entities = filtered.get("entities") if isinstance(filtered.get("entities"), list) else record.get("entities", [])
        relations = filtered.get("relations") if isinstance(filtered.get("relations"), list) else record.get("relations", [])

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        entity_name_to_node_id: Dict[str, str] = {}
        entity_id_to_node_id: Dict[str, str] = {}

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_id = first_text(entity.get("entity_id"), "")
            canonical_name = first_text(entity.get("canonical_name"), first_text(entity.get("name"), entity_id))
            entity_type = first_text(entity.get("entity_type"), "entity").lower()

            if not canonical_name:
                continue

            node_id = f"entity:{source_key}:{entity_id or safe_filename(canonical_name, 'entity')}"
            entity_id_to_node_id[entity_id] = node_id
            entity_name_to_node_id[canonical_name.lower()] = node_id

            if node_id not in nodes_by_id:
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "type": entity_type,
                    "taxonomyType": entity_type,
                    "label": canonical_name,
                    "text": first_text(entity.get("description"), ""),
                    "evidence": candidate_evidence_cards(entity, lookup),
                    "metadata": {
                        "kind": "entity",
                        "entityId": entity_id,
                        "entityType": entity_type,
                        "aliases": entity.get("aliases", []),
                        "temporal": entity.get("temporal", {}),
                        "postragDecision": candidate_decision(entity),
                        "groundingScore": candidate_grounding_score(entity),
                    },
                }

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            if not accepted_or_useful_candidate(relation):
                continue

            subject_id = first_text(relation.get("subject_id"), "")
            object_id = first_text(relation.get("object_id"), "")
            subject_name = first_text(relation.get("subject"), "")
            object_name = first_text(relation.get("object"), "")
            relation_id = first_text(relation.get("relation_id"), f"r_{len(edges) + 1}")
            relation_type = first_text(relation.get("relation_type"), "related_to")

            source_node = entity_id_to_node_id.get(subject_id) or entity_name_to_node_id.get(subject_name.lower())
            target_node = entity_id_to_node_id.get(object_id) or entity_name_to_node_id.get(object_name.lower())

            if not source_node and subject_name:
                source_node = f"entity:{source_key}:subject:{safe_filename(subject_name, 'subject')}"
                nodes_by_id.setdefault(source_node, {
                    "id": source_node,
                    "type": "entity",
                    "taxonomyType": "entity",
                    "label": subject_name,
                    "text": "",
                    "evidence": [],
                    "metadata": {"kind": "entity", "entityType": "entity"},
                })

            if not target_node and object_name:
                target_node = f"entity:{source_key}:object:{safe_filename(object_name, 'object')}"
                nodes_by_id.setdefault(target_node, {
                    "id": target_node,
                    "type": "entity",
                    "taxonomyType": "entity",
                    "label": object_name,
                    "text": "",
                    "evidence": [],
                    "metadata": {"kind": "entity", "entityType": "entity"},
                })

            if not source_node or not target_node:
                continue

            if len(nodes_by_id) > max_nodes or len(edges) > max_edges:
                truncated = True
                continue

            edges.append({
                "id": f"relation:{source_key}:{relation_id}",
                "source": source_node,
                "target": target_node,
                "relation": relation_type,
                "label": relation_type.replace("_", " "),
                "type": "relation",
                "text": first_text(relation.get("description"), ""),
                "evidence": candidate_evidence_cards(relation, lookup),
                "metadata": {
                    "kind": "relation",
                    "relationId": relation_id,
                    "subject": subject_name,
                    "object": object_name,
                    "relationType": relation_type,
                    "temporal": relation.get("temporal", {}),
                    "confidence": relation.get("confidence"),
                    "postragDecision": candidate_decision(relation),
                    "groundingScore": candidate_grounding_score(relation),
                },
            })

    nodes = list(nodes_by_id.values())

    return {
        "version": 1,
        "graphKind": "taxonomy_kg",
        "investigationId": meta["investigationId"],
        "updatedAt": now_iso(),
        "truncated": truncated,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "sourceRecordCount": len(postrag_records),
        },
    }

@app.get("/api/investigations/{investigation_id}/subgraph")
def get_subgraph(investigation_id: str) -> JSONDict:
    get_meta_or_404(investigation_id)
    sarg_graph = read_sarg_graph(investigation_id)
    if has_sarg_graph(sarg_graph):
        return sarg_graph_to_ui_subgraph(investigation_id, sarg_graph)
    return empty_local_kg(investigation_id)