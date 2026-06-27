#!/usr/bin/env python3
"""
Parallel thin orchestration runner for Lanthic Intelligence.

Owns:
- reading manifest.jsonl
- creating run/source artifact directories
- running stage scripts as subprocesses
- parallel per-source pipelines
- sequential stages inside each source
- Neo4j write semaphore
- resume/force/fail-fast/dry-run behavior
- per-source status.json
- run_summary.json aggregation

Does not own:
- source discovery
- acquisition
- dedupe
- extraction logic
- PostRAG logic
- Neo4j fusion logic
- KG-IRAG/SARG execution

Run:
    python src/mass_ingest.py --self-test
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


JSONDict = Dict[str, Any]

STAGES = ["ingest", "extract", "postrag", "neo4j_fusion"]
MODEL_STAGES = {"extract", "postrag", "neo4j_fusion"}


try:
    from cost_ledger import summarize_ledger
except Exception:
    summarize_ledger = None


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any, n: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:n]


def safe_name(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()

    if not text:
        text = fallback

    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")

    cleaned = "".join(out).strip("._")
    return cleaned or fallback


def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(data), indent=2, ensure_ascii=False, default=str, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def read_manifest_jsonl(path: Path) -> List[JSONDict]:
    rows: List[JSONDict] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSONL at {path}:{line_number}: {exc}") from exc

        if not isinstance(row, dict):
            raise ValueError(f"Manifest row at {path}:{line_number} is not a JSON object.")

        rows.append(row)

    return rows


def copy_manifest_to_run_dir(manifest: Path, run_dir: Path) -> Path:
    target = run_dir / "manifest.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if manifest.resolve() == target.resolve():
            return target
    except Exception:
        pass

    shutil.copyfile(manifest, target)
    return target


def duration_seconds(start: float) -> float:
    return round(time.time() - start, 6)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(str(x) for x in command)


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def row_source_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_id")

    if value:
        return str(value)

    payload = {
        "source": row.get("source"),
        "canonical_source": row.get("canonical_source"),
        "local_path": row.get("local_path"),
    }

    return "src_" + stable_hash(payload)


def row_source_to_ingest(row: Mapping[str, Any]) -> str:
    return str(row.get("local_path") or row.get("source") or "")


def row_tags(row: Mapping[str, Any]) -> List[str]:
    tags = row.get("tags")
    if isinstance(tags, list):
        return [str(tag) for tag in tags]
    return []


def selected_rows(
    rows: Sequence[JSONDict],
    *,
    source_ids: Sequence[str],
    tags: Sequence[str],
    limit: Optional[int],
) -> List[JSONDict]:
    source_filter = {str(x) for x in source_ids if str(x).strip()}
    tag_filter = {str(x) for x in tags if str(x).strip()}

    out: List[JSONDict] = []

    for row in rows:
        sid = row_source_id(row)

        if source_filter and sid not in source_filter:
            continue

        if tag_filter and not tag_filter.intersection(set(row_tags(row))):
            continue

        out.append(row)

        if limit is not None and len(out) >= limit:
            break

    return out


def stage_index(stage: Optional[str]) -> Optional[int]:
    if not stage:
        return None
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return STAGES.index(stage)


# ============================================================
# Paths / status
# ============================================================

@dataclass
class SourcePaths:
    source_id: str
    source_dir: Path
    logs_dir: Path
    status_path: Path
    document_json: Path
    extract_json: Path
    postrag_json: Path
    neo4j_summary_json: Path

    def output_for_stage(self, stage: str) -> Optional[Path]:
        if stage == "ingest":
            return self.document_json
        if stage == "extract":
            return self.extract_json
        if stage == "postrag":
            return self.postrag_json
        if stage == "neo4j_fusion":
            return self.neo4j_summary_json
        return None

    def log_for_stage(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}.log"


def source_paths(run_dir: Path, source_id: str) -> SourcePaths:
    folder = run_dir / "sources" / safe_name(source_id, fallback="source")
    return SourcePaths(
        source_id=source_id,
        source_dir=folder,
        logs_dir=folder / "logs",
        status_path=folder / "status.json",
        document_json=folder / "document.json",
        extract_json=folder / "extract.json",
        postrag_json=folder / "postrag.json",
        neo4j_summary_json=folder / "neo4j_summary.json",
    )


def initial_source_status(
    *,
    row: Mapping[str, Any],
    source_id: str,
    source_dir: Path,
    run_id: str,
    corpus_id: str,
    branch_id: str,
) -> JSONDict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_id": source_id,
        "source": row.get("source"),
        "local_path": row.get("local_path"),
        "canonical_source": row.get("canonical_source"),
        "source_kind": row.get("source_kind"),
        "corpus_id": row.get("corpus_id") or corpus_id,
        "branch_id": row.get("branch_id") or branch_id,
        "tags": row_tags(row),
        "source_dir": str(source_dir),
        "status": "pending",
        "current_stage": None,
        "started_at": utc_now(),
        "finished_at": None,
        "stages": {},
        "error": None,
    }


def load_existing_status(path: Path) -> Optional[JSONDict]:
    if not path.exists():
        return None

    try:
        data = read_json(path)
        if isinstance(data, dict):
            return data
    except Exception:
        return None

    return None


def stage_was_success(status: Mapping[str, Any], stage: str) -> bool:
    stages = status.get("stages")
    if not isinstance(stages, dict):
        return False

    item = stages.get(stage)
    return isinstance(item, dict) and item.get("status") == "success"


def write_stage_record(
    status: JSONDict,
    *,
    stage: str,
    stage_record: Mapping[str, Any],
    status_path: Path,
) -> None:
    stages = status.setdefault("stages", {})
    stages[stage] = dict(stage_record)
    status["current_stage"] = stage
    write_json(status_path, status)


# ============================================================
# Runtime context
# ============================================================

@dataclass
class RuntimeContext:
    args: argparse.Namespace
    scripts_dir: Path
    run_id: str
    run_dir: Path
    manifest_path: Path
    run_manifest_path: Path
    corpus_id: str
    branch_id: str
    cost_ledger_path: Path
    cache_dir: Path
    summary_output: Path

    neo4j_semaphore: threading.Semaphore
    cancel_event: threading.Event = field(default_factory=threading.Event)

    clear_neo4j_lock: threading.Lock = field(default_factory=threading.Lock)
    clear_neo4j_done: bool = False

    print_lock: threading.Lock = field(default_factory=threading.Lock)
    main_log_path: Optional[Path] = None

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}\n"
        with self.print_lock:
            if not getattr(self.args, "quiet", False):
                print(message, flush=True)
            if self.main_log_path:
                append_text(self.main_log_path, line)


def make_runtime_context(args: argparse.Namespace) -> RuntimeContext:
    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.run_dir or Path("runs") / run_id
    run_dir = Path(run_dir)

    scripts_dir = args.scripts_dir or Path(__file__).resolve().parent
    scripts_dir = Path(scripts_dir)

    cost_ledger_path = args.cost_ledger or (run_dir / "cost_ledger.jsonl")
    cache_dir = args.cache_dir or (run_dir / "cache")
    summary_output = args.summary_output or (run_dir / "run_summary.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "sources").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = copy_manifest_to_run_dir(args.manifest, run_dir)

    return RuntimeContext(
        args=args,
        scripts_dir=scripts_dir,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=args.manifest,
        run_manifest_path=run_manifest_path,
        corpus_id=args.corpus_id,
        branch_id=args.branch_id,
        cost_ledger_path=Path(cost_ledger_path),
        cache_dir=Path(cache_dir),
        summary_output=Path(summary_output),
        neo4j_semaphore=threading.Semaphore(max(1, int(args.neo4j_workers))),
        main_log_path=run_dir / "logs" / "mass_ingest.log",
    )


def source_env(ctx: RuntimeContext, row: Mapping[str, Any], source_id: str) -> JSONDict:
    env = dict(os.environ)

    src_dir = str(ctx.scripts_dir.resolve())
    extraction_dir = str((ctx.scripts_dir / "extraction").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        f"{src_dir}:{extraction_dir}:{existing_pythonpath}"
        if existing_pythonpath
        else f"{src_dir}:{extraction_dir}"
    )

    corpus_id = str(row.get("corpus_id") or ctx.corpus_id)
    branch_id = str(row.get("branch_id") or ctx.branch_id)
    canonical_source = str(
        first_nonempty(
            row.get("canonical_source"),
            row.get("source"),
            row.get("local_path"),
            "",
        )
    )

    env.update(
        {
            "LANTHIC_RUN_ID": ctx.run_id,
            "LANTHIC_SOURCE_ID": source_id,
            "LANTHIC_CORPUS_ID": corpus_id,
            "LANTHIC_BRANCH_ID": branch_id,
            "LANTHIC_CANONICAL_SOURCE": canonical_source,
            "LANTHIC_COST_LEDGER": str(ctx.cost_ledger_path),
            "LANTHIC_CACHE_DIR": str(ctx.cache_dir),
        }
    )

    if ctx.args.disable_cache:
        env["LANTHIC_DISABLE_CACHE"] = "1"

    if ctx.args.pricing_file:
        env["LANTHIC_PRICING_FILE"] = str(ctx.args.pricing_file)

    if ctx.args.self_test_event_log:
        env["MASS_INGEST_TEST_EVENT_LOG"] = str(ctx.args.self_test_event_log)

    if ctx.args.self_test_neo4j_log:
        env["MASS_INGEST_TEST_NEO4J_LOG"] = str(ctx.args.self_test_neo4j_log)

    return env


# ============================================================
# Command building
# ============================================================

def script_path(ctx: RuntimeContext, name: str) -> Path:
    if name in {"ingest.py", "extract.py"}:
        return ctx.scripts_dir / "extraction" / name

    return ctx.scripts_dir / name


def base_python(ctx: RuntimeContext) -> str:
    return str(ctx.args.python)


def build_ingest_command(ctx: RuntimeContext, row: Mapping[str, Any], paths: SourcePaths) -> List[str]:
    source = row_source_to_ingest(row)

    return [
        base_python(ctx),
        str(script_path(ctx, "ingest.py")),
        "--source",
        source,
        "--output",
        str(paths.document_json),
    ]


def build_extract_command(ctx: RuntimeContext, row: Mapping[str, Any], paths: SourcePaths) -> List[str]:
    command = [
        base_python(ctx),
        str(script_path(ctx, "extract.py")),
        "--input",
        str(paths.document_json),
        "--output",
        str(paths.extract_json),
        "--model",
        ctx.args.model,
        "--max-block-chars",
        str(ctx.args.max_block_chars),
        "--max-total-chars",
        str(ctx.args.max_total_chars),
    ]

    if ctx.args.include_evidence_text:
        command.append("--include-evidence-text")

    if ctx.args.ollama_chunking:
        command.append("--ollama-chunking")

    if ctx.args.patched_stage_args:
        command.extend(pipeline_cli_args(ctx, row, row_source_id(row)))

    return command


def build_postrag_command(ctx: RuntimeContext, row: Mapping[str, Any], paths: SourcePaths) -> List[str]:
    command = [
        base_python(ctx),
        str(script_path(ctx, "postrag.py")),
        "--input",
        str(paths.extract_json),
        "--output",
        str(paths.postrag_json),
        "--model",
        ctx.args.model,
        "--embed-model",
        ctx.args.embed_model,
        "--top-k",
        str(ctx.args.postrag_top_k),
    ]

    if ctx.args.ollama_chunking:
        command.append("--ollama-chunking")

    if ctx.args.patched_stage_args:
        command.extend(pipeline_cli_args(ctx, row, row_source_id(row)))

    return command


def pipeline_cli_args(ctx: RuntimeContext, row: Mapping[str, Any], source_id: str) -> List[str]:
    corpus_id = str(row.get("corpus_id") or ctx.corpus_id)
    branch_id = str(row.get("branch_id") or ctx.branch_id)
    canonical_source = str(
        first_nonempty(
            row.get("canonical_source"),
            row.get("source"),
            row.get("local_path"),
            "",
        )
    )

    command = [
        "--run-id",
        ctx.run_id,
        "--source-id",
        source_id,
        "--corpus-id",
        corpus_id,
        "--branch-id",
        branch_id,
        "--canonical-source",
        canonical_source,
        "--cost-ledger",
        str(ctx.cost_ledger_path),
        "--cache-dir",
        str(ctx.cache_dir),
    ]

    if ctx.args.disable_cache:
        command.append("--disable-cache")

    if ctx.args.pricing_file:
        command.extend(["--pricing-file", str(ctx.args.pricing_file)])

    return command


def build_neo4j_command(
    ctx: RuntimeContext,
    row: Mapping[str, Any],
    paths: SourcePaths,
    *,
    clear_neo4j: bool,
) -> List[str]:
    source_id = row_source_id(row)

    command = [
        base_python(ctx),
        str(script_path(ctx, "neo4j_fusion.py")),
        "--input",
        str(paths.postrag_json),
        "--uri",
        ctx.args.uri,
        "--user",
        ctx.args.user,
        "--password",
        ctx.args.password,
        "--database",
        ctx.args.database,
        "--model",
        ctx.args.model,
        "--embed-model",
        ctx.args.embed_model,
        "--embedding-dim",
        str(ctx.args.embedding_dim),
        "--entity-neighbor-k",
        str(ctx.args.entity_neighbor_k),
        "--summary-output",
        str(paths.neo4j_summary_json),
        *pipeline_cli_args(ctx, row, source_id),
    ]

    if clear_neo4j:
        command.append("--clear")

    return command


# ============================================================
# Stage execution
# ============================================================

def run_subprocess(
    *,
    command: Sequence[str],
    log_path: Path,
    env: Mapping[str, str],
    timeout_seconds: Optional[int],
) -> Tuple[int, float, Optional[str]]:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command_to_string(command) + "\n\n")
        log.flush()

        try:
            completed = subprocess.run(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(env),
                timeout=timeout_seconds,
            )
            runtime = duration_seconds(start)
            output = completed.stdout or ""
            log.write(output)
            if output and not output.endswith("\n"):
                log.write("\n")
            log.write(f"\n[exit_code] {completed.returncode}\n")
            log.write(f"[runtime_seconds] {runtime}\n")

            error = None if completed.returncode == 0 else tail_text(output, 3000)
            return completed.returncode, runtime, error

        except subprocess.TimeoutExpired as exc:
            runtime = duration_seconds(start)
            log.write(f"\n[TIMEOUT] command exceeded {timeout_seconds} seconds\n")
            if exc.stdout:
                log.write(str(exc.stdout))
            return 124, runtime, f"Timeout after {timeout_seconds} seconds"

        except Exception as exc:
            runtime = duration_seconds(start)
            log.write(f"\n[ERROR] {type(exc).__name__}: {exc}\n")
            return 1, runtime, f"{type(exc).__name__}: {exc}"


def tail_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def stage_should_force(args: argparse.Namespace, stage: str) -> bool:
    if args.force:
        return True

    force_i = stage_index(args.force_stage)
    if force_i is None:
        return False

    return STAGES.index(stage) >= force_i


def stage_should_skip(
    *,
    args: argparse.Namespace,
    stage: str,
    output_path: Optional[Path],
    existing_status: Optional[Mapping[str, Any]],
) -> bool:
    if not args.resume:
        return False

    if stage_should_force(args, stage):
        return False

    if output_path is None or not output_path.exists():
        return False

    if not existing_status:
        return False

    return stage_was_success(existing_status, stage)


def make_skipped_record(
    *,
    stage: str,
    output_path: Optional[Path],
    log_path: Path,
    reason: str,
) -> JSONDict:
    return {
        "stage": stage,
        "status": "skipped",
        "reason": reason,
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "runtime_seconds": 0.0,
        "command": None,
        "output": str(output_path) if output_path else None,
        "log": str(log_path),
        "error": None,
    }


def make_dry_run_record(
    *,
    stage: str,
    command: Sequence[str],
    output_path: Optional[Path],
    log_path: Path,
) -> JSONDict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + command_to_string(command) + "\n\n[DRY RUN] command not executed\n",
        encoding="utf-8",
    )

    return {
        "stage": stage,
        "status": "dry_run",
        "reason": "dry_run",
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "runtime_seconds": 0.0,
        "command": list(command),
        "command_string": command_to_string(command),
        "output": str(output_path) if output_path else None,
        "log": str(log_path),
        "error": None,
    }


def make_stage_record(
    *,
    stage: str,
    command: Sequence[str],
    output_path: Optional[Path],
    log_path: Path,
    exit_code: int,
    runtime_seconds: float,
    error: Optional[str],
) -> JSONDict:
    return {
        "stage": stage,
        "status": "success" if exit_code == 0 else "failed",
        "started_at": None,
        "finished_at": utc_now(),
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
        "command": list(command),
        "command_string": command_to_string(command),
        "output": str(output_path) if output_path else None,
        "log": str(log_path),
        "error": error,
    }


def run_stage(
    *,
    ctx: RuntimeContext,
    row: Mapping[str, Any],
    paths: SourcePaths,
    status: JSONDict,
    existing_status: Optional[Mapping[str, Any]],
    stage: str,
    env: Mapping[str, str],
    clear_neo4j: bool = False,
) -> bool:
    output_path = paths.output_for_stage(stage)
    log_path = paths.log_for_stage(stage)

    if stage == "neo4j_fusion" and not ctx.args.neo4j:
        record = make_skipped_record(
            stage=stage,
            output_path=output_path,
            log_path=log_path,
            reason="neo4j_disabled",
        )
        write_stage_record(status, stage=stage, stage_record=record, status_path=paths.status_path)
        return True

    if stage_should_skip(
        args=ctx.args,
        stage=stage,
        output_path=output_path,
        existing_status=existing_status,
    ):
        record = make_skipped_record(
            stage=stage,
            output_path=output_path,
            log_path=log_path,
            reason="resume_success",
        )
        write_stage_record(status, stage=stage, stage_record=record, status_path=paths.status_path)
        return True

    if stage == "ingest":
        command = build_ingest_command(ctx, row, paths)
    elif stage == "extract":
        command = build_extract_command(ctx, row, paths)
    elif stage == "postrag":
        command = build_postrag_command(ctx, row, paths)
    elif stage == "neo4j_fusion":
        command = build_neo4j_command(ctx, row, paths, clear_neo4j=clear_neo4j)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if ctx.args.dry_run:
        record = make_dry_run_record(
            stage=stage,
            command=command,
            output_path=output_path,
            log_path=log_path,
        )
        write_stage_record(status, stage=stage, stage_record=record, status_path=paths.status_path)
        return True

    started_at = utc_now()
    status["current_stage"] = stage
    write_json(paths.status_path, status)

    exit_code, runtime, error = run_subprocess(
        command=command,
        log_path=log_path,
        env=env,
        timeout_seconds=ctx.args.stage_timeout,
    )

    record = make_stage_record(
        stage=stage,
        command=command,
        output_path=output_path,
        log_path=log_path,
        exit_code=exit_code,
        runtime_seconds=runtime,
        error=error,
    )
    record["started_at"] = started_at

    write_stage_record(status, stage=stage, stage_record=record, status_path=paths.status_path)
    return exit_code == 0


def should_clear_neo4j_for_this_source(ctx: RuntimeContext) -> bool:
    if not ctx.args.clear_neo4j:
        return False

    with ctx.clear_neo4j_lock:
        if ctx.clear_neo4j_done:
            return False
        ctx.clear_neo4j_done = True
        return True


# ============================================================
# Source pipeline
# ============================================================

def process_source(
    *,
    ctx: RuntimeContext,
    row: JSONDict,
    ordinal: int,
    total: int,
) -> JSONDict:
    source_id = row_source_id(row)
    paths = source_paths(ctx.run_dir, source_id)
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    existing_status = load_existing_status(paths.status_path)

    status = initial_source_status(
        row=row,
        source_id=source_id,
        source_dir=paths.source_dir,
        run_id=ctx.run_id,
        corpus_id=str(row.get("corpus_id") or ctx.corpus_id),
        branch_id=str(row.get("branch_id") or ctx.branch_id),
    )

    status["ordinal"] = ordinal
    status["total_sources"] = total

    env = source_env(ctx, row, source_id)

    ctx.log(f"[{ordinal}/{total} started] {source_id}")

    try:
        if not row_source_to_ingest(row):
            raise ValueError("Manifest row has neither local_path nor source.")

        for stage in ["ingest", "extract", "postrag"]:
            if ctx.cancel_event.is_set() and ctx.args.fail_fast:
                status["status"] = "cancelled"
                status["finished_at"] = utc_now()
                status["error"] = "Cancelled by fail-fast."
                write_json(paths.status_path, status)
                return status

            ok = run_stage(
                ctx=ctx,
                row=row,
                paths=paths,
                status=status,
                existing_status=existing_status,
                stage=stage,
                env=env,
            )

            if not ok:
                status["status"] = "failed"
                status["failed_stage"] = stage
                status["finished_at"] = utc_now()
                status["error"] = (status.get("stages") or {}).get(stage, {}).get("error")
                write_json(paths.status_path, status)
                ctx.log(f"[{ordinal}/{total} failed:{stage}] {source_id}")
                return status

        if ctx.args.neo4j:
            with ctx.neo4j_semaphore:
                clear_for_this_source = should_clear_neo4j_for_this_source(ctx)

                ok = run_stage(
                    ctx=ctx,
                    row=row,
                    paths=paths,
                    status=status,
                    existing_status=existing_status,
                    stage="neo4j_fusion",
                    env=env,
                    clear_neo4j=clear_for_this_source,
                )

            if not ok:
                status["status"] = "failed"
                status["failed_stage"] = "neo4j_fusion"
                status["finished_at"] = utc_now()
                status["error"] = (status.get("stages") or {}).get("neo4j_fusion", {}).get("error")
                write_json(paths.status_path, status)
                ctx.log(f"[{ordinal}/{total} failed:neo4j_fusion] {source_id}")
                return status
        else:
            run_stage(
                ctx=ctx,
                row=row,
                paths=paths,
                status=status,
                existing_status=existing_status,
                stage="neo4j_fusion",
                env=env,
            )

        if ctx.args.dry_run:
            status["status"] = "dry_run"
        else:
            status["status"] = "success"

        status["finished_at"] = utc_now()
        status["error"] = None
        write_json(paths.status_path, status)

        ctx.log(f"[{ordinal}/{total} {status['status']}] {source_id}")
        return status

    except Exception as exc:
        status["status"] = "failed"
        status["failed_stage"] = status.get("current_stage")
        status["finished_at"] = utc_now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(paths.status_path, status)
        ctx.log(f"[{ordinal}/{total} failed:exception] {source_id} {status['error']}")
        return status


# ============================================================
# Parallel scheduler
# ============================================================

def run_sources_parallel(ctx: RuntimeContext, rows: Sequence[JSONDict]) -> List[JSONDict]:
    if not rows:
        return []

    total = len(rows)
    results: List[JSONDict] = []

    max_workers = max(1, int(ctx.args.workers))
    next_index = 0

    def submit_next(executor: concurrent.futures.ThreadPoolExecutor, active: Dict[Any, int]) -> bool:
        nonlocal next_index

        if next_index >= total:
            return False

        if ctx.cancel_event.is_set() and ctx.args.fail_fast:
            return False

        row = rows[next_index]
        ordinal = next_index + 1
        future = executor.submit(
            process_source,
            ctx=ctx,
            row=row,
            ordinal=ordinal,
            total=total,
        )
        active[future] = next_index
        next_index += 1
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        active: Dict[Any, int] = {}

        while len(active) < max_workers and submit_next(executor, active):
            pass

        while active:
            done, _ = concurrent.futures.wait(
                set(active.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                active.pop(future, None)

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                results.append(result)

                if result.get("status") == "failed" and ctx.args.fail_fast:
                    ctx.cancel_event.set()

            while len(active) < max_workers and submit_next(executor, active):
                pass

    results.sort(key=lambda item: int(item.get("ordinal") or 10**9))
    return results


# ============================================================
# Summary
# ============================================================

def source_status_counts(results: Sequence[Mapping[str, Any]]) -> JSONDict:
    counts = {
        "success": 0,
        "failed": 0,
        "partial": 0,
        "skipped": 0,
        "dry_run": 0,
        "cancelled": 0,
    }

    for result in results:
        status = str(result.get("status") or "partial")
        counts[status] = counts.get(status, 0) + 1

    return counts


def stage_counts(results: Sequence[Mapping[str, Any]]) -> JSONDict:
    counts: JSONDict = {
        stage: {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "dry_run": 0,
            "missing": 0,
        }
        for stage in STAGES
    }

    for result in results:
        stages = result.get("stages")
        if not isinstance(stages, dict):
            for stage in STAGES:
                counts[stage]["missing"] += 1
            continue

        for stage in STAGES:
            item = stages.get(stage)
            if not isinstance(item, dict):
                counts[stage]["missing"] += 1
                continue

            status = str(item.get("status") or "missing")
            counts[stage][status] = counts[stage].get(status, 0) + 1

    return counts


def build_run_summary(
    *,
    ctx: RuntimeContext,
    started_at: str,
    finished_at: str,
    selected: Sequence[JSONDict],
    results: Sequence[JSONDict],
) -> JSONDict:
    cost_summary = None

    if (
        summarize_ledger is not None
        and ctx.cost_ledger_path.exists()
        and ctx.cost_ledger_path.stat().st_size > 0
    ):
        try:
            cost_summary = summarize_ledger(ctx.cost_ledger_path)
        except Exception as exc:
            cost_summary = {
                "error": f"{type(exc).__name__}: {exc}",
                "ledger_path": str(ctx.cost_ledger_path),
            }

    source_counts = source_status_counts(results)

    return {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "corpus_id": ctx.corpus_id,
        "branch_id": ctx.branch_id,
        "manifest_path": str(ctx.manifest_path),
        "run_manifest_path": str(ctx.run_manifest_path),
        "run_dir": str(ctx.run_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "workers": ctx.args.workers,
        "neo4j_workers": ctx.args.neo4j_workers,
        "neo4j_enabled": ctx.args.neo4j,
        "dry_run": ctx.args.dry_run,
        "resume": ctx.args.resume,
        "force": ctx.args.force,
        "force_stage": ctx.args.force_stage,
        "sources_selected": len(selected),
        "sources_completed": len(results),
        "sources_success": source_counts.get("success", 0),
        "sources_failed": source_counts.get("failed", 0),
        "sources_dry_run": source_counts.get("dry_run", 0),
        "sources_cancelled": source_counts.get("cancelled", 0),
        "source_status_counts": source_counts,
        "stage_counts": stage_counts(results),
        "cost_ledger_path": str(ctx.cost_ledger_path),
        "cache_dir": str(ctx.cache_dir),
        "cost_summary": cost_summary,
        "source_results": list(results),
    }


# ============================================================
# Main run
# ============================================================

def run_mass_ingest(args: argparse.Namespace) -> JSONDict:
    started_at = utc_now()
    ctx = make_runtime_context(args)

    ctx.log(f"[mass_ingest] run_id={ctx.run_id}")
    ctx.log(f"[mass_ingest] run_dir={ctx.run_dir}")
    ctx.log(f"[mass_ingest] manifest={ctx.manifest_path}")
    ctx.log(f"[mass_ingest] workers={args.workers} neo4j_workers={args.neo4j_workers}")

    rows = read_manifest_jsonl(args.manifest)
    selected = selected_rows(
        rows,
        source_ids=args.source_id or [],
        tags=args.tag or [],
        limit=args.limit,
    )

    ctx.log(f"[mass_ingest] manifest_rows={len(rows)} selected={len(selected)}")

    if args.dry_run:
        ctx.log("[mass_ingest] dry_run enabled")

    results = run_sources_parallel(ctx, selected)

    finished_at = utc_now()

    summary = build_run_summary(
        ctx=ctx,
        started_at=started_at,
        finished_at=finished_at,
        selected=selected,
        results=results,
    )

    write_json(ctx.summary_output, summary)
    ctx.log(f"[mass_ingest] wrote summary: {ctx.summary_output}")

    return summary


# ============================================================
# Self-test support
# ============================================================

FAKE_STAGE_SCRIPT = r'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

def append_jsonl(path, data):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")

def write_json(path, data):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def main():
    stage = Path(__file__).stem

    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--summary-output")
    parser.add_argument("--clear", action="store_true")
    args, unknown = parser.parse_known_args()

    source_id = os.environ.get("LANTHIC_SOURCE_ID", "unknown")
    event_log = os.environ.get("MASS_INGEST_TEST_EVENT_LOG")
    neo4j_log = os.environ.get("MASS_INGEST_TEST_NEO4J_LOG")

    append_jsonl(event_log, {
        "event": "start",
        "stage": stage,
        "source_id": source_id,
        "time": time.time(),
        "run_id": os.environ.get("LANTHIC_RUN_ID"),
        "cost_ledger": os.environ.get("LANTHIC_COST_LEDGER"),
        "cache_dir": os.environ.get("LANTHIC_CACHE_DIR"),
    })

    if stage == "postrag" and "fail" in source_id:
        append_jsonl(event_log, {
            "event": "fail",
            "stage": stage,
            "source_id": source_id,
            "time": time.time(),
        })
        print("intentional postrag failure", file=sys.stderr)
        return 7

    if stage == "neo4j_fusion":
        append_jsonl(neo4j_log, {
            "event": "start",
            "source_id": source_id,
            "time": time.time(),
            "clear": bool(args.clear),
        })
        time.sleep(0.08)
        write_json(args.summary_output, {
            "stage": stage,
            "source_id": source_id,
            "clear": bool(args.clear),
            "input": args.input,
        })
        append_jsonl(neo4j_log, {
            "event": "end",
            "source_id": source_id,
            "time": time.time(),
        })
    else:
        time.sleep(0.03)
        write_json(args.output, {
            "stage": stage,
            "source_id": source_id,
            "source": args.source,
            "input": args.input,
            "run_id": os.environ.get("LANTHIC_RUN_ID"),
            "corpus_id": os.environ.get("LANTHIC_CORPUS_ID"),
            "branch_id": os.environ.get("LANTHIC_BRANCH_ID"),
        })

    append_jsonl(event_log, {
        "event": "end",
        "stage": stage,
        "source_id": source_id,
        "time": time.time(),
    })

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_fake_scripts(scripts_dir: Path) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for name in ["ingest.py", "extract.py", "postrag.py", "neo4j_fusion.py"]:
        path = scripts_dir / name
        path.write_text(FAKE_STAGE_SCRIPT, encoding="utf-8")
        try:
            path.chmod(0o755)
        except Exception:
            pass


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl_events(path: Path) -> List[JSONDict]:
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_no_neo4j_overlap(events: Sequence[Mapping[str, Any]]) -> None:
    intervals: Dict[str, Dict[str, float]] = {}

    for event in events:
        source_id = str(event.get("source_id"))
        intervals.setdefault(source_id, {})

        if event.get("event") == "start":
            intervals[source_id]["start"] = float(event["time"])
        elif event.get("event") == "end":
            intervals[source_id]["end"] = float(event["time"])

    ordered = sorted(
        [
            (source_id, item["start"], item["end"])
            for source_id, item in intervals.items()
            if "start" in item and "end" in item
        ],
        key=lambda x: x[1],
    )

    for (_, _s1, e1), (_, s2, _e2) in zip(ordered, ordered[1:]):
        assert_true(s2 >= e1, "neo4j stages overlapped despite neo4j_workers=1")


def run_self_test() -> int:
    print("[mass_ingest self-test] starting")

    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        scripts_dir = tmp / "scripts"
        write_fake_scripts(scripts_dir)

        manifest = tmp / "manifest.jsonl"
        event_log = tmp / "events.jsonl"
        neo4j_log = tmp / "neo4j_events.jsonl"

        rows = [
            {
                "source": "fake://one",
                "source_id": "src_one",
                "source_kind": "url",
                "canonical_source": "fake://one",
                "local_path": None,
                "corpus_id": "eval1",
                "branch_id": "staging_eval1",
                "tags": ["eval1"],
                "acquisition_status": "not_acquired",
                "metadata": {},
            },
            {
                "source": "fake://two",
                "source_id": "src_two",
                "source_kind": "url",
                "canonical_source": "fake://two",
                "local_path": None,
                "corpus_id": "eval1",
                "branch_id": "staging_eval1",
                "tags": ["eval1"],
                "acquisition_status": "not_acquired",
                "metadata": {},
            },
            {
                "source": "fake://three",
                "source_id": "src_three",
                "source_kind": "url",
                "canonical_source": "fake://three",
                "local_path": None,
                "corpus_id": "eval1",
                "branch_id": "staging_eval1",
                "tags": ["eval1"],
                "acquisition_status": "not_acquired",
                "metadata": {},
            },
        ]
        write_manifest(manifest, rows)

        run_dir = tmp / "run_parallel"

        summary = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(manifest),
                    "--run-id", "test_parallel",
                    "--run-dir", str(run_dir),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "3",
                    "--neo4j-workers", "1",
                    "--neo4j",
                    "--clear-neo4j",
                    "--self-test-event-log", str(event_log),
                    "--self-test-neo4j-log", str(neo4j_log),
                    "--quiet",
                ]
            )
        )

        assert_true(summary["sources_success"] == 3, "parallel run did not complete all sources")
        assert_true((run_dir / "run_summary.json").exists(), "run_summary.json missing")
        assert_true((run_dir / "sources" / "src_one" / "document.json").exists(), "document.json missing")
        assert_true((run_dir / "sources" / "src_one" / "extract.json").exists(), "extract.json missing")
        assert_true((run_dir / "sources" / "src_one" / "postrag.json").exists(), "postrag.json missing")
        assert_true((run_dir / "sources" / "src_one" / "neo4j_summary.json").exists(), "neo4j summary missing")

        neo_events = read_jsonl_events(neo4j_log)
        assert_true(len([e for e in neo_events if e.get("event") == "start"]) == 3, "neo4j stages not run")
        assert_true(
            len([e for e in neo_events if e.get("clear")]) == 1,
            "clear_neo4j should be passed exactly once",
        )
        assert_no_neo4j_overlap(neo_events)

        event_count_after_first = len(read_jsonl_events(event_log))

        summary_resume = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(manifest),
                    "--run-id", "test_parallel",
                    "--run-dir", str(run_dir),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "3",
                    "--neo4j-workers", "1",
                    "--neo4j",
                    "--self-test-event-log", str(event_log),
                    "--self-test-neo4j-log", str(neo4j_log),
                    "--quiet",
                ]
            )
        )

        assert_true(summary_resume["stage_counts"]["ingest"]["skipped"] == 3, "resume did not skip ingest")
        assert_true(
            len(read_jsonl_events(event_log)) == event_count_after_first,
            "resume should not execute fake scripts again",
        )

        summary_force = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(manifest),
                    "--run-id", "test_parallel",
                    "--run-dir", str(run_dir),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "3",
                    "--neo4j-workers", "1",
                    "--neo4j",
                    "--force",
                    "--self-test-event-log", str(event_log),
                    "--self-test-neo4j-log", str(neo4j_log),
                    "--quiet",
                ]
            )
        )

        assert_true(summary_force["sources_success"] == 3, "force rerun failed")
        assert_true(
            len(read_jsonl_events(event_log)) > event_count_after_first,
            "force did not rerun fake scripts",
        )

        # Failure should be isolated when fail-fast is off.
        failure_manifest = tmp / "manifest_failure.jsonl"
        failure_rows = [
            {**rows[0], "source_id": "src_ok_a", "source": "fake://ok-a"},
            {**rows[1], "source_id": "src_fail", "source": "fake://fail"},
            {**rows[2], "source_id": "src_ok_b", "source": "fake://ok-b"},
        ]
        write_manifest(failure_manifest, failure_rows)

        failure_summary = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(failure_manifest),
                    "--run-id", "test_failure",
                    "--run-dir", str(tmp / "run_failure"),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "2",
                    "--neo4j-workers", "1",
                    "--neo4j",
                    "--self-test-event-log", str(tmp / "failure_events.jsonl"),
                    "--self-test-neo4j-log", str(tmp / "failure_neo4j_events.jsonl"),
                    "--quiet",
                ]
            )
        )

        assert_true(failure_summary["sources_failed"] == 1, "failed source not counted")
        assert_true(failure_summary["sources_success"] == 2, "successful sources did not continue after failure")

        # Fail-fast with workers=1 should stop scheduling after first failure.
        fail_fast_manifest = tmp / "manifest_fail_fast.jsonl"
        fail_fast_rows = [
            {**rows[0], "source_id": "src_fail_first", "source": "fake://fail-first"},
            {**rows[1], "source_id": "src_should_not_run", "source": "fake://should-not-run"},
        ]
        write_manifest(fail_fast_manifest, fail_fast_rows)

        fail_fast_summary = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(fail_fast_manifest),
                    "--run-id", "test_fail_fast",
                    "--run-dir", str(tmp / "run_fail_fast"),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "1",
                    "--neo4j-workers", "1",
                    "--neo4j",
                    "--fail-fast",
                    "--self-test-event-log", str(tmp / "fail_fast_events.jsonl"),
                    "--self-test-neo4j-log", str(tmp / "fail_fast_neo4j_events.jsonl"),
                    "--quiet",
                ]
            )
        )

        assert_true(fail_fast_summary["sources_completed"] == 1, "fail-fast did not stop scheduling")
        assert_true(fail_fast_summary["sources_failed"] == 1, "fail-fast failure not counted")

        # Limit and dry-run.
        dry_run_summary = run_mass_ingest(
            parse_args(
                [
                    "--manifest", str(manifest),
                    "--run-id", "test_dry_run",
                    "--run-dir", str(tmp / "run_dry"),
                    "--corpus-id", "eval1",
                    "--branch-id", "staging_eval1",
                    "--scripts-dir", str(scripts_dir),
                    "--workers", "2",
                    "--limit", "2",
                    "--dry-run",
                    "--quiet",
                ]
            )
        )

        assert_true(dry_run_summary["sources_selected"] == 2, "limit failed")
        assert_true(dry_run_summary["sources_dry_run"] == 2, "dry-run source count wrong")
        assert_true(
            dry_run_summary["stage_counts"]["ingest"]["dry_run"] == 2,
            "dry-run stage count wrong",
        )

    print("[mass_ingest self-test] all tests passed")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel manifest-driven ingestion orchestrator.")

    parser.add_argument("--self-test", action="store_true")

    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--corpus-id", default="default_corpus")
    parser.add_argument("--branch-id", default="staging")
    parser.add_argument("--summary-output", type=Path, default=None)

    parser.add_argument("--scripts-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")

    parser.add_argument("--neo4j", dest="neo4j", action="store_true", default=False)
    parser.add_argument("--skip-neo4j", dest="neo4j", action="store_false")
    parser.add_argument("--clear-neo4j", action="store_true")

    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--entity-neighbor-k", type=int, default=8)

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--neo4j-workers", type=int, default=1)
    parser.add_argument("--stage-timeout", type=int, default=None)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])

    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stage", choices=STAGES, default=None)

    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--cost-ledger", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--pricing-file", type=Path, default=None)

    parser.add_argument("--include-evidence-text", dest="include_evidence_text", action="store_true", default=True)
    parser.add_argument("--no-include-evidence-text", dest="include_evidence_text", action="store_false")
    parser.add_argument("--ollama-chunking", action="store_true")
    parser.add_argument("--max-block-chars", type=int, default=2500)
    parser.add_argument("--max-total-chars", type=int, default=24000)
    parser.add_argument("--postrag-top-k", type=int, default=4)

    parser.add_argument(
        "--patched-stage-args",
        action="store_true",
        help=(
            "Pass pipeline/cost CLI args to extract.py and postrag.py. "
            "Keep off until those scripts accept the new arguments."
        ),
    )

    # Hidden/self-test-only hooks.
    parser.add_argument("--self-test-event-log", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--self-test-neo4j-log", type=Path, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if not args.self_test and args.manifest is None:
        parser.error("--manifest is required unless --self-test is used.")

    args.workers = max(1, int(args.workers))
    args.neo4j_workers = max(1, int(args.neo4j_workers))

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    summary = run_mass_ingest(args)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if summary.get("sources_failed", 0) > 0:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())