#!/usr/bin/env python3
"""
Neutral evaluation harness for Lanthic Intelligence.

This script does not implement retrieval, graph traversal, source filtering,
or any task-specific heuristics. It only runs fixed questions through fixed
systems, saves raw outputs, creates a manual score sheet, and aggregates scores.

Evaluation shape:
  3 task types x 5 questions x 3 systems = 45 outputs

Systems:
  - gpt_memo: direct GPT-style memo baseline
  - simple_rag: external command supplied by --simple-rag-command
  - lanthic: external SARG command supplied by --sarg-command

Human scoring remains manual.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QUESTIONS = [
    {
        "question_id": "H1",
        "task_type": "historical_disruption",
        "source_hint": "",
        "question": "How could disruption in Kachin rare-earth mining affect downstream magnet supply chains? Run a risk assessment.",
    },
    {
        "question_id": "H2",
        "task_type": "historical_disruption",
        "source_hint": "",
        "question": "What evidence supports the claim that Myanmar supply disruptions and Chinese rare-earth processing concentration create a compounding rare-earth supply risk?",
    },
    {
        "question_id": "H3",
        "task_type": "historical_disruption",
        "source_hint": "",
        "question": "How might Chinese rare-earth export controls affect downstream firms or sectors that depend on permanent magnets?",
    },
    {
        "question_id": "H4",
        "task_type": "historical_disruption",
        "source_hint": "",
        "question": "Which upstream rare-earth supply chokepoints appear most relevant to downstream permanent magnet risk, and why?",
    },
    {
        "question_id": "H5",
        "task_type": "historical_disruption",
        "source_hint": "",
        "question": "Compare the relative importance of mining disruption, processing concentration, and export-control risk in rare-earth supply-chain vulnerability.",
    },
    {
        "question_id": "U1",
        "task_type": "uploaded_document",
        "source_hint": "Global Witness Myanmar's poisoned mountains",
        "question": "For the Global Witness source on Myanmar's poisoned mountains, what rare-earth supply-chain claims are directly supported by the ingested evidence?",
    },
    {
        "question_id": "U2",
        "task_type": "uploaded_document",
        "source_hint": "Global Witness Myanmar's poisoned mountains",
        "question": "For the Global Witness source on Myanmar's poisoned mountains, which entities, regions, and organizations were extracted into the database?",
    },
    {
        "question_id": "U3",
        "task_type": "uploaded_document",
        "source_hint": "Global Witness Myanmar's poisoned mountains",
        "question": "For the Global Witness source on Myanmar's poisoned mountains, what evidence connects Myanmar rare-earth mining to Chinese rare-earth supply?",
    },
    {
        "question_id": "U4",
        "task_type": "uploaded_document",
        "source_hint": "USGS rare earths mineral commodity summary",
        "question": "For the USGS rare-earths source, what database evidence supports claims about rare-earth production, reserves, or import dependence?",
    },
    {
        "question_id": "U5",
        "task_type": "uploaded_document",
        "source_hint": "EU Critical Raw Materials Act Regulation 2024/1252",
        "question": "For the EU Critical Raw Materials Act source, what strategic raw-material dependency claims are directly supported, and what should not be over-inferred?",
    },
    {
        "question_id": "M1",
        "task_type": "missing_evidence",
        "source_hint": "",
        "question": "Where is the evidence base weakest for connecting upstream rare-earth mining disruption to downstream company-level exposure?",
    },
    {
        "question_id": "M2",
        "task_type": "missing_evidence",
        "source_hint": "",
        "question": "What additional evidence would be needed to prove that a Myanmar rare-earth disruption directly affects a named downstream manufacturer?",
    },
    {
        "question_id": "M3",
        "task_type": "missing_evidence",
        "source_hint": "",
        "question": "Which parts of the rare-earth supply-chain risk pathway are well supported, and which links remain uncertain?",
    },
    {
        "question_id": "M4",
        "task_type": "missing_evidence",
        "source_hint": "",
        "question": "What claims about rare-earth disruption risk should be treated cautiously because the available evidence is incomplete?",
    },
    {
        "question_id": "M5",
        "task_type": "missing_evidence",
        "source_hint": "",
        "question": "Identify the main missing evidence gaps that prevent a high-confidence forecast of downstream permanent magnet disruption.",
    },
]


SYSTEMS = ["gpt_memo", "simple_rag", "lanthic"]

METRICS = [
    "faithfulness",
    "citation_support",
    "multi_hop_completeness",
    "unsupported_claim_reduction",
    "missing_evidence_detection",
    "latency_and_token_cost",
    "analyst_workflow_usability",
]

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4.1-mini")
JUDGE_PROMPT_VERSION = "lanthic_eval_judge_v1"

JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for a compound-AI/agentic-system report.

You will score one system output for one evaluation question. The systems are:
- gpt_memo: plain GPT-style memo baseline with no access to Lanthic DB/KG.
- simple_rag: retrieval-only baseline over the ingested evidence corpus.
- lanthic: full Lanthic/SARG workflow using KG/evidence/agentic reasoning.

Score each metric from 1 to 5.

General scoring scale:
1 = poor / absent / failed
2 = weak
3 = adequate
4 = strong
5 = excellent

Do not reward fluent unsupported prose. Reward answers that are grounded, specific, appropriately cautious, and useful for analyst work.

Metric definitions:

1. faithfulness
Measures whether the answer stays within the evidence and system access available.
- 1: Mostly unsupported, fabricated, or contradicts available evidence.
- 2: Some grounded material but substantial unsupported claims.
- 3: Mostly plausible but with unclear grounding or mild overreach.
- 4: Strongly grounded with only minor over-compression or ambiguity.
- 5: Fully faithful to available evidence and clearly bounded.

2. citation_support
Measures whether important factual claims are tied to retrievable evidence, citations, source records, or explicit evidence labels.
- 1: No citations/evidence references, or references are unusable.
- 2: Some evidence references but many key claims are uncited or unclear.
- 3: Adequate support for main claims, but uneven citation coverage.
- 4: Most important claims have clear evidence support.
- 5: All important claims are tightly linked to clear evidence records.
For gpt_memo, which has no retrieval access, source-specific database questions should receive a low score unless the answer clearly states that it cannot inspect the source-specific evidence.

3. multi_hop_completeness
Measures whether the answer connects the necessary reasoning chain for the question.
For rare-earth disruption questions, this may include upstream disruption, processing/refining concentration, magnet inputs, downstream exposure, and risk consequence.
For source-specific questions, this means extracting and connecting the relevant source claims/entities/evidence.
- 1: No meaningful chain.
- 2: Mentions isolated facts but does not connect them.
- 3: Gives a partial chain.
- 4: Gives a mostly complete chain.
- 5: Gives a complete, well-structured chain with appropriate caveats.

4. unsupported_claim_reduction
Measures whether the system avoids confident claims not supported by evidence.
- 1: Many unsupported claims presented confidently.
- 2: Some unsupported claims or overstatements.
- 3: Mostly avoids unsupported claims, but some vague overreach remains.
- 4: Strong control of unsupported claims.
- 5: Explicitly separates supported claims, assumptions, and unknowns.

5. missing_evidence_detection
Measures whether the answer identifies weak, missing, or insufficient evidence.
- 1: Does not identify evidence gaps.
- 2: Mentions uncertainty vaguely.
- 3: Identifies some missing evidence.
- 4: Clearly identifies important missing links or weak support.
- 5: Gives precise, decision-relevant missing-evidence diagnostics.

6. latency_and_token_cost
Measures practical cost efficiency using the provided latency and token counts.
Use these objective bands unless the output failed:
- 5: <= 10 seconds and <= 4,000 total tokens.
- 4: <= 20 seconds and <= 8,000 total tokens.
- 3: <= 45 seconds and <= 15,000 total tokens.
- 2: <= 90 seconds and <= 30,000 total tokens.
- 1: slower than the above, missing runtime/cost data, or failed.
If a system is slower but produces a clearly more useful answer, do not increase this metric; that benefit belongs under other metrics.

7. analyst_workflow_usability
Measures whether the answer helps a professional analyst verify, inspect, and continue the investigation.
- 1: Not useful or failed.
- 2: Hard to inspect or act on.
- 3: Useful memo but limited inspectability.
- 4: Strong analyst utility with clear evidence/gap structure.
- 5: Excellent workflow support: inspectable evidence, reasoning structure, uncertainty, and clear next steps.

Failure rule:
If the system output is empty, errored, or unusable, assign 1 for all rubric metrics unless latency/token data alone clearly warrants a higher latency score. In most failed cases, all scores should be 1.

Return only valid JSON with this schema:
{
  "prompt_version": "lanthic_eval_judge_v1",
  "scores": {
    "faithfulness": 1,
    "citation_support": 1,
    "multi_hop_completeness": 1,
    "unsupported_claim_reduction": 1,
    "missing_evidence_detection": 1,
    "latency_and_token_cost": 1,
    "analyst_workflow_usability": 1
  },
  "rationales": {
    "faithfulness": "...",
    "citation_support": "...",
    "multi_hop_completeness": "...",
    "unsupported_claim_reduction": "...",
    "missing_evidence_detection": "...",
    "latency_and_token_cost": "...",
    "analyst_workflow_usability": "..."
  },
  "overall_comments": "...",
  "needs_human_review": false,
  "human_review_reason": ""
}
""".strip()

SCORE_COLUMNS = [
    "question_id",
    "task_type",
    "source_hint",
    "system",
    *METRICS,
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "notes",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def quote(value: str) -> str:
    return shlex.quote(value)


def fill_template(template: str, question: Dict[str, Any], output_path: Path) -> str:
    values = {
        "question": quote(question["question"]),
        "question_raw": question["question"],
        "question_id": quote(question["question_id"]),
        "question_id_raw": question["question_id"],
        "task_type": quote(question["task_type"]),
        "task_type_raw": question["task_type"],
        "source_hint": quote(question.get("source_hint", "")),
        "source_hint_raw": question.get("source_hint", ""),
        "output": quote(str(output_path)),
        "output_raw": str(output_path),
    }

    command = template
    for key, value in values.items():
        command = command.replace("{" + key + "}", value)
    return command


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def normalize_usage(usage: Dict[str, Any], question: str, answer: str) -> Dict[str, int]:
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("estimated_input_tokens")
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("estimated_output_tokens")
    )
    total_tokens = (
        usage.get("total_tokens")
        or usage.get("estimated_total_tokens")
    )

    if input_tokens is None:
        input_tokens = estimate_tokens(question)
    if output_tokens is None:
        output_tokens = estimate_tokens(answer)
    if total_tokens is None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def extract_text_from_json(obj: Any) -> str:
    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        for key in [
            "answer",
            "final_answer",
            "response",
            "content",
            "message",
            "text",
            "summary",
            "synthesis",
            "output",
        ]:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value

        return json.dumps(obj, indent=2, ensure_ascii=False)

    if isinstance(obj, list):
        return json.dumps(obj, indent=2, ensure_ascii=False)

    return str(obj)


def extract_usage_from_json(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}

    for key in ["usage", "token_usage", "tokens", "cost", "ledger"]:
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    return {}


def run_command_system(
    *,
    system: str,
    command_template: str,
    question: Dict[str, Any],
    output_path: Path,
    timeout_sec: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if not command_template:
        raise RuntimeError(f"Missing command template for {system}.")

    raw_output_path = output_path.with_suffix(f".{system}.raw.json")
    command = fill_template(command_template, question, raw_output_path)

    proc = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    parsed: Any = None
    answer = ""
    usage: Dict[str, Any] = {}

    if raw_output_path.exists():
        try:
            parsed = read_json(raw_output_path)
            answer = extract_text_from_json(parsed)
            usage = extract_usage_from_json(parsed)
        except Exception:
            answer = raw_output_path.read_text(encoding="utf-8", errors="ignore")

    if not answer.strip():
        try:
            parsed = json.loads(stdout)
            answer = extract_text_from_json(parsed)
            usage = extract_usage_from_json(parsed)
        except Exception:
            answer = stdout.strip()

    meta = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "raw_output_path": str(raw_output_path) if raw_output_path.exists() else "",
        "raw_output": parsed,
    }

    if proc.returncode != 0:
        raise RuntimeError(f"{system} command failed with return code {proc.returncode}.\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}")

    return answer, usage, meta

def run_gpt_memo(question: Dict[str, Any], model: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing OpenAI SDK. Install with: pip install openai") from exc

    client = OpenAI()

    system_prompt = (
        "You are the GPT-style risk memo baseline for an evaluation. "
        "You do not have access to the Lanthic database, knowledge graph, retrieval system, "
        "or evidence drawer. Answer as a concise analyst memo. Do not fabricate citations. "
        "If source-specific database evidence is required and unavailable to you, say so."
    )

    user_prompt = (
        f"QUESTION_ID: {question['question_id']}\n"
        f"TASK_TYPE: {question['task_type']}\n"
        f"SOURCE_HINT: {question.get('source_hint') or 'none'}\n\n"
        f"QUESTION:\n{question['question']}\n\n"
        "Format:\n"
        "1. Bottom line\n"
        "2. Evidence-supported points\n"
        "3. Uncertainties or missing evidence\n"
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

    return answer, usage, {"model": model}


def init_eval(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    ensure_dir(eval_dir / "runs")

    questions_path = eval_dir / "questions.jsonl"

    if questions_path.exists() and not args.force:
        print(f"[skip] {questions_path} exists. Use --force to overwrite.")
    else:
        write_jsonl(questions_path, QUESTIONS)
        print(f"[write] {questions_path}")

    make_score_sheet(args)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing questions file: {path}")
    return read_jsonl(path)


def run_eval(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    runs_dir = eval_dir / "runs"
    ensure_dir(runs_dir)

    questions = load_questions(Path(args.questions))

    for question in questions:
        for system in args.systems:
            out_path = runs_dir / f"{question['question_id']}_{system}.json"

            if out_path.exists() and not args.force:
                print(f"[skip] {out_path}")
                continue

            print(f"[run] {question['question_id']} {system}")

            started_at = now_utc()
            t0 = time.perf_counter()

            answer = ""
            usage: Dict[str, Any] = {}
            meta: Dict[str, Any] = {}
            error = ""

            try:
                if system == "gpt_memo":
                    answer, usage, meta = run_gpt_memo(question, args.model)

                elif system == "simple_rag":
                    answer, usage, meta = run_command_system(
                        system="simple_rag",
                        command_template=args.simple_rag_command,
                        question=question,
                        output_path=out_path,
                        timeout_sec=args.timeout_sec,
                    )

                elif system == "lanthic":
                    answer, usage, meta = run_command_system(
                        system="lanthic",
                        command_template=args.sarg_command,
                        question=question,
                        output_path=out_path,
                        timeout_sec=args.timeout_sec,
                    )

                else:
                    raise ValueError(f"Unknown system: {system}")

            except Exception as exc:
                error = repr(exc)

            latency_sec = round(time.perf_counter() - t0, 3)
            ended_at = now_utc()

            token_usage = normalize_usage(
                usage=usage,
                question=question["question"],
                answer=answer,
            )

            raw_output = meta.get("raw_output") if isinstance(meta, dict) else None

            retrieved_evidence = []
            if isinstance(raw_output, dict):
                retrieved_evidence = raw_output.get("retrieved_evidence") or []

            result = {
                "question_id": question["question_id"],
                "task_type": question["task_type"],
                "source_hint": question.get("source_hint", ""),
                "system": system,
                "question": question["question"],
                "answer": answer,
                "retrieved_evidence": retrieved_evidence,
                "started_at": started_at,
                "ended_at": ended_at,
                "latency_sec": latency_sec,
                **token_usage,
                "raw_usage": usage,
                "meta": meta,
                "error": error,
            }

            write_json(out_path, result)

            if error:
                print(f"[error] {out_path}: {error}")
            else:
                print(f"[write] {out_path}")

    make_score_sheet(args)


def make_score_sheet(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    questions_path = Path(getattr(args, "questions", eval_dir / "questions.jsonl"))
    scores_path = eval_dir / "eval_scores.csv"
    runs_dir = eval_dir / "runs"

    if not questions_path.exists():
        return

    questions = load_questions(questions_path)

    existing: Dict[Tuple[str, str], Dict[str, str]] = {}
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[(row["question_id"], row["system"])] = row

    rows = []

    for question in questions:
        for system in SYSTEMS:
            previous = existing.get((question["question_id"], system), {})
            run_path = runs_dir / f"{question['question_id']}_{system}.json"
            run_data = read_json(run_path) if run_path.exists() else {}

            row = {column: "" for column in SCORE_COLUMNS}
            row["question_id"] = question["question_id"]
            row["task_type"] = question["task_type"]
            row["source_hint"] = question.get("source_hint", "")
            row["system"] = system

            for metric in METRICS:
                row[metric] = previous.get(metric, "")

            row["latency_sec"] = previous.get("latency_sec") or run_data.get("latency_sec", "")
            row["input_tokens"] = previous.get("input_tokens") or run_data.get("input_tokens", "")
            row["output_tokens"] = previous.get("output_tokens") or run_data.get("output_tokens", "")
            row["total_tokens"] = previous.get("total_tokens") or run_data.get("total_tokens", "")
            row["notes"] = previous.get("notes", "")

            if run_data.get("error") and not row["notes"]:
                row["notes"] = f"ERROR: {run_data['error']}"

            rows.append(row)

    ensure_dir(scores_path.parent)
    with scores_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write] {scores_path}")


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def aggregate(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    scores_path = eval_dir / "eval_scores.csv"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing score sheet: {scores_path}")

    with scores_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summary_rows: List[Dict[str, Any]] = []

    def summarize(group: List[Dict[str, str]], group_name: str, task_type: str, system: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "group": group_name,
            "task_type": task_type,
            "system": system,
        }

        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            out[metric] = round(statistics.mean(values), 3) if values else ""

        for column in ["latency_sec", "input_tokens", "output_tokens", "total_tokens"]:
            values = [to_float(row.get(column)) for row in group]
            values = [value for value in values if value is not None]
            out[column] = round(statistics.mean(values), 3) if values else ""

        return out

    systems = sorted({row["system"] for row in rows})
    task_types = sorted({row["task_type"] for row in rows})

    for system in systems:
        group = [row for row in rows if row["system"] == system]
        summary_rows.append(summarize(group, "overall", "all", system))

    for task_type in task_types:
        for system in systems:
            group = [
                row for row in rows
                if row["task_type"] == task_type and row["system"] == system
            ]
            summary_rows.append(summarize(group, "by_task", task_type, system))

    summary_path = eval_dir / "results_summary.csv"
    fieldnames = [
        "group",
        "task_type",
        "system",
        *METRICS,
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[write] {summary_path}")

    latex_path = eval_dir / "results_table.tex"
    latex_path.write_text(make_latex_table(summary_rows), encoding="utf-8")
    print(f"[write] {latex_path}")


def make_latex_table(summary_rows: List[Dict[str, Any]]) -> str:
    overall = [row for row in summary_rows if row["group"] == "overall"]
    by_system = {row["system"]: row for row in overall}

    order = ["gpt_memo", "simple_rag", "lanthic"]

    labels = {
        "faithfulness": "Faithfulness",
        "citation_support": "Citation support",
        "multi_hop_completeness": "Multi-hop completeness",
        "unsupported_claim_reduction": "Unsupported-claim control",
        "missing_evidence_detection": "Missing-evidence detection",
        "latency_and_token_cost": "Latency/token cost",
        "analyst_workflow_usability": "Workflow usability",
    }

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & GPT memo & Simple RAG & Lanthic \\\\",
        "\\midrule",
    ]

    for metric in METRICS:
        values = []
        for system in order:
            value = by_system.get(system, {}).get(metric, "")
            values.append(str(value) if value != "" else "--")
        lines.append(f"{labels[metric]} & {values[0]} & {values[1]} & {values[2]} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Mean rubric scores across the fixed evaluation question set.}",
        "\\label{tab:evaluation-results}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)

def truncate_for_judge(value: Any, max_chars: int = 18000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 30] + "\n...[truncated for judge]"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge did not return JSON: {text[:1000]}")

    return json.loads(text[start : end + 1])


def coerce_judgement(obj: Dict[str, Any]) -> Dict[str, Any]:
    scores = obj.get("scores")
    rationales = obj.get("rationales")

    if not isinstance(scores, dict):
        raise ValueError("Judge output missing scores object.")

    if not isinstance(rationales, dict):
        rationales = {}

    clean_scores: Dict[str, int] = {}
    clean_rationales: Dict[str, str] = {}

    for metric in METRICS:
        raw_score = scores.get(metric)

        try:
            score = int(raw_score)
        except Exception as exc:
            raise ValueError(f"Missing/non-integer score for {metric}: {raw_score}") from exc

        if score < 1 or score > 5:
            raise ValueError(f"Out-of-range score for {metric}: {score}")

        clean_scores[metric] = score
        clean_rationales[metric] = str(rationales.get(metric, "")).strip()

    return {
        "prompt_version": obj.get("prompt_version", JUDGE_PROMPT_VERSION),
        "scores": clean_scores,
        "rationales": clean_rationales,
        "overall_comments": str(obj.get("overall_comments", "")).strip(),
        "needs_human_review": bool(obj.get("needs_human_review", False)),
        "human_review_reason": str(obj.get("human_review_reason", "")).strip(),
    }


def build_judge_payload(run_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": run_data.get("question_id", ""),
        "task_type": run_data.get("task_type", ""),
        "source_hint": run_data.get("source_hint", ""),
        "system": run_data.get("system", ""),
        "question": run_data.get("question", ""),
        "answer": run_data.get("answer", ""),
        "retrieved_evidence": run_data.get("retrieved_evidence", []),
        "latency_sec": run_data.get("latency_sec", ""),
        "input_tokens": run_data.get("input_tokens", ""),
        "output_tokens": run_data.get("output_tokens", ""),
        "total_tokens": run_data.get("total_tokens", ""),
        "error": run_data.get("error", ""),
    }


def call_judge(run_data: Dict[str, Any], judge_model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing OpenAI SDK. Install with: pip install openai") from exc

    client = OpenAI()

    payload = build_judge_payload(run_data)

    user_prompt = (
        "Score the following system output using the rubric. "
        "Return only the required JSON object.\n\n"
        f"{truncate_for_judge(payload)}"
    )

    response = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    judgement = coerce_judgement(extract_json_object(text))

    usage: Dict[str, Any] = {}
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return judgement, usage


def write_judge_prompt(eval_dir: Path) -> None:
    prompt_path = eval_dir / "judge_prompt.md"
    prompt_path.write_text(
        "# LLM-as-judge rubric\n\n"
        f"Prompt version: `{JUDGE_PROMPT_VERSION}`\n\n"
        "```text\n"
        f"{JUDGE_SYSTEM_PROMPT}\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"[write] {prompt_path}")


def update_scores_with_judgements(
    *,
    eval_dir: Path,
    questions_path: Path,
    systems: List[str],
    overwrite_scores: bool,
) -> None:
    scores_path = eval_dir / "eval_scores.csv"
    runs_dir = eval_dir / "runs"
    judgements_dir = eval_dir / "judgements"

    if not scores_path.exists():
        make_score_sheet(
            argparse.Namespace(
                eval_dir=str(eval_dir),
                questions=str(questions_path),
            )
        )

    with scores_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    row_by_key = {
        (row["question_id"], row["system"]): row
        for row in rows
    }

    questions = load_questions(questions_path)

    for question in questions:
        for system in systems:
            key = (question["question_id"], system)
            row = row_by_key.get(key)

            if row is None:
                continue

            judgement_path = judgements_dir / f"{question['question_id']}_{system}.json"

            if not judgement_path.exists():
                continue

            judgement_data = read_json(judgement_path)
            judgement = judgement_data.get("judgement", {})
            scores = judgement.get("scores", {})

            for metric in METRICS:
                if overwrite_scores or not str(row.get(metric, "")).strip():
                    row[metric] = str(scores.get(metric, ""))

            if not str(row.get("notes", "")).strip():
                comments = judgement.get("overall_comments", "")
                review = judgement.get("human_review_reason", "")
                if review:
                    row["notes"] = f"{comments} HUMAN REVIEW: {review}".strip()
                else:
                    row["notes"] = comments

            run_path = runs_dir / f"{question['question_id']}_{system}.json"
            if run_path.exists():
                run_data = read_json(run_path)
                for col in ["latency_sec", "input_tokens", "output_tokens", "total_tokens"]:
                    if not str(row.get(col, "")).strip():
                        row[col] = str(run_data.get(col, ""))

    with scores_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write] {scores_path}")


def judge(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    questions_path = Path(args.questions)
    runs_dir = eval_dir / "runs"
    judgements_dir = eval_dir / "judgements"

    ensure_dir(judgements_dir)
    write_judge_prompt(eval_dir)

    questions = load_questions(questions_path)

    for question in questions:
        for system in args.systems:
            run_path = runs_dir / f"{question['question_id']}_{system}.json"
            judgement_path = judgements_dir / f"{question['question_id']}_{system}.json"

            if not run_path.exists():
                print(f"[skip] missing run output: {run_path}")
                continue

            if judgement_path.exists() and not args.force:
                print(f"[skip] {judgement_path}")
                continue

            print(f"[judge] {question['question_id']} {system}")

            run_data = read_json(run_path)
            started_at = now_utc()
            t0 = time.perf_counter()

            try:
                judgement_obj, usage = call_judge(run_data, args.judge_model)
                error = ""
            except Exception as exc:
                judgement_obj = {
                    "prompt_version": JUDGE_PROMPT_VERSION,
                    "scores": {metric: 1 for metric in METRICS},
                    "rationales": {metric: "Judge failed; assigned failure score." for metric in METRICS},
                    "overall_comments": "Judge call failed.",
                    "needs_human_review": True,
                    "human_review_reason": repr(exc),
                }
                usage = {}
                error = repr(exc)

            ended_at = now_utc()
            latency_sec = round(time.perf_counter() - t0, 3)

            out = {
                "question_id": question["question_id"],
                "task_type": question["task_type"],
                "source_hint": question.get("source_hint", ""),
                "system": system,
                "judge_model": args.judge_model,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "started_at": started_at,
                "ended_at": ended_at,
                "latency_sec": latency_sec,
                "judgement": judgement_obj,
                "usage": usage,
                "error": error,
            }

            write_json(judgement_path, out)

            if error:
                print(f"[judge-error] {judgement_path}: {error}")
            else:
                print(f"[write] {judgement_path}")

    update_scores_with_judgements(
        eval_dir=eval_dir,
        questions_path=questions_path,
        systems=args.systems,
        overwrite_scores=args.overwrite_scores,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--eval-dir", default="eval")
    p_init.add_argument("--questions", default="eval/questions.jsonl")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=init_eval)

    p_run = sub.add_parser("run")
    p_run.add_argument("--eval-dir", default="eval")
    p_run.add_argument("--questions", default="eval/questions.jsonl")
    p_run.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    p_run.add_argument("--model", default=os.environ.get("EVAL_MODEL", "gpt-4.1-mini"))
    p_run.add_argument("--simple-rag-command", default=os.environ.get("SIMPLE_RAG_EVAL_COMMAND", ""))
    p_run.add_argument("--sarg-command", default=os.environ.get("SARG_EVAL_COMMAND", ""))
    p_run.add_argument("--timeout-sec", type=int, default=900)
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=run_eval)

    p_sheet = sub.add_parser("make-score-sheet")
    p_sheet.add_argument("--eval-dir", default="eval")
    p_sheet.add_argument("--questions", default="eval/questions.jsonl")
    p_sheet.set_defaults(func=make_score_sheet)

    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--eval-dir", default="eval")
    p_judge.add_argument("--questions", default="eval/questions.jsonl")
    p_judge.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    p_judge.add_argument("--judge-model", default=JUDGE_MODEL)
    p_judge.add_argument("--force", action="store_true")
    p_judge.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="Overwrite existing metric scores in eval_scores.csv. By default only blank cells are filled.",
    )
    p_judge.set_defaults(func=judge)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--eval-dir", default="eval")
    p_agg.set_defaults(func=aggregate)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()