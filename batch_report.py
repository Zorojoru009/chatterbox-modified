"""Validation report and chunk planning for chatterbox_batch_narrator.py.

Kept free of torch/gradio imports so it can be unit-tested and read by tools without the TTS stack.

The report (validation_report.json, next to the merged master) holds one entry per chunk:
index, id, text_sha (so a later partial run can tell whether a chunk's text changed), the chosen take's
seed / score / diagnosis / transcript, and every attempt. A partial run (--only-chunks) regenerates the
listed chunks, reuses the other chunk wavs as they are, and updates only the regenerated entries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def text_sha(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def load_report(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"chunks": {}}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    report.setdefault("chunks", {})
    return report


def failed_ids(report: dict[str, Any]) -> list[str]:
    entries = sorted(report.get("chunks", {}).values(), key=lambda e: int(e["index"]))
    return [e["id"] for e in entries if not e.get("passed")]


def plan_chunks(
    manifest: list[dict[str, Any]],
    only: str | None,
    report: dict[str, Any],
) -> set[int]:
    """Indices of the chunks to generate.

    only: None = every chunk; "failed" = the chunks the report lists as not passed; otherwise a
    comma-separated list of chunk ids (or 1-based indices). In a partial run every chunk left out must
    already have its wav, and its text must match the report, so a changed or shifted blueprint cannot
    merge stale audio.
    """
    if not only:
        return {item["index"] for item in manifest}

    wanted = set(failed_ids(report)) if only.strip().lower() == "failed" else {
        s.strip() for s in only.split(",") if s.strip()
    }
    by_id = {item["id"]: item["index"] for item in manifest}
    by_index = {str(item["index"]): item["index"] for item in manifest}
    unknown = [w for w in wanted if w not in by_id and w not in by_index]
    if unknown:
        raise ValueError(f"--only-chunks names chunks not in the blueprint: {', '.join(sorted(unknown))}")
    selected = {by_id.get(w) or by_index[w] for w in wanted}

    problems = []
    for item in manifest:
        if item["index"] in selected:
            continue
        entry = report.get("chunks", {}).get(str(item["index"]))
        if not Path(item["audio_path"]).exists():
            problems.append(f"{item['id']}: no wav from an earlier run")
        elif entry is None or entry.get("text_sha") != text_sha(item["text"]):
            problems.append(f"{item['id']}: text changed since its wav was made")
    if problems:
        raise ValueError(
            "a partial run can only reuse unchanged chunks; add these to --only-chunks or run all: "
            + "; ".join(problems[:10])
        )
    return selected


def record_chunk(report: dict[str, Any], item: dict[str, Any], result: dict[str, Any]) -> None:
    """result: from the worker: passed, score, seed, attempts, and the chosen take's diagnosis."""
    report["chunks"][str(item["index"])] = {
        "index": item["index"],
        "id": item["id"],
        "text": item["text"],
        "text_sha": text_sha(item["text"]),
        **result,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    entries = list(report.get("chunks", {}).values())
    failed = failed_ids(report)
    report["total"] = len(entries)
    report["passed"] = len(entries) - len(failed)
    report["failed"] = failed
    report["regenerated"] = sum(1 for e in entries if len(e.get("attempts") or []) > 1)
    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(summarize(report), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
