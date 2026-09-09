import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

try:
    import torchaudio as ta
except Exception:
    ta = None

from .config import (
    SESSION_ROOT,
    safe_name,
    safe_wav_filename,
)


def session_dir(session: dict[str, Any]) -> Path:
    return Path(session["session_dir"])


def session_json_path(session: dict[str, Any]) -> Path:
    return session_dir(session) / "session.json"


def save_session(session: dict[str, Any]) -> None:
    sdir = session_dir(session)
    sdir.mkdir(parents=True, exist_ok=True)
    tmp_path = sdir / "session.tmp.json"
    final_path = session_json_path(session)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    tmp_path.replace(final_path)


def load_session_from_path(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if path.is_dir():
        path = path / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        session = json.load(f)
    return session


from datetime import datetime


def get_next_session_number(session_root: Path = SESSION_ROOT) -> int:
    """Return the next chronological session index (e.g. 1, 2, 3...) based on existing sessions."""
    if not session_root.exists():
        return 1

    existing_numbers: list[int] = []
    session_dirs = [p for p in session_root.iterdir() if p.is_dir()]
    for p in session_dirs:
        # Match directory prefixes like 001_ or 0001_
        match = re.match(r"^(\d+)", p.name)
        if match:
            try:
                existing_numbers.append(int(match.group(1)))
            except ValueError:
                pass
        # Check inside session.json if available
        sess_file = p / "session.json"
        if sess_file.is_file():
            try:
                with sess_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data.get("session_number"), int):
                        existing_numbers.append(data["session_number"])
            except Exception:
                pass

    if not existing_numbers:
        return max(1, len(session_dirs) + 1)
    return max(max(existing_numbers), len(session_dirs)) + 1


def generate_chronological_session_id(project_name: str, session_root: Path = SESSION_ROOT) -> tuple[int, str]:
    """Generate a numbered chronological session ID like 001_youtube_narration_20260909_114530."""
    project = safe_name(project_name, fallback="narration")
    num = get_next_session_number(session_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = f"{num:03d}_{project}_{timestamp}"
    return num, sid


def list_session_choices(session_root: Path = SESSION_ROOT) -> list[tuple[str, str]]:
    """Return a list of (human_readable_label, session_path) tuples sorted chronologically (newest first)."""
    if not session_root.exists():
        return []
    session_paths = [path for path in session_root.glob("*/session.json") if path.is_file()]
    sorted_paths = sorted(session_paths, key=lambda item: item.stat().st_mtime, reverse=True)
    choices: list[tuple[str, str]] = []
    for p in sorted_paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            num = data.get("session_number")
            num_str = f"#{num:03d} " if num is not None else ""
            proj = data.get("project_name") or p.parent.name
            chunks_cnt = len(data.get("chunks", []))
            created = data.get("created_at") or datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            label = f"{num_str}{proj} ({chunks_cnt} chunks) — {created}"
            choices.append((label, str(p)))
        except Exception:
            choices.append((p.parent.name, str(p)))
    return choices


def list_session_paths(session_root: Path = SESSION_ROOT) -> list[str]:
    """Return a list of raw session.json paths sorted newest first."""
    if not session_root.exists():
        return []
    session_paths = [path for path in session_root.glob("*/session.json") if path.is_file()]
    return [str(path) for path in sorted(session_paths, key=lambda item: item.stat().st_mtime, reverse=True)]


def get_chunk(session: dict[str, Any], chunk_number: int) -> dict[str, Any]:
    if not session:
        raise ValueError("Create or load a session first.")
    idx = int(chunk_number or 1)
    chunks = session.get("chunks", [])
    if idx < 1 or idx > len(chunks):
        raise ValueError(f"Chunk number must be between 1 and {len(chunks)}.")
    return chunks[idx - 1]


def copy_reference_to_session(session: dict[str, Any], reference_audio_path: str | None) -> str | None:
    if not reference_audio_path:
        return session.get("reference_audio_path")

    ref_str = str(reference_audio_path)
    if ref_str.startswith("http://") or ref_str.startswith("https://"):
        session["reference_audio_path"] = ref_str
        return ref_str

    source = Path(reference_audio_path)
    if not source.exists():
        raise FileNotFoundError(f"Reference audio does not exist: {reference_audio_path}")

    if ta is not None:
        try:
            info = ta.info(str(source))
            duration_s = float(info.num_frames / info.sample_rate) if info.sample_rate else 0.0
            if duration_s <= 5.0:
                raise ValueError(f"Reference audio must be longer than 5 seconds; this file is {duration_s:.2f} seconds.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not inspect reference audio: {exc}") from exc

    ext = source.suffix or ".wav"
    dest = session_dir(session) / "reference" / f"reference{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    session["reference_audio_path"] = str(dest)
    return str(dest)


def make_session(
    project_name: str,
    output_filename: str,
    model_name: str,
    full_text: str,
    chunks: list[str],
) -> dict[str, Any]:
    project = safe_name(project_name, fallback="youtube_narration")
    session_num, sid = generate_chronological_session_id(project)
    sdir = SESSION_ROOT / sid
    now_chunks = []
    for index, chunk_text in enumerate(chunks, start=1):
        now_chunks.append(
            {
                "index": index,
                "id": f"chunk{index:04d}",
                "note": "",
                "text": chunk_text,
                "status": "pending",
                "audio_path": None,
                "transcript": None,
                "text_score": None,
                "voice_score": None,
                "validation_status": None,
                "validation_error": None,
                "validation_path": None,
                "audio_quality_status": None,
                "audio_quality_score": None,
                "audio_quality_error": None,
                "error": None,
                "model_name": model_name,
                "device": None,
                "settings": None,
            }
        )
    session = {
        "session_number": session_num,
        "session_id": sid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_name": project,
        "output_filename": safe_wav_filename(output_filename),
        "model_name": model_name,
        "full_text": full_text,
        "reference_audio_path": None,
        "generation_settings": {},
        "chunks": now_chunks,
        "final_output_path": None,
        "session_dir": str(sdir),
        "from_blueprint": False,
    }
    (sdir / "chunks").mkdir(parents=True, exist_ok=True)
    (sdir / "reference").mkdir(parents=True, exist_ok=True)
    (sdir / "final").mkdir(parents=True, exist_ok=True)
    for chunk in session["chunks"]:
        text_path = sdir / "chunks" / f"{chunk['index']:04d}.txt"
        text_path.write_text(chunk["text"], encoding="utf-8")
    save_session(session)
    return session


def format_chunk_table(session: dict[str, Any] | None) -> list[list[Any]]:
    if not session:
        return []
    rows = []
    for chunk in session.get("chunks", []):
        preview = chunk["text"].replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        rows.append(
            [
                chunk.get("id") or f"chunk{chunk['index']:04d}",
                chunk["index"],
                chunk["status"],
                chunk.get("note") or "",
                len(chunk["text"]),
                chunk.get("model_name") or "",
                chunk.get("device") or "",
                f"{chunk['text_score']:.3f}" if chunk.get("text_score") is not None else "",
                chunk.get("validation_status") or "",
                chunk.get("audio_quality_status") or "",
                preview,
                chunk.get("audio_path") or "",
                chunk.get("transcript") or chunk.get("validation_error") or chunk.get("error") or "",
            ]
        )
    return rows


def status_message(session: dict[str, Any] | None, prefix: str = "") -> str:
    if not session:
        return prefix or "No active session."
    counts: dict[str, int] = {}
    for chunk in session.get("chunks", []):
        counts[chunk["status"]] = counts.get(chunk["status"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no chunks"
    sid = session.get("session_id", session.get("project_name", "unnamed"))
    model = session.get("model_name", "default")
    sdir = session.get("session_dir", "in-memory")
    message = (
        f"{prefix + ' ' if prefix else ''}"
        f"Session `{sid}` | model: `{model}` | chunks: {summary} | "
        f"path: `{sdir}`"
    )
    return message


# Text splitting heuristics for raw scripts
def split_sentences_preserving_tags(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces: list[str] = []
    current: list[str] = []
    bracket_depth = 0

    for idx, char in enumerate(normalized):
        current.append(char)
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        next_char = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        at_sentence_end = char in ".!?" and bracket_depth == 0 and (not next_char or next_char.isspace())
        at_paragraph_end = char == "\n" and next_char == "\n" and bracket_depth == 0
        if at_sentence_end or at_paragraph_end:
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []

    tail = "".join(current).strip()
    if tail:
        pieces.append(tail)

    return pieces


def split_long_piece(piece: str, target_chars: int) -> list[str]:
    if len(piece) <= target_chars * 1.25:
        return [piece]

    words = piece.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > target_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_script(text: str, target_chars: int) -> list[str]:
    target = max(120, min(int(target_chars or 280), 800))
    sentences = split_sentences_preserving_tags(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        if len(sentence) > target * 1.25:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_piece(sentence, target))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks
