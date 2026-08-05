import json
import random
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

try:
    import torchaudio as ta
except Exception:  # pragma: no cover - surfaced at runtime in the UI
    ta = None

from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional Kaggle dependency
    WhisperModel = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SESSION_ROOT = Path("/kaggle/working/chatterbox_sessions") if Path("/kaggle/working").exists() else Path("outputs/chatterbox_sessions")

MODEL_TURBO = "Turbo"
MODEL_NANO = "Nano"
MODEL_ORIGINAL = "Original"
MODEL_CHOICES = [MODEL_TURBO, MODEL_NANO, MODEL_ORIGINAL]
PRESET_CHOICES = [
    "Reality Mechanism",
    "Clear Mental Model",
    "Investigative Case Study",
    "Philosophical Reflection",
    "High-Stakes Decision",
    "Custom",
]
GENERATION_PRESETS = {
    # Turbo/Nano primarily use temperature, top-p, top-k, repetition penalty,
    # and loudness normalization. Original also uses the remaining controls.
    "Reality Mechanism": (0.58, 0.92, 900, 1.18, 0.05, 0.48, 0.58, True),
    "Clear Mental Model": (0.52, 0.90, 800, 1.16, 0.05, 0.42, 0.62, True),
    "Investigative Case Study": (0.68, 0.94, 1000, 1.20, 0.05, 0.58, 0.52, True),
    "Philosophical Reflection": (0.48, 0.88, 700, 1.14, 0.05, 0.62, 0.48, True),
    "High-Stakes Decision": (0.74, 0.95, 1000, 1.20, 0.05, 0.70, 0.42, True),
    "Custom": (0.8, 0.95, 1000, 1.2, 0.05, 0.5, 0.5, True),
}
MODEL_ADAPTERS: dict[str, "ModelAdapter"] = {}
MODEL_ADAPTERS_LOCK = threading.Lock()
WHISPER_MODELS: dict[str, Any] = {}
WHISPER_MODELS_LOCK = threading.Lock()

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}

.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def safe_name(value: str, fallback: str = "narration") -> str:
    value = (value or "").strip()
    value = re.sub(r"[\\/]+", "-", value)
    value = re.sub(r"[^A-Za-z0-9._ -]+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ._-")
    return value or fallback


def safe_wav_filename(value: str) -> str:
    name = safe_name(value, fallback="narration")
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return name


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
    with path.open("r", encoding="utf-8") as f:
        session = json.load(f)
    return session


def list_session_paths() -> list[str]:
    if not SESSION_ROOT.exists():
        return []
    session_paths = [path for path in SESSION_ROOT.glob("*/session.json") if path.is_file()]
    return [str(path) for path in sorted(session_paths, key=lambda item: item.stat().st_mtime, reverse=True)]


def refresh_session_picker():
    choices = list_session_paths()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def apply_generation_preset(preset_name: str):
    return GENERATION_PRESETS.get(preset_name, GENERATION_PRESETS["Custom"])


def export_validation_report(session):
    if not session:
        raise gr.Error("Create or load a session first.")
    report_dir = session_dir(session) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    chunks = session.get("chunks", [])
    report = {
        "session_id": session.get("session_id"),
        "project_name": session.get("project_name"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_chunks": len(chunks),
            "generated": sum(bool(chunk.get("audio_path")) for chunk in chunks),
            "whisper_passed": sum(chunk.get("validation_status") == "passed" for chunk in chunks),
            "whisper_needs_review": sum(chunk.get("validation_status") == "needs_review" for chunk in chunks),
            "audio_passed": sum(chunk.get("audio_quality_status") == "passed" for chunk in chunks),
            "audio_needs_review": sum(chunk.get("audio_quality_status") == "needs_review" for chunk in chunks),
        },
        "chunks": chunks,
    }
    report_path = report_dir / f"{safe_name(session.get('project_name'), 'narration')}_validation_report.json"
    temp_path = report_path.with_suffix(".tmp.json")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(report_path)
    return str(report_path), status_message(session, f"Exported validation report: `{report_path.name}`.")


def make_session(project_name: str, output_filename: str, model_name: str, full_text: str, chunks: list[str]) -> dict[str, Any]:
    project = safe_name(project_name, fallback="youtube_narration")
    sid = f"{project}-{uuid.uuid4().hex[:8]}"
    sdir = SESSION_ROOT / sid
    now_chunks = []
    for index, chunk_text in enumerate(chunks, start=1):
        now_chunks.append(
            {
                "index": index,
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
                "model_name": None,
                "device": None,
            }
        )
    session = {
        "session_id": sid,
        "project_name": project,
        "output_filename": safe_wav_filename(output_filename),
        "model_name": model_name,
        "full_text": full_text,
        "reference_audio_path": None,
        "generation_settings": {},
        "chunks": now_chunks,
        "final_output_path": None,
        "session_dir": str(sdir),
    }
    (sdir / "chunks").mkdir(parents=True, exist_ok=True)
    (sdir / "reference").mkdir(parents=True, exist_ok=True)
    (sdir / "final").mkdir(parents=True, exist_ok=True)
    for chunk in session["chunks"]:
        text_path = sdir / "chunks" / f"{chunk['index']:04d}.txt"
        text_path.write_text(chunk["text"], encoding="utf-8")
    save_session(session)
    return session


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


def format_chunk_table(session: dict[str, Any] | None) -> list[list[Any]]:
    if not session:
        return []
    rows = []
    for chunk in session["chunks"]:
        preview = chunk["text"].replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        rows.append(
            [
                chunk["index"],
                chunk["status"],
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
    for chunk in session["chunks"]:
        counts[chunk["status"]] = counts.get(chunk["status"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no chunks"
    message = (
        f"{prefix + ' ' if prefix else ''}"
        f"Session `{session['session_id']}` | model: `{session['model_name']}` | chunks: {summary} | "
        f"path: `{session['session_dir']}`"
    )
    return message


def get_chunk(session: dict[str, Any], chunk_number: int) -> dict[str, Any]:
    if not session:
        raise gr.Error("Create or load a session first.")
    idx = int(chunk_number or 1)
    if idx < 1 or idx > len(session["chunks"]):
        raise gr.Error(f"Chunk number must be between 1 and {len(session['chunks'])}.")
    return session["chunks"][idx - 1]


def copy_reference_to_session(session: dict[str, Any], reference_audio_path: str | None) -> str | None:
    if not reference_audio_path:
        return session.get("reference_audio_path")

    if str(reference_audio_path).startswith("http://") or str(reference_audio_path).startswith("https://"):
        session["reference_audio_path"] = reference_audio_path
        return reference_audio_path

    source = Path(reference_audio_path)
    if not source.exists():
        raise gr.Error(f"Reference audio does not exist: {reference_audio_path}")

    ext = source.suffix or ".wav"
    dest = session_dir(session) / "reference" / f"reference{ext}"
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    session["reference_audio_path"] = str(dest)
    return str(dest)


class ModelAdapter:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.lock = threading.Lock()
        if model_name == MODEL_NANO:
            print(f"Loading Chatterbox Nano on {device}...")
            self.model = ChatterboxTurboTTS.from_pretrained(device, nano=True)
        elif model_name == MODEL_ORIGINAL:
            print(f"Loading original Chatterbox on {device}...")
            self.model = ChatterboxTTS.from_pretrained(device)
        else:
            print(f"Loading Chatterbox Turbo on {device}...")
            self.model = ChatterboxTurboTTS.from_pretrained(device, nano=False)

    @property
    def sr(self) -> int:
        return self.model.sr

    def generate(
        self,
        text: str,
        audio_prompt_path: str | None,
        temperature: float,
        min_p: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        exaggeration: float,
        cfg_weight: float,
        norm_loudness: bool,
    ) -> torch.Tensor:
        with self.lock:
            if self.model_name == MODEL_ORIGINAL:
                return self.model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

            return self.model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def available_generation_devices(enable_parallel: bool, max_parallel_devices: int) -> list[str]:
    if not torch.cuda.is_available():
        return ["cpu"]

    gpu_count = torch.cuda.device_count()
    if enable_parallel and gpu_count > 1:
        limit = max(1, min(int(max_parallel_devices or gpu_count), gpu_count))
        return [f"cuda:{idx}" for idx in range(limit)]

    return [default_device()]


def get_model_adapter(model_cache: dict[str, Any] | None, model_name: str, device: str | None = None) -> tuple[dict[str, Any], ModelAdapter]:
    """Return a cached adapter.

    The Gradio state argument is kept for callback compatibility, but model
    objects live in a process-global cache. Heavy torch modules should not be
    pushed through Gradio session state.
    """
    cache = model_cache or {}
    selected_device = device or default_device()
    cache_key = f"{model_name}:{selected_device}"
    with MODEL_ADAPTERS_LOCK:
        if cache_key not in MODEL_ADAPTERS:
            MODEL_ADAPTERS[cache_key] = ModelAdapter(model_name, selected_device)
    return cache, MODEL_ADAPTERS[cache_key]


def create_session(project_name, output_filename, model_name, full_text, target_chars):
    chunks = chunk_script(full_text or "", int(target_chars or 280))
    if not chunks:
        raise gr.Error("Add script text before splitting.")
    session = make_session(project_name, output_filename, model_name, full_text, chunks)
    return session, format_chunk_table(session), 1, chunks[0], status_message(session, "Created.")


def load_session(path_value):
    if not path_value:
        raise gr.Error("Enter a session directory or session.json path.")
    session = load_session_from_path(path_value)
    first_text = session["chunks"][0]["text"] if session.get("chunks") else ""
    return (
        session,
        format_chunk_table(session),
        1,
        first_text,
        session.get("full_text", ""),
        session.get("project_name", ""),
        session.get("output_filename", "narration.wav"),
        session.get("model_name", MODEL_TURBO),
        status_message(session, "Loaded."),
    )


def load_selected_chunk(session, chunk_number):
    chunk = get_chunk(session, int(chunk_number or 1))
    audio_path = chunk.get("audio_path")
    return (
        chunk["text"],
        audio_path,
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Selected chunk {chunk['index']}.")
    )


def move_selected_chunk(session, chunk_number, direction: int):
    if not session:
        raise gr.Error("Create or load a session first.")
    current = int(chunk_number or 1)
    next_number = max(1, min(len(session["chunks"]), current + int(direction)))
    return (next_number, *load_selected_chunk(session, next_number))


def save_selected_chunk(session, chunk_number, edited_text):
    chunk = get_chunk(session, int(chunk_number or 1))
    chunk["text"] = edited_text or ""
    chunk["status"] = "pending"
    chunk["audio_path"] = None
    chunk["transcript"] = None
    chunk["text_score"] = None
    chunk["voice_score"] = None
    chunk["validation_status"] = None
    chunk["validation_error"] = None
    chunk["validation_path"] = None
    chunk["audio_quality_status"] = None
    chunk["audio_quality_score"] = None
    chunk["audio_quality_error"] = None
    chunk["error"] = None
    chunk["model_name"] = None
    chunk["device"] = None
    text_path = session_dir(session) / "chunks" / f"{chunk['index']:04d}.txt"
    text_path.write_text(chunk["text"], encoding="utf-8")
    save_session(session)
    return (
        session,
        format_chunk_table(session),
        None,
        "",
        validation_details(chunk),
        status_message(session, f"Saved chunk {chunk['index']}. Existing audio cleared."),
    )


def generate_one_chunk(
    session: dict[str, Any],
    adapter: ModelAdapter,
    chunk: dict[str, Any],
    reference_audio_path: str | None,
    temperature: float,
    seed_num: int,
    min_p: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    exaggeration: float,
    cfg_weight: float,
    norm_loudness: bool,
):
    if seed_num:
        set_seed(int(seed_num) + int(chunk["index"]))

    chunk["status"] = "generating"
    chunk["error"] = None
    save_session(session)

    wav = adapter.generate(
        chunk["text"],
        audio_prompt_path=reference_audio_path,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        norm_loudness=norm_loudness,
    )

    audio_path = session_dir(session) / "chunks" / f"{chunk['index']:04d}.wav"
    if ta is None:
        raise gr.Error("torchaudio is required to save generated chunks but is not available.")
    ta.save(str(audio_path), wav.cpu(), adapter.sr)

    chunk["status"] = "generated"
    chunk["audio_path"] = str(audio_path)
    chunk["model_name"] = adapter.model_name
    chunk["transcript"] = None
    chunk["text_score"] = None
    chunk["validation_status"] = None
    chunk["validation_error"] = None
    chunk["validation_path"] = None
    chunk["audio_quality_status"] = None
    chunk["audio_quality_score"] = None
    chunk["audio_quality_error"] = None
    chunk["error"] = None
    save_session(session)


def generate_selected_chunk(
    session,
    model_cache,
    chunk_number,
    model_name,
    reference_audio_path,
    temperature,
    seed_num,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    exaggeration,
    cfg_weight,
    norm_loudness,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    session["model_name"] = model_name
    session["generation_settings"] = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    reference_audio_path = copy_reference_to_session(session, reference_audio_path)
    model_cache, adapter = get_model_adapter(model_cache, model_name)
    chunk = get_chunk(session, int(chunk_number or 1))
    try:
        generate_one_chunk(
            session, adapter, chunk, reference_audio_path, temperature, int(seed_num or 0),
            min_p, top_p, int(top_k), repetition_penalty, exaggeration, cfg_weight, norm_loudness
        )
    except Exception as exc:
        chunk["status"] = "failed"
        chunk["error"] = str(exc)
        save_session(session)
        raise
    return (
        session,
        model_cache,
        format_chunk_table(session),
        chunk.get("audio_path"),
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Generated chunk {chunk['index']}.")
    )


def collect_generation_settings(temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness):
    return {
        "temperature": float(temperature),
        "seed_num": int(seed_num or 0),
        "min_p": float(min_p),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "norm_loudness": bool(norm_loudness),
    }


def generate_chunk_wav(
    adapter: ModelAdapter,
    chunk_text: str,
    chunk_index: int,
    reference_audio_path: str | None,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, int, str]:
    seed_num = int(settings.get("seed_num") or 0)
    if seed_num:
        set_seed(seed_num + int(chunk_index))

    wav = adapter.generate(
        chunk_text,
        audio_prompt_path=reference_audio_path,
        temperature=float(settings["temperature"]),
        min_p=float(settings["min_p"]),
        top_p=float(settings["top_p"]),
        top_k=int(settings["top_k"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        exaggeration=float(settings["exaggeration"]),
        cfg_weight=float(settings["cfg_weight"]),
        norm_loudness=bool(settings["norm_loudness"]),
    )
    return wav.cpu(), adapter.sr, adapter.device


def save_chunk_audio(session: dict[str, Any], chunk: dict[str, Any], wav: torch.Tensor, sr: int, model_name: str, device: str) -> str:
    if ta is None:
        raise gr.Error("torchaudio is required to save generated chunks but is not available.")

    audio_path = session_dir(session) / "chunks" / f"{chunk['index']:04d}.wav"
    ta.save(str(audio_path), wav.cpu(), sr)
    chunk["status"] = "generated"
    chunk["audio_path"] = str(audio_path)
    chunk["model_name"] = model_name
    chunk["device"] = device
    chunk["transcript"] = None
    chunk["text_score"] = None
    chunk["validation_status"] = None
    chunk["validation_error"] = None
    chunk["validation_path"] = None
    chunk["audio_quality_status"] = None
    chunk["audio_quality_score"] = None
    chunk["audio_quality_error"] = None
    chunk["error"] = None
    return str(audio_path)


def generate_all_chunks(
    session,
    model_cache,
    model_name,
    reference_audio_path,
    temperature,
    seed_num,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    exaggeration,
    cfg_weight,
    norm_loudness,
    enable_parallel,
    max_parallel_devices,
    validation_enabled=False,
    auto_regenerate=False,
    whisper_model_name="small.en",
    whisper_device="cuda:0",
    validation_threshold=0.90,
    progress=gr.Progress(track_tqdm=True),
):
    if not session:
        raise gr.Error("Create or load a session first.")
    session["model_name"] = model_name
    settings = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    settings["enable_parallel"] = bool(enable_parallel)
    settings["max_parallel_devices"] = int(max_parallel_devices or 1)
    session["generation_settings"] = settings
    reference_audio_path = copy_reference_to_session(session, reference_audio_path)

    chunks_to_generate = [chunk for chunk in session["chunks"] if chunk["status"] != "excluded"]
    if not chunks_to_generate:
        return session, model_cache or {}, format_chunk_table(session), None, status_message(session, "No non-excluded chunks to generate.")

    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    last_audio = None
    for chunk in chunks_to_generate:
        chunk["status"] = "generating"
        chunk["error"] = None
    save_session(session)

    if len(devices) == 1:
        model_cache, adapter = get_model_adapter(model_cache, model_name, devices[0])
        for chunk_index, chunk in enumerate(chunks_to_generate, start=1):
            try:
                wav, sr, device = generate_chunk_wav(adapter, chunk["text"], chunk["index"], reference_audio_path, settings)
                last_audio = save_chunk_audio(session, chunk, wav, sr, adapter.model_name, device)
                save_session(session)
                progress(chunk_index / len(chunks_to_generate), desc=f"Generated chunk {chunk_index}/{len(chunks_to_generate)}")
            except Exception as exc:
                chunk["status"] = "failed"
                chunk["error"] = str(exc)
                save_session(session)
                break

        session, model_cache, auto_status = maybe_auto_validate_and_regenerate(
            session, model_cache, model_name, reference_audio_path, temperature, seed_num,
            min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness,
            enable_parallel, max_parallel_devices, validation_enabled, auto_regenerate,
            whisper_model_name, whisper_device, validation_threshold,
        )
        return (
            session,
            model_cache,
            format_chunk_table(session),
            last_audio,
            status_message(session, f"Batch generation finished on {devices[0]}. {auto_status}"),
        )

    adapters: list[ModelAdapter] = []
    for device in devices:
        model_cache, adapter = get_model_adapter(model_cache, model_name, device)
        adapters.append(adapter)

    future_to_chunk = {}
    max_workers = len(adapters)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job_index, chunk in enumerate(chunks_to_generate):
            adapter = adapters[job_index % len(adapters)]
            future = executor.submit(
                generate_chunk_wav,
                adapter,
                chunk["text"],
                chunk["index"],
                reference_audio_path,
                settings,
            )
            future_to_chunk[future] = chunk

        first_error = None
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            completed += 1
            try:
                wav, sr, device = future.result()
                last_audio = save_chunk_audio(session, chunk, wav, sr, model_name, device)
                save_session(session)
                progress(completed / len(chunks_to_generate), desc=f"Generated {completed}/{len(chunks_to_generate)} chunks")
            except Exception as exc:
                first_error = exc
                chunk["status"] = "failed"
                chunk["error"] = str(exc)
                save_session(session)

    if first_error is not None:
        for chunk in chunks_to_generate:
            if chunk["status"] == "generating":
                chunk["status"] = "failed"
                chunk["error"] = "Generation did not complete."
        save_session(session)
        return (
            session,
            model_cache,
            format_chunk_table(session),
            last_audio,
            status_message(session, f"Batch generation stopped after an error: {first_error}"),
        )

    session, model_cache, auto_status = maybe_auto_validate_and_regenerate(
        session, model_cache, model_name, reference_audio_path, temperature, seed_num,
        min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness,
        enable_parallel, max_parallel_devices, validation_enabled, auto_regenerate,
        whisper_model_name, whisper_device, validation_threshold,
    )
    return (
        session,
        model_cache,
        format_chunk_table(session),
        last_audio,
        status_message(session, f"Parallel batch generation finished across {len(devices)} device(s): {', '.join(devices)}. {auto_status}"),
    )


def gpu_status_text(enable_parallel, max_parallel_devices):
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    if torch.cuda.is_available():
        return f"Detected {torch.cuda.device_count()} CUDA device(s). Batch generation will use: `{', '.join(devices)}`."
    return "CUDA is not available. Batch generation will use CPU."


def warm_model_cache(model_cache, model_name, enable_parallel, max_parallel_devices):
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    for device in devices:
        model_cache, _adapter = get_model_adapter(model_cache, model_name, device)
    return model_cache, f"Loaded `{model_name}` on: `{', '.join(devices)}`."


def normalize_validation_text(value: str) -> str:
    """Normalize text for ASR comparison while ignoring Chatterbox event tags."""
    value = re.sub(r"\[[^\]]+\]", " ", value or "")
    value = value.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def validation_comparison(expected: str, transcript: str, threshold: float) -> dict[str, Any]:
    expected_normalized = normalize_validation_text(expected)
    transcript_normalized = normalize_validation_text(transcript)
    expected_words = expected_normalized.split()
    transcript_words = transcript_normalized.split()

    if not transcript_normalized:
        score = 0.0
    else:
        score = SequenceMatcher(None, expected_normalized, transcript_normalized).ratio()

    expected_counts = Counter(expected_words)
    transcript_counts = Counter(transcript_words)
    missing = list((expected_counts - transcript_counts).elements())
    extra = list((transcript_counts - expected_counts).elements())
    passed = bool(transcript_normalized) and score >= float(threshold)
    return {
        "expected_normalized": expected_normalized,
        "transcript_normalized": transcript_normalized,
        "score": round(score, 4),
        "threshold": float(threshold),
        "passed": passed,
        "missing_words": missing[:30],
        "extra_words": extra[:30],
        "expected_word_count": len(expected_words),
        "transcript_word_count": len(transcript_words),
    }


def get_whisper_model(model_name: str, device: str):
    if WhisperModel is None:
        raise gr.Error(
            "faster-whisper is not installed. Install it in Kaggle with `pip install faster-whisper`, then restart the app."
        )

    selected_device = str(device or "cpu")
    cache_key = f"{model_name}:{selected_device}"
    with WHISPER_MODELS_LOCK:
        if cache_key not in WHISPER_MODELS:
            if selected_device.startswith("cuda"):
                device_index = int(selected_device.split(":", 1)[1]) if ":" in selected_device else 0
                WHISPER_MODELS[cache_key] = WhisperModel(
                    model_name,
                    device="cuda",
                    device_index=device_index,
                    compute_type="float16",
                )
            else:
                WHISPER_MODELS[cache_key] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
    return WHISPER_MODELS[cache_key]


def transcribe_audio(audio_path: str, whisper_model_name: str, whisper_device: str) -> str:
    if not audio_path or not Path(audio_path).exists():
        raise gr.Error("Generate the chunk audio before validating it.")
    model = get_whisper_model(whisper_model_name, whisper_device)
    segments, _info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def validation_path(session: dict[str, Any], chunk: dict[str, Any]) -> Path:
    return session_dir(session) / "validation" / f"{chunk['index']:04d}.json"


def validate_chunk(
    session: dict[str, Any],
    chunk: dict[str, Any],
    whisper_model_name: str,
    whisper_device: str,
    validation_threshold: float,
) -> dict[str, Any]:
    transcript = transcribe_audio(chunk.get("audio_path"), whisper_model_name, whisper_device)
    comparison = validation_comparison(chunk["text"], transcript, float(validation_threshold))
    result = {
        "chunk_index": chunk["index"],
        "audio_path": chunk.get("audio_path"),
        "model_name": chunk.get("model_name"),
        "whisper_model": whisper_model_name,
        "whisper_device": whisper_device,
        "transcript": transcript,
        **comparison,
    }
    write_validation_report(session, chunk, "whisper", result)

    chunk["transcript"] = transcript
    chunk["text_score"] = comparison["score"]
    chunk["validation_status"] = "passed" if comparison["passed"] else "needs_review"
    chunk["validation_error"] = None if comparison["passed"] else (
        f"Missing: {', '.join(comparison['missing_words']) or 'none'}; "
        f"Extra: {', '.join(comparison['extra_words']) or 'none'}"
    )
    # A new validation result supersedes a previous approval.
    chunk["status"] = "validated" if comparison["passed"] else "needs_review"
    return result


def validation_details(chunk: dict[str, Any] | None) -> str:
    if not chunk:
        return "No chunk selected."
    sections = []
    if chunk.get("validation_status"):
        sections.append(
            f"**Whisper:** `{chunk['validation_status']}`  \n"
            f"**Text score:** `{chunk.get('text_score', 0):.3f}`  \n"
            f"**Transcript:** {chunk.get('transcript') or '(empty)'}  \n"
            f"**Details:** {chunk.get('validation_error') or 'No missing or extra words detected.'}"
        )
    if chunk.get("audio_quality_status"):
        sections.append(
            f"**Audio quality:** `{chunk['audio_quality_status']}`  \n"
            f"**Details:** {chunk.get('audio_quality_error') or 'No audio quality issues detected.'}"
        )
    return "\n\n".join(sections) or "This chunk has not been validated yet."


def write_validation_report(session: dict[str, Any], chunk: dict[str, Any], section: str, payload: dict[str, Any]):
    path = validation_path(session, chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    report.setdefault("chunk_index", chunk["index"])
    report[section] = payload
    temp_path = path.with_suffix(".tmp.json")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
    chunk["validation_path"] = str(path)


def audio_quality_details(chunk: dict[str, Any] | None) -> str:
    if not chunk or not chunk.get("audio_quality_status"):
        return "Audio quality has not been checked yet."
    return (
        f"**Audio quality:** `{chunk['audio_quality_status']}`  \n"
        f"**Details:** {chunk.get('audio_quality_error') or 'No audio quality issues detected.'}"
    )


def check_audio_quality(
    session: dict[str, Any],
    chunk: dict[str, Any],
    silence_threshold_db: float,
    max_silence_ms: float,
    max_clip_fraction: float,
    min_duration_s: float,
    max_duration_s: float,
    min_rms_dbfs: float,
) -> dict[str, Any]:
    if ta is None:
        raise gr.Error("torchaudio is required for audio quality checks.")
    audio_path = chunk.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise gr.Error("Generate the chunk audio before checking its quality.")

    wav, sample_rate = ta.load(audio_path)
    samples = wav.float().numpy()
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    duration_s = float(len(mono) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-8))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-8))
    silence_amplitude = 10.0 ** (float(silence_threshold_db) / 20.0)
    audible = np.abs(mono) > silence_amplitude
    if audible.any():
        first_audible = int(np.argmax(audible))
        last_audible = int(len(audible) - 1 - np.argmax(audible[::-1]))
        leading_silence_ms = first_audible / sample_rate * 1000.0
        trailing_silence_ms = (len(audible) - 1 - last_audible) / sample_rate * 1000.0
    else:
        leading_silence_ms = duration_s * 1000.0
        trailing_silence_ms = duration_s * 1000.0

    clip_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 1.0
    issues: list[str] = []
    if duration_s < float(min_duration_s):
        issues.append(f"duration too short ({duration_s:.2f}s)")
    if duration_s > float(max_duration_s):
        issues.append(f"duration too long ({duration_s:.2f}s)")
    if peak < 1e-5 or not audible.any():
        issues.append("near-silent audio")
    if rms_dbfs < float(min_rms_dbfs) and audible.any():
        issues.append(f"low loudness RMS {rms_dbfs:.1f}dBFS")
    if leading_silence_ms > float(max_silence_ms):
        issues.append(f"leading silence {leading_silence_ms:.0f}ms")
    if trailing_silence_ms > float(max_silence_ms):
        issues.append(f"trailing silence {trailing_silence_ms:.0f}ms")
    if clip_fraction > float(max_clip_fraction):
        issues.append(f"clipping fraction {clip_fraction:.5f}")

    score = max(0.0, 1.0 - min(1.0, len(issues) / 5.0))
    result = {
        "sample_rate": int(sample_rate),
        "duration_s": round(duration_s, 4),
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "leading_silence_ms": round(leading_silence_ms, 2),
        "trailing_silence_ms": round(trailing_silence_ms, 2),
        "clip_fraction": round(clip_fraction, 6),
        "silence_threshold_db": float(silence_threshold_db),
        "max_silence_ms": float(max_silence_ms),
        "min_rms_dbfs": float(min_rms_dbfs),
        "passed": not issues,
        "issues": issues,
    }
    chunk["audio_quality_status"] = "passed" if result["passed"] else "needs_review"
    chunk["audio_quality_score"] = round(score, 4)
    chunk["audio_quality_error"] = "; ".join(issues) if issues else None
    write_validation_report(session, chunk, "audio_quality", result)
    return result


def check_audio_selected(session, chunk_number, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs):
    if not session:
        raise gr.Error("Create or load a session first.")
    chunk = get_chunk(session, int(chunk_number or 1))
    result = check_audio_quality(session, chunk, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs)
    save_session(session)
    return session, format_chunk_table(session), audio_quality_details(chunk), status_message(
        session, f"Checked audio for chunk {chunk['index']}: {'passed' if result['passed'] else 'needs review'}."
    )


def check_audio_all(session, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs, progress=gr.Progress(track_tqdm=True)):
    if not session:
        raise gr.Error("Create or load a session first.")
    chunks = [chunk for chunk in session["chunks"] if chunk.get("audio_path") and chunk["status"] != "excluded"]
    if not chunks:
        raise gr.Error("Generate at least one chunk before checking audio quality.")
    passed = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            result = check_audio_quality(session, chunk, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs)
            passed += int(result["passed"])
        except Exception as exc:
            chunk["audio_quality_status"] = "error"
            chunk["audio_quality_error"] = str(exc)
        save_session(session)
        progress(chunk_index / len(chunks), desc=f"Checked audio {chunk_index}/{len(chunks)}")
    return session, format_chunk_table(session), status_message(session, f"Checked audio for {len(chunks)} chunk(s): {passed} passed, {len(chunks) - passed} need review.")


def ensure_validation_enabled(enabled: bool):
    if not enabled:
        raise gr.Error("Enable Whisper validation first, or continue without validation.")


def validate_selected_chunk(session, chunk_number, whisper_model_name, whisper_device, validation_threshold, enabled):
    if not session:
        raise gr.Error("Create or load a session first.")
    ensure_validation_enabled(enabled)
    chunk = get_chunk(session, int(chunk_number or 1))
    result = validate_chunk(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
    save_session(session)
    return (
        session,
        format_chunk_table(session),
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Validated chunk {chunk['index']} with score {result['score']:.3f}."),
    )


def validate_all_chunks(session, whisper_model_name, whisper_device, validation_threshold, enabled, progress=gr.Progress(track_tqdm=True)):
    if not session:
        raise gr.Error("Create or load a session first.")
    ensure_validation_enabled(enabled)
    chunks = [chunk for chunk in session["chunks"] if chunk.get("audio_path") and chunk["status"] != "excluded"]
    if not chunks:
        raise gr.Error("Generate at least one chunk before validating.")

    passed = 0
    first_error = None
    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            result = validate_chunk(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
            passed += int(result["passed"])
            save_session(session)
            progress(chunk_index / len(chunks), desc=f"Validated chunk {chunk_index}/{len(chunks)}")
        except Exception as exc:
            first_error = exc
            chunk["validation_status"] = "error"
            chunk["validation_error"] = str(exc)
            save_session(session)

    message = f"Validated {len(chunks)} chunk(s): {passed} passed, {len(chunks) - passed} need review."
    if first_error:
        message += f" First error: {first_error}"
    return session, format_chunk_table(session), status_message(session, message)


def regenerate_failed_chunks(
    session,
    model_cache,
    model_name,
    reference_audio_path,
    temperature,
    seed_num,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    exaggeration,
    cfg_weight,
    norm_loudness,
    enable_parallel,
    max_parallel_devices,
    validation_threshold,
    whisper_model_name,
    whisper_device,
    validation_enabled,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    ensure_validation_enabled(validation_enabled)
    failed = [
        chunk for chunk in session["chunks"]
        if chunk.get("validation_status") == "needs_review" and chunk["status"] != "excluded"
    ]
    if not failed:
        raise gr.Error("No chunks currently need review.")

    session["model_name"] = model_name
    settings = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    reference_audio_path = copy_reference_to_session(session, reference_audio_path)
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    model_cache, adapter = get_model_adapter(model_cache, model_name, devices[0])
    for chunk in failed:
        chunk["status"] = "generating"
        chunk["validation_status"] = None
        try:
            wav, sr, device = generate_chunk_wav(adapter, chunk["text"], chunk["index"], reference_audio_path, settings)
            save_chunk_audio(session, chunk, wav, sr, adapter.model_name, device)
            validate_chunk(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
            save_session(session)
        except Exception as exc:
            chunk["status"] = "failed"
            chunk["error"] = str(exc)
            save_session(session)

    return session, model_cache, format_chunk_table(session), status_message(session, f"Regenerated {len(failed)} chunk(s) needing review.")


def maybe_auto_validate_and_regenerate(
    session,
    model_cache,
    model_name,
    reference_audio_path,
    temperature,
    seed_num,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    exaggeration,
    cfg_weight,
    norm_loudness,
    enable_parallel,
    max_parallel_devices,
    validation_enabled,
    auto_regenerate,
    whisper_model_name,
    whisper_device,
    validation_threshold,
):
    if not validation_enabled or not auto_regenerate:
        return session, model_cache, ""

    session, _table, validation_status = validate_all_chunks(
        session,
        whisper_model_name,
        whisper_device,
        validation_threshold,
        True,
    )
    failed = [
        chunk for chunk in session["chunks"]
        if chunk.get("validation_status") == "needs_review" and chunk["status"] != "excluded"
    ]
    if not failed:
        return session, model_cache, validation_status

    session, model_cache, _table, regeneration_status = regenerate_failed_chunks(
        session,
        model_cache,
        model_name,
        reference_audio_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        exaggeration,
        cfg_weight,
        norm_loudness,
        enable_parallel,
        max_parallel_devices,
        validation_threshold,
        whisper_model_name,
        whisper_device,
        True,
    )
    return session, model_cache, f"{validation_status} Automatic regeneration: {regeneration_status}"


def exclude_selected_chunk(session, chunk_number):
    chunk = get_chunk(session, int(chunk_number or 1))
    chunk["status"] = "excluded"
    save_session(session)
    return session, format_chunk_table(session), status_message(session, f"Excluded chunk {chunk['index']}.")


def approve_selected_chunk(session, chunk_number):
    chunk = get_chunk(session, int(chunk_number or 1))
    if not chunk.get("audio_path"):
        raise gr.Error("Generate audio before approving this chunk.")
    chunk["status"] = "approved"
    save_session(session)
    return session, format_chunk_table(session), status_message(session, f"Approved chunk {chunk['index']}.")


def merge_chunks(session, output_filename, silence_ms, require_approved, export_mp3, mp3_bitrate):
    if not session:
        raise gr.Error("Create or load a session first.")
    if ta is None:
        raise gr.Error("torchaudio is required to merge chunks but is not available.")

    filename = safe_wav_filename(output_filename or session.get("output_filename"))
    session["output_filename"] = filename

    selected_chunks = [c for c in session["chunks"] if c["status"] != "excluded"]
    if not selected_chunks:
        raise gr.Error("No chunks available to merge.")
    missing = [c["index"] for c in selected_chunks if not c.get("audio_path")]
    if missing:
        raise gr.Error(f"These chunks have no audio yet: {missing}")
    if require_approved:
        not_approved = [c["index"] for c in selected_chunks if c["status"] != "approved"]
        if not_approved:
            raise gr.Error(f"Approve these chunks before finalizing: {not_approved}")

    waves = []
    sr = None
    for chunk in selected_chunks:
        wav, chunk_sr = ta.load(chunk["audio_path"])
        if sr is None:
            sr = chunk_sr
        elif chunk_sr != sr:
            raise gr.Error(f"Chunk {chunk['index']} sample rate {chunk_sr} does not match {sr}.")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        waves.append(wav)
        gap_samples = int((float(silence_ms or 0) / 1000.0) * sr)
        if gap_samples > 0:
            waves.append(torch.zeros((wav.shape[0], gap_samples), dtype=wav.dtype))

    if waves and float(silence_ms or 0) > 0:
        waves = waves[:-1]

    final_wav = torch.cat(waves, dim=1)
    final_path = session_dir(session) / "final" / filename
    ta.save(str(final_path), final_wav, sr)
    session["final_output_path"] = str(final_path)
    mp3_path = None
    if export_mp3:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise gr.Error("MP3 export requires ffmpeg. Disable MP3 export or install ffmpeg in the Kaggle runtime.")
        mp3_path = session_dir(session) / "final" / f"{Path(filename).stem}.mp3"
        try:
            subprocess.run(
                [ffmpeg_path, "-y", "-i", str(final_path), "-codec:a", "libmp3lame", "-b:a", str(mp3_bitrate), str(mp3_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
            raise gr.Error(f"MP3 export failed: {detail[0]}") from exc
        session["final_mp3_path"] = str(mp3_path)
    save_session(session)
    return session, str(final_path), (sr, final_wav.squeeze(0).numpy()), str(mp3_path) if mp3_path else None, status_message(session, f"Finalized `{filename}`.")


with gr.Blocks(title="Chatterbox Narration Suite", css=CUSTOM_CSS) as demo:
    gr.Markdown("# ⚡ Chatterbox Narration Suite")
    gr.Markdown("Phase 5 build: long-script chunking, resumable sessions, model selection, dual-GPU batch generation, optional Whisper/audio validation, presets, and final merge.")

    session_state = gr.State(None)
    model_cache_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Project / Session", open=True):
                project_name = gr.Textbox(value="youtube_narration", label="Project name")
                output_filename = gr.Textbox(value="narration.wav", label="Result file name")
                model_name = gr.Dropdown(MODEL_CHOICES, value=MODEL_TURBO, label="Model")
                session_path = gr.Textbox(label="Load existing session directory or session.json path", placeholder="outputs/chatterbox_sessions/...")
                session_picker = gr.Dropdown(
                    choices=list_session_paths(),
                    label="Saved sessions",
                    info="Select a previous session, then load it.",
                )
                with gr.Row():
                    load_session_btn = gr.Button("Load Session")
                    refresh_sessions_btn = gr.Button("Refresh Sessions")
                load_picked_session_btn = gr.Button("Load Selected Saved Session")

            text = gr.Textbox(
                value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                label="Full narration script",
                lines=10,
                elem_id="main_textbox",
            )

            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag, elem_classes=["tag-btn"])
                    btn.click(fn=None, inputs=[btn, text], outputs=text, js=INSERT_TAG_JS)

            with gr.Row():
                target_chars = gr.Slider(120, 800, step=10, value=280, label="Target chunk chars")
                split_btn = gr.Button("Split / New Session", variant="primary")

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
            )

        with gr.Column(scale=1):
            with gr.Accordion("Generation Options", open=True):
                preset_name = gr.Dropdown(PRESET_CHOICES, value="Reality Mechanism", label="Channel narration preset")
                apply_preset_btn = gr.Button("Apply Preset")
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K (Turbo/Nano)", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Original only)", value=0.05)
                exaggeration = gr.Slider(0.25, 2.0, step=.05, label="Exaggeration (Original only)", value=0.5)
                cfg_weight = gr.Slider(0.0, 1.0, step=.05, label="CFG/Pace (Original only)", value=0.5)
                norm_loudness = gr.Checkbox(value=True, label="Normalize reference loudness (Turbo/Nano)")

            with gr.Accordion("GPU / Batch Processing", open=True):
                enable_parallel = gr.Checkbox(value=True, label="Use multiple CUDA GPUs for Generate All when available")
                max_parallel_devices = gr.Slider(1, 8, step=1, value=2, label="Max GPU workers")
                gpu_status = gr.Markdown(gpu_status_text(True, 2))
                with gr.Row():
                    refresh_gpu_btn = gr.Button("Refresh GPU Status")
                    warm_models_btn = gr.Button("Load Selected Model on Batch Devices")

            with gr.Accordion("Whisper Validation", open=True):
                validation_enabled = gr.Checkbox(
                    value=False,
                    label="Enable Whisper validation (optional)",
                    info="Leave disabled to generate and merge without installing or loading Whisper.",
                )
                whisper_model_name = gr.Dropdown(
                    ["tiny.en", "base.en", "small.en", "medium.en"],
                    value="small.en",
                    label="Whisper model",
                )
                whisper_device = gr.Dropdown(
                    ["cpu", "cuda:0", "cuda:1"],
                    value="cuda:0" if torch.cuda.is_available() else "cpu",
                    label="Whisper device",
                )
                validation_threshold = gr.Slider(0.50, 0.99, step=0.01, value=0.90, label="Pass score threshold")
                auto_regenerate = gr.Checkbox(
                    value=False,
                    label="Auto-regenerate Generate All chunks below threshold",
                    info="Requires Whisper validation and retries each failed chunk once.",
                )
                with gr.Row():
                    validate_selected_btn = gr.Button("Validate Selected")
                    validate_all_btn = gr.Button("Validate All")
                regenerate_failed_btn = gr.Button("Regenerate Chunks Needing Review")

            with gr.Accordion("Audio Quality Checks (Optional)", open=False):
                silence_threshold_db = gr.Slider(-60, -20, step=1, value=-45, label="Silence threshold (dBFS)")
                max_silence_ms = gr.Slider(0, 2000, step=50, value=250, label="Maximum leading/trailing silence (ms)")
                max_clip_fraction = gr.Slider(0, 0.01, step=0.0005, value=0.001, label="Maximum clipping fraction")
                min_duration_s = gr.Slider(0.05, 5, step=0.05, value=0.20, label="Minimum chunk duration (s)")
                max_duration_s = gr.Slider(5, 180, step=1, value=120, label="Maximum chunk duration (s)")
                min_rms_dbfs = gr.Slider(-60, -10, step=1, value=-38, label="Minimum RMS loudness (dBFS)")
                with gr.Row():
                    check_audio_selected_btn = gr.Button("Check Audio Selected")
                    check_audio_all_btn = gr.Button("Check Audio All")

            status = gr.Markdown("No active session.")

    chunk_table = gr.Dataframe(
        headers=["#", "Status", "Chars", "Model", "Device", "Text Score", "Whisper", "Audio", "Text", "Audio Path", "Transcript / Error"],
        datatype=["number", "str", "number", "str", "str", "str", "str", "str", "str", "str", "str"],
        label="Chunks",
        interactive=False,
    )

    with gr.Row():
        with gr.Column():
            chunk_number = gr.Number(value=1, precision=0, label="Selected chunk number")
            with gr.Row():
                previous_chunk_btn = gr.Button("← Previous Chunk")
                next_chunk_btn = gr.Button("Next Chunk →")
            load_chunk_btn = gr.Button("Load Selected Chunk")
            chunk_editor = gr.Textbox(label="Selected chunk text", lines=5)
            save_chunk_btn = gr.Button("Save Edited Chunk")
        with gr.Column():
            chunk_audio = gr.Audio(label="Selected/generated chunk audio")
            chunk_transcript = gr.Textbox(label="Whisper transcript", lines=3, interactive=False)
            chunk_validation = gr.Markdown("This chunk has not been validated yet.")
            with gr.Row():
                generate_selected_btn = gr.Button("Generate Selected", variant="primary")
                generate_all_btn = gr.Button("Generate All")
            with gr.Row():
                approve_btn = gr.Button("Approve Selected")
                exclude_btn = gr.Button("Exclude Selected")

    with gr.Accordion("Finalize", open=True):
        with gr.Row():
            silence_ms = gr.Slider(0, 1000, step=25, value=200, label="Silence between chunks (ms)")
            require_approved = gr.Checkbox(value=False, label="Require all chunks approved")
            export_mp3 = gr.Checkbox(value=False, label="Also export MP3")
            mp3_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], value="192k", label="MP3 bitrate")
        merge_btn = gr.Button("Merge / Finalize", variant="primary")
        final_audio = gr.Audio(label="Final narration")
        final_file = gr.File(label="Download final WAV")
        final_mp3_file = gr.File(label="Download final MP3")
        export_report_btn = gr.Button("Export Validation Report")
        validation_report_file = gr.File(label="Validation report JSON")

    split_btn.click(
        fn=create_session,
        inputs=[project_name, output_filename, model_name, text, target_chars],
        outputs=[session_state, chunk_table, chunk_number, chunk_editor, status],
    )

    load_session_btn.click(
        fn=load_session,
        inputs=[session_path],
        outputs=[session_state, chunk_table, chunk_number, chunk_editor, text, project_name, output_filename, model_name, status],
    )

    refresh_sessions_btn.click(
        fn=refresh_session_picker,
        outputs=[session_picker],
    )

    load_picked_session_btn.click(
        fn=load_session,
        inputs=[session_picker],
        outputs=[session_state, chunk_table, chunk_number, chunk_editor, text, project_name, output_filename, model_name, status],
    )

    apply_preset_btn.click(
        fn=apply_generation_preset,
        inputs=[preset_name],
        outputs=[temp, top_p, top_k, repetition_penalty, min_p, exaggeration, cfg_weight, norm_loudness],
    )

    load_chunk_btn.click(
        fn=load_selected_chunk,
        inputs=[session_state, chunk_number],
        outputs=[chunk_editor, chunk_audio, chunk_transcript, chunk_validation, status],
    )

    previous_chunk_btn.click(
        fn=lambda session, number: move_selected_chunk(session, number, -1),
        inputs=[session_state, chunk_number],
        outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, status],
    )

    next_chunk_btn.click(
        fn=lambda session, number: move_selected_chunk(session, number, 1),
        inputs=[session_state, chunk_number],
        outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, status],
    )

    save_chunk_btn.click(
        fn=save_selected_chunk,
        inputs=[session_state, chunk_number, chunk_editor],
        outputs=[session_state, chunk_table, chunk_audio, chunk_transcript, chunk_validation, status],
    )

    generate_selected_btn.click(
        fn=generate_selected_chunk,
        inputs=[
            session_state,
            model_cache_state,
            chunk_number,
            model_name,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            exaggeration,
            cfg_weight,
            norm_loudness,
        ],
        outputs=[session_state, model_cache_state, chunk_table, chunk_audio, chunk_transcript, chunk_validation, status],
    )

    refresh_gpu_btn.click(
        fn=gpu_status_text,
        inputs=[enable_parallel, max_parallel_devices],
        outputs=[gpu_status],
    )

    warm_models_btn.click(
        fn=warm_model_cache,
        inputs=[model_cache_state, model_name, enable_parallel, max_parallel_devices],
        outputs=[model_cache_state, status],
    )

    validate_selected_btn.click(
        fn=validate_selected_chunk,
        inputs=[session_state, chunk_number, whisper_model_name, whisper_device, validation_threshold, validation_enabled],
        outputs=[session_state, chunk_table, chunk_transcript, chunk_validation, status],
    )

    validate_all_btn.click(
        fn=validate_all_chunks,
        inputs=[session_state, whisper_model_name, whisper_device, validation_threshold, validation_enabled],
        outputs=[session_state, chunk_table, status],
    )

    check_audio_selected_btn.click(
        fn=check_audio_selected,
        inputs=[
            session_state,
            chunk_number,
            silence_threshold_db,
            max_silence_ms,
            max_clip_fraction,
            min_duration_s,
            max_duration_s,
            min_rms_dbfs,
        ],
        outputs=[session_state, chunk_table, chunk_validation, status],
    )

    check_audio_all_btn.click(
        fn=check_audio_all,
        inputs=[session_state, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs],
        outputs=[session_state, chunk_table, status],
    )

    regenerate_failed_btn.click(
        fn=regenerate_failed_chunks,
        inputs=[
            session_state,
            model_cache_state,
            model_name,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            exaggeration,
            cfg_weight,
            norm_loudness,
            enable_parallel,
            max_parallel_devices,
            validation_threshold,
            whisper_model_name,
            whisper_device,
            validation_enabled,
        ],
        outputs=[session_state, model_cache_state, chunk_table, status],
    )

    generate_all_btn.click(
        fn=generate_all_chunks,
        inputs=[
            session_state,
            model_cache_state,
            model_name,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            exaggeration,
            cfg_weight,
            norm_loudness,
            enable_parallel,
            max_parallel_devices,
            validation_enabled,
            auto_regenerate,
            whisper_model_name,
            whisper_device,
            validation_threshold,
        ],
        outputs=[session_state, model_cache_state, chunk_table, chunk_audio, status],
    )

    approve_btn.click(
        fn=approve_selected_chunk,
        inputs=[session_state, chunk_number],
        outputs=[session_state, chunk_table, status],
    )

    exclude_btn.click(
        fn=exclude_selected_chunk,
        inputs=[session_state, chunk_number],
        outputs=[session_state, chunk_table, status],
    )

    merge_btn.click(
        fn=merge_chunks,
        inputs=[session_state, output_filename, silence_ms, require_approved, export_mp3, mp3_bitrate],
        outputs=[session_state, final_file, final_audio, final_mp3_file, status],
    )

    export_report_btn.click(
        fn=export_validation_report,
        inputs=[session_state],
        outputs=[validation_report_file, status],
    )


if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        share=True,
        allowed_paths=[str(SESSION_ROOT.resolve())],
    )
