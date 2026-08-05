import json
import random
import re
import shutil
import uuid
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SESSION_ROOT = Path("/kaggle/working/chatterbox_sessions") if Path("/kaggle/working").exists() else Path("outputs/chatterbox_sessions")

MODEL_TURBO = "Turbo"
MODEL_NANO = "Nano"
MODEL_ORIGINAL = "Original"
MODEL_CHOICES = [MODEL_TURBO, MODEL_NANO, MODEL_ORIGINAL]

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
                "error": None,
                "model_name": None,
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
                preview,
                chunk.get("audio_path") or "",
                chunk.get("error") or "",
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


def get_model_adapter(model_cache: dict[str, Any] | None, model_name: str) -> tuple[dict[str, Any], ModelAdapter]:
    cache = model_cache or {}
    cache_key = f"{model_name}:{DEVICE}"
    if cache_key not in cache:
        cache[cache_key] = ModelAdapter(model_name, DEVICE)
    return cache, cache[cache_key]


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
    return chunk["text"], audio_path, status_message(session, f"Selected chunk {chunk['index']}.")


def save_selected_chunk(session, chunk_number, edited_text):
    chunk = get_chunk(session, int(chunk_number or 1))
    chunk["text"] = edited_text or ""
    chunk["status"] = "pending"
    chunk["audio_path"] = None
    chunk["transcript"] = None
    chunk["text_score"] = None
    chunk["voice_score"] = None
    chunk["error"] = None
    chunk["model_name"] = None
    text_path = session_dir(session) / "chunks" / f"{chunk['index']:04d}.txt"
    text_path.write_text(chunk["text"], encoding="utf-8")
    save_session(session)
    return session, format_chunk_table(session), None, status_message(session, f"Saved chunk {chunk['index']}. Existing audio cleared.")


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
    return session, model_cache, format_chunk_table(session), chunk.get("audio_path"), status_message(session, f"Generated chunk {chunk['index']}.")


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
):
    if not session:
        raise gr.Error("Create or load a session first.")
    session["model_name"] = model_name
    session["generation_settings"] = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    reference_audio_path = copy_reference_to_session(session, reference_audio_path)
    model_cache, adapter = get_model_adapter(model_cache, model_name)

    last_audio = None
    for chunk in session["chunks"]:
        if chunk["status"] == "excluded":
            continue
        try:
            generate_one_chunk(
                session, adapter, chunk, reference_audio_path, temperature, int(seed_num or 0),
                min_p, top_p, int(top_k), repetition_penalty, exaggeration, cfg_weight, norm_loudness
            )
            last_audio = chunk.get("audio_path")
        except Exception as exc:
            chunk["status"] = "failed"
            chunk["error"] = str(exc)
            save_session(session)
            break

    return session, model_cache, format_chunk_table(session), last_audio, status_message(session, "Batch generation finished.")


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


def merge_chunks(session, output_filename, silence_ms, require_approved):
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
    save_session(session)
    return session, str(final_path), (sr, final_wav.squeeze(0).numpy()), status_message(session, f"Finalized `{filename}`.")


with gr.Blocks(title="Chatterbox Narration Suite", css=CUSTOM_CSS) as demo:
    gr.Markdown("# ⚡ Chatterbox Narration Suite")
    gr.Markdown("Phase 1 build: long-script chunking, session persistence, model selection, per-chunk regeneration, and final merge.")

    session_state = gr.State(None)
    model_cache_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Project / Session", open=True):
                project_name = gr.Textbox(value="youtube_narration", label="Project name")
                output_filename = gr.Textbox(value="narration.wav", label="Result file name")
                model_name = gr.Dropdown(MODEL_CHOICES, value=MODEL_TURBO, label="Model")
                session_path = gr.Textbox(label="Load existing session directory or session.json path", placeholder="outputs/chatterbox_sessions/...")
                with gr.Row():
                    load_session_btn = gr.Button("Load Session")

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
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K (Turbo/Nano)", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Original only)", value=0.05)
                exaggeration = gr.Slider(0.25, 2.0, step=.05, label="Exaggeration (Original only)", value=0.5)
                cfg_weight = gr.Slider(0.0, 1.0, step=.05, label="CFG/Pace (Original only)", value=0.5)
                norm_loudness = gr.Checkbox(value=True, label="Normalize reference loudness (Turbo/Nano)")

            status = gr.Markdown("No active session.")

    chunk_table = gr.Dataframe(
        headers=["#", "Status", "Chars", "Model", "Text", "Audio Path", "Error"],
        datatype=["number", "str", "number", "str", "str", "str", "str"],
        label="Chunks",
        interactive=False,
    )

    with gr.Row():
        with gr.Column():
            chunk_number = gr.Number(value=1, precision=0, label="Selected chunk number")
            load_chunk_btn = gr.Button("Load Selected Chunk")
            chunk_editor = gr.Textbox(label="Selected chunk text", lines=5)
            save_chunk_btn = gr.Button("Save Edited Chunk")
        with gr.Column():
            chunk_audio = gr.Audio(label="Selected/generated chunk audio")
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
        merge_btn = gr.Button("Merge / Finalize", variant="primary")
        final_audio = gr.Audio(label="Final narration")
        final_file = gr.File(label="Download final WAV")

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

    load_chunk_btn.click(
        fn=load_selected_chunk,
        inputs=[session_state, chunk_number],
        outputs=[chunk_editor, chunk_audio, status],
    )

    save_chunk_btn.click(
        fn=save_selected_chunk,
        inputs=[session_state, chunk_number, chunk_editor],
        outputs=[session_state, chunk_table, chunk_audio, status],
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
        outputs=[session_state, model_cache_state, chunk_table, chunk_audio, status],
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
        inputs=[session_state, output_filename, silence_ms, require_approved],
        outputs=[session_state, final_file, final_audio, status],
    )


if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
