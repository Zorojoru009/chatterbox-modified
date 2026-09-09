import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio as ta
except Exception:
    ta = None

from .config import safe_wav_filename
from .session import save_session, session_dir, status_message


def merge_chunks(
    session: dict[str, Any],
    output_filename: str,
    silence_ms: float,
    require_approved: bool,
    export_mp3: bool,
    mp3_bitrate: str,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    if ta is None:
        raise gr.Error("torchaudio is required to merge chunks but is not available.")

    existing_models = {
        chunk.get("model_name")
        for chunk in session.get("chunks", [])
        if chunk.get("audio_path") and chunk.get("model_name")
    }
    if len(existing_models) > 1 and not session.get("allow_mixed_models"):
        raise gr.Error(
            f"Cannot finalize a mixed-model session: {', '.join(sorted(existing_models))}. "
            "Set `allow_mixed_models: true` in the narration blueprint to allow this."
        )

    filename = safe_wav_filename(output_filename or session.get("output_filename"))
    session["output_filename"] = filename

    selected_chunks = [c for c in session.get("chunks", []) if c["status"] != "excluded"]
    if not selected_chunks:
        raise gr.Error("No chunks available to merge.")

    missing = [c.get("id") or c["index"] for c in selected_chunks if not c.get("audio_path")]
    if missing:
        raise gr.Error(f"These chunks have no audio yet: {missing}")

    if require_approved:
        not_approved = [c.get("id") or c["index"] for c in selected_chunks if c["status"] != "approved"]
        if not_approved:
            raise gr.Error(f"Approve these chunks before finalizing: {not_approved}")

    waves = []
    sr = None
    for chunk in selected_chunks:
        wav, chunk_sr = ta.load(chunk["audio_path"])
        if sr is None:
            sr = chunk_sr
        elif chunk_sr != sr:
            raise gr.Error(f"Chunk {chunk['index']} sample rate {chunk_sr} does not match project sample rate {sr}.")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        waves.append(wav)
        gap_samples = int((float(silence_ms or 0) / 1000.0) * sr)
        if gap_samples > 0:
            waves.append(torch.zeros((wav.shape[0], gap_samples), dtype=wav.dtype))

    # Remove the trailing silence gap if added
    if waves and float(silence_ms or 0) > 0:
        waves = waves[:-1]

    final_wav = torch.cat(waves, dim=1)
    final_path = session_dir(session) / "final" / filename
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(final_path), final_wav, sr)
    session["final_output_path"] = str(final_path)

    mp3_path = None
    if export_mp3:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise gr.Error("MP3 export requires ffmpeg. Disable MP3 export or install ffmpeg in the runtime environment.")
        mp3_path = session_dir(session) / "final" / f"{Path(filename).stem}.mp3"
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(final_path),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    str(mp3_bitrate or "192k"),
                    str(mp3_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
            raise gr.Error(f"MP3 export failed: {detail[0]}") from exc
        session["final_mp3_path"] = str(mp3_path)

    save_session(session)
    return (
        session,
        str(final_path),
        (sr, final_wav.squeeze(0).numpy()),
        str(mp3_path) if mp3_path else None,
        status_message(session, f"Finalized `{filename}`."),
    )
