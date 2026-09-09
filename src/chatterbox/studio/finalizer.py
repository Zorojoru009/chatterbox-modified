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
from .session import format_chunk_table, get_chunk, save_session, session_dir, status_message


def trim_silence_boundary(
    wav: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -45.0,
    head_pad_ms: float = 40.0,
    tail_pad_ms: float = 120.0,
) -> torch.Tensor:
    """Trim leading and trailing digital dead air while preserving internal pauses and natural tail decay."""
    if wav.numel() == 0:
        return wav

    mono = wav.abs().max(dim=0).values
    silence_amp = 10.0 ** (float(threshold_db) / 20.0)
    audible = mono > silence_amp

    if not audible.any():
        return wav

    first_idx = int(torch.argmax(audible.int()).item())
    reversed_audible = audible.flip(dims=[0])
    last_idx = int(len(audible) - 1 - torch.argmax(reversed_audible.int()).item())

    head_pad = int((float(head_pad_ms) / 1000.0) * sample_rate)
    tail_pad = int((float(tail_pad_ms) / 1000.0) * sample_rate)

    start = max(0, first_idx - head_pad)
    end = min(wav.shape[-1], last_idx + 1 + tail_pad)

    trimmed = wav[..., start:end].clone()

    fade_len = min(int(0.005 * sample_rate), trimmed.shape[-1])
    if fade_len > 1:
        fade_out = torch.linspace(1.0, 0.0, fade_len, dtype=trimmed.dtype, device=trimmed.device)
        trimmed[..., -fade_len:] *= fade_out
        fade_in = torch.linspace(0.0, 1.0, fade_len, dtype=trimmed.dtype, device=trimmed.device)
        trimmed[..., :fade_len] *= fade_in

    return trimmed


def trim_chunk_audio(
    session: dict[str, Any],
    chunk_number: int,
    threshold_db: float = -45.0,
    head_pad_ms: float = 40.0,
    tail_pad_ms: float = 120.0,
) -> tuple[dict[str, Any], list[list[Any]], str | None, str, str]:
    if not session:
        raise gr.Error("Create or load a session first.")
    if ta is None:
        raise gr.Error("torchaudio is required to trim audio.")

    chunk = get_chunk(session, int(chunk_number or 1))
    audio_path = chunk.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise gr.Error("Chunk has no generated audio to trim.")

    wav, sr = ta.load(audio_path)
    trimmed = trim_silence_boundary(wav, sr, threshold_db, head_pad_ms, tail_pad_ms)
    ta.save(audio_path, trimmed, sr)

    from .validator import check_audio_quality, validation_details
    check_audio_quality(
        session,
        chunk,
        silence_threshold_db=threshold_db,
        max_silence_ms=600.0,
        max_clip_fraction=0.001,
        min_duration_s=0.20,
        max_duration_s=120.0,
        min_rms_dbfs=-38.0,
    )
    save_session(session)
    return (
        session,
        format_chunk_table(session),
        audio_path,
        validation_details(chunk),
        status_message(session, f"Auto-trimmed dead air on chunk {chunk.get('id') or chunk['index']}. Acoustic checks updated."),
    )


def trim_all_chunks_audio(
    session: dict[str, Any],
    threshold_db: float = -45.0,
    head_pad_ms: float = 40.0,
    tail_pad_ms: float = 120.0,
    progress: Any = None,
) -> tuple[dict[str, Any], list[list[Any]], str]:
    if not session:
        raise gr.Error("Create or load a session first.")
    if ta is None:
        raise gr.Error("torchaudio is required to trim audio.")

    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    chunks = [c for c in session.get("chunks", []) if c.get("audio_path") and c["status"] != "excluded"]
    if not chunks:
        raise gr.Error("No audio chunks available to trim.")

    from .validator import check_audio_quality
    trimmed_count = 0
    for idx, chunk in enumerate(chunks, start=1):
        prog(idx / len(chunks), desc=f"Trimming dead air on chunk {idx}/{len(chunks)}...")
        p = chunk.get("audio_path")
        if p and Path(p).exists():
            wav, sr = ta.load(p)
            trimmed = trim_silence_boundary(wav, sr, threshold_db, head_pad_ms, tail_pad_ms)
            ta.save(p, trimmed, sr)
            check_audio_quality(
                session,
                chunk,
                silence_threshold_db=threshold_db,
                max_silence_ms=600.0,
                max_clip_fraction=0.001,
                min_duration_s=0.20,
                max_duration_s=120.0,
                min_rms_dbfs=-38.0,
            )
            trimmed_count += 1

    save_session(session)
    prog(1.0, desc=f"Trimmed {trimmed_count} chunk(s). All acoustic checks updated.")
    return (
        session,
        format_chunk_table(session),
        status_message(session, f"Auto-trimmed dead air on {trimmed_count} chunk(s). All acoustic checks updated."),
    )


def merge_chunks(
    session: dict[str, Any],
    output_filename: str,
    silence_ms: float,
    require_approved: bool,
    export_mp3: bool,
    mp3_bitrate: str,
    smart_trim: bool = True,
    progress: Any = None,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    if ta is None:
        raise gr.Error("torchaudio is required to merge chunks but is not available.")

    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    prog(0.05, desc="Checking session chunks for export...")

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
    for idx, chunk in enumerate(selected_chunks, start=1):
        prog(0.1 + 0.6 * (idx / len(selected_chunks)), desc=f"Loading audio {idx}/{len(selected_chunks)}...")
        wav, chunk_sr = ta.load(chunk["audio_path"])
        if sr is None:
            sr = chunk_sr
        elif chunk_sr != sr:
            raise gr.Error(f"Chunk {chunk['index']} sample rate {chunk_sr} does not match project sample rate {sr}.")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if smart_trim:
            wav = trim_silence_boundary(wav, sr)
        waves.append(wav)
        gap_samples = int((float(silence_ms or 0) / 1000.0) * sr)
        if gap_samples > 0:
            waves.append(torch.zeros((wav.shape[0], gap_samples), dtype=wav.dtype))

    # Remove the trailing silence gap if added
    if waves and float(silence_ms or 0) > 0:
        waves = waves[:-1]

    prog(0.75, desc="Concatenating and saving master WAV...")
    final_wav = torch.cat(waves, dim=1)
    final_path = session_dir(session) / "final" / filename
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(final_path), final_wav, sr)
    session["final_output_path"] = str(final_path)

    mp3_path = None
    if export_mp3:
        prog(0.85, desc=f"Exporting MP3 at {mp3_bitrate}...")
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
    prog(1.0, desc="Master narration export complete!")
    return (
        session,
        str(final_path),
        (sr, final_wav.squeeze(0).numpy()),
        str(mp3_path) if mp3_path else None,
        status_message(session, f"Finalized `{filename}`."),
    )
