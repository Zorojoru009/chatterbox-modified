import json
import uuid
from pathlib import Path
from typing import Any

from datetime import datetime

from .config import (
    BLUEPRINT_DEFAULT_SETTINGS,
    MODEL_CHOICES,
    MODEL_ORIGINAL,
    SESSION_ROOT,
    safe_name,
    safe_wav_filename,
)
from .session import generate_chronological_session_id


def normalize_blueprint_settings(
    settings: dict[str, Any] | None,
    model_name: str | None = None,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete generation settings dictionary.

    Per-chunk settings override `base` (typically the blueprint's default_settings),
    and any missing key falls back to `BLUEPRINT_DEFAULT_SETTINGS`.
    """
    normalized = dict(BLUEPRINT_DEFAULT_SETTINGS if base is None else base)
    if isinstance(settings, dict):
        for key in normalized:
            if key in settings and settings[key] is not None:
                normalized[key] = settings[key]

    try:
        normalized["temperature"] = float(normalized["temperature"])
    except (ValueError, TypeError):
        normalized["temperature"] = BLUEPRINT_DEFAULT_SETTINGS["temperature"]

    try:
        normalized["top_p"] = float(normalized["top_p"])
    except (ValueError, TypeError):
        normalized["top_p"] = BLUEPRINT_DEFAULT_SETTINGS["top_p"]

    try:
        normalized["top_k"] = int(normalized["top_k"])
    except (ValueError, TypeError):
        normalized["top_k"] = BLUEPRINT_DEFAULT_SETTINGS["top_k"]

    try:
        normalized["repetition_penalty"] = float(normalized["repetition_penalty"])
    except (ValueError, TypeError):
        normalized["repetition_penalty"] = BLUEPRINT_DEFAULT_SETTINGS["repetition_penalty"]

    try:
        normalized["min_p"] = float(normalized["min_p"])
    except (ValueError, TypeError):
        normalized["min_p"] = BLUEPRINT_DEFAULT_SETTINGS["min_p"]

    try:
        normalized["exaggeration"] = float(normalized["exaggeration"])
    except (ValueError, TypeError):
        normalized["exaggeration"] = BLUEPRINT_DEFAULT_SETTINGS["exaggeration"]

    try:
        normalized["cfg_weight"] = float(normalized["cfg_weight"])
    except (ValueError, TypeError):
        normalized["cfg_weight"] = BLUEPRINT_DEFAULT_SETTINGS["cfg_weight"]

    normalized["norm_loudness"] = bool(normalized.get("norm_loudness", True))
    try:
        normalized["seed_num"] = int(normalized.get("seed_num") or 0)
    except (ValueError, TypeError):
        normalized["seed_num"] = 0

    return normalized


def chunk_settings_for(
    session: dict[str, Any],
    chunk: dict[str, Any],
    ui_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve final generation settings for a single chunk.

    For blueprint-imported sessions, chunk-level `settings` override the blueprint's
    `default_settings`. For legacy/manual sessions, `ui_settings` are used.
    """
    ui_cfg = ui_settings or {}
    if session.get("from_blueprint"):
        model = chunk.get("model_name") or session.get("model_name") or MODEL_ORIGINAL
        base = session.get("generation_settings") or {}
        resolved = normalize_blueprint_settings(chunk.get("settings") or {}, model, base=base)
        resolved["enable_parallel"] = bool(ui_cfg.get("enable_parallel", True))
        resolved["max_parallel_devices"] = int(ui_cfg.get("max_parallel_devices") or 1)
        return resolved

    resolved = dict(ui_cfg)
    return resolved


def validate_narration_blueprint(blueprint: Any) -> str | None:
    """Return an error message if the blueprint is invalid, else None."""
    if not isinstance(blueprint, dict):
        return "Blueprint must be a JSON object."

    if blueprint.get("schema_version") not in (1, None):
        return f"Unsupported schema_version: {blueprint.get('schema_version')!r} (supported: 1)."

    chunks = blueprint.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return "Blueprint must contain a non-empty `chunks` array."

    default_model = blueprint.get("default_model", MODEL_ORIGINAL)
    if default_model not in MODEL_CHOICES:
        return f"`default_model` must be one of {MODEL_CHOICES}; got `{default_model}`."

    allow_mixed_models = bool(blueprint.get("allow_mixed_models", False))

    seen_ids = set()
    for index, entry in enumerate(chunks, start=1):
        if not isinstance(entry, dict):
            return f"chunks[{index}] must be a JSON object."

        chunk_text = str(entry.get("text") or "").strip()
        if not chunk_text:
            return f"chunks[{index}] is missing non-empty `text`."

        entry_model = entry.get("model") or default_model
        if entry_model not in MODEL_CHOICES:
            return f"chunks[{index}] `model` must be one of {MODEL_CHOICES}; got `{entry_model}`."

        if entry_model != default_model and not allow_mixed_models:
            return (
                f"chunks[{index}] uses `{entry_model}` but `default_model` is `{default_model}`. "
                "A blueprint must use a single model for every chunk — mixing models forces "
                "repeated GPU model loading/unloading and can cause VRAM thrashing. "
                "Omit `model` (or set it to the default) on every chunk, or set `allow_mixed_models: true`."
            )

        cid = entry.get("id")
        if cid:
            cid_str = str(cid)
            if cid_str in seen_ids:
                return f"Duplicate chunk ID `{cid_str}` found at chunk index {index}."
            seen_ids.add(cid_str)

    # Optional merge configuration validation
    if "merge" in blueprint and isinstance(blueprint["merge"], dict):
        merge_cfg = blueprint["merge"]
        if "silence_ms" in merge_cfg:
            try:
                val = float(merge_cfg["silence_ms"])
                if val < 0 or val > 10000:
                    return f"Invalid merge `silence_ms`: {val} (must be between 0 and 10000 ms)."
            except (ValueError, TypeError):
                return f"Invalid merge `silence_ms`: {merge_cfg['silence_ms']!r}"

        if "mp3_bitrate" in merge_cfg and str(merge_cfg["mp3_bitrate"]) not in ("128k", "192k", "256k", "320k"):
            return f"Invalid merge `mp3_bitrate`: {merge_cfg['mp3_bitrate']!r} (supported: 128k, 192k, 256k, 320k)."

    return None


def make_session_from_blueprint(blueprint: dict[str, Any]) -> dict[str, Any]:
    project = safe_name(blueprint.get("project_name"), fallback="youtube_narration")
    session_num, sid = generate_chronological_session_id(project)
    sdir = SESSION_ROOT / sid
    default_model = blueprint.get("default_model", MODEL_ORIGINAL)
    default_settings = normalize_blueprint_settings(blueprint.get("default_settings", {}), default_model)

    now_chunks = []
    for index, entry in enumerate(blueprint.get("chunks") or [], start=1):
        entry_model = entry.get("model") or default_model
        chunk_id = str(entry.get("id") or f"chunk{index:04d}")
        note = str(entry.get("note") or "")
        now_chunks.append(
            {
                "index": index,
                "id": chunk_id,
                "note": note,
                "text": str(entry.get("text") or "").strip(),
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
                "model_name": entry_model,
                "device": None,
                "settings": normalize_blueprint_settings(
                    entry.get("settings", {}), entry_model, base=default_settings
                ),
            }
        )

    session = {
        "session_number": session_num,
        "session_id": sid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_name": project,
        "output_filename": safe_wav_filename(blueprint.get("output_filename") or "narration.wav"),
        "model_name": default_model,
        "full_text": "\n\n".join(chunk["text"] for chunk in now_chunks),
        "reference_audio_path": blueprint.get("reference_audio_path") or None,
        "generation_settings": default_settings,
        "allow_mixed_models": bool(blueprint.get("allow_mixed_models", False)),
        "from_blueprint": True,
        "chunks": now_chunks,
        "final_output_path": None,
        "session_dir": str(sdir),
    }

    merge_cfg = blueprint.get("merge") if isinstance(blueprint.get("merge"), dict) else {}
    session["merge_settings"] = {
        "silence_ms": float(merge_cfg.get("silence_ms", 200)),
        "require_approved": bool(merge_cfg.get("require_approved", False)),
        "export_mp3": bool(merge_cfg.get("export_mp3", False)),
        "mp3_bitrate": str(merge_cfg.get("mp3_bitrate", "192k")),
    }

    (sdir / "chunks").mkdir(parents=True, exist_ok=True)
    (sdir / "reference").mkdir(parents=True, exist_ok=True)
    (sdir / "final").mkdir(parents=True, exist_ok=True)

    for chunk in session["chunks"]:
        text_path = sdir / "chunks" / f"{chunk['index']:04d}.txt"
        text_path.write_text(chunk["text"], encoding="utf-8")

    # Atomic save
    tmp_path = sdir / "session.tmp.json"
    final_path = sdir / "session.json"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    tmp_path.replace(final_path)

    return session


def export_blueprint_from_session(session: dict[str, Any]) -> dict[str, Any]:
    """Generate a blueprint JSON structure from an active session."""
    return {
        "schema_version": 1,
        "project_name": session.get("project_name", "narration_project"),
        "output_filename": session.get("output_filename", "narration.wav"),
        "default_model": session.get("model_name", MODEL_ORIGINAL),
        "reference_audio_path": session.get("reference_audio_path"),
        "default_settings": session.get("generation_settings", BLUEPRINT_DEFAULT_SETTINGS),
        "merge": session.get(
            "merge_settings",
            {
                "silence_ms": 200,
                "require_approved": False,
                "export_mp3": False,
                "mp3_bitrate": "192k",
            },
        ),
        "chunks": [
            {
                "id": chunk.get("id", f"chunk{chunk['index']:04d}"),
                "note": chunk.get("note", ""),
                "text": chunk.get("text", ""),
                "model": chunk.get("model_name"),
                "settings": chunk.get("settings"),
            }
            for chunk in session.get("chunks", [])
        ],
    }
