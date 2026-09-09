import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except Exception:
    ChatterboxTTS = None
    ChatterboxTurboTTS = None

from .blueprint import chunk_settings_for
from .config import (
    MODEL_NANO,
    MODEL_ORIGINAL,
    MODEL_TURBO,
    safe_wav_filename,
    set_seed,
)
from .session import (
    copy_reference_to_session,
    format_chunk_table,
    get_chunk,
    save_session,
    session_dir,
    status_message,
)

MODEL_ADAPTERS: dict[str, "ModelAdapter"] = {}
MODEL_ADAPTERS_LOCK = threading.Lock()


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


def gpu_status_text(enable_parallel: bool, max_parallel_devices: int) -> str:
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    if torch.cuda.is_available():
        return f"Detected {torch.cuda.device_count()} CUDA device(s). Batch generation will use: `{', '.join(devices)}`."
    return "CUDA is not available. Batch generation will use CPU."


def get_model_adapter(
    model_cache: dict[str, Any] | None,
    model_name: str,
    device: str | None = None,
) -> tuple[dict[str, Any], ModelAdapter]:
    cache = model_cache or {}
    selected_device = device or default_device()
    cache_key = f"{model_name}:{selected_device}"
    with MODEL_ADAPTERS_LOCK:
        if cache_key not in MODEL_ADAPTERS:
            MODEL_ADAPTERS[cache_key] = ModelAdapter(model_name, selected_device)
    return cache, MODEL_ADAPTERS[cache_key]


def clear_model_cache(model_cache: dict[str, Any] | None):
    with MODEL_ADAPTERS_LOCK:
        MODEL_ADAPTERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {}, "Cleared cached TTS models and released available GPU memory.", "No TTS models currently loaded."


def warm_model_cache(model_cache: dict[str, Any] | None, model_name: str, enable_parallel: bool, max_parallel_devices: int):
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    current_cache = model_cache or {}
    yield current_cache, f"Preparing to load `{model_name}` on `{', '.join(devices)}`...", "Model loading has started."
    for device_index, device in enumerate(devices, start=1):
        yield current_cache, f"Loading `{model_name}` on `{device}` ({device_index}/{len(devices)})...", f"Loading `{model_name}` on `{device}`..."
        current_cache, _adapter = get_model_adapter(current_cache, model_name, device)
        loaded = ", ".join(devices[:device_index])
        yield current_cache, f"Loaded `{model_name}` on `{loaded}`.", f"Loaded `{model_name}` on `{device}` ({device_index}/{len(devices)})."


def ensure_model_consistency(session: dict[str, Any], model_name: str):
    if session.get("allow_mixed_models"):
        return
    existing_models = {
        chunk.get("model_name")
        for chunk in session.get("chunks", [])
        if chunk.get("audio_path") and chunk.get("model_name")
    }
    if existing_models and existing_models != {model_name}:
        names = ", ".join(sorted(existing_models))
        raise gr.Error(
            f"This session already contains audio generated with `{names}`. "
            f"It cannot be mixed with `{model_name}`. Use the same model or create a new session."
        )


def collect_generation_settings(
    temperature: float,
    seed_num: int,
    min_p: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    exaggeration: float,
    cfg_weight: float,
    norm_loudness: bool,
) -> dict[str, Any]:
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


def save_chunk_audio(
    session: dict[str, Any],
    chunk: dict[str, Any],
    wav: torch.Tensor,
    sr: int,
    model_name: str,
    device: str,
) -> str:
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


def generate_selected_chunk(
    session: dict[str, Any],
    model_cache: dict[str, Any] | None,
    chunk_number: int,
    model_name: str,
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
    validation_details_fn: Any = None,
    progress: Any = None,
):
    if not session:
        raise gr.Error("Create or load a session first.")

    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    chunk = get_chunk(session, int(chunk_number or 1))
    target_model = chunk.get("model_name") or model_name
    ensure_model_consistency(session, target_model)

    prog(0.05, desc=f"Loading model {target_model}...")
    ui_settings = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    resolved_settings = chunk_settings_for(session, chunk, ui_settings)

    reference_audio_path = copy_reference_to_session(session, reference_audio_path)
    model_cache, adapter = get_model_adapter(model_cache, target_model)

    chunk["status"] = "generating"
    chunk["error"] = None
    save_session(session)

    prog(0.35, desc=f"Synthesizing chunk {chunk.get('id') or chunk['index']}...")
    try:
        wav, sr, device = generate_chunk_wav(
            adapter,
            chunk["text"],
            chunk["index"],
            reference_audio_path,
            resolved_settings,
        )
        prog(0.85, desc="Saving audio file...")
        audio_path = save_chunk_audio(session, chunk, wav, sr, adapter.model_name, device)
        save_session(session)
        prog(1.0, desc=f"Chunk {chunk.get('id') or chunk['index']} generated!")
    except Exception as exc:
        chunk["status"] = "failed"
        chunk["error"] = str(exc)
        save_session(session)
        raise

    details_str = validation_details_fn(chunk) if validation_details_fn else "Not validated yet."
    return (
        session,
        model_cache,
        format_chunk_table(session),
        chunk.get("audio_path"),
        chunk.get("transcript") or "",
        details_str,
        status_message(session, f"Generated chunk {chunk.get('id') or chunk['index']}."),
    )


def generate_all_chunks(
    session: dict[str, Any],
    model_cache: dict[str, Any] | None,
    model_name: str,
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
    enable_parallel: bool,
    max_parallel_devices: int,
    validation_enabled: bool = False,
    auto_regenerate: bool = False,
    whisper_model_name: str = "small.en",
    whisper_device: str = "cuda:0",
    validation_threshold: float = 0.90,
    skip_existing: bool = True,
    progress: Any = None,
    auto_validate_fn: Any = None,
):
    if not session:
        if gr is not None:
            raise gr.Error("Create or load a session first.")
        raise ValueError("Create or load a session first.")

    prog = progress if callable(progress) else (lambda *args, **kwargs: None)

    target_model = session.get("model_name") or model_name
    ensure_model_consistency(session, target_model)
    session["model_name"] = target_model

    ui_settings = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    ui_settings["enable_parallel"] = bool(enable_parallel)
    ui_settings["max_parallel_devices"] = int(max_parallel_devices or 1)
    if not session.get("from_blueprint"):
        session["generation_settings"] = ui_settings

    reference_audio_path = copy_reference_to_session(session, reference_audio_path)

    if skip_existing:
        chunks_to_generate = [
            chunk for chunk in session.get("chunks", [])
            if chunk["status"] != "excluded" and (not chunk.get("audio_path") or chunk["status"] in ("pending", "failed"))
        ]
        if not chunks_to_generate:
            return (
                session,
                model_cache or {},
                format_chunk_table(session),
                None,
                status_message(session, "All chunks are already generated. Disable 'Skip already generated' to re-synthesize all chunks."),
            )
    else:
        chunks_to_generate = [chunk for chunk in session.get("chunks", []) if chunk["status"] != "excluded"]
        if not chunks_to_generate:
            return (
                session,
                model_cache or {},
                format_chunk_table(session),
                None,
                status_message(session, "No non-excluded chunks to generate."),
            )

    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    last_audio = None
    for chunk in chunks_to_generate:
        chunk["status"] = "generating"
        chunk["error"] = None
    save_session(session)

    # Single-device execution
    if len(devices) == 1:
        prog(0, desc=f"Loading {target_model} on {devices[0]}...")
        model_cache, adapter = get_model_adapter(model_cache, target_model, devices[0])
        prog(0.05, desc=f"Loaded {target_model} on {devices[0]}; generating chunks...")

        first_error = None
        for chunk_index, chunk in enumerate(chunks_to_generate, start=1):
            try:
                resolved_settings = chunk_settings_for(session, chunk, ui_settings)
                wav, sr, device = generate_chunk_wav(
                    adapter, chunk["text"], chunk["index"], reference_audio_path, resolved_settings
                )
                last_audio = save_chunk_audio(session, chunk, wav, sr, adapter.model_name, device)
                save_session(session)
                prog(chunk_index / len(chunks_to_generate), desc=f"Generated chunk {chunk_index}/{len(chunks_to_generate)}")
            except Exception as exc:
                first_error = exc
                chunk["status"] = "failed"
                chunk["error"] = str(exc)
                save_session(session)
                break

        if first_error is not None:
            return (
                session,
                model_cache,
                format_chunk_table(session),
                last_audio,
                status_message(session, f"Batch generation stopped after error: {first_error}"),
            )

        auto_status = ""
        if auto_validate_fn and validation_enabled and auto_regenerate:
            session, model_cache, auto_status = auto_validate_fn(
                session=session,
                model_cache=model_cache,
                model_name=target_model,
                reference_audio_path=reference_audio_path,
                temperature=temperature,
                seed_num=seed_num,
                min_p=min_p,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                norm_loudness=norm_loudness,
                enable_parallel=enable_parallel,
                max_parallel_devices=max_parallel_devices,
                validation_enabled=validation_enabled,
                auto_regenerate=auto_regenerate,
                whisper_model_name=whisper_model_name,
                whisper_device=whisper_device,
                validation_threshold=validation_threshold,
            )

        return (
            session,
            model_cache,
            format_chunk_table(session),
            last_audio,
            status_message(session, f"Batch generation finished on {devices[0]}. {auto_status}".strip()),
        )

    # Multi-device parallel execution
    adapters: list[ModelAdapter] = []
    prog(0, desc=f"Loading {target_model} on {', '.join(devices)}...")
    for device in devices:
        model_cache, adapter = get_model_adapter(model_cache, target_model, device)
        adapters.append(adapter)
        prog(len(adapters) / len(devices) * 0.1, desc=f"Loaded {target_model} on {device} ({len(adapters)}/{len(devices)})")

    future_to_chunk = {}
    max_workers = len(adapters)
    first_error = None
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job_index, chunk in enumerate(chunks_to_generate):
            adapter = adapters[job_index % len(adapters)]
            resolved_settings = chunk_settings_for(session, chunk, ui_settings)
            future = executor.submit(
                generate_chunk_wav,
                adapter,
                chunk["text"],
                chunk["index"],
                reference_audio_path,
                resolved_settings,
            )
            future_to_chunk[future] = chunk

        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            completed += 1
            try:
                wav, sr, device = future.result()
                last_audio = save_chunk_audio(session, chunk, wav, sr, target_model, device)
                save_session(session)
                prog(completed / len(chunks_to_generate), desc=f"Generated {completed}/{len(chunks_to_generate)} chunks")
            except Exception as exc:
                if first_error is None:
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

    auto_status = ""
    if auto_validate_fn and validation_enabled and auto_regenerate:
        session, model_cache, auto_status = auto_validate_fn(
            session=session,
            model_cache=model_cache,
            model_name=target_model,
            reference_audio_path=reference_audio_path,
            temperature=temperature,
            seed_num=seed_num,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            norm_loudness=norm_loudness,
            enable_parallel=enable_parallel,
            max_parallel_devices=max_parallel_devices,
            validation_enabled=validation_enabled,
            auto_regenerate=auto_regenerate,
            whisper_model_name=whisper_model_name,
            whisper_device=whisper_device,
            validation_threshold=validation_threshold,
        )

    return (
        session,
        model_cache,
        format_chunk_table(session),
        last_audio,
        status_message(
            session,
            f"Parallel batch generation finished across {len(devices)} device(s): {', '.join(devices)}. {auto_status}".strip(),
        ),
    )


def regenerate_failed_chunks(
    session: dict[str, Any],
    model_cache: dict[str, Any] | None,
    model_name: str,
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
    enable_parallel: bool,
    max_parallel_devices: int,
    validation_threshold: float,
    whisper_model_name: str,
    whisper_device: str,
    validation_enabled: bool,
    validate_chunk_fn: Any = None,
    progress: Any = None,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    if not validation_enabled:
        raise gr.Error("Enable Whisper validation first, or continue without validation.")

    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    target_model = session.get("model_name") or model_name
    ensure_model_consistency(session, target_model)

    failed = [
        chunk for chunk in session.get("chunks", [])
        if chunk.get("validation_status") == "needs_review" and chunk["status"] != "excluded"
    ]
    if not failed:
        raise gr.Error("No chunks currently need review.")

    prog(0.05, desc=f"Regenerating {len(failed)} chunk(s) needing review...")
    session["model_name"] = target_model
    ui_settings = collect_generation_settings(
        temperature, seed_num, min_p, top_p, top_k, repetition_penalty, exaggeration, cfg_weight, norm_loudness
    )
    reference_audio_path = copy_reference_to_session(session, reference_audio_path)
    devices = available_generation_devices(bool(enable_parallel), int(max_parallel_devices or 1))
    model_cache, adapter = get_model_adapter(model_cache, target_model, devices[0])

    for idx, chunk in enumerate(failed, start=1):
        prog(idx / len(failed), desc=f"Regenerating chunk {chunk.get('id') or chunk['index']} ({idx}/{len(failed)})...")
        chunk["status"] = "generating"
        chunk["validation_status"] = None
        try:
            resolved_settings = chunk_settings_for(session, chunk, ui_settings)
            wav, sr, device = generate_chunk_wav(
                adapter, chunk["text"], chunk["index"], reference_audio_path, resolved_settings
            )
            save_chunk_audio(session, chunk, wav, sr, adapter.model_name, device)
            if validate_chunk_fn:
                validate_chunk_fn(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
            save_session(session)
        except Exception as exc:
            chunk["status"] = "failed"
            chunk["error"] = str(exc)
            save_session(session)

    prog(1.0, desc=f"Finished regenerating {len(failed)} chunk(s).")
    return (
        session,
        model_cache,
        format_chunk_table(session),
        status_message(session, f"Regenerated {len(failed)} chunk(s) needing review."),
    )
