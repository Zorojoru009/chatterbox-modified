#!/usr/bin/env python3
"""Chatterbox - High-Throughput Headless Dual-GPU Batch Narrator.

Purpose-built for The Sovereign Mind narration production on Kaggle Dual-T4
accelerators (and local multi-GPU workstations).

Key Architectural Features:
1. True Process-Level Parallelism: Spawns independent worker processes across
   cuda:0 and cuda:1 using torch.multiprocessing (spawn method), completely
   bypassing the Python GIL for linear 2x throughput.
2. Zero-Redundancy Conditional Caching: Pre-computes reference voice embeddings
   and S3Gen prompt dict once per GPU at worker initialization, eliminating
   redundant librosa/VoiceEncoder passes on every chunk.
3. Sovereign Mind Blueprint Ingestion: Natively consumes narration_blueprint.json
   with per-chunk model selection, delivery settings (temperature, top_p,
   repetition_penalty, exaggeration, cfg_weight), and pause_after_ms.
4. Signal Health Guard: Automatic detection of digital silence, amplitude collapse,
   or clipping with automated seed re-rolling.
4b. Whisper Validation (--validate): each chunk is transcribed on its own GPU with the
   studio validator (same scoring as the Gradio app) and regenerated with a new seed
   until it passes or --regen-rounds runs out; the best take is kept and every take is
   logged in validation_report.json. --only-chunks reruns just some chunks (or "failed")
   and re-merges with the earlier chunk wavs.
5. Smart Boundary Trimming: Trims digital dead-air and applies smooth 5ms fades
   before inserting exact pause_after_ms silence intervals.
6. Clean 24kHz 16-bit PCM Export: Direct master WAV export ready for WhisperX
   alignment and GPU Video Engine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure local repository 'src' directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio as ta

from batch_report import load_report, plan_chunks, record_chunk, save_report, summarize
from chatterbox.studio.config import (
    MODEL_CHOICES,
    MODEL_NANO,
    MODEL_ORIGINAL,
    MODEL_TURBO,
    S3GEN_SR,
    safe_name,
    safe_wav_filename,
    set_seed,
)
from chatterbox.studio.blueprint import (
    normalize_blueprint_settings,
    validate_narration_blueprint,
)
from chatterbox.studio.finalizer import trim_silence_boundary
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS


def configure_worker_gpu(device_str: str) -> None:
    """Configures cuDNN and TensorCore optimizations for NVIDIA GPUs."""
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        try:
            device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
            torch.cuda.set_device(device_idx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as exc:
            print(f"[{device_str}] Note: GPU optimization flag configuration skipped: {exc}")


def check_signal_health(wav_tensor: torch.Tensor) -> tuple[bool, float, float]:
    """Inspects generated audio for digital clipping, collapse, or dead air."""
    if wav_tensor.numel() == 0:
        return False, 0.0, 0.0

    mono = wav_tensor.abs().max(dim=0).values.cpu().numpy()
    if np.isnan(mono).any() or np.isinf(mono).any():
        return False, float("nan"), 0.0

    max_amp = float(np.max(mono))
    std_amp = float(np.std(mono))

    # Health criteria:
    # 1. Not completely silent (max_amp > 0.01, std_amp > 0.005)
    # 2. No hard digital clipping (max_amp <= 0.999)
    is_healthy = (max_amp > 0.01) and (std_amp > 0.005) and (max_amp <= 0.999)
    return is_healthy, max_amp, std_amp


def _narrator_worker(
    device_str: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    ref_audio_path: str,
    ckpt_dir: str | None,
    apply_watermark: bool,
    validate: dict[str, Any] | None = None,
) -> None:
    """Dedicated worker process bound to a single GPU device.

    validate: None, or {"whisper_model", "threshold", "regen_rounds"} to Whisper-check every chunk
    on this worker's GPU and regenerate the chunks that fail.
    """
    configure_worker_gpu(device_str)

    try:
        print(f"[{device_str}] Loading {model_name}...")
        t0_load = time.perf_counter()

        if ckpt_dir and Path(ckpt_dir).exists():
            if model_name == MODEL_ORIGINAL:
                model = ChatterboxTTS.from_local(ckpt_dir, device_str)
            else:
                model = ChatterboxTurboTTS.from_local(ckpt_dir, device_str, nano=(model_name == MODEL_NANO))
        else:
            if model_name == MODEL_ORIGINAL:
                model = ChatterboxTTS.from_pretrained(device_str)
            elif model_name == MODEL_NANO:
                model = ChatterboxTurboTTS.from_pretrained(device_str, nano=True)
            else:
                model = ChatterboxTurboTTS.from_pretrained(device_str, nano=False)

        load_sec = time.perf_counter() - t0_load
        print(f"[{device_str}] {model_name} initialized in {load_sec:.2f}s.")

        # Pre-compute and cache reference voice conditionals ONCE
        if ref_audio_path and Path(ref_audio_path).exists():
            print(f"[{device_str}] Pre-computing voice conditionals from '{Path(ref_audio_path).name}'...")
            t0_cond = time.perf_counter()
            if model_name == MODEL_ORIGINAL:
                model.prepare_conditionals(ref_audio_path, exaggeration=0.5)
            else:
                model.prepare_conditionals(ref_audio_path, exaggeration=0.0, norm_loudness=True)
            cond_sec = time.perf_counter() - t0_cond
            print(f"[{device_str}] Reference voice cached in {cond_sec:.2f}s (0ms per-chunk prompt overhead).")
        else:
            print(f"[{device_str}] Warning: No reference audio provided; using model defaults.")

        if validate:
            from chatterbox.studio.validator import get_whisper_model, transcribe_audio, validation_comparison

            t0_whisper = time.perf_counter()
            get_whisper_model(validate["whisper_model"], device_str)  # load now, so a bad install fails early
            print(f"[{device_str}] Whisper {validate['whisper_model']} loaded in {time.perf_counter() - t0_whisper:.2f}s.")

    except Exception as exc:
        print(f"[{device_str}] FATAL: Worker initialization failed: {exc}", file=sys.stderr)
        result_queue.put(("WORKER_INIT_ERROR", None, None, 0.0, str(exc), device_str, None))
        return

    # Notify main process that worker is fully warm and ready
    result_queue.put(("WORKER_READY", None, None, 0.0, None, device_str, None))

    # Processing loop: the main process runs the phases (generate all, check all, regenerate the failures),
    # so a worker only ever does one of two things
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break

        if task[0] == "VAL":
            _, chunk_idx, chunk_id, text, take_path = task
            try:
                transcript = transcribe_audio(str(take_path), validate["whisper_model"], device_str)
                cmp = validation_comparison(text, transcript, float(validate["threshold"]))
                check = {
                    "passed": cmp["passed"],
                    "score": cmp["score"],
                    "category": cmp["diagnostic_category"],
                    "message": cmp["diagnostic_message"],
                    "transcript": transcript,
                    "missing_words": cmp["missing_words"][:10],
                    "extra_words": cmp["extra_words"][:10],
                    "number_errors": cmp["number_errors"],
                }
            except Exception as exc:
                check = {"passed": False, "score": 0.0, "category": "validator_error", "message": str(exc)}
            result_queue.put(("VAL_DONE", chunk_idx, chunk_id, 0.0, None, device_str, check))
            continue

        _, chunk_idx, chunk_id, text, settings, take_path, round_no = task
        # Seeds: seed_base for the first take (as without validation), then +101 per synthesis, so a
        # regeneration never repeats a take that failed. Up to 3 health-guard attempts per round.
        seed_base = int(settings.get("seed_num") or 0) + int(chunk_idx)
        last_error = ""
        wav = None
        for attempt in range(3):
            current_seed = seed_base + (round_no * 3 + attempt) * 101
            set_seed(current_seed)
            try:
                candidate = _synthesize(model, model_name, text, settings, apply_watermark)
            except Exception as exc:
                last_error = str(exc)
                print(f"[{device_str}] Generation error on chunk {chunk_id} (attempt {attempt + 1}): {exc}", file=sys.stderr)
                continue
            is_healthy, max_amp, std_amp = check_signal_health(candidate)
            if is_healthy:
                wav = candidate
                break
            print(
                f"[{device_str}] Health Guard: Chunk {chunk_id} attempt {attempt + 1} unhealthy "
                f"(max_amp={max_amp:.3f}, std={std_amp:.3f}). Retrying...",
                file=sys.stderr,
            )
        if wav is None:
            err_msg = last_error or "Signal health guard failed after max retries."
            result_queue.put(("GEN_FAILED", chunk_idx, chunk_id, 0.0, err_msg, device_str, None))
            continue
        ta.save(str(take_path), wav.cpu(), model.sr)
        duration_s = round(wav.shape[-1] / float(model.sr), 3)
        take = {"seed": current_seed, "path": str(take_path), "duration_s": duration_s}
        result_queue.put(("GEN_DONE", chunk_idx, chunk_id, duration_s, None, device_str, take))


def _synthesize(model, model_name: str, text: str, settings: dict[str, Any], apply_watermark: bool) -> torch.Tensor:
    with torch.inference_mode():
        if model_name == MODEL_ORIGINAL:
            return model.generate(
                text=text,
                audio_prompt_path=None,  # Uses cached conditionals
                repetition_penalty=float(settings.get("repetition_penalty", 1.2)),
                min_p=float(settings.get("min_p", 0.05)),
                top_p=float(settings.get("top_p", 1.0)),
                exaggeration=float(settings.get("exaggeration", 0.5)),
                cfg_weight=float(settings.get("cfg_weight", 0.5)),
                temperature=float(settings.get("temperature", 0.8)),
                apply_watermark=apply_watermark,
            )
        return model.generate(
            text=text,
            audio_prompt_path=None,
            repetition_penalty=float(settings.get("repetition_penalty", 1.2)),
            min_p=float(settings.get("min_p", 0.00)),
            top_p=float(settings.get("top_p", 0.95)),
            top_k=int(settings.get("top_k") or 1000),
            temperature=float(settings.get("temperature", 0.8)),
            norm_loudness=bool(settings.get("norm_loudness", True)),
            apply_watermark=apply_watermark,
        )


def _next_result(result_queue: mp.Queue, workers: list[mp.Process], devices: list[str], poll_s: float = 30.0):
    """result_queue.get() that cannot hang: a worker killed outright (CUDA abort, OOM kill, segfault) never
    reports back, so check every poll_s that every worker is still alive and fail loudly if one died."""
    import queue as _queue

    while True:
        try:
            return result_queue.get(timeout=poll_s)
        except _queue.Empty:
            dead = [f"{dev} (exit code {w.exitcode})" for w, dev in zip(workers, devices) if not w.is_alive()]
            if dead:
                for w in workers:
                    if w.is_alive():
                        w.terminate()
                raise RuntimeError(
                    f"STOP: worker on {', '.join(dead)} died without reporting; see its last lines above"
                )


def detect_devices(device_arg: str | None) -> list[str]:
    """Resolves target compute devices from argument or hardware detection."""
    if device_arg and device_arg.strip().lower() != "auto":
        devices = [d.strip() for d in device_arg.split(",") if d.strip()]
        return devices if devices else ["cuda:0" if torch.cuda.is_available() else "cpu"]

    if not torch.cuda.is_available():
        return ["cpu"]

    count = torch.cuda.device_count()
    if count >= 2:
        return ["cuda:0", "cuda:1"]
    return ["cuda:0"]


def merge_synthesized_chunks(
    chunk_manifest: list[dict[str, Any]],
    output_wav_path: Path,
    default_silence_ms: float = 200.0,
    smart_trim: bool = True,
    threshold_db: float = -45.0,
) -> tuple[int, float]:
    """Merges ordered chunk audio files with smart silence trimming and pause intervals."""
    waves: list[torch.Tensor] = []
    sr: int | None = None
    total_samples = 0

    num_chunks = len(chunk_manifest)
    for idx, item in enumerate(chunk_manifest, start=1):
        chunk_path = Path(item["audio_path"])
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing audio file for chunk {item['id']}: {chunk_path}")

        wav, chunk_sr = ta.load(str(chunk_path))
        if sr is None:
            sr = chunk_sr
        elif chunk_sr != sr:
            raise ValueError(f"Chunk sample rate mismatch: expected {sr}, got {chunk_sr} on {chunk_path}")

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Smart trim leading and trailing dead air
        if smart_trim:
            wav = trim_silence_boundary(wav, sr, threshold_db=threshold_db)

        waves.append(wav)
        total_samples += wav.shape[-1]

        # Insert pause gap between chunks (never after the final chunk)
        if idx < num_chunks:
            pause_ms = item.get("pause_after_ms")
            if pause_ms is None:
                pause_ms = default_silence_ms
            gap_samples = int((float(pause_ms or 0) / 1000.0) * sr)
            if gap_samples > 0:
                silence = torch.zeros((wav.shape[0], gap_samples), dtype=wav.dtype)
                waves.append(silence)
                total_samples += gap_samples

    # Concatenate all segments
    master_wav = torch.cat(waves, dim=-1)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_wav_path), master_wav, sr)

    total_duration_s = total_samples / float(sr)
    return sr, total_duration_s


def run_batch_narrator(
    blueprint_path: str | Path,
    ref_audio_path: str | Path | None = None,
    output_path: str | Path | None = None,
    device_arg: str | None = None,
    ckpt_dir: str | Path | None = None,
    smart_trim: bool = True,
    apply_watermark: bool = False,
    test_limit: int | None = None,
    validate: dict[str, Any] | None = None,
    only_chunks: str | None = None,
    report_path: str | Path | None = None,
) -> tuple[Path, list[str]]:
    """Main orchestrator for parallel batch narration. Returns the master path and the ids of chunks
    that still fail validation (always empty without --validate)."""
    bp_path = Path(blueprint_path).resolve()
    if not bp_path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {bp_path}")

    with open(bp_path, "r", encoding="utf-8") as f:
        blueprint = json.load(f)

    # Validate blueprint schema
    validation_err = validate_narration_blueprint(blueprint)
    if validation_err:
        raise ValueError(f"Invalid blueprint: {validation_err}")

    # Determine reference audio path
    resolved_ref_audio = ""
    if ref_audio_path and Path(ref_audio_path).exists():
        resolved_ref_audio = str(Path(ref_audio_path).resolve())
    elif blueprint.get("reference_audio_path") and Path(blueprint["reference_audio_path"]).exists():
        resolved_ref_audio = str(Path(blueprint["reference_audio_path"]).resolve())
    else:
        # Search workspace or relative directories for standard reference audio
        standard_ref = bp_path.parent / "Authority_voice.mp3"
        if standard_ref.exists():
            resolved_ref_audio = str(standard_ref)

    project_name = safe_name(blueprint.get("project_name") or bp_path.stem)
    default_model = blueprint.get("default_model", MODEL_ORIGINAL)
    default_settings = normalize_blueprint_settings(blueprint.get("default_settings", {}), default_model)

    # Determine output directories
    if output_path:
        final_master_path = Path(output_path).resolve()
        work_dir = final_master_path.parent
    else:
        work_dir = bp_path.parent
        out_filename = safe_wav_filename(blueprint.get("output_filename") or f"narration_{project_name}.wav")
        final_master_path = work_dir / out_filename

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path) if report_path else final_master_path.with_name("validation_report.json")
    report = load_report(report_path) if only_chunks else {"chunks": {}}

    # Extract chunks
    raw_chunks = blueprint.get("chunks", [])
    if test_limit and test_limit > 0:
        raw_chunks = raw_chunks[:test_limit]
        print(f"⚠️  TEST LIMIT APPLIED: Generating first {test_limit} chunk(s) only.")

    num_chunks = len(raw_chunks)
    devices = detect_devices(device_arg)

    print("=" * 70)
    print(f"🎙️  CHATTERBOX HIGH-THROUGHPUT NARRATION ENGINE")
    print(f"• Project:           {project_name}")
    print(f"• Blueprint:         {bp_path.name} ({num_chunks} chunks)")
    print(f"• Model:             {default_model}")
    print(f"• Reference Audio:   {Path(resolved_ref_audio).name if resolved_ref_audio else 'None (Default)'}")
    print(f"• Active Devices:    {', '.join(devices)} ({len(devices)} parallel worker processes)")
    print(f"• Watermark:         {'Enabled' if apply_watermark else 'Bypassed (0% vocoder loss, max speed)'}")
    if validate:
        print(
            f"• Whisper Check:     {validate['whisper_model']} on each worker's GPU, pass >= {validate['threshold']}, "
            f"up to {validate['regen_rounds']} regeneration(s) per chunk"
        )
    else:
        print("• Whisper Check:     Off")
    print(f"• Output Master:     {final_master_path}")
    print("=" * 70)

    # Prepare chunk manifest and tasks
    tasks = []
    chunk_manifest = []

    for index, entry in enumerate(raw_chunks, start=1):
        chunk_id = str(entry.get("id") or f"b{index:03d}")
        text = str(entry.get("text", "")).strip()
        pause_ms = entry.get("pause_after_ms")
        if pause_ms is None and isinstance(blueprint.get("merge"), dict):
            pause_ms = blueprint["merge"].get("silence_ms", 200.0)

        chunk_settings = normalize_blueprint_settings(
            entry.get("settings", {}), default_model, base=default_settings
        )

        out_chunk_path = chunks_dir / f"{index:04d}_{chunk_id}.wav"
        tasks.append((index, chunk_id, text, chunk_settings, out_chunk_path))

        chunk_manifest.append(
            {
                "index": index,
                "id": chunk_id,
                "text": text,
                "pause_after_ms": pause_ms,
                "audio_path": str(out_chunk_path),
            }
        )

    selected = plan_chunks(chunk_manifest, only_chunks, report)
    tasks = [t for t in tasks if t[0] in selected]
    manifest_by_index = {item["index"]: item for item in chunk_manifest}
    if only_chunks:
        print(f"• Partial Run:       {len(tasks)} of {num_chunks} chunk(s): {', '.join(str(t[1]) for t in tasks)}")
    num_jobs = len(tasks)

    # Start multi-process worker pool using torch.multiprocessing (spawn method)
    mp.set_start_method("spawn", force=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = []
    for dev in devices:
        p = mp.Process(
            target=_narrator_worker,
            args=(
                dev,
                task_queue,
                result_queue,
                default_model,
                resolved_ref_audio,
                str(ckpt_dir) if ckpt_dir else None,
                apply_watermark,
                validate,
            ),
        )
        p.start()
        workers.append(p)

    # Wait for all workers to report ready
    ready_workers = 0
    while ready_workers < len(devices):
        msg = _next_result(result_queue, workers, devices)
        event_type = msg[0]
        if event_type == "WORKER_READY":
            ready_workers += 1
        elif event_type == "WORKER_INIT_ERROR":
            _, _, _, _, err, dev, _ = msg
            print(f"❌ Worker on {dev} failed to initialize: {err}", file=sys.stderr)
            for w in workers:
                w.terminate()
            raise RuntimeError(f"Worker on {dev} initialization failed: {err}")

    print(f"⚡ All {len(devices)} GPU worker processes online. Dispatching {num_jobs} chunks...")
    t_start = time.perf_counter()

    def run_phase(phase_tasks: list[tuple], label: str) -> list[tuple]:
        """Queue one phase's tasks on both GPUs and wait for all of them."""
        for t in phase_tasks:
            task_queue.put(t)
        results, t0 = [], time.perf_counter()
        for n in range(1, len(phase_tasks) + 1):
            msg = _next_result(result_queue, workers, devices)
            results.append(msg)
            if n % 10 == 0 or n == len(phase_tasks):
                elapsed = time.perf_counter() - t0
                print(f"[{label}] {n}/{len(phase_tasks)} | Elapsed: {elapsed:.1f}s | "
                      f"ETA: {elapsed / n * (len(phase_tasks) - n):.1f}s")
        return results

    def take_path(index: int, round_no: int) -> Path:
        out = Path(manifest_by_index[index]["audio_path"])
        return out.with_name(f"{out.stem}.take{round_no}.wav")

    # Like the Gradio studio: generate every chunk first, then Whisper-check them all, then regenerate only the
    # failures (and check those), round after round. Here the checks run on both GPUs, not one.
    takes: dict[int, list[dict[str, Any]]] = {t[0]: [] for t in tasks}
    gen_errors: dict[int, str] = {}
    task_by_index = {t[0]: t for t in tasks}
    pending = [t[0] for t in tasks]
    rounds = 1 + (int(validate["regen_rounds"]) if validate else 0)

    for round_no in range(rounds):
        label = "Generate" if round_no == 0 else f"Regenerate {round_no}/{rounds - 1}"
        print(f"\n🎙️  {label}: {len(pending)} chunk(s)")
        gen = run_phase(
            [("GEN", i, task_by_index[i][1], task_by_index[i][2], task_by_index[i][3], take_path(i, round_no), round_no)
             for i in pending],
            label,
        )
        made = []
        for event_type, chunk_idx, chunk_id, _, err, dev, take in gen:
            if event_type == "GEN_DONE":
                take["passed"], take["score"] = (False, None) if validate else (True, None)
                takes[chunk_idx].append(take)
                made.append(chunk_idx)
            else:
                print(f"[{dev}] ❌ Chunk {chunk_id} FAILED: {err}", file=sys.stderr)
                gen_errors[chunk_idx] = err
        if not validate:
            break

        print(f"\n🔍 Whisper check: {len(made)} chunk(s)")
        checks = run_phase(
            [("VAL", i, task_by_index[i][1], task_by_index[i][2], takes[i][-1]["path"]) for i in made],
            f"Whisper {round_no + 1}",
        )
        for _, chunk_idx, chunk_id, _, _, dev, check in checks:
            takes[chunk_idx][-1].update(check)
            if not check["passed"]:
                print(f"[{dev}] Whisper: chunk {chunk_id} take {len(takes[chunk_idx])} scored {check['score']:.3f} "
                      f"({check['category']}: {check['message']})", file=sys.stderr)

        next_pending = []
        for i in pending:
            history = takes[i]
            if history and history[-1]["passed"]:
                continue
            # The same transcript twice means the text or the validator is the problem, not the take: another
            # seed cannot fix it (b034/b042, 2026-10-30: 4 identical takes), so stop and let the report flag it
            if len(history) > 1 and history[-1].get("transcript") and history[-1].get("transcript") == history[-2].get("transcript"):
                history[-1]["message"] = f"{history[-1]['message']} (same transcript as the previous take; retries stopped)"
                print(f"Whisper: chunk {task_by_index[i][1]} gave the same transcript twice; not retrying", file=sys.stderr)
                continue
            next_pending.append(i)
        passed_now = sum(1 for i in takes if takes[i] and takes[i][-1]["passed"])
        print(f"✔️  {passed_now}/{len(takes)} chunk(s) pass after round {round_no + 1}")
        pending = next_pending
        if not pending:
            break

    # Keep a passing take, else the best-scoring one; the report shows every take
    failed_chunks = []
    total_spoken_seconds = 0.0
    for i, history in takes.items():
        item = manifest_by_index[i]
        if not history:
            failed_chunks.append((i, item["id"], gen_errors.get(i, "no take generated")))
            continue
        best = max(history, key=lambda t: (bool(t["passed"]), t["score"] or 0.0))
        for t in history:
            if t is best:
                Path(t["path"]).replace(item["audio_path"])
            else:
                Path(t["path"]).unlink(missing_ok=True)
        total_spoken_seconds += best["duration_s"]
        record_chunk(report, item, {
            "passed": best["passed"],
            "score": best["score"],
            "seed": best["seed"],
            "category": best.get("category"),
            "message": best.get("message"),
            "transcript": best.get("transcript"),
            "attempts": [{k: v for k, v in t.items() if k != "path"} for t in history],
        })

    # Send poison pill to workers
    for _ in devices:
        task_queue.put(None)

    for w in workers:
        w.join(timeout=10)

    total_wall_sec = time.perf_counter() - t_start
    report.update(
        blueprint=bp_path.name,
        whisper_model=validate["whisper_model"] if validate else None,
        threshold=validate["threshold"] if validate else None,
        regen_rounds=validate["regen_rounds"] if validate else None,
        updated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    save_report(report, report_path)

    if failed_chunks:
        print(f"\n⚠️  {len(failed_chunks)} chunk(s) failed generation:", file=sys.stderr)
        for idx, cid, err in failed_chunks:
            print(f"  • Chunk {cid} (Index {idx}): {err}", file=sys.stderr)
        raise RuntimeError(f"Batch generation completed with {len(failed_chunks)} failure(s).")

    print("\n🔗 Merging chunks into master narration track...")
    default_silence = 200.0
    if isinstance(blueprint.get("merge"), dict):
        default_silence = float(blueprint["merge"].get("silence_ms", 200.0))

    sample_rate, final_duration_s = merge_synthesized_chunks(
        chunk_manifest=chunk_manifest,
        output_wav_path=final_master_path,
        default_silence_ms=default_silence,
        smart_trim=smart_trim,
    )

    rtf = total_wall_sec / final_duration_s if final_duration_s > 0 else 0.0
    speedup = final_duration_s / total_wall_sec if total_wall_sec > 0 else 0.0

    print("=" * 70)
    print("🏆 BATCH NARRATION COMPLETE")
    print(f"• Master Output:     {final_master_path}")
    print(f"• Sample Rate:       {sample_rate} Hz (16-bit PCM)")
    print(f"• Spoken Duration:   {final_duration_s / 60.0:.2f} min ({final_duration_s:.1f} seconds)")
    print(f"• Wall-Clock Time:   {total_wall_sec / 60.0:.2f} min ({total_wall_sec:.1f} seconds)")
    print(f"• Throughput Factor: {speedup:.2f}x Real-Time (RTF: {rtf:.3f})")
    summary = summarize(report)
    if validate:
        print(
            f"• Whisper Check:     {summary['passed']}/{summary['total']} passed, "
            f"{summary['regenerated']} regenerated, report {report_path.name}"
        )
        if summary["failed"]:
            print(f"• STILL FAILING:     {', '.join(summary['failed'])} (best take merged; listen or rerun them)")
    print("=" * 70)

    return final_master_path, summary["failed"] if validate else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-Throughput Headless Dual-GPU Batch Narrator for Chatterbox."
    )
    parser.add_argument(
        "--blueprint",
        "-b",
        type=str,
        required=True,
        help="Path to narration_blueprint.json file.",
    )
    parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        default=None,
        help="Path to reference audio file (Authority_voice.mp3 / wav).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination path for the final merged master WAV file.",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        default="auto",
        help="Comma-separated devices (e.g. 'cuda:0,cuda:1', 'cuda:0', or 'auto').",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional local checkpoint directory containing model weights.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable smart dead-air silence trimming on chunk boundaries.",
    )
    parser.add_argument(
        "--apply-watermark",
        action="store_true",
        help="Enable Perth CPU watermarking (disabled by default for maximum speed).",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit execution to the first N chunks (for testing and calibration).",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whisper-check every chunk on its worker's GPU and regenerate the ones that fail.",
    )
    parser.add_argument("--whisper-model", type=str, default="large-v3-turbo", help="faster-whisper model for --validate.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Pass score for --validate (Gradio default 0.90).")
    parser.add_argument(
        "--regen-rounds",
        type=int,
        default=3,
        help="Extra takes (new seed each) for a chunk that fails --validate; the best take is kept.",
    )
    parser.add_argument(
        "--only-chunks",
        type=str,
        default=None,
        help="Regenerate only these chunk ids (comma-separated) or 'failed' (from the report), reuse the other "
        "chunk wavs in <output dir>/chunks, and re-merge.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="validation_report.json path (default: next to the output). Read by --only-chunks, then updated.",
    )

    args = parser.parse_args()
    validate = (
        {"whisper_model": args.whisper_model, "threshold": args.threshold, "regen_rounds": args.regen_rounds}
        if args.validate
        else None
    )

    _, still_failing = run_batch_narrator(
        blueprint_path=args.blueprint,
        ref_audio_path=args.ref_audio,
        output_path=args.output,
        device_arg=args.devices,
        ckpt_dir=args.ckpt_dir,
        smart_trim=not args.no_trim,
        apply_watermark=args.apply_watermark,
        test_limit=args.test_limit,
        validate=validate,
        only_chunks=args.only_chunks,
        report_path=args.report,
    )
    if still_failing:
        sys.exit(3)  # merged with the best takes, but some chunks never passed Whisper


if __name__ == "__main__":
    main()
