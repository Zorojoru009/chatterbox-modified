import json
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

from .blueprint import (
    export_blueprint_from_session,
    make_session_from_blueprint,
    validate_narration_blueprint,
)
from .config import (
    CUSTOM_CSS,
    DEFAULT_REFERENCE_AUDIO,
    EVENT_TAGS,
    INSERT_TAG_JS,
    MODEL_CHOICES,
    MODEL_ORIGINAL,
    PRESET_CHOICES,
    safe_wav_filename,
)
from .engine import (
    available_generation_devices,
    clear_model_cache,
    collect_generation_settings,
    default_device,
    ensure_model_consistency,
    generate_all_chunks,
    generate_selected_chunk,
    get_model_adapter,
    gpu_status_text,
    regenerate_failed_chunks,
    warm_model_cache,
)
from .finalizer import merge_chunks
from .session import (
    chunk_script,
    copy_reference_to_session,
    format_chunk_table,
    get_chunk,
    list_session_choices,
    list_session_paths,
    load_session_from_path,
    make_session,
    save_session,
    session_dir,
    status_message,
)
from .validator import (
    build_full_report_markdown,
    build_validation_overview,
    check_audio_all,
    check_audio_selected,
    export_validation_report,
    validate_all_chunks,
    validate_chunk,
    validate_selected_chunk,
    validation_details,
)


def format_blueprint_summary(session: dict[str, Any] | None) -> str:
    if not session:
        return "*(No blueprint loaded)*"
    chunks = session.get("chunks", [])
    total_chars = sum(len(c.get("text", "")) for c in chunks)
    return (
        f"**Project:** `{session.get('project_name')}` | "
        f"**Model:** `{session.get('model_name')}` | "
        f"**Chunks:** `{len(chunks)}` | "
        f"**Total Chars:** `{total_chars:,}` | "
        f"**Target WAV:** `{session.get('output_filename')}`"
    )


def import_narration_blueprint(file_path: str):
    if not file_path:
        raise gr.Error("Upload a narration blueprint JSON file first.")
    path = Path(file_path)
    if not path.exists():
        raise gr.Error(f"Blueprint file not found: {file_path}")
    try:
        blueprint = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Blueprint is not valid JSON: {exc}") from exc

    problem = validate_narration_blueprint(blueprint)
    if problem:
        raise gr.Error(f"Invalid narration blueprint: {problem}")

    session = make_session_from_blueprint(blueprint)
    merge = session.get("merge_settings", {})
    first_text = session["chunks"][0]["text"] if session.get("chunks") else ""
    summary = format_blueprint_summary(session)

    return (
        session,
        format_chunk_table(session),
        1,
        first_text,
        session.get("project_name", ""),
        session.get("output_filename", "narration.wav"),
        session.get("model_name", MODEL_ORIGINAL),
        merge.get("silence_ms", 200),
        merge.get("require_approved", False),
        merge.get("export_mp3", False),
        merge.get("mp3_bitrate", "192k"),
        summary,
        status_message(session, f"Imported blueprint `{path.name}` ({len(session['chunks'])} chunks)."),
    )


def load_session(path_value: str, current_ref_audio: str | None = None):
    if not path_value:
        raise gr.Error("Enter a session directory or session.json path.")
    session = load_session_from_path(path_value)
    first_chunk = session["chunks"][0] if session.get("chunks") else {}
    first_text = first_chunk.get("text", "")
    first_audio = first_chunk.get("audio_path")
    first_transcript = first_chunk.get("transcript") or ""
    first_validation = validation_details(first_chunk) if first_chunk else "No chunk selected."
    merge = session.get("merge_settings", {})
    summary = format_blueprint_summary(session)

    # Restore reference audio from session if saved, otherwise keep current
    ref_audio = session.get("reference_audio_path") or current_ref_audio
    final_wav = session.get("final_output_path")
    final_mp3 = session.get("final_mp3_path")

    return (
        session,
        format_chunk_table(session),
        1,
        first_text,
        first_audio,
        first_transcript,
        first_validation,
        session.get("project_name", ""),
        session.get("output_filename", "narration.wav"),
        session.get("model_name", MODEL_ORIGINAL),
        merge.get("silence_ms", 200),
        merge.get("require_approved", False),
        merge.get("export_mp3", False),
        merge.get("mp3_bitrate", "192k"),
        summary,
        ref_audio,
        final_wav,
        final_wav,
        final_mp3,
        status_message(session, "Session loaded successfully with all assets restored."),
    )


def on_table_select(session: dict[str, Any] | None = None, evt: gr.SelectData | None = None):
    # Defensively handle either argument order from Gradio
    if isinstance(session, gr.SelectData):
        evt, session = session, evt
    if not session or not session.get("chunks"):
        return (1, "", None, "", "No active session.", "No active session.")
    if evt is None:
        return (1, "", None, "", "No chunk selected.", "No chunk selected.")
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    chunk_num = row_idx + 1
    chunk = get_chunk(session, chunk_num)
    return (
        chunk_num,
        chunk["text"],
        chunk.get("audio_path"),
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Selected chunk {chunk.get('id') or chunk['index']} from manifest."),
    )


def update_validation_scorecard(session: dict[str, Any] | None):
    """Generate updated scorecard markdown, problem choices, full report, and button label."""
    scorecard_md, problem_choices, btn_text = build_validation_overview(session)
    full_md = build_full_report_markdown(session)
    if gr is not None:
        dropdown_update = gr.Dropdown(choices=problem_choices, value=None)
        btn_update = gr.Button(value=btn_text)
    else:
        dropdown_update = problem_choices
        btn_update = btn_text
    return scorecard_md, dropdown_update, full_md, btn_update


def on_jump_to_problem(session: dict[str, Any] | None = None, selected_choice: Any = None):
    """Load the selected problem chunk directly into the inspector & editor."""
    if not session or not session.get("chunks"):
        return (1, "", None, "", "No active session.", "No active session.")
    if selected_choice is None:
        return (1, "", None, "", "No chunk selected.", "No chunk selected.")
    try:
        chunk_num = int(selected_choice)
    except (ValueError, TypeError):
        import re
        match = re.search(r"Chunk\s+(\d+)", str(selected_choice))
        if match:
            chunk_num = int(match.group(1))
        else:
            return (1, "", None, "", "Invalid chunk selection.", "Invalid chunk selection.")
    chunk = get_chunk(session, chunk_num)
    return (
        chunk_num,
        chunk["text"],
        chunk.get("audio_path"),
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Jumped to problem chunk {chunk.get('id') or chunk['index']}."),
    )


def refresh_session_picker():
    choices = list_session_choices()
    val = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=val)


def load_selected_chunk(session: dict[str, Any], chunk_number: int):
    chunk = get_chunk(session, int(chunk_number or 1))
    audio_path = chunk.get("audio_path")
    return (
        chunk["text"],
        audio_path,
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Selected chunk {chunk.get('id') or chunk['index']}."),
    )


def move_selected_chunk(session: dict[str, Any], chunk_number: int, direction: int):
    if not session:
        raise gr.Error("Create or load a session first.")
    current = int(chunk_number or 1)
    next_number = max(1, min(len(session["chunks"]), current + int(direction)))
    return (next_number, *load_selected_chunk(session, next_number))


def save_selected_chunk(session: dict[str, Any], chunk_number: int, edited_text: str):
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
        status_message(session, f"Saved chunk {chunk.get('id') or chunk['index']}. Existing audio reset."),
    )


def exclude_selected_chunk(session: dict[str, Any], chunk_number: int):
    chunk = get_chunk(session, int(chunk_number or 1))
    chunk["status"] = "excluded"
    save_session(session)
    return session, format_chunk_table(session), status_message(session, f"Excluded chunk {chunk.get('id') or chunk['index']}.")


def approve_selected_chunk(session: dict[str, Any], chunk_number: int):
    chunk = get_chunk(session, int(chunk_number or 1))
    if not chunk.get("audio_path"):
        raise gr.Error("Generate audio before approving this chunk.")
    chunk["status"] = "approved"
    save_session(session)
    return session, format_chunk_table(session), status_message(session, f"Approved chunk {chunk.get('id') or chunk['index']}.")


def create_session_from_script(project_name: str, output_filename: str, model_name: str, full_text: str, target_chars: int):
    chunks = chunk_script(full_text or "", int(target_chars or 280))
    if not chunks:
        raise gr.Error("Add script text before splitting.")
    session = make_session(project_name, output_filename, model_name, full_text, chunks)
    summary = format_blueprint_summary(session)
    return (
        session,
        format_chunk_table(session),
        1,
        chunks[0],
        session.get("project_name", ""),
        session.get("output_filename", "narration.wav"),
        session.get("model_name", MODEL_ORIGINAL),
        summary,
        status_message(session, f"Created new session with {len(chunks)} chunks."),
    )


def export_session_to_blueprint_file(session: dict[str, Any]):
    if not session:
        raise gr.Error("Create or load a session first.")
    blueprint_dict = export_blueprint_from_session(session)
    sdir = session_dir(session)
    out_path = sdir / f"{session.get('project_name', 'narration')}_blueprint.json"
    out_path.write_text(json.dumps(blueprint_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path), status_message(session, f"Exported blueprint to `{out_path.name}`.")


def auto_validate_and_regenerate_callback(
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
    validation_enabled: bool,
    auto_regenerate: bool,
    whisper_model_name: str,
    whisper_device: str,
    validation_threshold: float,
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

    session, model_cache, _table, reg_msg = regenerate_failed_chunks(
        session=session,
        model_cache=model_cache,
        model_name=model_name,
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
        validation_threshold=validation_threshold,
        whisper_model_name=whisper_model_name,
        whisper_device=whisper_device,
        validation_enabled=True,
        validate_chunk_fn=validate_chunk,
    )
    return session, model_cache, f"{validation_status} Automatic regeneration: {reg_msg}"


def build_studio_app(default_reference_audio: str = DEFAULT_REFERENCE_AUDIO) -> gr.Blocks:
    blocks_kwargs: dict[str, Any] = {"title": "Chatterbox Narration Studio"}
    # Gradio 5.0+ moved css from Blocks constructor to launch()
    try:
        if gr is not None and hasattr(gr, "__version__"):
            major_ver = int(str(gr.__version__).split(".")[0])
            if major_ver < 5:
                blocks_kwargs["css"] = CUSTOM_CSS
    except Exception:
        pass

    with gr.Blocks(**blocks_kwargs) as demo:
        session_state = gr.State(None)
        model_cache_state = gr.State({})

        gr.Markdown(
            "# 🎙️ Chatterbox Narration Studio\n"
            "*Blueprint-First Production Suite for Long-Form Voice Generation with Dual-GPU Acceleration*"
        )

        global_status = gr.Markdown("No active session. Import a Narration Blueprint JSON to begin.")

        with gr.Tabs():
            # =========================================================================
            # TAB 1: BLUEPRINT STUDIO (PRIMARY WORKFLOW)
            # =========================================================================
            with gr.Tab("🎬 Blueprint Studio", id="blueprint_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 1. Import Narration Blueprint")
                            blueprint_file = gr.File(
                                label="Upload Narration Blueprint JSON",
                                file_types=[".json"],
                                type="filepath",
                            )
                            import_blueprint_btn = gr.Button("📥 Import Blueprint & Load Session", variant="primary")
                            blueprint_summary = gr.Markdown("*(No blueprint loaded)*")

                            with gr.Accordion("📂 Or Resume Previous Session", open=False):
                                with gr.Row():
                                    tab1_session_picker = gr.Dropdown(
                                        choices=list_session_choices(),
                                        label="Saved Session",
                                        scale=3,
                                    )
                                    refresh_tab1_sessions_btn = gr.Button("🔄", scale=1)
                                    load_tab1_session_btn = gr.Button("📂 Resume", variant="secondary", scale=2)

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 2. Reference Voice (Clone Prompt)")
                            ref_wav = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="Voice Audio (passed through UI)",
                                value=default_reference_audio,
                            )
                            gr.Markdown(
                                "<small>💡 Audio loaded here conditions all generated chunks. "
                                "You can swap reference audio anytime without touching the blueprint.</small>"
                            )

                with gr.Accordion("⚡ Multi-GPU & Advanced Generation Settings", open=False):
                    with gr.Row():
                        enable_parallel = gr.Checkbox(
                            value=True,
                            label="Use Multiple CUDA GPUs (Batch Parallel)",
                            info="Optimized for Kaggle Dual T4 (cuda:0 and cuda:1).",
                        )
                        max_parallel_devices = gr.Slider(1, 8, step=1, value=2, label="Max GPU workers")
                        skip_existing = gr.Checkbox(
                            value=True,
                            label="Skip already generated chunks (Resume mode)",
                            info="Only synthesizes pending or failed chunks; preserves existing chunk audio.",
                        )
                    gpu_status = gr.Markdown(gpu_status_text(True, 2))
                    model_load_status = gr.Markdown("No TTS models currently loaded in memory.")
                    with gr.Row():
                        refresh_gpu_btn = gr.Button("Refresh GPU Status")
                        warm_models_btn = gr.Button("Pre-load Model Cache")
                        clear_models_btn = gr.Button("Clear Model VRAM Cache")

                    with gr.Row():
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 2.0, step=0.05, label="Temperature", value=0.8)
                        top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                        top_k = gr.Slider(0, 1000, step=10, label="Top K (Turbo/Nano)", value=1000)
                    with gr.Row():
                        repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                        min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Original only)", value=0.05)
                        exaggeration = gr.Slider(0.25, 2.0, step=0.05, label="Exaggeration (Original only)", value=0.5)
                        cfg_weight = gr.Slider(0.0, 1.0, step=0.05, label="CFG/Pace (Original only)", value=0.5)
                        norm_loudness = gr.Checkbox(value=True, label="Normalize Reference Loudness (Turbo/Nano)")

                with gr.Row():
                    generate_all_btn = gr.Button("🚀 Generate All Chunks (Batch)", variant="primary", scale=2)

                # =====================================================================
                # VALIDATION DASHBOARD & SCORECARD
                # =====================================================================
                validation_scorecard_md = gr.Markdown("### 📊 Validation Health Scorecard\n*(No session active)*")
                with gr.Row():
                    failed_chunks_dropdown = gr.Dropdown(
                        label="⚠️ Jump Directly to Problem Chunk",
                        choices=[],
                        scale=3,
                        info="Select a flagged chunk to jump immediately into Chunk Inspector & Editor without hunting row-by-row.",
                    )
                    regenerate_problems_btn = gr.Button("🔄 Regenerate Problem Chunks", variant="secondary", scale=2)
                    refresh_scorecard_btn = gr.Button("🔄 Refresh Scorecard", scale=1)

                with gr.Accordion("📋 View Full Inspection Manifest Table", open=False):
                    full_report_md = gr.Markdown("*(No session active)*")

                gr.Markdown("### 3. Narration Chunks Manifest")
                chunk_table = gr.Dataframe(
                    headers=[
                        "ID",
                        "#",
                        "Status",
                        "Note",
                        "Chars",
                        "Model",
                        "Device",
                        "Text Score",
                        "Whisper",
                        "Audio",
                        "Text Preview",
                        "Audio Path",
                        "Transcript / Error",
                    ],
                    datatype=["str", "number", "str", "str", "number", "str", "str", "str", "str", "str", "str", "str", "str"],
                    label="Chunks Manifest",
                    interactive=False,
                )

                gr.Markdown("### 4. Chunk Inspector & Editor")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            chunk_number = gr.Number(value=1, precision=0, label="Selected Chunk #", scale=1)
                            previous_chunk_btn = gr.Button("← Previous")
                            next_chunk_btn = gr.Button("Next →")
                            load_chunk_btn = gr.Button("Load Chunk")
                        chunk_editor = gr.Textbox(label="Chunk Text (Editable)", lines=4)
                        save_chunk_btn = gr.Button("💾 Save Edited Text")
                    with gr.Column():
                        chunk_audio = gr.Audio(label="Chunk Generated Audio")
                        chunk_transcript = gr.Textbox(label="Whisper ASR Transcript", lines=2, interactive=False)
                        chunk_validation = gr.Markdown("This chunk has not been validated yet.")
                        with gr.Row():
                            generate_selected_btn = gr.Button("⚡ Generate Selected Chunk", variant="primary")
                            approve_btn = gr.Button("✅ Approve")
                            exclude_btn = gr.Button("⛔ Exclude")

                with gr.Accordion("🔍 Quality Assurance & Whisper Validation", open=False):
                    with gr.Row():
                        validation_enabled = gr.Checkbox(
                            value=True,
                            label="Enable Whisper Validation",
                            info="Validates text accuracy and flags hallucinated or missed words.",
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
                        validation_threshold = gr.Slider(0.50, 0.99, step=0.01, value=0.90, label="Pass Score Threshold")
                        auto_regenerate = gr.Checkbox(
                            value=False,
                            label="Auto-regenerate chunks below threshold",
                        )
                    with gr.Row():
                        validate_selected_btn = gr.Button("Validate Selected")
                        validate_all_btn = gr.Button("Validate All")
                        regenerate_failed_btn = gr.Button("Regenerate Chunks Needing Review")

                    gr.Markdown("#### Acoustic Audio Checks")
                    with gr.Row():
                        silence_threshold_db = gr.Slider(-60, -20, step=1, value=-45, label="Silence threshold (dBFS)")
                        max_silence_ms = gr.Slider(0, 2000, step=50, value=250, label="Max leading/trailing silence (ms)")
                        max_clip_fraction = gr.Slider(0, 0.01, step=0.0005, value=0.001, label="Max clipping fraction")
                        min_duration_s = gr.Slider(0.05, 5, step=0.05, value=0.20, label="Min chunk duration (s)")
                        max_duration_s = gr.Slider(5, 180, step=1, value=120, label="Max chunk duration (s)")
                        min_rms_dbfs = gr.Slider(-60, -10, step=1, value=-38, label="Min RMS loudness (dBFS)")
                    with gr.Row():
                        check_audio_selected_btn = gr.Button("Check Audio Selected")
                        check_audio_all_btn = gr.Button("Check Audio All")
                        export_report_btn = gr.Button("Export Validation Report JSON")
                    validation_report_file = gr.File(label="Exported Validation Report")

                with gr.Accordion("📦 Finalize & Export Narration", open=True):
                    with gr.Row():
                        output_filename = gr.Textbox(value="narration.wav", label="Final Output Filename")
                        silence_ms = gr.Slider(0, 1000, step=25, value=200, label="Silence between chunks (ms)")
                        require_approved = gr.Checkbox(value=False, label="Require all chunks approved")
                        export_mp3 = gr.Checkbox(value=False, label="Also export MP3")
                        mp3_bitrate = gr.Dropdown(["128k", "192k", "256k", "320k"], value="192k", label="MP3 bitrate")
                    merge_btn = gr.Button("🎉 Merge & Finalize Narration", variant="primary")
                    with gr.Row():
                        final_audio = gr.Audio(label="Final Merged Audio Player")
                    with gr.Row():
                        final_file = gr.File(label="Download Final WAV")
                        final_mp3_file = gr.File(label="Download Final MP3")

            # =========================================================================
            # TAB 2: SCRIPT SPLITTER (BLUEPRINT GENERATOR)
            # =========================================================================
            with gr.Tab("✍️ Script Splitter & Blueprint Builder", id="splitter_tab"):
                gr.Markdown(
                    "### Split Raw Script into Chunks\n"
                    "Use this tool if you have an unchunked text script and want to create a new session or export a blueprint JSON."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        script_project_name = gr.Textbox(value="youtube_narration", label="Project Name")
                        script_output_filename = gr.Textbox(value="narration.wav", label="Target Output Filename")
                        script_model_name = gr.Dropdown(MODEL_CHOICES, value=MODEL_ORIGINAL, label="Model")
                        raw_script_text = gr.Textbox(
                            value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                            label="Full Narration Script",
                            lines=8,
                            elem_id="main_textbox",
                        )
                        with gr.Row(elem_classes=["tag-container"]):
                            for tag in EVENT_TAGS:
                                btn = gr.Button(tag, elem_classes=["tag-btn"])
                                btn.click(fn=None, inputs=[btn, raw_script_text], outputs=raw_script_text, js=INSERT_TAG_JS)
                        target_chars = gr.Slider(120, 800, step=10, value=280, label="Target Chunk Characters")
                        with gr.Row():
                            split_script_btn = gr.Button("✂️ Split Script & Load Into Studio", variant="primary")
                            export_blueprint_btn = gr.Button("💾 Export as Blueprint JSON")
                        exported_blueprint_file = gr.File(label="Exported Blueprint JSON")

            # =========================================================================
            # TAB 3: SAVED SESSIONS
            # =========================================================================
            with gr.Tab("📂 Saved Sessions", id="sessions_tab"):
                gr.Markdown("### Browse and Resume Saved Sessions")
                session_path = gr.Textbox(
                    label="Direct Session Path or session.json",
                    placeholder="outputs/chatterbox_sessions/... or /kaggle/working/chatterbox_sessions/...",
                )
                session_picker = gr.Dropdown(
                    choices=list_session_choices(),
                    label="Discovered Sessions",
                    info="Select a previous session to resume progress.",
                )
                with gr.Row():
                    refresh_sessions_btn = gr.Button("🔄 Refresh Sessions List")
                    load_picked_session_btn = gr.Button("📂 Load Selected Session", variant="primary")
                    load_session_path_btn = gr.Button("Load Direct Path")

        # Hidden shared fields for model consistency
        hidden_project_name = gr.State("youtube_narration")
        hidden_model_name = gr.State(MODEL_ORIGINAL)

        # -------------------------------------------------------------------------
        # EVENT BINDINGS
        # -------------------------------------------------------------------------

        scorecard_outputs = [
            validation_scorecard_md,
            failed_chunks_dropdown,
            full_report_md,
            regenerate_problems_btn,
        ]

        # Jump directly to problem chunk when selected from dropdown
        failed_chunks_dropdown.change(
            fn=on_jump_to_problem,
            inputs=[session_state, failed_chunks_dropdown],
            outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, global_status],
        )

        # Quick refresh button for scorecard
        refresh_scorecard_btn.click(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Blueprint Import
        import_blueprint_btn.click(
            fn=import_narration_blueprint,
            inputs=[blueprint_file],
            outputs=[
                session_state,
                chunk_table,
                chunk_number,
                chunk_editor,
                hidden_project_name,
                output_filename,
                hidden_model_name,
                silence_ms,
                require_approved,
                export_mp3,
                mp3_bitrate,
                blueprint_summary,
                global_status,
            ],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Progress default hook helper
        prog_hook = gr.Progress(track_tqdm=True) if gr is not None else None

        # Generate All Chunks (with live progress tracking)
        def on_generate_all(
            session, cache, model, ref, t, s, mp, tp, tk, rp, ex, cfg, nl, ep, mpd, se, ve, ar, wm, wd, vt,
            progress=prog_hook,
        ):
            return generate_all_chunks(
                session=session,
                model_cache=cache,
                model_name=model,
                reference_audio_path=ref,
                temperature=t,
                seed_num=s,
                min_p=mp,
                top_p=tp,
                top_k=tk,
                repetition_penalty=rp,
                exaggeration=ex,
                cfg_weight=cfg,
                norm_loudness=nl,
                enable_parallel=ep,
                max_parallel_devices=mpd,
                skip_existing=se,
                validation_enabled=ve,
                auto_regenerate=ar,
                whisper_model_name=wm,
                whisper_device=wd,
                validation_threshold=vt,
                progress=progress,
                auto_validate_fn=auto_validate_and_regenerate_callback,
            )

        generate_all_btn.click(
            fn=on_generate_all,
            inputs=[
                session_state,
                model_cache_state,
                hidden_model_name,
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
                skip_existing,
                validation_enabled,
                auto_regenerate,
                whisper_model_name,
                whisper_device,
                validation_threshold,
            ],
            outputs=[session_state, model_cache_state, chunk_table, chunk_audio, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Click table row to inspect chunk immediately
        chunk_table.select(
            fn=on_table_select,
            inputs=[session_state],
            outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, global_status],
        )

        # Chunk Inspector Navigation & Editing
        load_chunk_btn.click(
            fn=load_selected_chunk,
            inputs=[session_state, chunk_number],
            outputs=[chunk_editor, chunk_audio, chunk_transcript, chunk_validation, global_status],
        )

        previous_chunk_btn.click(
            fn=lambda session, number: move_selected_chunk(session, number, -1),
            inputs=[session_state, chunk_number],
            outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, global_status],
        )

        next_chunk_btn.click(
            fn=lambda session, number: move_selected_chunk(session, number, 1),
            inputs=[session_state, chunk_number],
            outputs=[chunk_number, chunk_editor, chunk_audio, chunk_transcript, chunk_validation, global_status],
        )

        save_chunk_btn.click(
            fn=save_selected_chunk,
            inputs=[session_state, chunk_number, chunk_editor],
            outputs=[session_state, chunk_table, chunk_audio, chunk_transcript, chunk_validation, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Generate Selected Chunk (with live progress tracking)
        def on_generate_selected(
            session, cache, num, model, ref, t, s, mp, tp, tk, rp, ex, cfg, nl,
            progress=prog_hook,
        ):
            return generate_selected_chunk(
                session=session,
                model_cache=cache,
                chunk_number=num,
                model_name=model,
                reference_audio_path=ref,
                temperature=t,
                seed_num=s,
                min_p=mp,
                top_p=tp,
                top_k=tk,
                repetition_penalty=rp,
                exaggeration=ex,
                cfg_weight=cfg,
                norm_loudness=nl,
                validation_details_fn=validation_details,
                progress=progress,
            )

        generate_selected_btn.click(
            fn=on_generate_selected,
            inputs=[
                session_state,
                model_cache_state,
                chunk_number,
                hidden_model_name,
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
            outputs=[session_state, model_cache_state, chunk_table, chunk_audio, chunk_transcript, chunk_validation, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        approve_btn.click(
            fn=approve_selected_chunk,
            inputs=[session_state, chunk_number],
            outputs=[session_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        exclude_btn.click(
            fn=exclude_selected_chunk,
            inputs=[session_state, chunk_number],
            outputs=[session_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Validation & Audio Checks (with live progress tracking)
        def on_validate_selected(
            session, num, wm, wd, vt, ve,
            progress=prog_hook,
        ):
            return validate_selected_chunk(
                session=session,
                chunk_number=num,
                whisper_model_name=wm,
                whisper_device=wd,
                validation_threshold=vt,
                enabled=ve,
                progress=progress,
            )

        validate_selected_btn.click(
            fn=on_validate_selected,
            inputs=[session_state, chunk_number, whisper_model_name, whisper_device, validation_threshold, validation_enabled],
            outputs=[session_state, chunk_table, chunk_transcript, chunk_validation, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        def on_validate_all(
            session, wm, wd, vt, ve,
            progress=prog_hook,
        ):
            return validate_all_chunks(
                session=session,
                whisper_model_name=wm,
                whisper_device=wd,
                validation_threshold=vt,
                enabled=ve,
                progress=progress,
            )

        validate_all_btn.click(
            fn=on_validate_all,
            inputs=[session_state, whisper_model_name, whisper_device, validation_threshold, validation_enabled],
            outputs=[session_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        def on_regenerate_failed(
            session, cache, model, ref, t, s, mp, tp, tk, rp, ex, cfg, nl, ep, mpd, vt, wm, wd, ve,
            progress=prog_hook,
        ):
            return regenerate_failed_chunks(
                session=session,
                model_cache=cache,
                model_name=model,
                reference_audio_path=ref,
                temperature=t,
                seed_num=s,
                min_p=mp,
                top_p=tp,
                top_k=tk,
                repetition_penalty=rp,
                exaggeration=ex,
                cfg_weight=cfg,
                norm_loudness=nl,
                enable_parallel=ep,
                max_parallel_devices=mpd,
                validation_threshold=vt,
                whisper_model_name=wm,
                whisper_device=wd,
                validation_enabled=ve,
                validate_chunk_fn=validate_chunk,
                progress=progress,
            )

        regenerate_failed_btn.click(
            fn=on_regenerate_failed,
            inputs=[
                session_state,
                model_cache_state,
                hidden_model_name,
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
            outputs=[session_state, model_cache_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Also wire the scorecard problem chunk regeneration button
        regenerate_problems_btn.click(
            fn=on_regenerate_failed,
            inputs=[
                session_state,
                model_cache_state,
                hidden_model_name,
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
            outputs=[session_state, model_cache_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        def on_check_audio_selected(
            session, num, st, ms, cf, mind, maxd, mrms,
            progress=prog_hook,
        ):
            return check_audio_selected(
                session=session,
                chunk_number=num,
                silence_threshold_db=st,
                max_silence_ms=ms,
                max_clip_fraction=cf,
                min_duration_s=mind,
                max_duration_s=maxd,
                min_rms_dbfs=mrms,
                progress=progress,
            )

        check_audio_selected_btn.click(
            fn=on_check_audio_selected,
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
            outputs=[session_state, chunk_table, chunk_validation, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        def on_check_audio_all(
            session, st, ms, cf, mind, maxd, mrms,
            progress=prog_hook,
        ):
            return check_audio_all(
                session=session,
                silence_threshold_db=st,
                max_silence_ms=ms,
                max_clip_fraction=cf,
                min_duration_s=mind,
                max_duration_s=maxd,
                min_rms_dbfs=mrms,
                progress=progress,
            )

        check_audio_all_btn.click(
            fn=on_check_audio_all,
            inputs=[
                session_state,
                silence_threshold_db,
                max_silence_ms,
                max_clip_fraction,
                min_duration_s,
                max_duration_s,
                min_rms_dbfs,
            ],
            outputs=[session_state, chunk_table, global_status],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        export_report_btn.click(
            fn=export_validation_report,
            inputs=[session_state],
            outputs=[validation_report_file, global_status],
        )

        # GPU Caching & Status
        refresh_gpu_btn.click(
            fn=gpu_status_text,
            inputs=[enable_parallel, max_parallel_devices],
            outputs=[gpu_status],
        )

        warm_models_btn.click(
            fn=warm_model_cache,
            inputs=[model_cache_state, hidden_model_name, enable_parallel, max_parallel_devices],
            outputs=[model_cache_state, global_status, model_load_status],
        )

        clear_models_btn.click(
            fn=clear_model_cache,
            inputs=[model_cache_state],
            outputs=[model_cache_state, global_status, model_load_status],
        )

        # Finalize & Merge (with live progress tracking)
        def on_merge_chunks(
            session, fn, sms, req_appr, exp_mp3, mp3_br,
            progress=prog_hook,
        ):
            return merge_chunks(
                session=session,
                output_filename=fn,
                silence_ms=sms,
                require_approved=req_appr,
                export_mp3=exp_mp3,
                mp3_bitrate=mp3_br,
                progress=progress,
            )

        merge_btn.click(
            fn=on_merge_chunks,
            inputs=[session_state, output_filename, silence_ms, require_approved, export_mp3, mp3_bitrate],
            outputs=[session_state, final_file, final_audio, final_mp3_file, global_status],
        )

        # Script Splitter Tab
        split_script_btn.click(
            fn=create_session_from_script,
            inputs=[script_project_name, script_output_filename, script_model_name, raw_script_text, target_chars],
            outputs=[
                session_state,
                chunk_table,
                chunk_number,
                chunk_editor,
                hidden_project_name,
                output_filename,
                hidden_model_name,
                blueprint_summary,
                global_status,
            ],
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        export_blueprint_btn.click(
            fn=export_session_to_blueprint_file,
            inputs=[session_state],
            outputs=[exported_blueprint_file, global_status],
        )

        # Session Loading & Resuming
        session_load_outputs = [
            session_state,
            chunk_table,
            chunk_number,
            chunk_editor,
            chunk_audio,
            chunk_transcript,
            chunk_validation,
            hidden_project_name,
            output_filename,
            hidden_model_name,
            silence_ms,
            require_approved,
            export_mp3,
            mp3_bitrate,
            blueprint_summary,
            ref_wav,
            final_file,
            final_audio,
            final_mp3_file,
            global_status,
        ]

        # Tab 1 Quick Session Resume
        refresh_tab1_sessions_btn.click(
            fn=refresh_session_picker,
            outputs=[tab1_session_picker],
        )

        load_tab1_session_btn.click(
            fn=load_session,
            inputs=[tab1_session_picker, ref_wav],
            outputs=session_load_outputs,
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        # Tab 3 Sessions Browser
        refresh_sessions_btn.click(
            fn=refresh_session_picker,
            outputs=[session_picker],
        )

        load_picked_session_btn.click(
            fn=load_session,
            inputs=[session_picker, ref_wav],
            outputs=session_load_outputs,
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

        load_session_path_btn.click(
            fn=load_session,
            inputs=[session_path, ref_wav],
            outputs=session_load_outputs,
        ).then(
            fn=update_validation_scorecard,
            inputs=[session_state],
            outputs=scorecard_outputs,
        )

    return demo
