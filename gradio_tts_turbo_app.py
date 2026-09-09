#!/usr/bin/env python3
"""Chatterbox Narration Studio Launcher

Blueprint-first production suite for long-form voice narration with Chatterbox.
Optimized for Kaggle Dual T4 accelerators with parallel generation, ASR validation,
and seamless blueprint importing.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on python path for Kaggle notebook direct execution
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatterbox.studio.blueprint import (
    chunk_settings_for,
    export_blueprint_from_session,
    make_session_from_blueprint,
    normalize_blueprint_settings,
    validate_narration_blueprint,
)
from chatterbox.studio.config import (
    CUSTOM_CSS,
    DEFAULT_REFERENCE_AUDIO,
    EVENT_TAGS,
    GENERATION_PRESETS,
    MODEL_CHOICES,
    MODEL_NANO,
    MODEL_ORIGINAL,
    MODEL_TURBO,
    PRESET_CHOICES,
    SESSION_ROOT,
    safe_name,
    safe_wav_filename,
    set_seed,
)
from chatterbox.studio.engine import (
    ModelAdapter,
    available_generation_devices,
    clear_model_cache,
    collect_generation_settings,
    default_device,
    ensure_model_consistency,
    generate_all_chunks,
    generate_chunk_wav,
    generate_selected_chunk,
    get_model_adapter,
    gpu_status_text,
    regenerate_failed_chunks,
    save_chunk_audio,
    warm_model_cache,
)
from chatterbox.studio.finalizer import merge_chunks
from chatterbox.studio.session import (
    chunk_script,
    copy_reference_to_session,
    format_chunk_table,
    get_chunk,
    list_session_paths,
    load_session_from_path,
    make_session,
    save_session,
    session_dir,
    session_json_path,
    status_message,
)
from chatterbox.studio.ui import (
    build_studio_app,
    import_narration_blueprint,
    load_selected_chunk,
    load_session,
)
from chatterbox.studio.validator import (
    check_audio_all,
    check_audio_quality,
    check_audio_selected,
    export_validation_report,
    transcribe_audio,
    validate_all_chunks,
    validate_chunk,
    validate_selected_chunk,
    validation_comparison,
    validation_details,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox Narration Studio")
    parser.add_argument(
        "reference_audio",
        nargs="?",
        default=None,
        help="Path or URL to use as the default reference voice audio.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="Create a public Gradio share link (default: True for Kaggle).",
    )
    parser.add_argument(
        "--no-share",
        action="store_false",
        dest="share",
        help="Disable Gradio public share link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the Gradio app on.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name/host to bind (default: 0.0.0.0).",
    )
    return parser.parse_known_args()[0]


args = parse_args()
active_ref_audio = args.reference_audio or DEFAULT_REFERENCE_AUDIO

# Build the studio UI
demo = build_studio_app(default_reference_audio=active_ref_audio)


if __name__ == "__main__":
    launch_kwargs = {
        "share": args.share,
        "allowed_paths": [str(SESSION_ROOT.resolve())],
        "server_name": args.server_name,
        "css": CUSTOM_CSS,
    }
    if args.port:
        launch_kwargs["server_port"] = args.port

    try:
        demo.queue(
            max_size=50,
            default_concurrency_limit=1,
        ).launch(**launch_kwargs)
    except TypeError as exc:
        if "css" in str(exc) and "css" in launch_kwargs:
            launch_kwargs.pop("css", None)
            demo.queue(
                max_size=50,
                default_concurrency_limit=1,
            ).launch(**launch_kwargs)
        else:
            raise
