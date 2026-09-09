import os
import random
import re
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

# Paths
SESSION_ROOT = Path("/kaggle/working/chatterbox_sessions") if Path("/kaggle/working").exists() else Path("outputs/chatterbox_sessions")
DEFAULT_REFERENCE_AUDIO = os.environ.get(
    "CHATTERBOX_REFERENCE_AUDIO",
    "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
)

# Models
MODEL_TURBO = "Turbo"
MODEL_NANO = "Nano"
MODEL_ORIGINAL = "Original"
MODEL_CHOICES = [MODEL_TURBO, MODEL_NANO, MODEL_ORIGINAL]

# Generation Presets
PRESET_CHOICES = [
    "Reality Mechanism",
    "Clear Mental Model",
    "Investigative Case Study",
    "Philosophical Reflection",
    "High-Stakes Decision",
    "Custom",
]
GENERATION_PRESETS = {
    "Reality Mechanism": (0.58, 0.92, 900, 1.18, 0.05, 0.48, 0.58, True),
    "Clear Mental Model": (0.52, 0.90, 800, 1.16, 0.05, 0.42, 0.62, True),
    "Investigative Case Study": (0.68, 0.94, 1000, 1.20, 0.05, 0.58, 0.52, True),
    "Philosophical Reflection": (0.48, 0.88, 700, 1.14, 0.05, 0.62, 0.48, True),
    "High-Stakes Decision": (0.74, 0.95, 1000, 1.20, 0.05, 0.70, 0.42, True),
    "Custom": (0.8, 0.95, 1000, 1.2, 0.05, 0.5, 0.5, True),
}

# Default Blueprint Settings
BLUEPRINT_DEFAULT_SETTINGS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "norm_loudness": True,
    "seed_num": 0,
}

# Paralinguistic emotion event tags
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

# UI Custom CSS
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

.badge-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
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
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    if np is not None:
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
