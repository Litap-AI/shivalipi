"""
Shivalipi Configuration
Author: Rohit Manikrao Patil
"""

from __future__ import annotations

import os
from pathlib import Path

# Package cache directory (defaults to ~/.cache/shivalipi)
CACHE_DIR: Path = Path(
    os.getenv("SHIVALIPI_CACHE_DIR", Path.home() / ".cache" / "shivalipi")
)

# Default Hugging Face repository containing fine-tuned weights
DEFAULT_MODEL_ID: str = os.getenv("SHIVALIPI_MODEL_ID", "Litap007/shivalipi-vlm")

# Maximum sequence length for transcription output
MAX_TARGET_LENGTH: int = 128

# Device configuration fallback
DEVICE_OVERRIDE: str | None = os.getenv("SHIVALIPI_DEVICE", None)
