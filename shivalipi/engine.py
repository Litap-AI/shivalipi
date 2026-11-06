"""
Shivalipi Core Inference Engine
Author: Rohit Manikrao Patil
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import CACHE_DIR, DEFAULT_MODEL_ID
from .metrics import normalize_devanagari


class ShivalipiResult:
    """Encapsulates transcription outputs and normalized formatting."""

    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text.strip()
        self.text = normalize_devanagari(self.raw_text)

    def __repr__(self) -> str:
        return f"<ShivalipiResult text='{self.text}'>"

    def __str__(self) -> str:
        return self.text


class Shivalipi:
    """Main pipeline class managing model loading, device placement, and inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.processor = TrOCRProcessor.from_pretrained(
            model_id, cache_dir=CACHE_DIR
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id, cache_dir=CACHE_DIR
        ).to(self.device)
        self.model.eval()

    def transcribe(
        self, image_input: str | Path | Image.Image
    ) -> ShivalipiResult:
        """Transcribes a Modi Lipi document image into Devanagari Marathi text."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("Input must be a file path string, Path object, or PIL Image.")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return ShivalipiResult(raw_text=raw_output)


_default_engine: Shivalipi | None = None


def transcribe(
    image: str | Path | Image.Image,
    model_id: str | None = None,
    device: str | None = None,
) -> str:
    """Direct transcription function returning normalized Devanagari Marathi text."""
    global _default_engine
    if _default_engine is None or model_id is not None:
        _default_engine = Shivalipi(
            model_id=model_id or DEFAULT_MODEL_ID, device=device
        )
    return str(_default_engine.transcribe(image))
