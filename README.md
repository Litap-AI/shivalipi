# Shivalipi (शिवलिपी) 📜

[![PyPI Version](https://img.shields.io/pypi/v/shivalipi)](https://pypi.org/project/shivalipi/)
[![Author](https://img.shields.io/badge/Author-Rohit%20Manikrao%20Patil-blue)](https://www.rohitmanikraopatil.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A dedicated Python vision engine and linguistic mapping library engineered to decipher historical handwritten Modi Lipi (मोडी लिपी) documents directly into standard Devanagari Marathi (देवनागरी मराठी).

---

## 🚀 Key Features

- **Pure Direct Mapping:** Direct image-to-Devanagari pipeline without requiring intermediate external transliteration engines.
- **Historical Administrative Lexicon:** Optimized for Peshwa, Maratha Empire, and colonial-era administrative documents, sanads (सनद), and court letters (कौलनामा).
- **Dual Pipeline Support:**
  - Fast, lightweight Vision-Encoder-Decoder for standard manuscript line strips.
  - Native Devanagari Unicode NFC normalization and ligature handling.
- **Developer First:** Clean, minimalist Python API.

---

## 📦 Installation

```bash
pip install shivalipi
```

---

## Quickstart
Python
import shivalipi

#Transcribe a Modi manuscript image to Devanagari Marathi
result = shivalipi.transcribe("path/to/modi_document.jpg")
print("Transcription:")
print(result)

---

## Advanced Usage
Python
from shivalipi import Shivalipi

#Initialize the engine on a specific device (cuda / mps / cpu)
engine = Shivalipi(device="cuda")

#Run inference with structured metadata
output = engine.transcribe("peshwa_letter_line.png")

print("Clean Marathi Text :", output.text)
print("Raw Model Output   :", output.raw_text)

---

## Evaluation & Metrics
shivalipi provides built-in metrics tailored for Devanagari scripts:

Python
from shivalipi import compute_cer, compute_wer

predictions = ["राजमान्य राजश्री", "कौलनामा"]
ground_truth = ["राजमान्य राजश्री", "कौलनामा"]

cer = compute_cer(predictions, ground_truth)
wer = compute_wer(predictions, ground_truth)

print(f"CER: {cer * 100:.2f}% | WER: {wer * 100:.2f}%")

---

## Author & Maintainer
Developed and maintained by Rohit Manikrao Patil

Website: www.rohitmanikraopatil.com

PyPI project: https://pypi.org/project/shivalipi/


---

## License
This project is licensed under the Apache License 2.0.

--- 
