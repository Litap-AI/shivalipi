"""
Shivalipi: Modi Lipi to Devanagari Marathi Recognition Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Author: Rohit Manikrao Patil
"""

__version__ = "0.1.3"
__author__ = "Rohit Manikrao Patil"
__license__ = "Apache-2.0"

from .engine import Shivalipi, ShivalipiResult, transcribe
from .metrics import compute_cer, compute_wer, normalize_devanagari

__all__ = [
    "Shivalipi",
    "ShivalipiResult",
    "__author__",
    "__license__",
    "__version__",
    "compute_cer",
    "compute_wer",
    "normalize_devanagari",
    "transcribe",
]
