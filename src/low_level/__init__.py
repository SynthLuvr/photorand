"""Low-level modular primitives for entropy extraction and expansion."""

from __future__ import annotations

from src.low_level.capture import WebcamCaptureError, capture_webcam_noise, generate_from_webcam
from src.low_level.csprng import expand_entropy_chacha20
from src.low_level.entropy import EntropyAssessment, estimate_entropy
from src.low_level.generate import (
    condition_entropy_pool,
    generate_true_random_number,
    generate_with_assessment,
)
from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import reduce_fixed_pattern_noise, sample_entropy_grid

__all__ = [
    "EntropyAssessment",
    "WebcamCaptureError",
    "capture_webcam_noise",
    "condition_entropy_pool",
    "estimate_entropy",
    "expand_entropy_chacha20",
    "generate_from_webcam",
    "generate_true_random_number",
    "generate_with_assessment",
    "hash_entropy_pool",
    "ingest_raw_image",
    "reduce_fixed_pattern_noise",
    "sample_entropy_grid",
]
