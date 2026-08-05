"""photorand — a True Random Number Generator using raw camera sensor data."""

from __future__ import annotations

from src.high_level.engine import PhotoRandEngine
from src.high_level.seed import PhotoRandSeed
from src.low_level.csprng import expand_entropy_chacha20
from src.low_level.entropy import EntropyAssessment, estimate_entropy
from src.low_level.generate import generate_true_random_number, generate_with_assessment
from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import reduce_fixed_pattern_noise, sample_entropy_grid

__version__ = "1.2.0"

__all__ = [
    "EntropyAssessment",
    "PhotoRandEngine",
    "PhotoRandSeed",
    "estimate_entropy",
    "expand_entropy_chacha20",
    "generate_true_random_number",
    "generate_with_assessment",
    "hash_entropy_pool",
    "ingest_raw_image",
    "reduce_fixed_pattern_noise",
    "sample_entropy_grid",
]
