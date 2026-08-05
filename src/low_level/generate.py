"""Functional pipeline — the master entropy extraction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.low_level.entropy import EntropyAssessment, estimate_entropy
from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


def condition_entropy_pool(
    raw_image_data: np.ndarray,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Sample, assess, and condition a raw sensor array into a TRNG seed.

    This is the source-agnostic core of the pipeline: any ingest function that
    yields a 2-D sensor array (a RAW file, a webcam capture, …) funnels through
    here so the estimator and conditioner are applied identically everywhere.

    Args:
        raw_image_data: A 2-D array of sensor values from any ingest source.
        sample_fn: Samples the array and returns a byte stream of raw entropy.
            Defaults to :func:`sample_entropy_grid`.
        hash_fn: Hashes the entropy pool into a fixed-length, uniformly
            distributed byte string. Defaults to :func:`hash_entropy_pool`.

    Returns:
        ``(seed, entropy_pool, assessment)`` where *seed* is the 64-byte
        SHA3-512 digest, *entropy_pool* is the raw pre-hash bytes, and
        *assessment* is the :class:`EntropyAssessment`.
    """
    entropy_pool = sample_fn(raw_image_data)
    assessment = estimate_entropy(entropy_pool)
    secure_hash_bytes = hash_fn(entropy_pool)
    return secure_hash_bytes, entropy_pool, assessment


def generate_with_assessment(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Extract entropy from a RAW image and measure its quality.

    Runs the full pipeline with NIST SP 800-90B entropy estimation applied
    to the raw pool before SHA3-512 conditioning.

    Args:
        image_path: Path to the RAW image file used as the entropy source.
        ingest_fn: Reads the RAW file and returns a 2D array of raw sensor
            values. Defaults to :func:`ingest_raw_image`.
        sample_fn: Samples the sensor array and returns a byte stream of
            raw entropy. Defaults to :func:`sample_entropy_grid`.
        hash_fn: Hashes the entropy pool into a fixed-length, uniformly
            distributed byte string. Defaults to :func:`hash_entropy_pool`.

    Returns:
        ``(seed, entropy_pool, assessment)`` where *seed* is the 64-byte
        SHA3-512 digest, *entropy_pool* is the raw pre-hash bytes, and
        *assessment* is the :class:`EntropyAssessment`.
    """
    raw_image_data = ingest_fn(image_path)
    return condition_entropy_pool(raw_image_data, sample_fn=sample_fn, hash_fn=hash_fn)


def generate_true_random_number(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> bytes:
    """Extract physical entropy from a RAW image and return pure, raw bytes.

    Thin wrapper around :func:`generate_with_assessment` that returns only
    the 64-byte seed.
    """
    seed, _pool, _assessment = generate_with_assessment(image_path, ingest_fn, sample_fn, hash_fn)
    return seed
