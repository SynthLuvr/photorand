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


def generate_with_assessment(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Extract entropy from a RAW image **and** measure its quality.

    This is the full pipeline with NIST SP 800-90B entropy estimation
    applied to the raw pool *before* SHA3-512 conditioning.  The hash is a
    whitener — it distributes entropy but cannot create it — so the
    measurement must happen on the unconditioned source.

    Args:
        image_path: Path to the RAW image file used as the entropy source.
        ingest_fn: Function that reads the RAW file and returns a 2D array of raw
            sensor values. Defaults to :func:`ingest_raw_image`.
        sample_fn: Function that samples the sensor array and returns a byte stream
            of raw entropy. Defaults to :func:`sample_entropy_grid`.
        hash_fn: Function that hashes the entropy pool into a fixed-length, uniformly
            distributed byte string. Defaults to :func:`hash_entropy_pool`.

    Returns:
        A tuple of ``(seed, entropy_pool, assessment)`` where *seed* is the
        64-byte SHA3-512 digest, *entropy_pool* is the raw pre-hash bytes,
        and *assessment* is the :class:`EntropyAssessment`.
    """
    raw_image_data = ingest_fn(image_path)
    entropy_pool = sample_fn(raw_image_data)
    assessment = estimate_entropy(entropy_pool)
    secure_hash_bytes = hash_fn(entropy_pool)

    if not assessment.sufficient_for_seed:
        logger_msg = (
            "[generate] WARNING: measured min-entropy is %.1f bits but "
            "the SHA3-512 output is 512 bits.  The digest cannot contain "
            "more entropy than the source.  Use a dark-frame capture for "
            "high-value keys."
        )
        from src.logger import logger

        logger.warning(logger_msg, assessment.total_entropy_bits)

    return secure_hash_bytes, entropy_pool, assessment


def generate_true_random_number(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> bytes:
    """Extract physical entropy from a RAW image and return pure, raw bytes.

    Thin wrapper around :func:`generate_with_assessment` that returns only
    the 64-byte seed.  The entropy assessment is still computed internally
    (and logged) but discarded from the return value for backward
    compatibility.

    Args:
        image_path: Path to the RAW image file used as the entropy source.
        ingest_fn: Function that reads the RAW file and returns a 2D array of raw
            sensor values. Defaults to :func:`ingest_raw_image`.
        sample_fn: Function that samples the sensor array and returns a byte stream
            of raw entropy. Defaults to :func:`sample_entropy_grid`.
        hash_fn: Function that hashes the entropy pool into a fixed-length, uniformly
            distributed byte string. Defaults to :func:`hash_entropy_pool`.

    Returns:
        A fixed-length byte string of cryptographically secure entropy.
    """
    seed, _pool, _assessment = generate_with_assessment(image_path, ingest_fn, sample_fn, hash_fn)
    return seed
