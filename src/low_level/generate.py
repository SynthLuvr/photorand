"""Functional pipeline — the master entropy extraction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.logger import logger
from src.low_level.entropy import (
    EntropyAssessment,
    EntropyHealthError,
    InsufficientEntropyError,
    estimate_entropy,
)
from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

# ChaCha20 needs a 32-byte key + 16-byte nonce; a seed shorter than 48 bytes
# can't drive the expander, so a weak seed below this is refused (not padded).
_MIN_EXPANDABLE_SEED_BYTES = 48


def _enforce_floor(
    assessment: EntropyAssessment,
    min_entropy_bits: int,
    allow_weak: bool,
) -> None:
    """Raise if the source is too faulty or too weak to emit a seed.

    Health-check failures are never overridable.  Below the floor the source is
    refused unless *allow_weak* is set, in which case the output is truncated to
    the measured entropy bound (but still refused if too short for ChaCha20).
    """
    if not assessment.passed_health_checks:
        raise EntropyHealthError(
            "Entropy health checks FAILED (repetition count / adaptive "
            "proportion). A faulty source is never conditioned."
        )

    if assessment.total_entropy_bits >= min_entropy_bits:
        return

    if not allow_weak:
        raise InsufficientEntropyError(
            f"Measured entropy ({assessment.total_entropy_bits:.1f} bits) is below "
            f"the {min_entropy_bits}-bit floor. Re-capture with more data, or use "
            "allow_weak=True to emit a truncated seed."
        )

    if assessment.max_seed_bytes < _MIN_EXPANDABLE_SEED_BYTES:
        raise InsufficientEntropyError(
            f"Measured entropy ({assessment.total_entropy_bits:.1f} bits => "
            f"{assessment.max_seed_bytes} bytes) is too short to seed the ChaCha20 "
            f"expander (needs >= {_MIN_EXPANDABLE_SEED_BYTES} bytes). Capture more entropy."
        )

    logger.warning(
        "[generate] Emitting a weak seed: %.1f bits is below the %d-bit floor; "
        "output truncated to %d measured bytes.",
        assessment.total_entropy_bits,
        min_entropy_bits,
        assessment.max_seed_bytes,
    )


def condition_entropy_pool(
    raw_image_data: np.ndarray,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
    *,
    min_entropy_bits: int = 512,
    allow_weak: bool = False,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Sample, assess, and condition a raw sensor array into a TRNG seed.

    Enforces NIST SP 800-90B: the output never claims more entropy than the
    source contains.  Any ingest function yielding a 2-D sensor array funnels
    through here so the estimator and conditioner are applied identically.

    Args:
        raw_image_data: 2-D array of raw sensor values.
        sample_fn: Samples the array into a raw entropy byte stream.
        hash_fn: Compresses the pool into a uniformly distributed digest.
        min_entropy_bits: Entropy floor in bits.
        allow_weak: Emit a truncated seed below the floor instead of refusing.

    Returns:
        ``(seed, entropy_pool, assessment)``.  *seed* is the conditioned digest,
        truncated to ``max_seed_bytes`` (64 when sufficient).

    Raises:
        EntropyHealthError: If a startup health check fails (never overridable).
        InsufficientEntropyError: If below the floor and ``allow_weak`` is off,
            or a weak seed would be too short for ChaCha20.
    """
    entropy_pool = sample_fn(raw_image_data)
    assessment = estimate_entropy(entropy_pool)
    _enforce_floor(assessment, min_entropy_bits, allow_weak)
    seed = hash_fn(entropy_pool)[: assessment.max_seed_bytes]
    return seed, entropy_pool, assessment


def generate_with_assessment(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
    *,
    min_entropy_bits: int = 512,
    allow_weak: bool = False,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Extract entropy from a RAW image and measure its quality.

    Runs the full pipeline with NIST SP 800-90B entropy estimation applied
    to the raw pool before SHA3-512 conditioning, enforcing the measured
    entropy floor.

    Args:
        image_path: Path to the RAW image file used as the entropy source.
        ingest_fn: Reads the RAW file and returns a 2D array of raw sensor
            values. Defaults to :func:`ingest_raw_image`.
        sample_fn: Samples the sensor array and returns a byte stream of
            raw entropy. Defaults to :func:`sample_entropy_grid`.
        hash_fn: Hashes the entropy pool into a fixed-length, uniformly
            distributed byte string. Defaults to :func:`hash_entropy_pool`.
        min_entropy_bits: Entropy floor in bits.
        allow_weak: Emit a truncated seed below the floor instead of refusing.

    Returns:
        ``(seed, entropy_pool, assessment)``.

    Raises:
        EntropyHealthError: If a startup health check fails.
        InsufficientEntropyError: If measured entropy is below the floor (and
            ``allow_weak`` is off or the seed would be too short).
    """
    raw_image_data = ingest_fn(image_path)
    return condition_entropy_pool(
        raw_image_data,
        sample_fn=sample_fn,
        hash_fn=hash_fn,
        min_entropy_bits=min_entropy_bits,
        allow_weak=allow_weak,
    )


def generate_true_random_number(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
    *,
    min_entropy_bits: int = 512,
    allow_weak: bool = False,
) -> bytes:
    """Extract physical entropy from a RAW image and return the conditioned seed.

    Thin wrapper around :func:`generate_with_assessment` (see it for parameter
    and exception details) that discards the pool and assessment.
    """
    seed, _pool, _assessment = generate_with_assessment(
        image_path,
        ingest_fn,
        sample_fn,
        hash_fn,
        min_entropy_bits=min_entropy_bits,
        allow_weak=allow_weak,
    )
    return seed
