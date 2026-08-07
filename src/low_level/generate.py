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

# The ChaCha20 expander needs a 32-byte key + 16-byte nonce, so any seed
# shorter than 48 bytes cannot drive the CSPRNG in deterministic mode.  A weak
# seed that collapses below this size is refused rather than padded, forcing a
# longer capture instead of silently weakening the output.
_MIN_EXPANDABLE_SEED_BYTES = 48


def condition_entropy_pool(
    raw_image_data: np.ndarray,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
    *,
    min_entropy_bits: int = 512,
    allow_weak: bool = False,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Sample, assess, and condition a raw sensor array into a TRNG seed.

    Enforces NIST SP 800-90B: the entropy claimed in the output never exceeds
    the entropy measured in the source.

    Source-agnostic pipeline core: any ingest function yielding a 2-D sensor
    array funnels through here so the estimator and conditioner are applied
    identically everywhere.

    Args:
        raw_image_data: 2-D array of raw sensor values.
        sample_fn: Samples the array into a raw entropy byte stream.
        hash_fn: Compresses the pool into a uniformly distributed digest.
        min_entropy_bits: Entropy floor in bits.  Measured entropy below this
            is treated as weak.
        allow_weak: When ``True``, emit a weak seed truncated to the measured
            entropy bound instead of refusing.  The output never carries more
            bits than the source contains.  A weak seed too short
            (< 48 bytes) to seed the ChaCha20 expander is still refused.

    Returns:
        ``(seed, entropy_pool, assessment)`` where *seed* is the conditioned
        digest (64 bytes when the source is sufficient, otherwise truncated to
        ``max_seed_bytes``), *entropy_pool* is the raw pre-hash bytes, and
        *assessment* is the :class:`EntropyAssessment`.

    Raises:
        EntropyHealthError: If a startup health check fails.  A faulty source
            is always refused, even with ``allow_weak=True``.
        InsufficientEntropyError: If measured entropy is below the floor and
            ``allow_weak`` is ``False``, or a weak seed would be shorter than
            the ChaCha20 minimum.
    """
    entropy_pool = sample_fn(raw_image_data)
    assessment = estimate_entropy(entropy_pool)

    # Health checks are non-negotiable: a stuck-at or wildly biased source must
    # never produce output, regardless of the allow_weak override.
    if not assessment.passed_health_checks:
        raise EntropyHealthError(
            "Entropy health checks FAILED (repetition count / adaptive "
            "proportion). Refusing to condition a faulty source."
        )

    sufficient = assessment.total_entropy_bits >= min_entropy_bits

    if not sufficient and not allow_weak:
        raise InsufficientEntropyError(
            f"Measured entropy ({assessment.total_entropy_bits:.1f} bits) is "
            f"below the {min_entropy_bits}-bit floor. Refusing to emit more "
            "bits than the source contains. Re-capture with more data, or "
            "pass allow_weak=True to emit a seed truncated to the measured "
            "entropy bound."
        )

    max_bytes = assessment.max_seed_bytes

    if not sufficient:
        # allow_weak=True: a weak seed too short to drive the ChaCha20 expander
        # is refused rather than padded, forcing a longer capture.
        if max_bytes < _MIN_EXPANDABLE_SEED_BYTES:
            raise InsufficientEntropyError(
                f"Measured entropy ({assessment.total_entropy_bits:.1f} bits => "
                f"{max_bytes} bytes) is too short to seed the ChaCha20 expander "
                f"(needs >= {_MIN_EXPANDABLE_SEED_BYTES} bytes), even with "
                "allow_weak=True. Capture more entropy."
            )
        logger.warning(
            "[generate] EMITTING A WEAK SEED: measured %.1f bits is below the "
            "%d-bit floor. Output truncated to %d measured bytes so it never "
            "carries more bits than the source contains.",
            assessment.total_entropy_bits,
            min_entropy_bits,
            max_bytes,
        )

    # Never emit more bits than were measured: truncate the digest to
    # max_seed_bytes.  With the default 512-bit floor this is a no-op
    # (max_seed_bytes == 64), but it holds unconditionally.
    digest = hash_fn(entropy_pool)
    seed = digest[:max_bytes]
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
        min_entropy_bits: Entropy floor in bits enforced by the conditioner.
        allow_weak: When ``True``, emit a seed truncated to the measured
            entropy bound instead of refusing a weak source.

    Returns:
        ``(seed, entropy_pool, assessment)``.

    Raises:
        EntropyHealthError: If a startup health check fails.
        InsufficientEntropyError: If measured entropy is below the floor (and
            ``allow_weak`` is ``False`` or the seed would be too short).
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
    """Extract physical entropy from a RAW image and return pure, raw bytes.

    Thin wrapper around :func:`generate_with_assessment` that returns only
    the conditioned seed.

    Args:
        image_path: Path to the RAW image file.
        ingest_fn: Ingest function returning a 2-D sensor array.
        sample_fn: Sampler returning a raw entropy byte stream.
        hash_fn: Conditioning hash function.
        min_entropy_bits: Entropy floor in bits enforced by the conditioner.
        allow_weak: When ``True``, emit a truncated seed from a weak source.

    Returns:
        The conditioned seed bytes.

    Raises:
        EntropyHealthError: If a startup health check fails.
        InsufficientEntropyError: If measured entropy is below the floor.
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
