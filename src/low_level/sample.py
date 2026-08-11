"""Entropy sampling — isolate noisy LSBs from a sensor grid."""

from __future__ import annotations

import numpy as np

from src.logger import logger

# Number of bits extracted per sampled pixel (the low N bits of the noise).
_BITS_PER_SYMBOL = 4

# A non-degenerate pool must contain at least this many distinct symbols.
# Below it every byte is identical — zero usable entropy — so the pool is
# rejected immediately rather than relying on downstream estimation to notice.
_MIN_DISTINCT_SYMBOLS = 2


class DegenerateEntropyPoolError(RuntimeError):
    """Raised when the LSB extraction yields a degenerate (zero-entropy) pool.

    Every symbol collapses to a single value, which means the sensor produced
    no usable noise: a stuck-at source, a fully saturated/flat capture, or an
    over-aggressive FPN reduction that zeroed the entire grid.  This is caught
    at the sample stage — before conditioning — so the failure is obvious and
    early rather than silently hashed into a uniform-looking but empty digest.
    """


def reduce_fixed_pattern_noise(sampled_matrix: np.ndarray) -> np.ndarray:
    """Remove additive fixed-pattern noise (FPN) from a sampled sensor grid.

    FPN is a deterministic per-sensor signature (DSNU, PRNU) that
    reproduces across captures.  Subtracting row and column medians
    removes the dominant additive components, leaving the stochastic
    noise floor that constitutes genuine entropy.

    Args:
        sampled_matrix: A 2-D array of raw pixel values sampled from the
            sensor grid.

    Returns:
        A float64 residual matrix with row and column bias removed.
    """
    residual = sampled_matrix.astype(np.float64)
    residual -= np.median(residual, axis=1, keepdims=True)
    residual -= np.median(residual, axis=0, keepdims=True)
    return residual


def _validate_extraction(compact_matrix: np.ndarray, bits_per_symbol: int) -> None:
    """Reject a degenerate pool whose LSB extraction carries no usable noise.

    After masking the low *bits_per_symbol* bits, every value in a healthy
    noise floor should vary.  If the entire pool collapses to a single symbol
    (or is empty) the source produced no entropy at all — a stuck sensor, a
    flat/saturated capture, or FPN reduction that zeroed the grid.  Rejecting
    here, before conditioning, prevents a uniform-looking-but-empty digest.

    Args:
        compact_matrix: The uint8 symbol matrix just before flattening.
        bits_per_symbol: Bit-width of each extracted symbol.

    Raises:
        DegenerateEntropyPoolError: If fewer than ``_MIN_DISTINCT_SYMBOLS``
            distinct values are present.
    """
    alphabet_size = 1 << bits_per_symbol
    distinct_count = int(np.unique(compact_matrix).size)

    if distinct_count < _MIN_DISTINCT_SYMBOLS:
        raise DegenerateEntropyPoolError(
            f"The {bits_per_symbol}-LSB extraction produced a degenerate pool: "
            f"only {distinct_count} distinct symbol(s) of a {alphabet_size}-symbol "
            f"alphabet. The sensor produced no usable noise — the source is faulty "
            f"or the capture contains no entropy."
        )

    logger.info("[sample] Symbol diversity: %d/%d distinct values.", distinct_count, alphabet_size)


def sample_entropy_grid(
    raw_sensor_data: np.ndarray,
    grid_spacing: int = 64,
    *,
    reduce_fpn: bool = True,
) -> bytes:
    """Extract a deterministic grid of pixels, isolate noisy LSBs, and dump to bytes.

    Args:
        raw_sensor_data: The 2D array from ingest_fn.
        grid_spacing: The stride/step size. 64 means we grab 1 pixel out of every
            64x64 block.
        reduce_fpn: When ``True`` (default), remove additive fixed-pattern
            noise before extracting LSBs.  Set to ``False`` for legacy behaviour.

    Returns:
        A byte string representing the entropy pool.

    Raises:
        DegenerateEntropyPoolError: If the LSB extraction collapses to a single
            symbol (all-same) — the sensor produced no usable noise.
    """
    # 1. Slice the grid (excellent for reducing spatial correlation)
    sampled_matrix = raw_sensor_data[::grid_spacing, ::grid_spacing]

    if reduce_fpn:
        residual = reduce_fixed_pattern_noise(sampled_matrix)
        # Absolute value keeps the noise magnitude unsigned; the low 4
        # bits of that magnitude carry the unpredictable content.
        lsb_source = np.abs(residual)
    else:
        lsb_source = sampled_matrix.astype(np.float64)

    # 2. Isolate the bottom 4 bits using a bitwise AND mask (0x0F is 00001111)
    # This turns a pixel value like 14_253 into just its bottom 4 noisy bits.
    lsb_mask = (1 << _BITS_PER_SYMBOL) - 1
    lsb_matrix = lsb_source.astype(np.uint16) & lsb_mask

    # 3. Cast to an 8-bit unsigned integer.
    # Since our max value is now 15, we don't need 16-bit memory slots.
    compact_matrix = lsb_matrix.astype(np.uint8)

    # 4. Validate the extraction: a degenerate (all-same-symbol) pool means
    #    the sensor produced no usable noise.  Reject before conditioning so
    #    an empty source never becomes a uniform-looking digest.
    _validate_extraction(compact_matrix, _BITS_PER_SYMBOL)

    # 5. Flatten and dump straight to bytes
    entropy_bytes = compact_matrix.flatten().tobytes()

    logger.info(
        "[sample] Sampled grid at %dpx spacing (FPN reduction: %s), "
        "harvested %d bytes of dense entropy.",
        grid_spacing,
        "on" if reduce_fpn else "off",
        len(entropy_bytes),
    )

    return entropy_bytes
