"""Entropy sampling — isolate noisy LSBs from a sensor grid."""

from __future__ import annotations

import numpy as np

from src.logger import logger


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
    lsb_matrix = lsb_source.astype(np.uint16) & 0x0F

    # 3. Cast to an 8-bit unsigned integer.
    # Since our max value is now 15, we don't need 16-bit memory slots.
    compact_matrix = lsb_matrix.astype(np.uint8)

    # 4. Flatten and dump straight to bytes
    entropy_bytes = compact_matrix.flatten().tobytes()

    logger.info(
        "[sample] Sampled grid at %dpx spacing (FPN reduction: %s), "
        "harvested %d bytes of dense entropy.",
        grid_spacing,
        "on" if reduce_fpn else "off",
        len(entropy_bytes),
    )

    return entropy_bytes
