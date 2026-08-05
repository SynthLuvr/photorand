"""Entropy sampling — isolate noisy LSBs from a sensor grid."""

from __future__ import annotations

import numpy as np

from src.logger import logger


def reduce_fixed_pattern_noise(sampled_matrix: np.ndarray) -> np.ndarray:
    """Remove additive fixed-pattern noise (FPN) from a sampled sensor grid.

    Real camera sensors exhibit **fixed-pattern noise** — dark-signal
    non-uniformity (DSNU), row/column offsets, and pixel-response
    non-uniformity (PRNU).  This noise is a *deterministic* per-sensor
    signature: it reproduces identically across captures of the same
    camera.  Without removal, these deterministic bits masquerade as
    entropy when in fact they are predictable.

    This function subtracts the row-wise and column-wise medians, which
    removes the dominant *additive* FPN components (horizontal/vertical
    banding, dark-row/dark-column offsets).  The residual that remains is
    dominated by the stochastic noise floor — shot noise, read noise, and
    thermal noise — which is genuine entropy.

    Args:
        sampled_matrix: A 2-D array of raw pixel values sampled from the
            sensor grid.

    Returns:
        A float64 residual matrix with row and column bias removed.
    """
    residual = sampled_matrix.astype(np.float64)

    # Remove row bias (horizontal banding, dark-column offsets).
    row_medians = np.median(residual, axis=1, keepdims=True)
    residual -= row_medians

    # Remove column bias (vertical banding, dark-row offsets).
    col_medians = np.median(residual, axis=0, keepdims=True)
    residual -= col_medians

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
        reduce_fpn: When ``True`` (default), subtract row and column medians to
            remove additive fixed-pattern noise — the deterministic per-sensor
            fingerprint — before extracting LSBs.  This ensures the sampled
            pool contains *stochastic* noise rather than a reproducible
            sensor signature.  Set to ``False`` to reproduce legacy behaviour.

    Returns:
        A byte string representing the entropy pool.
    """
    # 1. Slice the grid (excellent for reducing spatial correlation)
    sampled_matrix = raw_sensor_data[::grid_spacing, ::grid_spacing]

    if reduce_fpn:
        # Remove the deterministic fixed-pattern component so that only
        # the stochastic noise floor feeds into the LSB extraction.
        residual = reduce_fixed_pattern_noise(sampled_matrix)
        # Absolute value keeps the noise magnitude unsigned; the low 4
        # bits of that magnitude carry the unpredictable content.
        lsb_source = np.abs(residual)
    else:
        # Legacy mode: operate directly on the raw pixel values.
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
