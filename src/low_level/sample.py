"""Entropy sampling — isolate noisy LSBs from a sensor grid."""

from __future__ import annotations

import numpy as np

from src.logger import logger


def sample_entropy_grid(raw_sensor_data: np.ndarray, grid_spacing: int = 64) -> bytes:
    """Extract a deterministic grid of pixels, isolate noisy LSBs, and dump to bytes.

    Args:
        raw_sensor_data: The 2D array from ingest_fn.
        grid_spacing: The stride/step size. 64 means we grab 1 pixel out of every
            64x64 block.

    Returns:
        A byte string representing the entropy pool.
    """
    # 1. Slice the grid (excellent for reducing spatial correlation)
    sampled_matrix = raw_sensor_data[::grid_spacing, ::grid_spacing]

    # 2. Isolate the bottom 4 bits using a bitwise AND mask (0x0F is 00001111)
    # This turns a pixel value like 14_253 into just its bottom 4 noisy bits.
    lsb_matrix = sampled_matrix & 0x0F

    # 3. Cast to an 8-bit unsigned integer.
    # Since our max value is now 15, we don't need 16-bit memory slots.
    compact_matrix = lsb_matrix.astype(np.uint8)

    # 4. Flatten and dump straight to bytes
    entropy_bytes = compact_matrix.flatten().tobytes()

    logger.info(
        "[sample] Sampled grid at %dpx spacing, harvested %d bytes of dense entropy.",
        grid_spacing,
        len(entropy_bytes),
    )

    return entropy_bytes
