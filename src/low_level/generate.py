"""Functional pipeline — the master entropy extraction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


def generate_true_random_number(
    image_path: str,
    ingest_fn: Callable[[str], np.ndarray] = ingest_raw_image,
    sample_fn: Callable[[np.ndarray], bytes] = sample_entropy_grid,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> bytes:
    """Extract physical entropy from a RAW image and return pure, raw bytes.

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
    raw_image_data = ingest_fn(image_path)
    entropy_pool = sample_fn(raw_image_data)
    secure_hash_bytes = hash_fn(entropy_pool)

    return secure_hash_bytes
