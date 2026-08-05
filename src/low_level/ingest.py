"""RAW image ingestion — extract unprocessed sensor data."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import rawpy

from src.logger import logger

if TYPE_CHECKING:
    import numpy as np


def ingest_raw_image(image_path: str) -> np.ndarray:
    """Read a RAW image file and extract the unprocessed, pure sensor data.

    Supports .ARW, .CR2, .NEF, .DNG, and most other RAW formats.

    Args:
        image_path: The path to the RAW file.

    Returns:
        A 2D array representing the raw light values hitting the sensor.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If the file is not a valid RAW image.
    """
    logger.info("[ingest] Reading raw sensor data from: %s", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    if not os.path.isfile(image_path):
        raise IsADirectoryError(f"Path is a directory, not a file: {image_path}")

    try:
        # We use a context manager to ensure the file is safely closed after reading.
        with rawpy.imread(image_path) as raw:
            # 'raw.raw_image' grabs the 2D array BEFORE any color processing or denoising.
            # We use .copy() so the data persists in memory after the file closes.
            raw_sensor_data: np.ndarray = raw.raw_image.copy()
    except Exception as e:
        raise ValueError(
            f"Not a valid RAW image file or unsupported format: {image_path} ({e})"
        ) from e

    logger.info("[ingest] Extracted sensor grid of shape: %s", raw_sensor_data.shape)

    return raw_sensor_data
