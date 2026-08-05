"""Tests for src.low_level.ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.low_level.ingest import ingest_raw_image


def test_ingest_real_image(example_image_path: str) -> None:
    # Ensure the file actually exists where we expect it
    assert Path(example_image_path).exists()

    # Run the ingestion
    result = ingest_raw_image(example_image_path)

    # Assertions
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2  # Should be a 2D sensor grid
    assert result.size > 0  # Should have actual data


def test_ingest_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        ingest_raw_image("path/to/nonexistent/image.ARW")


def test_ingest_directory() -> None:
    with pytest.raises(IsADirectoryError):
        ingest_raw_image("src")


def test_ingest_invalid_raw() -> None:
    # Use a file that exists but is not a RAW image
    with pytest.raises(ValueError, match="Not a valid RAW image file"):
        ingest_raw_image("README.md")
