"""Tests for src.low_level.sample."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    import numpy as np


def test_sample_entropy_grid_type(mock_image_data: np.ndarray) -> None:
    result = sample_entropy_grid(mock_image_data, grid_spacing=10)
    assert isinstance(result, bytes)


def test_sample_entropy_grid_length(mock_image_data: np.ndarray) -> None:
    # A 100x100 grid sampled every 10 pixels is a 10x10 grid (100 pixels).
    # LSBs are masked to 4 bits and cast to uint8, so each pixel becomes 1 byte -> 100 bytes.
    result = sample_entropy_grid(mock_image_data, grid_spacing=10)
    assert len(result) == 100


def test_sample_entropy_grid_deterministic(mock_image_data: np.ndarray) -> None:
    result1 = sample_entropy_grid(mock_image_data, grid_spacing=5)
    result2 = sample_entropy_grid(mock_image_data, grid_spacing=5)
    assert result1 == result2
