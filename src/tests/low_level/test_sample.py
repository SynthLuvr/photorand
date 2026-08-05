"""Tests for src.low_level.sample — grid sampling and FPN reduction."""

from __future__ import annotations

import numpy as np
import pytest

from src.low_level.sample import reduce_fixed_pattern_noise, sample_entropy_grid

# ---------------------------------------------------------------------------
# Legacy grid-sampling behaviour
# ---------------------------------------------------------------------------


class TestSampleEntropyGrid:
    def test_returns_bytes(self, mock_image_data: np.ndarray) -> None:
        result = sample_entropy_grid(mock_image_data, grid_spacing=10)
        assert isinstance(result, bytes)

    def test_length(self, mock_image_data: np.ndarray) -> None:
        # A 100x100 grid sampled every 10 pixels is a 10x10 grid (100 pixels).
        result = sample_entropy_grid(mock_image_data, grid_spacing=10)
        assert len(result) == 100

    def test_deterministic(self, mock_image_data: np.ndarray) -> None:
        result1 = sample_entropy_grid(mock_image_data, grid_spacing=5)
        result2 = sample_entropy_grid(mock_image_data, grid_spacing=5)
        assert result1 == result2

    def test_no_fpn_matches_legacy(self, mock_image_data: np.ndarray) -> None:
        """With reduce_fpn=False, output should match the original algorithm."""
        sampled = mock_image_data[::10, ::10]
        expected = (sampled & 0x0F).astype(np.uint8).flatten().tobytes()
        result = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=False)
        assert result == expected


# ---------------------------------------------------------------------------
# Fixed-pattern noise reduction
# ---------------------------------------------------------------------------


class TestReduceFixedPatternNoise:
    def test_removes_constant_offset(self) -> None:
        """A uniform offset should be completely removed."""
        matrix = np.full((20, 20), 1000, dtype=np.uint16)
        residual = reduce_fixed_pattern_noise(matrix)
        assert np.all(residual == pytest.approx(0.0))

    def test_removes_row_bias(self) -> None:
        """Per-row offsets should be removed."""
        base = np.full((20, 20), 500, dtype=np.uint16)
        for i in range(20):
            base[i] += i * 10  # each row has a different offset
        residual = reduce_fixed_pattern_noise(base)
        row_medians = np.median(residual, axis=1)
        assert np.allclose(row_medians, 0.0)

    def test_removes_column_bias(self) -> None:
        """Per-column offsets should be removed."""
        base = np.full((20, 20), 500, dtype=np.uint16)
        for j in range(20):
            base[:, j] += j * 10  # each column has a different offset
        residual = reduce_fixed_pattern_noise(base)
        col_medians = np.median(residual, axis=0)
        assert np.allclose(col_medians, 0.0)

    def test_preserves_stochastic_noise(self) -> None:
        """Random noise should be preserved (not zeroed out)."""
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 5, size=(50, 50)).astype(np.uint16)
        base = np.full((50, 50), 2048, dtype=np.uint16) + noise
        residual = reduce_fixed_pattern_noise(base)
        # The residual should not be all zeros (noise is preserved)
        assert np.count_nonzero(residual) > 0
        # And the magnitude should be similar to the noise magnitude
        assert np.max(np.abs(residual)) > 2


class TestFPNIntegration:
    def test_fpn_on_by_default(self, mock_image_data: np.ndarray) -> None:
        """FPN reduction should be the default behaviour."""
        result_default = sample_entropy_grid(mock_image_data, grid_spacing=10)
        result_explicit = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=True)
        assert result_default == result_explicit

    def test_fpn_changes_output(self, mock_image_data: np.ndarray) -> None:
        """FPN reduction should produce different output than legacy mode."""
        with_fpn = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=True)
        without_fpn = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=False)
        assert with_fpn != without_fpn

    def test_fpn_output_is_valid_range(self, mock_image_data: np.ndarray) -> None:
        """All output bytes should be in [0, 15] (4-bit symbols)."""
        result = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=True)
        assert all(b <= 15 for b in result)
