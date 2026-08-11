"""Tests for src.low_level.sample — grid sampling and FPN reduction."""

from __future__ import annotations

import numpy as np
import pytest

from src.low_level.sample import (
    DegenerateEntropyPoolError,
    reduce_fixed_pattern_noise,
    sample_entropy_grid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def noisy_sensor_data() -> np.ndarray:
    """A grid with DC offset + stochastic noise that survives FPN reduction.

    Unlike the perfectly separable ``mock_image_data``, this has independently
    random per-pixel noise — the realistic case the sampler is designed for.
    """
    rng = np.random.default_rng(42)
    base = np.full((100, 100), 2048, dtype=np.uint16)
    return base + rng.integers(0, 16, size=(100, 100)).astype(np.uint16)


# ---------------------------------------------------------------------------
# Legacy grid-sampling behaviour
# ---------------------------------------------------------------------------


class TestSampleEntropyGrid:
    def test_returns_bytes(self, mock_image_data: np.ndarray) -> None:
        # FPN reduction zeroes the perfectly separable mock grid, so use legacy mode.
        result = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=False)
        assert isinstance(result, bytes)

    def test_length(self, mock_image_data: np.ndarray) -> None:
        # A 100x100 grid sampled every 10 pixels is a 10x10 grid (100 pixels).
        result = sample_entropy_grid(mock_image_data, grid_spacing=10, reduce_fpn=False)
        assert len(result) == 100

    def test_deterministic(self, mock_image_data: np.ndarray) -> None:
        result1 = sample_entropy_grid(mock_image_data, grid_spacing=5, reduce_fpn=False)
        result2 = sample_entropy_grid(mock_image_data, grid_spacing=5, reduce_fpn=False)
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
    def test_fpn_on_by_default(self, noisy_sensor_data: np.ndarray) -> None:
        """FPN reduction should be the default behaviour."""
        result_default = sample_entropy_grid(noisy_sensor_data, grid_spacing=10)
        result_explicit = sample_entropy_grid(noisy_sensor_data, grid_spacing=10, reduce_fpn=True)
        assert result_default == result_explicit

    def test_fpn_changes_output(self, noisy_sensor_data: np.ndarray) -> None:
        """FPN reduction should produce different output than legacy mode."""
        with_fpn = sample_entropy_grid(noisy_sensor_data, grid_spacing=10, reduce_fpn=True)
        without_fpn = sample_entropy_grid(noisy_sensor_data, grid_spacing=10, reduce_fpn=False)
        assert with_fpn != without_fpn

    def test_fpn_output_is_valid_range(self, noisy_sensor_data: np.ndarray) -> None:
        """All output bytes should be in [0, 15] (4-bit symbols)."""
        result = sample_entropy_grid(noisy_sensor_data, grid_spacing=10, reduce_fpn=True)
        assert all(b <= 15 for b in result)


# ---------------------------------------------------------------------------
# Degenerate pool rejection (all-same-symbol)
# ---------------------------------------------------------------------------


class TestDegeneratePoolRejection:
    """The sampler rejects zero-entropy pools before conditioning."""

    def test_uniform_input_no_fpn_rejected(self) -> None:
        """A flat sensor reading collapses to one symbol — rejected."""
        flat = np.full((100, 100), 1000, dtype=np.uint16)
        with pytest.raises(DegenerateEntropyPoolError, match="degenerate pool"):
            sample_entropy_grid(flat, grid_spacing=10, reduce_fpn=False)

    def test_uniform_input_with_fpn_rejected(self) -> None:
        """FPN reduction of a flat grid zeros the residual — rejected."""
        flat = np.full((100, 100), 2048, dtype=np.uint16)
        with pytest.raises(DegenerateEntropyPoolError, match="degenerate pool"):
            sample_entropy_grid(flat, grid_spacing=10, reduce_fpn=True)

    def test_single_pixel_grid_rejected(self) -> None:
        """A grid spacing larger than the array yields a single symbol — rejected."""
        tiny = np.array([[42]], dtype=np.uint16)
        with pytest.raises(DegenerateEntropyPoolError):
            sample_entropy_grid(tiny, grid_spacing=64, reduce_fpn=False)

    def test_two_symbol_pool_not_rejected(self) -> None:
        """Two distinct symbols is not degenerate — just biased."""
        data = np.zeros((20, 20), dtype=np.uint16)
        data[10:] = 7  # bottom half is 7, top half is 0
        result = sample_entropy_grid(data, grid_spacing=2, reduce_fpn=False)
        assert len(set(result)) == 2

    def test_healthy_pool_not_rejected(self, noisy_sensor_data: np.ndarray) -> None:
        """A varied sensor grid passes the degeneracy check."""
        result = sample_entropy_grid(noisy_sensor_data, grid_spacing=10)
        assert isinstance(result, bytes)
        assert len(set(result)) >= 2

    def test_error_is_runtime_error(self) -> None:
        """DegenerateEntropyPoolError is a RuntimeError for CLI catch-all safety."""
        flat = np.full((50, 50), 7, dtype=np.uint16)
        with pytest.raises(RuntimeError):
            sample_entropy_grid(flat, grid_spacing=5, reduce_fpn=False)
