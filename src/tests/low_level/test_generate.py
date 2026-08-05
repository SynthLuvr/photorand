"""Tests for src.low_level.generate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.low_level.generate import generate_true_random_number

# Determine the path to the test data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_IMAGE = str(DATA_DIR / "DSC02111.ARW")


def test_generate_true_random_number_basic() -> None:
    """Verify the high-level functional pipeline returns 64 bytes."""
    seed = generate_true_random_number(TEST_IMAGE)
    assert isinstance(seed, bytes)
    assert len(seed) == 64


def test_generate_true_random_number_deterministic() -> None:
    """Verify that same image produces same seed."""
    seed1 = generate_true_random_number(TEST_IMAGE)
    seed2 = generate_true_random_number(TEST_IMAGE)
    assert seed1 == seed2


def test_generate_true_random_number_custom_fns() -> None:
    """Verify the pipeline works with injected mock functions."""

    def mock_ingest(path: str) -> np.ndarray:
        return np.zeros((10, 10), dtype=np.uint16)

    def mock_sample(data: np.ndarray) -> bytes:
        assert isinstance(data, np.ndarray)
        return b"mock_entropy"

    def mock_hash(pool: bytes) -> bytes:
        assert pool == b"mock_entropy"
        return b"mock_seed"

    seed = generate_true_random_number(
        "dummy_path",
        ingest_fn=mock_ingest,
        sample_fn=mock_sample,
        hash_fn=mock_hash,
    )
    assert seed == b"mock_seed"
