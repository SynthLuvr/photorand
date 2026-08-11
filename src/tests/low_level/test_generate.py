"""Tests for src.low_level.generate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.low_level import generate
from src.low_level.entropy import (
    EntropyAssessment,
    EntropyHealthError,
    InsufficientEntropyError,
)
from src.low_level.generate import (
    condition_entropy_pool,
    generate_true_random_number,
    generate_with_assessment,
)
from src.tests.conftest import requires_raw_data

# Determine the path to the test data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_IMAGE = str(DATA_DIR / "DSC02111.ARW")


def _assessment(
    *,
    total_entropy_bits: float = 600.0,
    max_seed_bytes: int = 64,
    repetition_count_passed: bool = True,
    adaptive_proportion_passed: bool = True,
) -> EntropyAssessment:
    """Build an EntropyAssessment with controlled gate-relevant fields."""
    return EntropyAssessment(
        sample_count=256,
        bits_per_symbol=4,
        symbol_alphabet_size=16,
        most_common_value_estimate=2.4,
        collision_estimate=2.6,
        markov_estimate=2.5,
        shannon_entropy=2.9,
        min_entropy=2.4,
        total_entropy_bits=total_entropy_bits,
        max_seed_bytes=max_seed_bytes,
        chi_square_statistic=12.0,
        chi_square_p_value=0.6,
        is_uniform=True,
        repetition_count_passed=repetition_count_passed,
        repetition_count_max_run=4,
        repetition_count_cutoff=12,
        adaptive_proportion_passed=adaptive_proportion_passed,
        adaptive_proportion_max_count=40,
        adaptive_proportion_cutoff=59,
        adaptive_proportion_windows=1,
    )


# 280 bytes; the content is irrelevant once the estimate is mocked.
_POOL = b"entropy" * 40
_DATA = np.zeros((10, 10), dtype=np.uint16)


def _sample(_data: np.ndarray) -> bytes:
    """Sample stub returning a fixed pool (the estimate is mocked)."""
    return _POOL


def _ingest(_path: str) -> np.ndarray:
    """Ingest stub returning a fixed array (the estimate is mocked)."""
    return _DATA


def _est_sufficient(_data: bytes) -> EntropyAssessment:
    return _assessment()


def _est_low_truncatable(_data: bytes) -> EntropyAssessment:
    return _assessment(total_entropy_bits=400.0, max_seed_bytes=50)


def _est_health_fail(_data: bytes) -> EntropyAssessment:
    return _assessment(repetition_count_passed=False)


def _est_very_low(_data: bytes) -> EntropyAssessment:
    return _assessment(total_entropy_bits=100.0, max_seed_bytes=12)


@requires_raw_data
def test_generate_true_random_number_basic() -> None:
    """Verify the high-level functional pipeline returns 64 bytes."""
    seed = generate_true_random_number(TEST_IMAGE)
    assert isinstance(seed, bytes)
    assert len(seed) == 64


@requires_raw_data
def test_generate_true_random_number_deterministic() -> None:
    """Verify that same image produces same seed."""
    seed1 = generate_true_random_number(TEST_IMAGE)
    seed2 = generate_true_random_number(TEST_IMAGE)
    assert seed1 == seed2


def test_generate_true_random_number_custom_fns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the pipeline works with injected mock functions."""
    monkeypatch.setattr(generate, "estimate_entropy", _est_sufficient)

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


class TestConditionEntropyPool:
    """The shared funnel enforces the NIST SP 800-90B entropy floor."""

    def test_sufficient_source_emits_full_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(generate, "estimate_entropy", _est_sufficient)
        seed, _pool, _assessment_out = condition_entropy_pool(_DATA, sample_fn=_sample)
        assert len(seed) == 64

    def test_refuses_low_entropy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(generate, "estimate_entropy", _est_low_truncatable)
        with pytest.raises(InsufficientEntropyError, match="below the 512-bit floor"):
            condition_entropy_pool(_DATA, sample_fn=_sample)

    def test_failed_health_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(generate, "estimate_entropy", _est_health_fail)
        with pytest.raises(EntropyHealthError, match="health checks FAILED"):
            condition_entropy_pool(_DATA, sample_fn=_sample)

    def test_min_entropy_bits_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 400 measured bits is insufficient at the default floor but sufficient
        # at a lowered floor; output is still capped at the measured bound.
        monkeypatch.setattr(generate, "estimate_entropy", _est_low_truncatable)
        seed, _pool, _assessment_out = condition_entropy_pool(
            _DATA, sample_fn=_sample, min_entropy_bits=300
        )
        assert len(seed) == 50


class TestGenerateWithAssessment:
    """The RAW ingest path inherits the funnel's entropy gate."""

    def test_refuses_low_entropy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(generate, "estimate_entropy", _est_very_low)
        with pytest.raises(InsufficientEntropyError, match="below the 512-bit floor"):
            generate_with_assessment("fake.arw", ingest_fn=_ingest, sample_fn=_sample)
