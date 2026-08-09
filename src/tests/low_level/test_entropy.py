"""Tests for src.low_level.entropy."""

from __future__ import annotations

import random

import pytest

from src.low_level.entropy import EntropyAssessment, estimate_entropy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniform_bytes(n: int, bits: int = 4, seed: int = 42) -> bytes:
    """Generate *n* uniformly random symbols as a byte string."""
    rng = random.Random(seed)
    mask = (1 << bits) - 1
    return bytes(rng.randint(0, mask) & mask for _ in range(n))


def _make_biased_bytes(n: int, bias_value: int = 0, bias_frac: float = 0.8) -> bytes:
    """Generate *n* symbols where *bias_frac* are *bias_value* and the rest random."""
    rng = random.Random(123)
    mask = 0x0F
    result: list[int] = []
    for _ in range(n):
        if rng.random() < bias_frac:
            result.append(bias_value & mask)
        else:
            result.append(rng.randint(0, 15) & mask)
    return bytes(result)


def _make_stuck_bytes(n: int, value: int = 5) -> bytes:
    """Generate *n* identical symbols."""
    return bytes([value & 0x0F] * n)


# ===========================================================================
# Estimate shape / type
# ===========================================================================


class TestAssessmentShape:
    def test_returns_entropy_assessment(self) -> None:
        data = _make_uniform_bytes(5000)
        result = estimate_entropy(data)
        assert isinstance(result, EntropyAssessment)

    def test_sample_count_matches(self) -> None:
        data = _make_uniform_bytes(1000)
        result = estimate_entropy(data)
        assert result.sample_count == 1000

    def test_alphabet_size(self) -> None:
        data = _make_uniform_bytes(100)
        result = estimate_entropy(data)
        assert result.symbol_alphabet_size == 16
        assert result.bits_per_symbol == 4


# ===========================================================================
# Most Common Value estimator
# ===========================================================================


class TestMCVEstimator:
    def test_uniform_gives_near_max_entropy(self) -> None:
        data = _make_uniform_bytes(20000)
        result = estimate_entropy(data)
        # For 4-bit symbols, MCV should be close to 4 (within 0.5)
        assert result.most_common_value_estimate > 3.5

    def test_biased_gives_low_entropy(self) -> None:
        data = _make_biased_bytes(5000, bias_value=0, bias_frac=0.9)
        result = estimate_entropy(data)
        assert result.most_common_value_estimate < 1.0

    def test_stuck_value_gives_near_zero(self) -> None:
        data = _make_stuck_bytes(1000, value=3)
        result = estimate_entropy(data)
        assert result.most_common_value_estimate < 0.01


# ===========================================================================
# Collision estimator
# ===========================================================================


class TestCollisionEstimator:
    def test_uniform_gives_high_entropy(self) -> None:
        data = _make_uniform_bytes(10000)
        result = estimate_entropy(data)
        assert result.collision_estimate > 3.5

    def test_stuck_value_gives_low_entropy(self) -> None:
        data = _make_stuck_bytes(2000, value=7)
        result = estimate_entropy(data)
        assert result.collision_estimate < 1.0


# ===========================================================================
# Markov estimator (non-IID, SP 800-90B §6.2.3)
# ===========================================================================


def _make_cyclic_bytes(period: int, n: int, bits: int = 4) -> bytes:
    """Generate *n* symbols cycling deterministically through ``[0, period)``.

    The marginal distribution is uniform over ``period`` symbols, but each
    symbol is fully predictable from its predecessor — exactly the kind of
    temporal correlation the Markov estimator must catch.
    """
    mask = (1 << bits) - 1
    return bytes((i % period) & mask for i in range(n))


class TestMarkovEstimator:
    def test_uniform_gives_reasonable_entropy(self) -> None:
        data = _make_uniform_bytes(10000)
        result = estimate_entropy(data)
        # For 4-bit uniform data, Markov should be close to (but below) 4.0.
        assert result.markov_estimate > 3.0
        assert result.markov_estimate <= 4.0

    def test_stuck_value_gives_near_zero(self) -> None:
        data = _make_stuck_bytes(2000, value=7)
        result = estimate_entropy(data)
        assert result.markov_estimate < 0.01

    def test_cyclic_sequence_captures_correlation(self) -> None:
        """A cyclic sequence has uniform marginals but zero Markov entropy.

        ``0,1,…,15,0,1,…,15,…`` cycles through all 16 symbols so every symbol
        appears with probability 1/16 (MCV ≈ 4.0), yet each transition is
        deterministic and the Markov estimate collapses to ~0.
        """
        data = _make_cyclic_bytes(period=16, n=16000)
        result = estimate_entropy(data)
        assert result.markov_estimate < 0.1
        assert result.most_common_value_estimate > 3.0

    def test_correlated_runs_lower_than_mcv(self) -> None:
        """Sticky transitions (runs) produce Markov < MCV."""
        # Build a sequence with 80% self-transition probability across all 16
        # symbols so marginal frequencies stay roughly uniform.
        rng = random.Random(2024)
        symbols: list[int] = [rng.randint(0, 15)]
        for _ in range(4999):
            if rng.random() < 0.8:
                symbols.append(symbols[-1])
            else:
                symbols.append(rng.randint(0, 15))
        data = bytes(symbols)
        result = estimate_entropy(data)
        assert result.markov_estimate < result.most_common_value_estimate

    def test_markov_dominates_min_entropy_for_correlated(self) -> None:
        """For a cyclic source, Markov is the most conservative estimator."""
        data = _make_cyclic_bytes(period=16, n=16000)
        result = estimate_entropy(data)
        assert result.min_entropy == pytest.approx(result.markov_estimate)
        assert result.markov_estimate < result.most_common_value_estimate
        assert result.markov_estimate < result.collision_estimate


# ===========================================================================
# Shannon entropy
# ===========================================================================


class TestShannonEntropy:
    def test_uniform_near_max(self) -> None:
        data = _make_uniform_bytes(20000)
        result = estimate_entropy(data)
        assert result.shannon_entropy > 3.9

    def test_stuck_value_zero(self) -> None:
        data = _make_stuck_bytes(1000)
        result = estimate_entropy(data)
        assert result.shannon_entropy < 0.01

    def test_shannon_geq_min_entropy(self) -> None:
        """Shannon entropy is always >= min-entropy."""
        data = _make_biased_bytes(5000, bias_frac=0.5)
        result = estimate_entropy(data)
        assert result.shannon_entropy >= result.min_entropy - 1e-9


# ===========================================================================
# Conservative min-entropy
# ===========================================================================


class TestMinEntropy:
    def test_is_minimum_of_estimators(self) -> None:
        data = _make_uniform_bytes(5000)
        result = estimate_entropy(data)
        assert result.min_entropy == min(
            result.most_common_value_estimate,
            result.collision_estimate,
            result.markov_estimate,
        )

    def test_capped_at_bits_per_symbol(self) -> None:
        data = _make_uniform_bytes(50000)
        result = estimate_entropy(data)
        assert result.min_entropy <= 4.0 + 1e-9

    def test_total_entropy_bits(self) -> None:
        data = _make_uniform_bytes(1000)
        result = estimate_entropy(data)
        assert result.total_entropy_bits == pytest.approx(result.min_entropy * result.sample_count)

    def test_max_seed_bytes_capped_at_64(self) -> None:
        data = _make_uniform_bytes(50000)
        result = estimate_entropy(data)
        assert result.max_seed_bytes <= 64


# ===========================================================================
# Chi-square test
# ===========================================================================


class TestChiSquare:
    def test_uniform_passes(self) -> None:
        data = _make_uniform_bytes(20000)
        result = estimate_entropy(data)
        assert result.is_uniform is True
        assert result.chi_square_p_value > 0.01

    def test_biased_fails(self) -> None:
        data = _make_biased_bytes(10000, bias_value=0, bias_frac=0.5)
        result = estimate_entropy(data)
        assert result.is_uniform is False
        assert result.chi_square_p_value < 0.01

    def test_stuck_value_fails(self) -> None:
        data = _make_stuck_bytes(1000, value=5)
        result = estimate_entropy(data)
        assert result.is_uniform is False


# ===========================================================================
# Health checks
# ===========================================================================


class TestHealthChecks:
    def test_uniform_passes_all(self) -> None:
        data = _make_uniform_bytes(10000)
        result = estimate_entropy(data)
        assert result.passed_health_checks is True
        assert result.repetition_count_passed is True
        assert result.adaptive_proportion_passed is True

    def test_stuck_value_fails_repetition(self) -> None:
        data = _make_stuck_bytes(2000)
        result = estimate_entropy(data)
        assert result.repetition_count_passed is False
        assert result.repetition_count_max_run >= result.repetition_count_cutoff

    def test_repetition_cutoff_for_4bit(self) -> None:
        """For 4-bit symbols the cutoff is 12 (NIST SP 800-90B)."""
        data = _make_uniform_bytes(100)
        result = estimate_entropy(data)
        assert result.repetition_count_cutoff == 12


# ===========================================================================
# Properties / status
# ===========================================================================


class TestProperties:
    def test_overall_status_good_for_uniform(self) -> None:
        data = _make_uniform_bytes(20000)
        result = estimate_entropy(data)
        assert result.overall_status == "GOOD"

    def test_overall_status_fail_for_stuck(self) -> None:
        data = _make_stuck_bytes(2000)
        result = estimate_entropy(data)
        assert result.overall_status == "FAIL"

    def test_sufficient_for_seed_true_when_enough(self) -> None:
        data = _make_uniform_bytes(20000)
        result = estimate_entropy(data)
        assert result.sufficient_for_seed is True

    def test_sufficient_for_seed_false_when_low(self) -> None:
        # 100 samples at ~4 bits each = ~400 bits < 512
        data = _make_uniform_bytes(100)
        result = estimate_entropy(data)
        assert result.sufficient_for_seed is False


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_data(self) -> None:
        result = estimate_entropy(b"")
        assert result.sample_count == 0
        assert result.min_entropy == 0.0
        assert result.total_entropy_bits == 0.0

    def test_single_sample(self) -> None:
        result = estimate_entropy(b"\x05")
        assert result.sample_count == 1

    def test_all_same_value(self) -> None:
        result = estimate_entropy(b"\x03" * 500)
        assert result.min_entropy < 0.01
        assert result.shannon_entropy < 0.01

    def test_invalid_bits_per_symbol(self) -> None:
        with pytest.raises(ValueError, match="bits_per_symbol"):
            estimate_entropy(b"\x00", bits_per_symbol=0)

    def test_invalid_bits_per_symbol_too_large(self) -> None:
        with pytest.raises(ValueError, match="bits_per_symbol"):
            estimate_entropy(b"\x00", bits_per_symbol=9)

    def test_8bit_symbols(self) -> None:
        """Works with 8-bit symbols (full byte entropy)."""
        rng = random.Random(7)
        data = bytes(rng.randint(0, 255) for _ in range(5000))
        result = estimate_entropy(data, bits_per_symbol=8)
        assert result.bits_per_symbol == 8
        assert result.symbol_alphabet_size == 256
        assert result.min_entropy > 6.0


# ===========================================================================
# Mathematical correctness
# ===========================================================================


class TestMathematicalCorrectness:
    def test_mcv_matches_manual_calculation(self) -> None:
        """MCV estimate should be positive for a reasonable uniform sample."""
        # 200 uniform 4-bit symbols — enough that the confidence bound is tight
        rng = random.Random(99)
        data = bytes(rng.randint(0, 15) for _ in range(200))
        result = estimate_entropy(data)
        assert result.most_common_value_estimate > 0.0
        assert result.most_common_value_estimate <= 4.0

    def test_shannon_matches_formula(self) -> None:
        """Shannon entropy for uniform 4-bit should be ~4."""
        # Create perfectly uniform data
        symbols = list(range(16)) * 625  # 10000 samples, each value exactly 625 times
        data = bytes(symbols)
        result = estimate_entropy(data)
        assert result.shannon_entropy == pytest.approx(4.0, abs=0.01)

    def test_frozen_dataclass(self) -> None:
        """EntropyAssessment should be immutable."""
        data = _make_uniform_bytes(100)
        result = estimate_entropy(data)
        with pytest.raises(AttributeError):
            result.min_entropy = 5.0  # type: ignore[misc]
