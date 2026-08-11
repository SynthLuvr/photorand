"""Tests for src.low_level.drbg.HMACDRBG (NIST SP 800-90A §10.1.2).

Strategy:
  - Use a fixed 64-byte seed for determinism.
  - Verify output shape, determinism, and basic statistical sanity.
  - Verify backtracking resistance (post-generate state update).
  - Verify reseeding changes the output stream.
  - Verify prediction resistance adds non-determinism.
  - Verify long-stream output passes continuous health checks.
"""

from __future__ import annotations

import os

import pytest

from src.low_level.drbg import HMACDRBG, MAX_BYTES_PER_REQUEST, MIN_ENTROPY
from src.low_level.health import HealthMonitor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_64 = bytes(range(64))  # deterministic 64-byte seed
SEED_ALT = bytes(reversed(range(64)))  # different 64-byte seed


# ===========================================================================
# Output shape
# ===========================================================================


class TestOutputShape:
    def test_returns_bytes(self) -> None:
        result = HMACDRBG(SEED_64).generate(32)
        assert isinstance(result, bytes)

    def test_requested_length_is_exact(self) -> None:
        drbg = HMACDRBG(SEED_64)
        for n in [1, 16, 64, 128, 1024]:
            assert len(drbg.generate(n)) == n

    def test_zero_bytes_returns_empty(self) -> None:
        assert HMACDRBG(SEED_64).generate(0) == b""


# ===========================================================================
# Determinism (no nonce, no prediction resistance)
# ===========================================================================


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        a = HMACDRBG(SEED_64).generate(128)
        b = HMACDRBG(SEED_64).generate(128)
        assert a == b

    def test_different_seeds_different_output(self) -> None:
        a = HMACDRBG(SEED_64).generate(64)
        b = HMACDRBG(SEED_ALT).generate(64)
        assert a != b

    def test_generate_is_not_a_continuous_stream(self) -> None:
        """HMAC-DRBG output depends on request boundaries, unlike a stream cipher.

        Each generate call runs a post-generation Update that mixes the state,
        so splitting a request into two calls produces different output than
        one large call.  This is the correct SP 800-90A DRBG behaviour.
        """
        drbg_single = HMACDRBG(SEED_64)
        single = drbg_single.generate(128)

        drbg_split = HMACDRBG(SEED_64)
        split = drbg_split.generate(64) + drbg_split.generate(64)

        assert single != split

    def test_nonce_produces_different_output(self) -> None:
        a = HMACDRBG(SEED_64, nonce=b"nonce-a").generate(64)
        b = HMACDRBG(SEED_64, nonce=b"nonce-b").generate(64)
        assert a != b


# ===========================================================================
# Backtracking resistance
# ===========================================================================


class TestBacktrackingResistance:
    """After every generate, the internal state is irreversibly updated.

    This is the HMAC-DRBG Update step (SP 800-90A §10.1.2.2) that provides
    backtracking resistance: an adversary capturing the post-generate state
    cannot reconstruct the pre-generate state or the already-consumed output.
    """

    def test_state_changes_after_generate(self) -> None:
        """K and V must differ after a generate call."""
        drbg = HMACDRBG(SEED_64)
        k_before = drbg.key
        v_before = drbg.value

        drbg.generate(32)

        assert drbg.key != k_before
        assert drbg.value != v_before

    def test_post_state_cannot_replay_past_output(self) -> None:
        """An adversary holding the post-generate state cannot replay past output."""
        drbg = HMACDRBG(SEED_64)
        past_output = drbg.generate(32)

        # Adversary captures the post-generation state.
        adversary = HMACDRBG(SEED_64)
        adversary._key = drbg.key  # type: ignore[reportPrivateUsage]
        adversary._value = drbg.value  # type: ignore[reportPrivateUsage]

        # Adversary can only produce *future* output, never the consumed past.
        future_output = adversary.generate(32)
        assert future_output != past_output

    def test_two_generates_produce_different_output(self) -> None:
        """Sequential generates from the same DRBG must differ."""
        drbg = HMACDRBG(SEED_64)
        first = drbg.generate(32)
        second = drbg.generate(32)
        assert first != second


# ===========================================================================
# Reseeding
# ===========================================================================


class TestReseeding:
    def test_reseed_changes_the_stream(self) -> None:
        """After reseeding, the output must differ from the pre-reseed continuation."""
        drbg_reseeded = HMACDRBG(SEED_64)
        drbg_reseeded.generate(32)  # skip first block
        drbg_reseeded.reseed(os.urandom(48))
        reseeded_output = drbg_reseeded.generate(32)

        drbg_reference = HMACDRBG(SEED_64)
        drbg_reference.generate(32)  # skip first block (no reseed)
        reference_output = drbg_reference.generate(32)

        assert reseeded_output != reference_output

    def test_reseed_resets_counter(self) -> None:
        drbg = HMACDRBG(SEED_64, reseed_interval=10)
        drbg.generate(32)
        drbg.generate(32)
        assert drbg.reseed_counter == 3

        drbg.reseed(os.urandom(48))
        assert drbg.reseed_counter == 1

    def test_auto_reseed_on_interval(self) -> None:
        """The DRBG auto-reseeds when the reseed counter exceeds the interval."""
        drbg = HMACDRBG(SEED_64, reseed_interval=3)

        # Three generates: counter goes 1→2→3→4, none exceeding 3.
        for _ in range(3):
            drbg.generate(32)
        assert drbg.reseed_counter == 4

        # Fourth generate: counter (4) > interval (3), triggers auto-reseed
        # (resets counter to 1), then increments to 2.
        drbg.generate(32)
        assert drbg.reseed_counter == 2

    def test_reseed_with_short_entropy_raises(self) -> None:
        drbg = HMACDRBG(SEED_64)
        with pytest.raises(ValueError, match="at least 32 bytes"):
            drbg.reseed(b"too short")


# ===========================================================================
# Prediction resistance
# ===========================================================================


class TestPredictionResistance:
    def test_prediction_resistance_makes_output_nondeterministic(self) -> None:
        """With prediction resistance, same seed produces different output each run."""
        # prediction_resistance reseeds from OS entropy before every generate,
        # so two instances with the same seed must differ.
        a = HMACDRBG(SEED_64, prediction_resistance=True).generate(64)
        b = HMACDRBG(SEED_64, prediction_resistance=True).generate(64)
        assert a != b

    def test_prediction_resistance_property(self) -> None:
        drbg = HMACDRBG(SEED_64, prediction_resistance=True)
        assert drbg.prediction_resistance is True

    def test_no_prediction_resistance_by_default(self) -> None:
        drbg = HMACDRBG(SEED_64)
        assert drbg.prediction_resistance is False


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_seed_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 32 bytes"):
            HMACDRBG(b"short")

    def test_seed_exactly_min_length_accepted(self) -> None:
        drbg = HMACDRBG(bytes(range(MIN_ENTROPY)))
        assert len(drbg.generate(32)) == 32

    def test_negative_num_bytes_raises(self) -> None:
        drbg = HMACDRBG(SEED_64)
        with pytest.raises(ValueError, match="non-negative"):
            drbg.generate(-1)

    def test_excessive_num_bytes_raises(self) -> None:
        drbg = HMACDRBG(SEED_64)
        with pytest.raises(ValueError, match="must not exceed"):
            drbg.generate(MAX_BYTES_PER_REQUEST + 1)

    def test_invalid_reseed_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            HMACDRBG(SEED_64, reseed_interval=0)


# ===========================================================================
# Statistical sanity (weak but fast)
# ===========================================================================


class TestStatisticalSanity:
    def test_output_is_not_all_zeros(self) -> None:
        result = HMACDRBG(SEED_64).generate(64)
        assert any(b != 0 for b in result)

    def test_all_byte_values_appear_in_large_output(self) -> None:
        """With 4096 bytes all 256 byte values should appear."""
        result = HMACDRBG(SEED_64).generate(4096)
        assert len(set(result)) == 256

    def test_mean_byte_value_is_centred(self) -> None:
        result = HMACDRBG(SEED_64).generate(4096)
        mean = sum(result) / len(result)
        assert 100 < mean < 155

    def test_long_stream_passes_continuous_health(self) -> None:
        """A long DRBG output stream must pass NIST SP 800-90B health tests."""
        drbg = HMACDRBG(SEED_64)
        monitor = HealthMonitor(window=1024)

        for _ in range(20):  # well over several AP windows
            block = drbg.generate(1024)
            monitor.observe(block)

        assert monitor.failed is False
        status = monitor.status
        assert status.adaptive_proportion_windows >= 18
        assert status.total_samples == 20 * 1024

    def test_long_stream_no_repetition_health_error(self) -> None:
        """Generating a large batch must not trip the repetition count test."""
        drbg = HMACDRBG(SEED_64)
        monitor = HealthMonitor()

        # Generate 64 KiB in one shot — no health fault.
        block = drbg.generate(65536)
        monitor.observe(block)
        assert monitor.failed is False
