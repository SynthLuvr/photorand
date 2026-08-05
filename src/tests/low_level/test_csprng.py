"""Tests for src.low_level.csprng.expand_entropy_chacha20.

Strategy:
  - Use a fixed 64-byte seed for determinism.
  - Verify output length, type, and basic randomness properties.
  - Verify that the same seed always produces the same keystream (determinism).
  - Verify that different seed lengths produce different keystreams.
  - Verify input validation (seed too short raises ValueError).
"""

from __future__ import annotations

import pytest

from src.low_level.csprng import expand_entropy_chacha20

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_64 = bytes(range(64))  # deterministic 64-byte seed
SEED_48 = bytes(range(48))  # minimum valid seed (48 bytes)
SEED_SHORT = bytes(range(47))  # one byte too short — should fail


# ===========================================================================
# Output shape
# ===========================================================================


class TestOutputShape:
    def test_returns_bytes(self) -> None:
        result = expand_entropy_chacha20(SEED_64, 32)
        assert isinstance(result, bytes)

    def test_requested_length_is_exact(self) -> None:
        for n in [1, 16, 64, 128, 1024]:
            assert len(expand_entropy_chacha20(SEED_64, n)) == n

    def test_zero_bytes_returns_empty(self) -> None:
        assert expand_entropy_chacha20(SEED_64, 0) == b""


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same seed must always yield the same keystream."""

    def test_same_seed_same_output(self) -> None:
        a = expand_entropy_chacha20(SEED_64, 128)
        b = expand_entropy_chacha20(SEED_64, 128)
        assert a == b

    def test_different_seeds_different_output(self) -> None:
        seed_a = bytes(range(64))
        seed_b = bytes(reversed(range(64)))
        a = expand_entropy_chacha20(seed_a, 64)
        b = expand_entropy_chacha20(seed_b, 64)
        assert a != b

    def test_salting_produces_different_output_for_same_seed(self) -> None:
        a = expand_entropy_chacha20(SEED_64, 128, salt=False)
        b = expand_entropy_chacha20(SEED_64, 128, salt=True)
        assert a != b

    def test_longer_request_is_prefix_of_same_seed(self) -> None:
        """Requesting N bytes must be a prefix of requesting 2*N bytes."""
        short = expand_entropy_chacha20(SEED_64, 64)
        long_ = expand_entropy_chacha20(SEED_64, 128)
        assert long_[:64] == short


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_seed_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="48 bytes"):
            expand_entropy_chacha20(SEED_SHORT, 64)

    def test_seed_exactly_48_bytes_is_accepted(self) -> None:
        result = expand_entropy_chacha20(SEED_48, 32)
        assert len(result) == 32

    def test_seed_longer_than_64_bytes_is_accepted(self) -> None:
        long_seed = bytes(range(128))
        result = expand_entropy_chacha20(long_seed, 64)
        assert len(result) == 64


# ===========================================================================
# Statistical sanity (weak but fast)
# ===========================================================================


class TestStatisticalSanity:
    """Very light checks — real randomness tests belong to dedicated NIST suites."""

    def test_output_is_not_all_zeros(self) -> None:
        result = expand_entropy_chacha20(SEED_64, 64)
        assert any(b != 0 for b in result)

    def test_all_byte_values_appear_in_large_output(self) -> None:
        """With 4096 bytes of keystream all 256 byte values should appear."""
        result = expand_entropy_chacha20(SEED_64, 4096)
        assert len(set(result)) == 256

    def test_output_has_reasonable_entropy(self) -> None:
        """Mean byte value should sit near 127 for a random-looking stream."""
        result = expand_entropy_chacha20(SEED_64, 4096)
        mean = sum(result) / len(result)
        # Allow generous tolerance: flat distribution has mean 127.5
        assert 100 < mean < 155
