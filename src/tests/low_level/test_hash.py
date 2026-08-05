"""Tests for src.low_level.hash."""

from __future__ import annotations

from src.low_level.hash import hash_entropy_pool


def test_hash_length(mock_entropy_bytes: bytes) -> None:
    # Ensure it always spits out 64 bytes
    result = hash_entropy_pool(mock_entropy_bytes)
    assert len(result) == 64
    assert isinstance(result, bytes)


def test_hash_deterministic(mock_entropy_bytes: bytes) -> None:
    # Same bytes in should equal same bytes out
    result1 = hash_entropy_pool(mock_entropy_bytes)
    result2 = hash_entropy_pool(mock_entropy_bytes)
    assert result1 == result2


def test_hash_avalanche(mock_entropy_bytes: bytes) -> None:
    # Changing one byte should drastically change the output
    modified_bytes = bytearray(mock_entropy_bytes)
    modified_bytes[0] = (modified_bytes[0] + 1) % 256

    result1 = hash_entropy_pool(mock_entropy_bytes)
    result2 = hash_entropy_pool(bytes(modified_bytes))

    assert result1 != result2
