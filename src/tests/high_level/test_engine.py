"""Tests for src.high_level.engine."""

from __future__ import annotations

import time
from pathlib import Path

from src.high_level.engine import PhotoRandEngine
from src.high_level.seed import PhotoRandSeed

# Determine the path to the test data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_IMAGE = str(DATA_DIR / "DSC02111.ARW")


class TestPhotoRandEngine:
    """Tests for the PhotoRandEngine (CSPRNG) class."""

    def test_init_with_path(self) -> None:
        """Verify initialization directly from an image path string."""
        engine = PhotoRandEngine(TEST_IMAGE)
        assert isinstance(engine.seed, PhotoRandSeed)
        assert len(engine.seed.to_bytes()) == 64

    def test_init_with_seed_object(self) -> None:
        """Verify initialization from an existing PhotoRandSeed object."""
        seed = PhotoRandSeed(TEST_IMAGE)
        engine = PhotoRandEngine(seed)
        assert engine.seed is seed

    def test_salting_uniqueness(self) -> None:
        """Verify that two engines with the SAME seed produce DIFFERENT streams."""
        seed = PhotoRandSeed(TEST_IMAGE)
        engine1 = PhotoRandEngine(seed)

        # Ensure a tiny time difference if the resolution is coarse
        time.sleep(0.001)
        engine2 = PhotoRandEngine(seed)

        bytes1 = engine1.next_bytes(32)
        bytes2 = engine2.next_bytes(32)

        assert bytes1 != bytes2, "Engines with same seed must produce unique streams."

    def test_next_bytes(self) -> None:
        """Verify raw byte generation."""
        engine = PhotoRandEngine(TEST_IMAGE)
        b = engine.next_bytes(10)
        assert len(b) == 10
        assert isinstance(b, bytes)

    def test_next_int(self) -> None:
        """Verify integer generation with specific byte lengths."""
        engine = PhotoRandEngine(TEST_IMAGE)

        # 4-byte int (32-bit)
        val = engine.next_int(length=4)
        assert 0 <= val < (1 << 32)

        # 8-byte int (64-bit)
        val64 = engine.next_int(length=8)
        assert 0 <= val64 < (1 << 64)

    def test_next_int_digits(self) -> None:
        """Verify integer generation with specific number of decimal digits."""
        engine = PhotoRandEngine(TEST_IMAGE)

        # 5-digit int
        val5 = engine.next_int_digits(digits=5)
        assert len(str(val5)) == 5

        # 1-digit int
        val1 = engine.next_int_digits(digits=1)
        assert len(str(val1)) == 1
        assert 0 <= val1 <= 9

        # Large digit int (e.g., 100 digits)
        val100 = engine.next_int_digits(digits=100)
        assert len(str(val100)) == 100

    def test_next_int_range(self) -> None:
        """Verify range-bounded integer generation with rejection sampling."""
        engine = PhotoRandEngine(TEST_IMAGE)

        # Small range
        for _ in range(50):
            val = engine.next_int_range(1, 6)
            assert 1 <= val <= 6

        # Large range spanning multiple bytes
        large_val = engine.next_int_range(10**12, 10**13)
        assert 10**12 <= large_val <= 10**13

    def test_next_string(self) -> None:
        """Verify random string generation with various charsets."""
        engine = PhotoRandEngine(TEST_IMAGE)

        # Numeric
        s_num = engine.next_string(length=12, charset="numeric")
        assert len(s_num) == 12
        assert s_num.isdigit()

        # Hex
        s_hex = engine.next_string(length=16, charset="hex")
        assert len(s_hex) == 16
        assert all(c in "0123456789abcdef" for c in s_hex)

        # Alphanumeric
        s_alpha = engine.next_string(length=20, charset="alphanumeric")
        assert len(s_alpha) == 20
        assert s_alpha.isalnum()

        # Custom charset
        custom = "ABC"
        s_custom = engine.next_string(length=50, charset=custom)
        assert all(c in custom for c in s_custom)

    def test_next_bool(self) -> None:
        """Verify boolean generation."""
        engine = PhotoRandEngine(TEST_IMAGE)
        # Probability of 100 same booleans in a row is 2^-100
        results = {engine.next_bool() for _ in range(100)}
        assert {True, False} == results

    def test_next_float(self) -> None:
        """Verify float generation."""
        engine = PhotoRandEngine(TEST_IMAGE)
        for _ in range(100):
            val_float = engine.next_float()
            assert isinstance(val_float, float)
            assert 0.0 <= val_float < 1.0, f"Value {val_float} not in [0.0, 1.0)"

    def test_next_float_range(self) -> None:
        """Verify float range generation."""
        engine = PhotoRandEngine(TEST_IMAGE)
        for _ in range(100):
            val_float = engine.next_float_range(10.0, 20.0)
            assert 10.0 <= val_float < 20.0, f"Value {val_float} not in [10.0, 20.0)"

        # Test negative range
        for _ in range(100):
            val_neg = engine.next_float_range(-2.0, -1.0)
            assert -2.0 <= val_neg < -1.0, f"Value {val_neg} not in [-2.0, -1.0)"

        # Batch of floats
        batch = engine.generate_batch(engine.next_float, 20)
        assert len(batch) == 20
        assert all(isinstance(f, float) and 0.0 <= f <= 1.0 for f in batch)

    def test_generate_batch(self) -> None:
        """Verify the batch helper method."""
        engine = PhotoRandEngine(TEST_IMAGE)

        # Batch of strings
        batch = engine.generate_batch(engine.next_string, 10, length=5, charset="hex")
        assert len(batch) == 10
        assert all(len(s) == 5 for s in batch)

        # Batch of integers in range
        batch_ints = engine.generate_batch(engine.next_int_range, 20, min_val=0, max_val=1)
        assert len(batch_ints) == 20
        assert all(x in [0, 1] for x in batch_ints)
