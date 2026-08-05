"""PhotoRandEngine — Cryptographically Secure Pseudo-Random Number Generator (CSPRNG)."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from src.high_level.seed import PhotoRandSeed
from src.low_level.csprng import generate_chacha20_encryptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class PhotoRandEngine:
    """Cryptographically Secure Pseudo-Random Number Generator (CSPRNG).

    This class uses a :class:`PhotoRandSeed` as its absolute source of entropy,
    combined with environmental salting (nanosecond timestamp and process ID),
    to fuel a continuous ChaCha20 stream cipher. It provides a stateful stream
    of secure random values.
    """

    def __init__(self, source: PhotoRandSeed | str, salt: bool = True) -> None:
        """Initialize the CSPRNG using a PhotoRandSeed or an image path.

        Args:
            source: A :class:`PhotoRandSeed` object or path to a RAW image.
            salt: When True (default), append environmental entropy (timestamp,
                PID) to ensure a unique sequence for every execution. When False,
                use the seed exactly as provided for a deterministic, reproducible
                sequence.
        """
        if isinstance(source, str):
            self.seed = PhotoRandSeed(source)
        else:
            self.seed = source

        raw_seed = self.seed.to_bytes()
        self._encryptor = generate_chacha20_encryptor(raw_seed, salt=salt)

    def _get_bytes(self, n: int) -> bytes:
        """Push *n* null bytes through the ChaCha20 encryptor and return the keystream.

        Args:
            n: Number of bytes to generate.

        Returns:
            *n* secret random bytes.
        """
        null_payload = b"\x00" * n
        return self._encryptor.update(null_payload)

    def next_bytes(self, length: int) -> bytes:
        """Generate a sequence of random bytes.

        Args:
            length: Number of bytes to generate.

        Returns:
            Random bytes.
        """
        return self._get_bytes(length)

    def next_int(self, length: int = 8) -> int:
        """Generate an integer with the specified number of bytes.

        Args:
            length: Number of bytes (not digits) to use for the integer.
                Defaults to 8 (64 bits).

        Returns:
            A random integer.
        """
        raw_bytes = self._get_bytes(length)
        return int.from_bytes(raw_bytes, byteorder="big")

    def next_int_digits(self, digits: int) -> int:
        """Generate a random integer with exactly the specified number of decimal digits.

        Args:
            digits: The number of decimal digits for the generated integer.

        Returns:
            A random integer with the specified number of digits.

        Raises:
            ValueError: If digits is not greater than 0.
        """
        if digits <= 0:
            raise ValueError("digits must be greater than 0")
        min_val = 10 ** (digits - 1) if digits > 1 else 0
        max_val = 10**digits - 1
        return self.next_int_range(min_val, max_val)

    def next_int_range(self, min_val: int, max_val: int) -> int:
        """Generate a random integer within the range ``[min_val, max_val]`` (inclusive).

        Args:
            min_val: The lower bound of the range.
            max_val: The upper bound of the range.

        Returns:
            A random integer.

        Raises:
            ValueError: If max_val is less than min_val.
        """
        range_size = max_val - min_val + 1
        if range_size <= 0:
            raise ValueError("max_val must be greater than or equal to min_val")

        # The exact number of bits needed to represent the range
        num_bits = range_size.bit_length()
        num_bytes = (num_bits + 7) // 8

        while True:
            raw_bytes = self._get_bytes(num_bytes)
            val = int.from_bytes(raw_bytes, byteorder="big")

            # Mask the value to the exact bit length needed
            val &= (1 << num_bits) - 1

            if val < range_size:
                return min_val + val

    def next_string(self, length: int = 16, charset: str = "all") -> str:
        """Generate a random string using the specified character set via base conversion.

        Args:
            length: Length of the string.
            charset: ``'all'``, ``'alphanumeric'``, ``'numeric'``, ``'hex'``, or a
                custom string of characters.

        Returns:
            Random string.

        Raises:
            ValueError: If charset is a custom string and is empty.
        """
        if length <= 0:
            return ""

        if charset == "all":
            chars = string.ascii_letters + string.digits + string.punctuation
        elif charset == "alphanumeric":
            chars = string.ascii_letters + string.digits
        elif charset == "numeric":
            chars = string.digits
        elif charset == "hex":
            chars = "0123456789abcdef"
        else:
            if not charset:
                raise ValueError("charset cannot be empty")
            chars = charset

        base = len(chars)

        # Calculate the exact number of possible string permutations
        max_val = base**length - 1

        # Pull ONE massive integer that perfectly represents our string
        val = self.next_int_range(0, max_val)

        # Convert the integer into base-N to extract the characters
        result: list[str] = []
        for _ in range(length):
            result.append(chars[val % base])
            val //= base

        return "".join(result)

    def next_bool(self) -> bool:
        """Generate a random boolean value.

        Returns:
            True or False.
        """
        return (self._get_bytes(1)[0] & 1) == 1

    def next_float(self) -> float:
        """Generate a random float between 0.0 (inclusive) and 1.0 (exclusive).

        Returns:
            A random float in ``[0.0, 1.0)``.
        """
        # Pull 7 bytes (56 bits), shift right by 3 to get exactly 53 bits
        raw_bytes = self._get_bytes(7)
        val = int.from_bytes(raw_bytes, byteorder="big") >> 3

        # Divide by 2^53. This is exact and introduces zero rounding error.
        return val * (2.0**-53)

    def next_float_range(self, min_val: float, max_val: float) -> float:
        """Generate a random float within the range ``[min_val, max_val)`` (half-open).

        Args:
            min_val: The lower bound of the range (inclusive).
            max_val: The upper bound of the range (exclusive).

        Returns:
            A random float in ``[min_val, max_val)``.

        Raises:
            ValueError: If max_val is less than min_val.
        """
        if max_val < min_val:
            raise ValueError("max_val must be greater than or equal to min_val")

        factor = self.next_float()
        return min_val + (factor * (max_val - min_val))

    def generate_batch[T](self, type_func: Callable[..., T], n: int, **kwargs: Any) -> list[T]:
        """Generate a list of *n* items using one of the generation methods.

        Args:
            type_func: The method to call (e.g., :meth:`next_int_range`).
            n: Number of items to generate.
            **kwargs: Arguments for *type_func*.

        Returns:
            A list of generated items.
        """
        return [type_func(**kwargs) for _ in range(n)]
