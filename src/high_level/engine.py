"""PhotoRandEngine — Cryptographically Secure Pseudo-Random Number Generator (CSPRNG)."""

from __future__ import annotations

import os
import string
import time
from typing import TYPE_CHECKING

from src.high_level.seed import PhotoRandSeed
from src.low_level.drbg import HMACDRBG
from src.low_level.health import HealthMonitor, HealthStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def _build_uniqueness_tweak() -> bytes:
    """Build a non-security uniqueness nonce from environmental data.

    .. warning::

        This is **not** a source of cryptographic entropy.  A nanosecond
        timestamp and process ID are predictable by an attacker who can
        observe process creation.  The nonce exists only so that two
        :class:`PhotoRandEngine` instances initialised with the *same*
        :class:`PhotoRandSeed` produce different output streams — a
        uniqueness tweak, not a security measure.  All real security comes
        from the seed entropy and the HMAC-DRBG construction.
    """
    timestamp = str(time.time_ns()).encode()
    pid = str(os.getpid()).encode()
    return timestamp + pid


class PhotoRandEngine:
    """Cryptographically Secure Pseudo-Random Number Generator (CSPRNG).

    This class uses a :class:`PhotoRandSeed` as its absolute source of entropy
    to seed an :class:`~src.low_level.drbg.HMACDRBG` (NIST SP 800-90A §10.1.2).
    The DRBG provides backtracking resistance (post-generation state update),
    periodic reseeding from OS entropy, and optional prediction resistance.
    """

    def __init__(
        self,
        source: PhotoRandSeed | str,
        salt: bool = True,
        *,
        continuous_health: bool = True,
        prediction_resistance: bool = False,
        reseed_interval: int | None = None,
    ) -> None:
        """Initialize the CSPRNG using a PhotoRandSeed or an image path.

        The engine is backed by an :class:`~src.low_level.drbg.HMACDRBG`
        (NIST SP 800-90A §10.1.2), which provides backtracking resistance,
        periodic reseeding from OS entropy, and optional prediction
        resistance.

        Args:
            source: A :class:`PhotoRandSeed` object or path to a RAW image.
            salt: When True (default), mix in a nanosecond-timestamp / PID
                **uniqueness tweak** so that two engines with the same seed
                produce different streams.  This is *not* a security measure
                (see :func:`_build_uniqueness_tweak`); all cryptographic
                security comes from the seed entropy and the HMAC-DRBG.
                When False the sequence is deterministic and reproducible.
            continuous_health: When True (default), run NIST SP 800-90B
                continuous health tests on output, raising
                :class:`RuntimeHealthError` on a fault.
            prediction_resistance: When True, reseed the DRBG from OS
                entropy before *every* generate call (SP 800-90A prediction
                resistance).  Defaults to False.
            reseed_interval: Maximum number of generate calls between
                automatic DRBG reseeds.  ``None`` uses the DRBG default
                (2³²).

        Raises:
            RuntimeHealthError: If a continuous health test fails after generation
                has started (raised by the monitor during ``next_*`` calls).
        """
        if isinstance(source, str):
            self.seed = PhotoRandSeed(source)
        else:
            self.seed = source

        raw_seed = self.seed.to_bytes()
        nonce = _build_uniqueness_tweak() if salt else b""

        self._drbg = HMACDRBG(
            raw_seed,
            nonce=nonce,
            prediction_resistance=prediction_resistance,
            reseed_interval=reseed_interval,
        )
        self._monitor: HealthMonitor | None = HealthMonitor() if continuous_health else None

    @property
    def drbg(self) -> HMACDRBG:
        """The underlying HMAC-DRBG instance (NIST SP 800-90A §10.1.2)."""
        return self._drbg

    @property
    def health_monitor(self) -> HealthMonitor | None:
        """The continuous health monitor, or ``None`` if disabled."""
        return self._monitor

    @property
    def health_status(self) -> HealthStatus | None:
        """A snapshot of the continuous health-test counters, or ``None`` if disabled."""
        return self._monitor.status if self._monitor is not None else None

    def reseed(self, entropy_input: bytes) -> None:
        """Reseed the DRBG with fresh entropy.

        Mixing fresh entropy into the DRBG state breaks the link between old
        and future output (forward security) and is useful after a suspected
        state compromise or simply to add freshness during long sessions.

        Args:
            entropy_input: Fresh entropy bytes (at least 32 bytes).
        """
        self._drbg.reseed(entropy_input)

    def _get_bytes(self, n: int) -> bytes:
        """Generate *n* bytes from the HMAC-DRBG.

        Every byte produced passes through the continuous health monitor (when
        enabled) before being returned.

        Args:
            n: Number of bytes to generate.

        Returns:
            *n* secret random bytes.

        Raises:
            RuntimeHealthError: If the continuous health monitor detects a fault.
        """
        out = self._drbg.generate(n)
        if self._monitor is not None:
            self._monitor.observe(out)
        return out

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
