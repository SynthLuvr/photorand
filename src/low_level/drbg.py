"""NIST SP 800-90A Rev. 1 HMAC-DRBG (SHA-256) — a reseedable DRBG.

This module implements a vetted, reseedable deterministic random bit generator
per NIST SP 800-90A §10.1.2.  It replaces the bare one-shot ChaCha20 keystream
that ``PhotoRandEngine`` previously used, adding three properties the old
approach lacked:

1. **Backtracking resistance** — after every ``generate`` call the internal
   HMAC-DRBG_Update mixes the state (§10.1.2.2).  An adversary who compromises
   the current state therefore *cannot* reconstruct past output.

2. **Periodic reseeding** — the generator reseeds itself from OS cryptographic
   entropy after ``reseed_interval`` generate calls, limiting the amount of
   output any single state can produce and enabling recovery from state
   compromise (forward security).

3. **Prediction resistance** — when enabled, the generator reseeds from OS
   entropy *before* every generate call, providing maximum forward security at
   the cost of an OS syscall per request.

Reseed entropy is drawn from :func:`os.urandom`, which on modern platforms
reads ``/dev/urandom`` or ``getrandom()`` — the operating system's own
cryptographic entropy pool.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from src.logger import logger

# SHA-256 output length in bytes (the *outlen* parameter in SP 800-90A).
_OUTLEN = 32

# Minimum entropy_input length accepted at instantiation (SP 800-90A §10.1.2
# Table 2 requires at least *security_strength* bits; SHA-256 DRBG supports a
# 256-bit security strength).
MIN_ENTROPY = 32

# Maximum output per generate call before the DRBG must reseed or error
# (SP 800-90A §10.1.2 Table 2).
MAX_BYTES_PER_REQUEST = 65536

# Default reseed interval in generate calls.  SP 800-90A permits up to 2^48,
# but a far smaller value gives stronger forward-security guarantees in
# practice.
_DEFAULT_RESEED_INTERVAL = 1 << 32

# Amount of OS entropy to pull on each automatic reseed (≥ MIN_ENTROPY).
_RESEED_ENTROPY_BYTES = 48


def _hmac_sha256(key: bytes, data: bytes) -> bytes:
    """Compute HMAC-SHA-256 of *data* under *key*."""
    return hmac.new(key, data, hashlib.sha256).digest()


class HMACDRBG:
    """NIST SP 800-90A Rev. 1 HMAC-DRBG using SHA-256.

    A stateful, reseedable DRBG with backtracking resistance.  Instantiate
    once with a high-entropy seed and call :meth:`generate` to obtain
    pseudo-random bytes.  The DRBG automatically reseeds from OS entropy
    when the reseed interval is exceeded.

    Args:
        entropy_input: The initial entropy (e.g. a 64-byte TRNG seed).
        nonce: Optional nonce mixed into the seed material at instantiation.
            A nonce is **not** a security measure on its own — it provides
            uniqueness so that two DRBGs initialised with the same seed
            produce different streams.
        personalization_string: Optional application-specific string mixed
            into the seed material.
        prediction_resistance: When ``True``, reseed from OS entropy before
            *every* generate call (SP 800-90A prediction resistance request).
        reseed_interval: Maximum number of generate calls between automatic
            reseeds.  ``None`` uses the default (2³²).

    Raises:
        ValueError: If *entropy_input* is shorter than 32 bytes.
    """

    # Exposed as class attributes for documentation and test access.
    _outlen: int = _OUTLEN

    def __init__(
        self,
        entropy_input: bytes,
        *,
        nonce: bytes = b"",
        personalization_string: bytes = b"",
        prediction_resistance: bool = False,
        reseed_interval: int | None = None,
    ) -> None:
        if len(entropy_input) < MIN_ENTROPY:
            raise ValueError(
                f"entropy_input must be at least {MIN_ENTROPY} bytes (got {len(entropy_input)})."
            )

        interval = reseed_interval if reseed_interval is not None else _DEFAULT_RESEED_INTERVAL
        if interval < 1:
            raise ValueError("reseed_interval must be at least 1.")

        self._prediction_resistance: bool = prediction_resistance
        self._reseed_interval: int = interval

        # SP 800-90A §10.1.2.3 — HMAC_DRBG_Instantiate.
        seed_material = entropy_input + nonce + personalization_string
        self._key = b"\x00" * _OUTLEN
        self._value = b"\x01" * _OUTLEN
        self._update(seed_material)
        self._reseed_counter: int = 1

        logger.info(
            "[drbg] HMAC-DRBG initialised with %d bytes of entropy "
            "(prediction_resistance=%s, reseed_interval=%d).",
            len(entropy_input),
            prediction_resistance,
            interval,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def reseed(self, entropy_input: bytes, additional_input: bytes = b"") -> None:
        """Reseed the DRBG (SP 800-90A §10.1.2.4 — HMAC_DRBG_Reseed).

        Mixing fresh entropy into the state resets the reseed counter and
        breaks the link between old and future output (forward security).

        Args:
            entropy_input: Fresh entropy to mix in (at least 32 bytes).
            additional_input: Optional additional data to include.

        Raises:
            ValueError: If *entropy_input* is shorter than 32 bytes.
        """
        if len(entropy_input) < MIN_ENTROPY:
            raise ValueError(
                f"entropy_input must be at least {MIN_ENTROPY} bytes (got {len(entropy_input)})."
            )

        seed_material = entropy_input + additional_input
        self._update(seed_material)
        self._reseed_counter = 1
        logger.info("[drbg] Reseeded from external entropy.")

    def generate(self, num_bytes: int, additional_input: bytes = b"") -> bytes:
        """Generate *num_bytes* of pseudo-random output (SP 800-90A §10.1.2.5).

        Performs automatic reseeding (from OS entropy) when the reseed
        interval is exceeded or when prediction resistance is enabled.
        After generating, the state is updated so that the output cannot be
        recovered from a future state compromise (backtracking resistance).

        Args:
            num_bytes: Number of bytes to generate (0 – 65 536).
            additional_input: Optional data mixed into both the pre- and
                post-generation state update.

        Returns:
            *num_bytes* of pseudo-random bytes.

        Raises:
            ValueError: If *num_bytes* is negative or exceeds 65 536.
        """
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative.")
        if num_bytes > MAX_BYTES_PER_REQUEST:
            raise ValueError(f"num_bytes must not exceed {MAX_BYTES_PER_REQUEST} per request.")
        if num_bytes == 0:
            return b""

        # --- Automatic reseed (§10.1.2.5 step 1) ---------------------------
        if self._prediction_resistance or self._reseed_counter > self._reseed_interval:
            self._auto_reseed()

        # --- Pre-generation additional-input update (step 2.2) -------------
        if additional_input:
            self._update(additional_input)

        # --- Generate blocks (step 3) --------------------------------------
        temp = bytearray()
        while len(temp) < num_bytes:
            self._value = _hmac_sha256(self._key, self._value)
            temp.extend(self._value)
        output = bytes(temp[:num_bytes])

        # --- Post-generation update — backtracking resistance (step 4) -----
        self._update(additional_input)

        self._reseed_counter += 1
        return output

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _update(self, provided_data: bytes) -> None:
        """SP 800-90A §10.1.2.2 — HMAC_DRBG_Update.

        Irreversibly mixes *provided_data* into the HMAC key and value.
        This is what provides backtracking resistance: after the update the
        state can no longer be used to reconstruct previous output.
        """
        # Step 1: K = HMAC(K, V ‖ 0x00 ‖ provided_data)
        self._key = _hmac_sha256(self._key, self._value + b"\x00" + provided_data)
        # Step 2: V = HMAC(K, V)
        self._value = _hmac_sha256(self._key, self._value)

        # Steps 3.1–3.2 (only when there is data to mix).
        if provided_data:
            self._key = _hmac_sha256(self._key, self._value + b"\x01" + provided_data)
            self._value = _hmac_sha256(self._key, self._value)

    def _auto_reseed(self) -> None:
        """Pull OS entropy and feed it to :meth:`reseed`."""
        self.reseed(os.urandom(_RESEED_ENTROPY_BYTES))

    # ------------------------------------------------------------------ #
    #  Read-only properties (useful for inspection / tests)               #
    # ------------------------------------------------------------------ #

    @property
    def reseed_counter(self) -> int:
        """Number of generate calls since the last reseed."""
        return self._reseed_counter

    @property
    def prediction_resistance(self) -> bool:
        """Whether prediction resistance (reseed-before-generate) is enabled."""
        return self._prediction_resistance

    @property
    def reseed_interval(self) -> int:
        """Configured reseed interval in generate calls."""
        return self._reseed_interval

    @property
    def key(self) -> bytes:
        """Current HMAC key (for backtracking-resistance tests)."""
        return self._key

    @property
    def value(self) -> bytes:
        """Current HMAC value (for backtracking-resistance tests)."""
        return self._value
