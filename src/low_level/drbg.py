"""NIST SP 800-90A Rev. 1 HMAC-DRBG (SHA-256).

A reseedable DRBG (SP 800-90A §10.1.2) that supersedes the engine's earlier
one-shot ChaCha20 keystream, adding:

* **Backtracking resistance** — the state is mixed after every generate
  (§10.1.2.2), so a later compromise cannot reveal past output.
* **Periodic reseeding** — reseeds from OS entropy after ``reseed_interval``
  calls, bounding the output tied to any single state.
* **Prediction resistance** — when enabled, reseeds before every generate.

Reseed entropy is drawn from :func:`os.urandom`.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from src.logger import logger

# outlen for the SHA-256 DRBG (SP 800-90A §10.1.2).
_OUTLEN = 32

# SHA-256 supports a 256-bit security strength; SP 800-90A §10.1.2 Table 2
# requires entropy of at least that strength.
MIN_ENTROPY = 32

# SP 800-90A §10.1.2 Table 2: maximum output before a reseed is forced.
MAX_BYTES_PER_REQUEST = 65536

# SP 800-90A allows up to 2^48; a smaller value tightens forward security.
_DEFAULT_RESEED_INTERVAL = 1 << 32

# OS entropy drawn per automatic reseed (>= MIN_ENTROPY).
_RESEED_ENTROPY_BYTES = 48


def _hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _require_min_entropy(entropy_input: bytes) -> None:
    if len(entropy_input) < MIN_ENTROPY:
        raise ValueError(
            f"entropy_input must be at least {MIN_ENTROPY} bytes (got {len(entropy_input)})."
        )


class HMACDRBG:
    """NIST SP 800-90A Rev. 1 HMAC-DRBG using SHA-256.

    Instantiate once with a high-entropy seed and call :meth:`generate` for
    pseudo-random bytes.  The DRBG auto-reseeds from OS entropy once the
    reseed interval is exceeded.

    Args:
        entropy_input: Initial entropy (e.g. a 64-byte TRNG seed).
        nonce: Uniqueness material so two DRBGs sharing a seed diverge.
            Not a security measure on its own; security comes from the seed.
        personalization_string: Optional application-specific string mixed
            into the seed material.
        prediction_resistance: Reseed from OS entropy before every generate.
        reseed_interval: Max generates between automatic reseeds;
            ``None`` uses the default (2³²).

    Raises:
        ValueError: If *entropy_input* is shorter than 32 bytes.
    """

    def __init__(
        self,
        entropy_input: bytes,
        *,
        nonce: bytes = b"",
        personalization_string: bytes = b"",
        prediction_resistance: bool = False,
        reseed_interval: int | None = None,
    ) -> None:
        _require_min_entropy(entropy_input)

        interval = reseed_interval if reseed_interval is not None else _DEFAULT_RESEED_INTERVAL
        if interval < 1:
            raise ValueError("reseed_interval must be at least 1.")

        self._prediction_resistance = prediction_resistance
        self._reseed_interval = interval

        # HMAC_DRBG_Instantiate (SP 800-90A §10.1.2.3).
        self._key = b"\x00" * _OUTLEN
        self._value = b"\x01" * _OUTLEN
        self._update(entropy_input + nonce + personalization_string)
        self._reseed_counter = 1

        logger.info(
            "[drbg] HMAC-DRBG initialised: %d entropy bytes, "
            "prediction_resistance=%s, reseed_interval=%d.",
            len(entropy_input),
            prediction_resistance,
            interval,
        )

    def reseed(self, entropy_input: bytes, additional_input: bytes = b"") -> None:
        """Mix fresh entropy into the state (SP 800-90A §10.1.2.4).

        Resets the reseed counter and breaks the link between old and future
        output (forward security).

        Raises:
            ValueError: If *entropy_input* is shorter than 32 bytes.
        """
        _require_min_entropy(entropy_input)
        self._update(entropy_input + additional_input)
        self._reseed_counter = 1
        logger.info("[drbg] Reseeded from external entropy.")

    def generate(self, num_bytes: int, additional_input: bytes = b"") -> bytes:
        """Generate *num_bytes* of pseudo-random output (SP 800-90A §10.1.2.5).

        Reseeds automatically when the interval lapses or prediction
        resistance is on, then mixes the state afterwards so the output
        cannot be recovered from a future compromise (backtracking
        resistance).

        Raises:
            ValueError: If *num_bytes* is negative or exceeds 65 536.
        """
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative.")
        if num_bytes > MAX_BYTES_PER_REQUEST:
            raise ValueError(f"num_bytes must not exceed {MAX_BYTES_PER_REQUEST} per request.")
        if num_bytes == 0:
            return b""

        # Auto-reseed before generating (SP 800-90A §10.1.2.5 step 1).
        if self._prediction_resistance or self._reseed_counter > self._reseed_interval:
            self.reseed(os.urandom(_RESEED_ENTROPY_BYTES))

        if additional_input:
            self._update(additional_input)

        temp = bytearray()
        while len(temp) < num_bytes:
            self._value = _hmac_sha256(self._key, self._value)
            temp.extend(self._value)
        output = bytes(temp[:num_bytes])

        # Post-generation update provides backtracking resistance (step 4).
        self._update(additional_input)

        self._reseed_counter += 1
        return output

    def _update(self, provided_data: bytes) -> None:
        """HMAC_DRBG_Update (SP 800-90A §10.1.2.2).

        Irreversibly mixes *provided_data* into the key and value — the step
        that provides backtracking resistance.
        """
        self._key = _hmac_sha256(self._key, self._value + b"\x00" + provided_data)
        self._value = _hmac_sha256(self._key, self._value)
        if provided_data:
            self._key = _hmac_sha256(self._key, self._value + b"\x01" + provided_data)
            self._value = _hmac_sha256(self._key, self._value)

    @property
    def reseed_counter(self) -> int:
        """Generate calls since the last reseed."""
        return self._reseed_counter

    @property
    def prediction_resistance(self) -> bool:
        """Whether prediction resistance (reseed-before-generate) is on."""
        return self._prediction_resistance

    @property
    def reseed_interval(self) -> int:
        """Configured reseed interval in generate calls."""
        return self._reseed_interval

    @property
    def key(self) -> bytes:
        """Current HMAC key (exposed for backtracking-resistance tests)."""
        return self._key

    @property
    def value(self) -> bytes:
        """Current HMAC value (exposed for backtracking-resistance tests)."""
        return self._value
