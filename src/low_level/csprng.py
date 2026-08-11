"""Legacy one-shot ChaCha20 keystream expansion.

.. deprecated::

    :class:`~src.high_level.engine.PhotoRandEngine` now uses the vetted,
    reseedable :class:`~src.low_level.drbg.HMACDRBG` (NIST SP 800-90A) which
    provides backtracking resistance and periodic reseeding.  This module
    remains for backwards compatibility but should not be used in new code.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import TYPE_CHECKING

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from src.logger import logger

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.ciphers import CipherContext


def generate_chacha20_encryptor(raw_seed: bytes, salt: bool) -> CipherContext:
    """Initialize a ChaCha20 encryptor from a raw seed, optionally with environmental salting.

    Args:
        raw_seed: The raw seed bytes (at least 48 bytes in deterministic mode).
        salt: When True, append a nanosecond timestamp and PID before hashing.

    Returns:
        A ChaCha20 encryptor context ready to produce a keystream.
    """
    if salt:
        # Salting Logic: [Camera Seed] + [Time] + [PID]
        timestamp = str(time.time_ns()).encode()
        pid = str(os.getpid()).encode()
        combined = raw_seed + timestamp + pid
        session_hash = hashlib.sha3_512(combined).digest()
        logger.info("[csprng] CSPRNG initialized with salted session key.")
    else:
        # Deterministic Mode: Use the raw 64-byte seed directly
        if len(raw_seed) < 48:
            raise ValueError("ChaCha20 requires at least 48 bytes of seed (32 key, 16 nonce).")
        session_hash = raw_seed
        logger.info("[csprng] CSPRNG initialized in deterministic mode.")

    key = session_hash[:32]
    nonce = session_hash[32:48]

    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    return cipher.encryptor()


def expand_entropy_chacha20(trng_seed: bytes, num_bytes_needed: int, salt: bool = False) -> bytes:
    """Expand a photorand seed into arbitrary CSPRNG bytes via ChaCha20.

    Args:
        trng_seed: The 64-byte true random output from the hash function.
        num_bytes_needed: How many random bytes to generate.
        salt: Whether to apply environmental salting. Defaults to False.

    Returns:
        A highly secure, mathematically random byte string.
    """
    encryptor = generate_chacha20_encryptor(trng_seed, salt=salt)

    # Generate the keystream (encrypting zeros)
    null_payload = b"\x00" * num_bytes_needed

    logger.info("[csprng] Expanding physical seed into %d bytes...", num_bytes_needed)

    # Passing the zeros through the cipher extracts the pure random keystream
    csprng_stream = encryptor.update(null_payload)

    logger.info("[csprng] Expansion complete.")
    return csprng_stream
