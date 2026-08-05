"""Entropy conditioning — compress a raw entropy pool with SHA3-512."""

from __future__ import annotations

import hashlib

from src.logger import logger


def hash_entropy_pool(entropy_pool: bytes) -> bytes:
    """Compress a raw byte stream of physical entropy into 64 uniformly distributed bytes.

    Args:
        entropy_pool: The raw entropy byte stream produced by the sampler.

    Returns:
        A 64-byte (512-bit) SHA3-512 digest of the input entropy.
    """
    logger.info("[hash] Compressing entropy pool with SHA3-512...")

    # Run the blender directly on the NumPy byte dump
    secure_bytes = hashlib.sha3_512(entropy_pool).digest()

    logger.info("[hash] Generated %d bytes of secure entropy.", len(secure_bytes))
    return secure_bytes
