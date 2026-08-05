"""PhotoRandSeed — True Random Number Generator (TRNG) interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.logger import logger
from src.low_level.generate import generate_with_assessment

if TYPE_CHECKING:
    from src.low_level.entropy import EntropyAssessment


class PhotoRandSeed:
    """True Random Number Generator (TRNG) interface.

    Extracts physical entropy from a RAW image to create a 64-byte seed.
    Following NIST SP 800-90B, it also assesses the raw sampled pool before
    SHA3-512 conditioning — accessible via the :attr:`assessment` property.
    """

    _raw_seed: bytes
    _assessment: EntropyAssessment

    def __init__(self, image_path: str) -> None:
        """Initialize the TRNG by ingesting a RAW image and extracting entropy.

        Args:
            image_path: Path to the RAW image file.
        """
        logger.info("[PhotoRandSeed] Extracting TRNG entropy from: %s", image_path)

        self._raw_seed, _, self._assessment = generate_with_assessment(image_path)

        logger.info(
            "[PhotoRandSeed] Seed generated. Measured min-entropy: %.3f bits/symbol "
            "(%.1f total bits). Status: %s.",
            self._assessment.min_entropy,
            self._assessment.total_entropy_bits,
            self._assessment.overall_status,
        )

    @classmethod
    def from_webcam(
        cls,
        duration: float = 5.0,
        camera_index: int = 0,
    ) -> PhotoRandSeed:
        """Build a seed from a webcam capture instead of a RAW file.

        Requires the optional ``capture`` extra (``opencv-python-headless``).

        Raises:
            WebcamCaptureError: If capture cannot proceed.
        """
        from src.low_level.capture import generate_from_webcam

        logger.info(
            "[PhotoRandSeed] Extracting TRNG entropy from webcam %d (%.1fs)",
            camera_index,
            duration,
        )
        raw_seed, _pool, assessment = generate_from_webcam(
            duration=duration, camera_index=camera_index
        )

        obj = cls.__new__(cls)
        obj._raw_seed = raw_seed
        obj._assessment = assessment
        logger.info(
            "[PhotoRandSeed] Webcam seed generated. Measured min-entropy: %.3f bits/symbol "
            "(%.1f total bits). Status: %s.",
            assessment.min_entropy,
            assessment.total_entropy_bits,
            assessment.overall_status,
        )
        return obj

    @property
    def assessment(self) -> EntropyAssessment:
        """Return the NIST SP 800-90B entropy assessment of the sampled pool."""
        return self._assessment

    def to_bytes(self) -> bytes:
        """Return the pure 64-byte seed.

        Returns:
            The 64-byte TRNG seed.
        """
        return self._raw_seed

    def to_hex_string(self) -> str:
        """Return the seed as a hex string.

        Returns:
            128-character hex representation of the seed.
        """
        return self._raw_seed.hex()

    def to_int(self) -> int:
        """Convert the 64 bytes into a massive integer.

        Returns:
            The seed as a large integer.
        """
        return int.from_bytes(self._raw_seed, byteorder="big")

    def to_int_range(self, min_val: int, max_val: int) -> int:
        """Safely bound the TRNG entropy using rejection sampling.

        Because this is a pure TRNG with a strict 64-byte (512-bit) entropy limit,
        it cannot "redraw" a number if the initial sample falls into the modulo
        rejection zone. To preserve absolute cryptographic purity, no pseudo-random
        fallbacks are used. Instead, in the astronomically unlikely event of a
        rejection, a RuntimeError is raised.

        The mathematical hard limit for absolute fairness is calculated as:
        ``Limit = 2^512 - (2^512 mod range_size)``

        Args:
            min_val: The lower bound (inclusive).
            max_val: The upper bound (inclusive).

        Returns:
            A uniformly distributed random integer within ``[min_val, max_val]``.

        Raises:
            ValueError: If max_val is less than min_val.
            RuntimeError: If the physical entropy falls into the rejection zone.
        """
        range_size = max_val - min_val + 1
        if range_size <= 0:
            raise ValueError("max_val must be greater than or equal to min_val")

        large_int = self.to_int()

        # Calculate the mathematical boundary for a perfectly uniform distribution
        limit = (1 << 512) - ((1 << 512) % range_size)

        if large_int < limit:
            return min_val + (large_int % range_size)
        raise RuntimeError(
            "TRNG Exhaustion: The physical entropy from the image fell into the "
            f"rejection zone (value >= {limit}). To maintain strict TRNG purity, "
            "no PRNG fallback was used. Please extract a seed from a new image."
        )

    def to_bool(self) -> bool:
        """Convert the exact pure TRNG seed into a single boolean value.

        Returns:
            True or False.
        """
        return (self._raw_seed[0] & 1) == 1

    def to_float(self) -> float:
        """Convert the TRNG seed into a float in ``[0.0, 1.0)``.

        Returns:
            Random float in ``[0.0, 1.0)``.
        """
        # Pull the first 7 bytes, shift right by 3 to get exactly 53 bits
        val = int.from_bytes(self._raw_seed[:7], byteorder="big") >> 3

        # Divide by 2^53 for exact, zero-rounding-error floating point math
        return val * (2.0**-53)

    def to_float_range(self, min_val: float, max_val: float) -> float:
        """Convert the TRNG seed into a float within ``[min_val, max_val)`` (half-open).

        Args:
            min_val: Lower bound (inclusive).
            max_val: Upper bound (exclusive).

        Returns:
            Random float in ``[min_val, max_val)``.

        Raises:
            ValueError: If max_val is less than min_val.
        """
        if max_val < min_val:
            raise ValueError("max_val must be greater than or equal to min_val")

        factor = self.to_float()
        return min_val + (factor * (max_val - min_val))
