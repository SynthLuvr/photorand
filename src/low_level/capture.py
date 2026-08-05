"""Webcam entropy capture — harvest temporal sensor noise via frame differencing.

This is an *optional* entropy source.  It depends on ``opencv-python-headless``
(the ``capture`` extra) and imports it lazily, so the core package keeps working
for users who do not install it.

A webcam is treated as a new **ingest** function: it returns a 2-D ``np.ndarray``
of accumulated temporal noise that the existing ``sample → estimate → hash`` chain
consumes unchanged.  Consecutive-frame differencing cancels the static scene and
fixed-pattern noise, leaving the stochastic read/shot-noise floor that constitutes
genuine entropy.  See ``docs/webcam-capture-library.md``.
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING

import numpy as np

from src.logger import logger
from src.low_level.entropy import EntropyAssessment, estimate_entropy
from src.low_level.hash import hash_entropy_pool
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# Consecutive read() failures before giving up on a camera mid-capture.
_MAX_READ_FAILURES = 10

# Frames discarded up front so auto-exposure / white-balance can settle.
_WARMUP_FRAMES = 5


class WebcamCaptureError(RuntimeError):
    """Raised when webcam entropy capture cannot proceed.

    Covers the three failure cases specific to this source: the optional
    dependency is missing, no camera is present or accessible, or too few
    usable frames were captured.
    """


def _import_cv2() -> Any:
    """Import cv2 lazily, raising a helpful error if the optional extra is missing.

    ``importlib.import_module`` (rather than a top-level ``import cv2``) keeps the
    module importable — and pyright's reportMissingImports quiet — on systems
    without the ``capture`` extra installed.
    """
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise WebcamCaptureError(
            "Webcam capture requires the optional 'opencv-python-headless' package.\n"
            "Install it with one of:\n"
            "    uv sync --extra capture\n"
            "    pip install 'photorand[capture]'"
        ) from exc


def capture_webcam_noise(
    duration: float = 5.0,
    camera_index: int = 0,
) -> np.ndarray:
    """Capture temporal sensor noise from a webcam for *duration* seconds.

    Requests an uncompressed (``YUYV``) pixel format where the backend allows it,
    then accumulates the absolute differences between consecutive frames.
    Differencing cancels the static scene and fixed-pattern noise, so what
    remains — and is summed — is the temporal noise floor, i.e. the genuine
    entropy.  The resulting array is ready to feed into
    :func:`sample_entropy_grid`.

    Args:
        duration: How long to capture, in seconds.
        camera_index: OpenCV device index (``0`` is the default camera).

    Returns:
        A 2-D ``float64`` array of accumulated per-pixel temporal noise.

    Raises:
        WebcamCaptureError: If the optional dependency is missing, the camera
            cannot be opened, or fewer than two usable frame differences were
            captured.
    """
    cv2 = _import_cv2()

    cap: Any = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise WebcamCaptureError(
            f"Could not open webcam at index {camera_index}. "
            "Check that the device is connected and not in use by another application."
        )

    # Best-effort: request uncompressed YUYV.  CAP_PROP_FOURCC is backend-dependent
    # and may be silently ignored (notably on Windows CAP_DSHOW).  cap.set() returns
    # a success flag rather than raising; if it reports failure we log and rely on
    # the entropy estimator as the downstream safety net for compressed input.
    fourcc: Any = cv2.VideoWriter_fourcc("Y", "U", "Y", "V")
    if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        logger.warning(
            "[capture] Backend would not commit to YUYV; relying on the "
            "entropy estimator to reject any silently-compressed frames."
        )

    try:
        # Let the camera's auto-exposure / white-balance settle before measuring.
        for _ in range(_WARMUP_FRAMES):
            cap.read()

        accumulator: np.ndarray | None = None
        prev: np.ndarray | None = None
        differences = 0
        failures = 0

        start = time.monotonic()
        while time.monotonic() - start < duration:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= _MAX_READ_FAILURES:
                    break
                continue
            failures = 0

            gray = np.asarray(frame, dtype=np.float64)
            if gray.ndim == 3:  # BGR (or similar) -> single-channel luminance
                gray = gray.mean(axis=2)

            if prev is not None:
                diff = np.abs(gray - prev)
                if accumulator is None:
                    accumulator = diff.copy()
                else:
                    accumulator += diff
                differences += 1
            prev = gray
    finally:
        cap.release()

    if differences < 2 or accumulator is None:
        raise WebcamCaptureError(
            f"Captured only {differences} usable frame difference(s) from camera "
            f"{camera_index}; need at least two. Is the camera producing frames?"
        )

    logger.info(
        "[capture] %d frame difference(s) over %.2fs from camera %d -> noise grid of shape %s.",
        differences,
        duration,
        camera_index,
        accumulator.shape,
    )
    return accumulator


def generate_from_webcam(
    duration: float = 5.0,
    camera_index: int = 0,
    *,
    sample_fn: Callable[..., bytes] = sample_entropy_grid,
    sample_grid_spacing: int = 4,
    reduce_fpn: bool = False,
    hash_fn: Callable[[bytes], bytes] = hash_entropy_pool,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Capture webcam noise and run it through the standard entropy pipeline.

    This mirrors :func:`src.low_level.generate.generate_with_assessment` but
    ingests from a webcam instead of a RAW file, so the captured data flows
    through the *same* sample → estimate → hash chain.

    Args:
        duration: Capture length in seconds.
        camera_index: OpenCV device index.
        sample_fn: Sampler callable.  Defaults to :func:`sample_entropy_grid`.
        sample_grid_spacing: Grid stride passed to the sampler.  Webcams expose
            far fewer pixels than a RAW sensor, so a tighter spacing than the
            RAW default (64) is used to harvest enough symbols.
        reduce_fpn: Whether the sampler should reduce fixed-pattern noise.
            Defaults to ``False`` because frame differencing already cancels
            the static FPN, and re-subtracting medians from near-uniform
            temporal noise can over-concentrate the residuals.
        hash_fn: Conditioning hash.  Defaults to :func:`hash_entropy_pool`.

    Returns:
        ``(seed, entropy_pool, assessment)`` — the 64-byte SHA3-512 seed, the
        raw pre-hash entropy bytes, and the :class:`EntropyAssessment`.
    """
    noise = capture_webcam_noise(duration=duration, camera_index=camera_index)
    entropy_pool = sample_fn(noise, grid_spacing=sample_grid_spacing, reduce_fpn=reduce_fpn)
    assessment = estimate_entropy(entropy_pool)
    secure_hash_bytes = hash_fn(entropy_pool)
    return secure_hash_bytes, entropy_pool, assessment
