"""Webcam entropy capture — harvest temporal sensor noise via frame differencing.

Backed by ``opencv-python-headless``. Consecutive-frame differencing cancels the
static scene and fixed-pattern noise, leaving the read/shot-noise floor that is
genuine entropy.
"""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.logger import logger
from src.low_level.generate import condition_entropy_pool
from src.low_level.hash import hash_entropy_pool
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from src.low_level.entropy import EntropyAssessment

_MAX_READ_FAILURES = 10
_WARMUP_FRAMES = 5
# MJPG is tried before YUYV: it is near-universal and OpenCV decodes it into valid
# BGR. Requesting YUYV from an MJPG-only sensor instead yields a corrupt, near-flat
# green frame with no usable entropy, so YUYV is only a fallback for sensors lacking
# MJPG.
_PREFERRED_FOURCCS: tuple[str, ...] = ("MJPG", "YUYV")


def _fourcc(label: str) -> int:
    """Pack a 4-char FOURCC label into the int OpenCV expects.

    Pure-Python equivalent of ``cv2.VideoWriter_fourcc(*label)``, a dynamically
    generated symbol without a type stub.
    """
    return sum(ord(c) << (8 * i) for i, c in enumerate(label))


class WebcamCaptureError(RuntimeError):
    """Raised when webcam entropy capture cannot proceed."""


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    gray = np.asarray(frame, dtype=np.float64)
    return gray.mean(axis=2) if gray.ndim == 3 else gray


def _negotiate_pixel_format(cap: Any, camera_index: int) -> None:
    """Ask the camera for MJPG, falling back to YUYV, so frames decode cleanly."""
    chosen: str | None = None
    for label in _PREFERRED_FOURCCS:
        if cap.set(cv2.CAP_PROP_FOURCC, _fourcc(label)):
            chosen = label
            break

    if chosen is None:
        logger.warning(
            "[capture] Camera %d accepted none of %s; using its default format.",
            camera_index,
            ", ".join(_PREFERRED_FOURCCS),
        )
    else:
        logger.info("[capture] Camera %d negotiated pixel format %s.", camera_index, chosen)


def _accumulate_frame_differences(cap: Any, duration: float) -> tuple[np.ndarray | None, int]:
    """Discard warmup frames, then sum the absolute difference of each consecutive
    grayscale frame pair for *duration* seconds.

    Returns the accumulated noise array (or ``None``) and the difference count.
    """
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

        gray = _to_grayscale(frame)
        if prev is not None:
            diff = np.abs(gray - prev)
            accumulator = diff if accumulator is None else accumulator + diff
            differences += 1
        prev = gray

    return accumulator, differences


def capture_webcam_noise(
    duration: float = 5.0,
    camera_index: int = 0,
) -> np.ndarray:
    """Capture temporal sensor noise from a webcam for *duration* seconds.

    Returns a 2-D ``float64`` array of accumulated consecutive-frame differences,
    ready for :func:`sample_entropy_grid`.

    Raises:
        WebcamCaptureError: If the camera cannot be opened or fewer than two frame
            differences were captured.
    """
    cap: Any = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise WebcamCaptureError(
            f"Could not open webcam at index {camera_index}. "
            "Check that the device is connected and not in use."
        )

    _negotiate_pixel_format(cap, camera_index)

    try:
        accumulator, differences = _accumulate_frame_differences(cap, duration)
    finally:
        cap.release()

    if differences < 2 or accumulator is None:
        raise WebcamCaptureError(
            f"Captured only {differences} usable frame difference(s) from camera "
            f"{camera_index}; need at least two."
        )

    logger.info(
        "[capture] %d frame difference(s) over %.2fs from camera %d -> %s.",
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
    min_entropy_bits: int = 512,
    allow_weak: bool = False,
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Capture webcam noise and run it through the standard entropy pipeline.

    A tighter grid spacing (4 vs 64) compensates for the lower pixel count, and
    fixed-pattern-noise reduction is off by default because frame differencing
    already cancels the static FPN.
    """
    noise = capture_webcam_noise(duration=duration, camera_index=camera_index)
    sampler = partial(sample_fn, grid_spacing=sample_grid_spacing, reduce_fpn=reduce_fpn)
    return condition_entropy_pool(
        noise,
        sample_fn=sampler,
        hash_fn=hash_fn,
        min_entropy_bits=min_entropy_bits,
        allow_weak=allow_weak,
    )
