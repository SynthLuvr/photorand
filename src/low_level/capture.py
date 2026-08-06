"""Webcam entropy capture — harvest temporal sensor noise via frame differencing.

Optional entropy source depending on ``opencv-python-headless`` (the ``capture``
extra).  ``cv2`` is imported lazily (at call time, not module load) so that
``import src`` never requires the extra — see the ADR
(``docs/webcam-capture-library.md``) for the full rationale.

Consecutive-frame differencing cancels the static scene and fixed-pattern noise,
leaving the stochastic read/shot-noise floor that constitutes genuine entropy.
"""

from __future__ import annotations

import importlib
import time
from functools import partial
from typing import TYPE_CHECKING

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


class WebcamCaptureError(RuntimeError):
    """Raised when webcam entropy capture cannot proceed."""


def _import_cv2() -> Any:
    """Import cv2 lazily, or raise a helpful error if the extra is missing.

    Using ``importlib`` instead of a top-level ``import cv2`` keeps the module
    importable on systems without the ``capture`` extra.
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

    Accumulates absolute differences between consecutive grayscale frames.
    Returns a 2-D ``float64`` array ready for :func:`sample_entropy_grid`.

    Raises:
        WebcamCaptureError: If the optional dependency is missing, the camera
            cannot be opened, or fewer than two frame differences were captured.
    """
    cv2 = _import_cv2()

    cap: Any = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise WebcamCaptureError(
            f"Could not open webcam at index {camera_index}. "
            "Check that the device is connected and not in use."
        )

    # Best-effort: request uncompressed YUYV. The backend may silently ignore
    # this (notably Windows CAP_DSHOW); the entropy estimator is the safety net
    # for compressed input.
    fourcc: Any = cv2.VideoWriter_fourcc("Y", "U", "Y", "V")
    if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        logger.warning(
            "[capture] Backend would not commit to YUYV; relying on the "
            "entropy estimator to reject silently-compressed frames."
        )

    try:
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
            if gray.ndim == 3:
                gray = gray.mean(axis=2)

            if prev is not None:
                diff = np.abs(gray - prev)
                accumulator = diff.copy() if accumulator is None else accumulator + diff
                differences += 1
            prev = gray
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
) -> tuple[bytes, bytes, EntropyAssessment]:
    """Capture webcam noise and run it through the standard entropy pipeline.

    Like :func:`generate_with_assessment` but ingests from a webcam.  A tighter
    grid spacing (4 vs 64) compensates for the lower pixel count.  FPN reduction
    is off by default because frame differencing already cancels the static FPN.
    """
    noise = capture_webcam_noise(duration=duration, camera_index=camera_index)
    sampler = partial(sample_fn, grid_spacing=sample_grid_spacing, reduce_fpn=reduce_fpn)
    return condition_entropy_pool(noise, sample_fn=sampler, hash_fn=hash_fn)