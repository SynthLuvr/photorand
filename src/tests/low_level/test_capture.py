"""Tests for src.low_level.capture — webcam entropy capture.

External hardware is mocked: a fake ``cv2`` module feeds synthetic frames so no
real camera is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.low_level import capture as cap
from src.low_level import generate
from src.low_level.capture import WebcamCaptureError, capture_webcam_noise, generate_from_webcam
from src.low_level.entropy import EntropyAssessment
from src.low_level.sample import DegenerateEntropyPoolError

# This test file mocks package internals (capture.cv2 / _WARMUP_FRAMES).
# pyright: reportPrivateUsage=false


def _assessment(
    *,
    total_entropy_bits: float = 600.0,
    max_seed_bytes: int = 64,
    repetition_count_passed: bool = True,
    adaptive_proportion_passed: bool = True,
) -> EntropyAssessment:
    """Build an EntropyAssessment with controlled gate-relevant fields."""
    return EntropyAssessment(
        sample_count=256,
        bits_per_symbol=4,
        symbol_alphabet_size=16,
        most_common_value_estimate=2.4,
        collision_estimate=2.6,
        markov_estimate=2.5,
        shannon_entropy=2.9,
        min_entropy=2.4,
        total_entropy_bits=total_entropy_bits,
        max_seed_bytes=max_seed_bytes,
        chi_square_statistic=12.0,
        chi_square_p_value=0.6,
        is_uniform=True,
        repetition_count_passed=repetition_count_passed,
        repetition_count_max_run=4,
        repetition_count_cutoff=12,
        adaptive_proportion_passed=adaptive_proportion_passed,
        adaptive_proportion_max_count=40,
        adaptive_proportion_cutoff=59,
        adaptive_proportion_windows=1,
    )


def _est_sufficient(_data: bytes) -> EntropyAssessment:
    return _assessment()


# ---------------------------------------------------------------------------
# Fake cv2 / VideoCapture
# ---------------------------------------------------------------------------


class _FakeVideoCapture:
    """Mimics the handful of cv2.VideoCapture methods capture_webcam_noise uses."""

    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        open_ok: bool = True,
        set_ok: bool = True,
        unsupported_fourccs: set[int] | None = None,
    ) -> None:
        self._frames = list(frames)
        self._i = 0
        self._open_ok = open_ok
        self._set_ok = set_ok
        self._unsupported_fourccs = unsupported_fourccs or set()
        self.released = False
        self.requested_index: int | None = None
        self.set_calls: list[tuple[int, int]] = []

    def isOpened(self) -> bool:
        return self._open_ok

    def set(self, prop: int, value: int) -> bool:
        self.set_calls.append((prop, value))
        if prop == _FakeCV2.CAP_PROP_FOURCC and value in self._unsupported_fourccs:
            return False
        return self._set_ok

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return True, frame
        return False, None

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FOURCC = 6

    def __init__(self, capture: _FakeVideoCapture) -> None:
        self._capture = capture

    def VideoCapture(self, index: int) -> _FakeVideoCapture:
        self._capture.requested_index = index
        return self._capture


def _noise_frames(
    n: int,
    shape: tuple[int, int, int] = (64, 64, 3),
    *,
    seed: int = 0,
    hi: int = 256,
    scene: int = 128,
) -> list[np.ndarray]:
    """Build *n* frames = static scene + per-frame random noise (clipped to uint8)."""
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    for _ in range(n):
        base = np.full(shape, scene, dtype=np.int16)
        noise = rng.integers(0, hi, size=shape).astype(np.int16)
        frames.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    return frames


def _install_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[np.ndarray],
    *,
    open_ok: bool = True,
    set_ok: bool = True,
    unsupported_fourccs: set[int] | None = None,
) -> _FakeVideoCapture:
    """Inject a fake cv2 module (backed by *frames*) into the capture module."""
    capture = _FakeVideoCapture(
        frames,
        open_ok=open_ok,
        set_ok=set_ok,
        unsupported_fourccs=unsupported_fourccs,
    )
    monkeypatch.setattr(cap, "cv2", _FakeCV2(capture))
    return capture


def _fourcc(label: str) -> int:
    """Encode a FOURCC label the way the (fake and real) cv2 does."""
    return sum(ord(c) << (8 * i) for i, c in enumerate(label))


def _gray(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def _reference_accumulator(frames: list[np.ndarray]) -> np.ndarray:
    """Reproduce the expected accumulation, skipping the warmup frames."""
    grays = [_gray(f) for f in frames[cap._WARMUP_FRAMES :]]
    acc: np.ndarray | None = None
    for prev, cur in zip(grays, grays[1:], strict=False):
        diff = np.abs(cur - prev)
        acc = diff.copy() if acc is None else acc + diff
    assert acc is not None
    return acc


# ---------------------------------------------------------------------------
# capture_webcam_noise
# ---------------------------------------------------------------------------


class TestCaptureWebcamNoise:
    def test_returns_2d_noise_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_cv2(monkeypatch, _noise_frames(20))
        noise = capture_webcam_noise(duration=5.0, camera_index=0)

        assert isinstance(noise, np.ndarray)
        assert noise.ndim == 2
        assert noise.shape == (64, 64)
        assert noise.dtype == np.float64

    def test_accumulates_frame_differences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frames = _noise_frames(12)
        _install_fake_cv2(monkeypatch, frames)

        noise = capture_webcam_noise(duration=5.0)

        # Matching the reference (which skips warmup) proves both that warmup
        # frames are discarded and that differencing/accumulation is correct.
        np.testing.assert_allclose(noise, _reference_accumulator(frames))

    def test_constant_scene_is_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A static scene must cancel out, leaving only the temporal noise."""
        scene = np.full((32, 32, 3), 200, dtype=np.uint8)
        rng = np.random.default_rng(7)
        frames = [
            np.clip(scene.astype(np.int16) + rng.integers(0, 16, size=(32, 32, 3)), 0, 255).astype(
                np.uint8
            )
            for _ in range(20)
        ]
        _install_fake_cv2(monkeypatch, frames)

        noise = capture_webcam_noise(duration=5.0)

        # Differences are small (0..15 per channel); the accumulated noise is modest.
        assert np.all(np.isfinite(noise))
        assert noise.max() < 20 * 16

    def test_releases_camera(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _install_fake_cv2(monkeypatch, _noise_frames(20))
        capture_webcam_noise(duration=5.0)
        assert capture.released

    def test_releases_camera_even_on_open_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _install_fake_cv2(monkeypatch, _noise_frames(5), open_ok=False)

        with pytest.raises(WebcamCaptureError, match="Could not open webcam"):
            capture_webcam_noise(duration=5.0)

        assert capture.released

    def test_negotiates_mjpg_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _install_fake_cv2(monkeypatch, _noise_frames(20))
        capture_webcam_noise(duration=5.0)

        fourcc_values = [v for p, v in capture.set_calls if p == _FakeCV2.CAP_PROP_FOURCC]
        # MJPG is preferred and accepted, so YUYV is never tried.
        assert _fourcc("MJPG") in fourcc_values
        assert _fourcc("YUYV") not in fourcc_values

    def test_falls_back_to_yuyv_when_mjpg_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture = _install_fake_cv2(
            monkeypatch, _noise_frames(20), unsupported_fourccs={_fourcc("MJPG")}
        )
        capture_webcam_noise(duration=5.0)

        fourcc_values = [v for p, v in capture.set_calls if p == _FakeCV2.CAP_PROP_FOURCC]
        assert _fourcc("MJPG") in fourcc_values
        assert _fourcc("YUYV") in fourcc_values

    def test_camera_index_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _install_fake_cv2(monkeypatch, _noise_frames(20))

        capture_webcam_noise(duration=5.0, camera_index=3)

        assert capture.requested_index == 3

    def test_too_few_frames_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # warmup consumes 5 frames; 2 more -> only 1 difference -> error.
        _install_fake_cv2(monkeypatch, _noise_frames(7))

        with pytest.raises(WebcamCaptureError, match="usable frame difference"):
            capture_webcam_noise(duration=5.0)

    def test_camera_not_opened_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_cv2(monkeypatch, _noise_frames(5), open_ok=False)

        with pytest.raises(WebcamCaptureError, match="Could not open webcam"):
            capture_webcam_noise(duration=5.0)


# ---------------------------------------------------------------------------
# generate_from_webcam
# ---------------------------------------------------------------------------


class TestGenerateFromWebcam:
    def test_returns_seed_pool_assessment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 128×128 frames give enough samples to exceed the 512-bit floor
        # despite the conservative Markov estimator.
        _install_fake_cv2(monkeypatch, _noise_frames(40, shape=(128, 128, 3)))

        seed, pool, assessment = generate_from_webcam(duration=5.0)

        assert isinstance(seed, bytes)
        assert len(seed) == 64
        assert isinstance(pool, bytes)
        assert len(pool) > 0
        assert isinstance(assessment, EntropyAssessment)
        assert assessment.passed_health_checks
        assert assessment.sufficient_for_seed

    def test_constant_frames_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        identical = [np.zeros((64, 64, 3), dtype=np.uint8)] * 20
        _install_fake_cv2(monkeypatch, identical)

        # A stuck-at source yields a degenerate pool, rejected before health checks.
        with pytest.raises(DegenerateEntropyPoolError, match="degenerate pool"):
            generate_from_webcam(duration=5.0)

    def test_custom_functions_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_cv2(monkeypatch, _noise_frames(20))
        # A 4-byte pool is far below the floor; mock the assessment as
        # sufficient so the plumbing (custom fns) is what is exercised.
        monkeypatch.setattr(generate, "estimate_entropy", _est_sufficient)

        def sample_fn(data: np.ndarray, grid_spacing: int = 4, reduce_fpn: bool = True) -> bytes:
            assert isinstance(data, np.ndarray)
            return b"\x01\x02\x03\x04"

        def hash_fn(pool: bytes) -> bytes:
            assert pool == b"\x01\x02\x03\x04"
            return b"x" * 64

        seed, pool, _assessment = generate_from_webcam(
            duration=5.0, sample_fn=sample_fn, hash_fn=hash_fn
        )

        assert seed == b"x" * 64
        assert pool == b"\x01\x02\x03\x04"
