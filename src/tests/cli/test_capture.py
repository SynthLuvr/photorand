"""Tests for the 'capture' CLI subcommand.

Strategy mirrors test_main.py: ``PhotoRandSeed`` is patched so no real camera
or RAW file is touched.  The assessment is injected to exercise the
warn-vs-refuse policy.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import patch

from src.cli.main import main
from src.low_level.capture import WebcamCaptureError
from src.low_level.entropy import EntropyAssessment, EntropyHealthError, InsufficientEntropyError

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MOCK_SEED: bytes = bytes(range(64))


def _assessment(
    *,
    repetition_count_passed: bool = True,
    adaptive_proportion_passed: bool = True,
    total_entropy_bits: float = 600.0,
    is_uniform: bool = True,
    min_entropy: float = 2.5,
) -> EntropyAssessment:
    """Build an EntropyAssessment for CLI policy tests."""
    return EntropyAssessment(
        sample_count=256,
        bits_per_symbol=4,
        symbol_alphabet_size=16,
        most_common_value_estimate=2.4,
        collision_estimate=2.6,
        shannon_entropy=2.9,
        min_entropy=min_entropy,
        total_entropy_bits=total_entropy_bits,
        max_seed_bytes=64,
        chi_square_statistic=12.0,
        chi_square_p_value=0.6,
        is_uniform=is_uniform,
        repetition_count_passed=repetition_count_passed,
        repetition_count_max_run=4,
        repetition_count_cutoff=12,
        adaptive_proportion_passed=adaptive_proportion_passed,
        adaptive_proportion_max_count=40,
        adaptive_proportion_cutoff=59,
        adaptive_proportion_windows=1,
    )


def _good_assessment() -> EntropyAssessment:
    return _assessment()


class CaptureResult(NamedTuple):
    """Captured stdout/stderr plus any SystemExit code."""

    out: str
    err: str
    exit_code: int | str | None


def _run_capture(
    args_list: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    assessment: EntropyAssessment | None = None,
    from_webcam_side_effect: BaseException | None = None,
) -> CaptureResult:
    """Invoke the capture subcommand with PhotoRandSeed mocked."""
    monkeypatch.setattr(sys, "argv", ["photorand", *args_list])

    with patch("src.cli.capture.PhotoRandSeed") as mock_seed_cls:
        mock_seed = mock_seed_cls.return_value
        if from_webcam_side_effect is not None:
            mock_seed_cls.from_webcam.side_effect = from_webcam_side_effect
        else:
            mock_seed_cls.from_webcam.return_value = mock_seed

        mock_seed.to_bytes.return_value = MOCK_SEED
        mock_seed.to_hex_string.return_value = MOCK_SEED.hex()
        mock_seed.to_int.return_value = int.from_bytes(MOCK_SEED, "big")
        mock_seed.to_int_range.return_value = 50
        mock_seed.to_float_range.return_value = 1.5
        mock_seed.to_bool.return_value = True
        mock_seed.to_float.return_value = 0.5
        mock_seed.assessment = assessment if assessment is not None else _good_assessment()

        exit_code: int | str | None = None
        try:
            main()
        except SystemExit as exc:
            exit_code = exc.code

    captured = capsys.readouterr()
    return CaptureResult(captured.out, captured.err, exit_code)


# ===========================================================================
# Output formats — stdout
# ===========================================================================


class TestCaptureStdout:
    def test_hex(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        result = _run_capture(["capture", "hex"], monkeypatch, capsys)
        assert result.exit_code is None
        assert result.out.strip() == MOCK_SEED.hex()

    def test_int(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        result = _run_capture(["capture", "int"], monkeypatch, capsys)
        assert int(result.out.strip()) == int.from_bytes(MOCK_SEED, "big")

    def test_int_range(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(
            ["capture", "int-range", "--min", "1", "--max", "100"], monkeypatch, capsys
        )
        assert 1 <= int(result.out.strip()) <= 100

    def test_float_range(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(
            ["capture", "float-range", "--min", "1.0", "--max", "2.0"], monkeypatch, capsys
        )
        assert 1.0 <= float(result.out.strip()) <= 2.0

    def test_bool(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture", "bool"], monkeypatch, capsys)
        assert result.out.strip() == "True"

    def test_float(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture", "float"], monkeypatch, capsys)
        assert float(result.out.strip()) == 0.5


# ===========================================================================
# Summary line (stderr)
# ===========================================================================


class TestCaptureSummary:
    def test_summary_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture", "hex"], monkeypatch, capsys)
        assert "[capture]" in result.err
        assert "GOOD" in result.err

    def test_stdout_stays_clean(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture", "hex"], monkeypatch, capsys)
        # Only the seed value reaches stdout; the summary is on stderr.
        assert result.out.strip() == MOCK_SEED.hex()


# ===========================================================================
# File output
# ===========================================================================


class TestCaptureFileOutput:
    def test_text_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "seed.txt"
        result = _run_capture(["capture", "hex", "--out", str(out_file)], monkeypatch, capsys)

        assert result.exit_code is None
        assert result.out == ""
        assert out_file.read_text() == MOCK_SEED.hex() + "\n"

    def test_binary_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "seed.bin"
        _run_capture(["capture", "hex", "--out", str(out_file), "--binary"], monkeypatch, capsys)
        assert out_file.read_bytes() == MOCK_SEED


# ===========================================================================
# Weak-source policy (warn vs refuse)
# ===========================================================================


class TestCaptureWeakSource:
    def test_health_failure_refuses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        result = _run_capture(
            ["capture", "hex"],
            monkeypatch,
            capsys,
            from_webcam_side_effect=EntropyHealthError("health checks FAILED"),
        )

        assert result.exit_code == 1
        assert result.out == ""  # no seed emitted
        assert "FAILED" in caplog.text

    def test_low_entropy_refuses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        result = _run_capture(
            ["capture", "hex"],
            monkeypatch,
            capsys,
            from_webcam_side_effect=InsufficientEntropyError("below the 512-bit floor"),
        )

        assert result.exit_code == 1
        assert result.out == ""  # no seed emitted
        assert "below the 512-bit" in caplog.text


# ===========================================================================
# Error handling
# ===========================================================================


class TestCaptureErrors:
    def test_webcam_capture_error_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        result = _run_capture(
            ["capture", "hex"],
            monkeypatch,
            capsys,
            from_webcam_side_effect=WebcamCaptureError("no camera"),
        )

        assert result.exit_code == 1
        assert "no camera" in caplog.text

    def test_missing_format_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture"], monkeypatch, capsys)
        assert result.exit_code == 2
        assert "error" in result.err.lower()

    def test_int_range_requires_min_and_max(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_capture(["capture", "int-range"], monkeypatch, capsys)
        assert result.exit_code == 2
        assert "error" in result.err.lower()


# ===========================================================================
# Argument pass-through
# ===========================================================================


class TestCaptureArgs:
    def test_duration_and_camera_passed_to_from_webcam(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["photorand", "capture", "hex", "--duration", "3", "--camera", "1"]
        )

        with (
            patch("src.cli.capture.PhotoRandSeed") as mock_seed_cls,
            contextlib.suppress(SystemExit),
        ):
            mock_seed = mock_seed_cls.return_value
            mock_seed_cls.from_webcam.return_value = mock_seed
            mock_seed.to_hex_string.return_value = MOCK_SEED.hex()
            mock_seed.assessment = _good_assessment()

            main()

        mock_seed_cls.from_webcam.assert_called_once_with(duration=3.0, camera_index=1)
        capsys.readouterr()  # drain

    def test_duration_defaults_to_five(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["photorand", "capture", "hex"])

        with (
            patch("src.cli.capture.PhotoRandSeed") as mock_seed_cls,
            contextlib.suppress(SystemExit),
        ):
            mock_seed = mock_seed_cls.return_value
            mock_seed_cls.from_webcam.return_value = mock_seed
            mock_seed.to_hex_string.return_value = MOCK_SEED.hex()
            mock_seed.assessment = _good_assessment()

            main()

        mock_seed_cls.from_webcam.assert_called_once_with(duration=5.0, camera_index=0)
        capsys.readouterr()
