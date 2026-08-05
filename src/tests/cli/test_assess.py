"""Tests for the 'assess' CLI subcommand."""

from __future__ import annotations

import contextlib
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from src.cli.main import main
from src.low_level.entropy import EntropyAssessment


def _make_mock_assessment() -> EntropyAssessment:
    """Create a realistic mock EntropyAssessment for CLI testing."""
    return EntropyAssessment(
        sample_count=5000,
        bits_per_symbol=4,
        symbol_alphabet_size=16,
        most_common_value_estimate=3.8,
        collision_estimate=3.95,
        shannon_entropy=3.98,
        min_entropy=3.8,
        total_entropy_bits=19000.0,
        max_seed_bytes=64,
        chi_square_statistic=12.5,
        chi_square_p_value=0.637,
        is_uniform=True,
        repetition_count_passed=True,
        repetition_count_max_run=4,
        repetition_count_cutoff=12,
        adaptive_proportion_passed=True,
        adaptive_proportion_max_count=49,
        adaptive_proportion_cutoff=59,
        adaptive_proportion_windows=9,
    )


def _run_assess(
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Run the assess CLI with mocked dependencies."""
    monkeypatch.setattr(sys, "argv", ["photorand", *args])
    mock_data = np.zeros((10, 10), dtype=np.uint16)

    with (
        patch("src.cli.assess.ingest_raw_image", return_value=mock_data),
        patch("src.cli.assess.sample_entropy_grid", return_value=b"\x00" * 100),
        patch("src.cli.assess.estimate_entropy", return_value=_make_mock_assessment()),
        contextlib.suppress(SystemExit),
    ):
        main()

    return capsys.readouterr().out


class TestAssessReport:
    """assess prints a human-readable report."""

    def test_report_contains_key_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = _run_assess(["assess", "--from", "fake.arw"], monkeypatch, capsys)
        assert "Entropy Assessment" in out
        assert "Min-Entropy" in out
        assert "Chi-square" in out
        assert "Repetition Count" in out
        assert "Adaptive Proportion" in out
        assert "GOOD" in out

    def test_report_shows_sample_count(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = _run_assess(["assess", "--from", "fake.arw"], monkeypatch, capsys)
        assert "5,000" in out


class TestAssessJSON:
    """assess --json outputs valid JSON."""

    def test_json_is_valid(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = _run_assess(["assess", "--from", "fake.arw", "--json"], monkeypatch, capsys)
        data = json.loads(out)
        assert data["sample_count"] == 5000
        assert data["min_entropy"] == 3.8
        assert data["overall_status"] == "GOOD"
        assert data["sufficient_for_seed"] is True

    def test_json_contains_health_checks(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = _run_assess(["assess", "--from", "fake.arw", "--json"], monkeypatch, capsys)
        data = json.loads(out)
        assert data["passed_health_checks"] is True
        assert data["repetition_count_passed"] is True
        assert data["adaptive_proportion_passed"] is True


class TestAssessFPNFlag:
    """assess --no-fpn disables FPN reduction."""

    def test_no_fpn_flag_passes_through(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["photorand", "assess", "--from", "fake.arw", "--no-fpn"])
        mock_data = np.zeros((10, 10), dtype=np.uint16)

        with (
            patch("src.cli.assess.ingest_raw_image", return_value=mock_data) as mock_ingest,
            patch("src.cli.assess.sample_entropy_grid", return_value=b"\x00" * 100) as mock_sample,
            patch("src.cli.assess.estimate_entropy", return_value=_make_mock_assessment()),
            contextlib.suppress(SystemExit),
        ):
            main()

        capsys.readouterr()
        # sample_entropy_grid should have been called with reduce_fpn=False
        mock_sample.assert_called_once_with(mock_ingest.return_value, reduce_fpn=False)


class TestAssessErrors:
    """assess handles errors gracefully."""

    def test_file_not_found_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["photorand", "assess", "--from", "missing.arw"])

        with (
            patch("src.cli.assess.ingest_raw_image", side_effect=FileNotFoundError("nope")),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        assert "nope" in caplog.text
