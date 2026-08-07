"""Tests for src.cli.main CLI (subparser architecture).

Strategy:
  - Patch ``src.cli.extract.PhotoRandSeed`` so tests never hit the disk.
  - Patch ``src.cli.generate.PhotoRandEngine`` for ``generate`` subcommand tests
    so the CSPRNG expansion is also deterministic and instant.
  - Use monkeypatch to control sys.argv.
  - Each test targets one distinct behaviour of the CLI.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import patch

from src.cli.main import main
from src.low_level.entropy import InsufficientEntropyError

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ---------------------------------------------------------------------------
# Shared mock values
# ---------------------------------------------------------------------------

# A fixed 64-byte seed returned by `generate_true_random_number` mock.
MOCK_SEED: bytes = bytes(range(64))  # b'\x00\x01\x02...\x3f'

# A fixed 256-byte keystream returned by `expand_entropy_chacha20` mock.
MOCK_KEYSTREAM: bytes = bytes(range(256))


class CaptureResult(NamedTuple):
    """Captured stdout/stderr pair (mirrors pytest's internal type)."""

    out: str
    err: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run_cli(
    args_list: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    seed_val: bytes = MOCK_SEED,
    keystream: bytes = MOCK_KEYSTREAM,
) -> CaptureResult:
    """Set sys.argv and invoke main() with both core dependencies mocked.

    Returns the captured stdout/stderr pair.
    """
    monkeypatch.setattr(sys, "argv", ["photorand", *args_list])

    with (
        patch("src.cli.extract.PhotoRandSeed") as MockSeed,
        patch("src.cli.generate.PhotoRandEngine") as MockEngine,
    ):
        # Setup mock_seed instance
        mock_seed = MockSeed.return_value
        mock_seed.to_bytes.return_value = seed_val
        mock_seed.to_hex_string.return_value = seed_val.hex()
        mock_seed.to_int.return_value = int.from_bytes(seed_val, "big")
        # Simple deterministic mock for range
        mock_seed.to_int_range.return_value = 10
        mock_seed.to_float_range.return_value = 1.5

        # Setup mock_engine instance
        mock_engine = MockEngine.return_value

        def mock_next_bytes(length: int) -> bytes:
            return keystream[:length]

        def mock_next_int(length: int = 8) -> int:
            return int.from_bytes(keystream[:length], "big")

        def mock_next_string(length: int, charset: str) -> str:
            return "a" * length

        mock_engine.next_bytes.side_effect = mock_next_bytes
        mock_engine.next_int.side_effect = mock_next_int
        mock_engine.next_string.side_effect = mock_next_string
        mock_engine.next_int_range.return_value = 5
        mock_engine.next_float_range.return_value = 1.5

        with contextlib.suppress(SystemExit):
            main()

    captured = capsys.readouterr()
    return CaptureResult(out=captured.out, err=captured.err)


# ===========================================================================
# 'extract' subcommand  –  stdout
# ===========================================================================


class TestExtractStdout:
    """extract writes formatted entropy to stdout."""

    def test_hex_is_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["extract", "hex", "--from", "fake.arw"], monkeypatch, capsys)
        assert out.out.strip() == MOCK_SEED.hex()

    def test_explicit_hex_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["extract", "hex", "--from", "fake.arw"], monkeypatch, capsys)
        assert out.out.strip() == MOCK_SEED.hex()

    def test_int_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["extract", "int", "--from", "fake.arw"], monkeypatch, capsys)
        expected = int.from_bytes(MOCK_SEED, "big")
        assert int(out.out.strip()) == expected

    def test_int_range_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["extract", "int-range", "--from", "fake.arw", "--min", "10", "--max", "20"],
            monkeypatch,
            capsys,
        )
        value = int(out.out.strip())
        assert 10 <= value <= 20

    def test_float_range_format(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["extract", "float-range", "--from", "fake.arw", "--min", "1.0", "--max", "2.0"],
            monkeypatch,
            capsys,
        )
        value = float(out.out.strip())
        assert 1.0 <= value <= 2.0


# ===========================================================================
# 'extract' subcommand  –  file output
# ===========================================================================


class TestExtractFileOutput:
    """extract writes to file when --out is provided."""

    def test_text_file_int(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "result.txt"
        run_cli(
            ["extract", "int", "--from", "fake.arw", "--out", str(out_file)],
            monkeypatch,
            capsys,
        )
        expected = str(int.from_bytes(MOCK_SEED, "big")) + "\n"
        assert out_file.read_text() == expected

    def test_text_file_hex(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "result.txt"
        run_cli(
            ["extract", "hex", "--from", "fake.arw", "--out", str(out_file)],
            monkeypatch,
            capsys,
        )
        assert out_file.read_text() == MOCK_SEED.hex() + "\n"

    def test_binary_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "result.bin"
        run_cli(
            ["extract", "hex", "--from", "fake.arw", "--out", str(out_file), "--binary"],
            monkeypatch,
            capsys,
        )
        assert out_file.read_bytes() == MOCK_SEED

    def test_text_file_nothing_on_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "result.txt"
        out = run_cli(
            ["extract", "hex", "--from", "fake.arw", "--out", str(out_file)],
            monkeypatch,
            capsys,
        )
        assert out.out == ""


# ===========================================================================
# 'extract' subcommand  –  validation errors
# ===========================================================================


class TestExtractValidation:
    """extract raises argparse errors for missing required arguments."""

    def test_int_range_without_min_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["extract", "int-range", "--from", "fake.arw", "--max", "20"],
            monkeypatch,
            capsys,
        )
        assert "error" in out.err.lower()

    def test_int_range_without_max_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["extract", "int-range", "--from", "fake.arw", "--min", "10"],
            monkeypatch,
            capsys,
        )
        assert "error" in out.err.lower()

    def test_int_range_without_both_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["extract", "int-range", "--from", "fake.arw"], monkeypatch, capsys)
        assert "error" in out.err.lower()

    def test_missing_image_path_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["extract", "hex"], monkeypatch, capsys)
        assert "error" in out.err.lower()


class TestExtractEntropyFloor:
    """extract honours the entropy floor and refuses weak sources."""

    def test_insufficient_entropy_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["photorand", "extract", "hex", "--from", "fake.arw"])
        exit_code: int | str | None = None
        with patch(
            "src.cli.extract.PhotoRandSeed",
            side_effect=InsufficientEntropyError("below the 512-bit floor"),
        ):
            try:
                main()
            except SystemExit as exc:
                exit_code = exc.code

        assert exit_code == 1
        assert "below the 512-bit" in caplog.text


# ===========================================================================
# 'generate' subcommand  –  stdout
# ===========================================================================


class TestGenerateStdout:
    """generate writes CSPRNG output to stdout."""

    def test_bytes_type(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "bytes", "--from", "fake.arw", "-n", "2", "-l", "4"],
            monkeypatch,
            capsys,
        )
        lines = out.out.strip().splitlines()
        assert len(lines) == 2
        # Each line is a hex string of length 8 (4 bytes * 2 hex chars)
        for line in lines:
            assert len(line) == 8
            int(line, 16)  # must be valid hex

    def test_int_type(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "int", "--from", "fake.arw", "-n", "3"],
            monkeypatch,
            capsys,
        )
        lines = out.out.strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            int(line)  # must be a valid integer

    def test_string_type(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "string", "--from", "fake.arw", "-n", "2", "-l", "8"],
            monkeypatch,
            capsys,
        )
        lines = out.out.strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert len(line) == 8

    def test_string_charset_hex(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "string", "--from", "fake.arw", "-n", "1", "-l", "16", "--charset", "hex"],
            monkeypatch,
            capsys,
        )
        line = out.out.strip()
        assert len(line) == 16
        assert all(c in "0123456789abcdef" for c in line)

    def test_string_charset_alpha(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            [
                "generate",
                "string",
                "--from",
                "fake.arw",
                "-n",
                "1",
                "-l",
                "10",
                "--charset",
                "alpha",
            ],
            monkeypatch,
            capsys,
        )
        line = out.out.strip()
        assert line.isalpha()

    def test_int_range_type(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "int-range", "--from", "fake.arw", "--min", "1", "--max", "6", "-n", "5"],
            monkeypatch,
            capsys,
        )
        lines = out.out.strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            assert 1 <= int(line) <= 6

    def test_float_range_type(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            [
                "generate",
                "float-range",
                "--from",
                "fake.arw",
                "--min",
                "1.0",
                "--max",
                "6.0",
                "-n",
                "5",
            ],
            monkeypatch,
            capsys,
        )
        lines = out.out.strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            assert 1.0 <= float(line) <= 6.0

    def test_count_defaults_to_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "bytes", "--from", "fake.arw", "-l", "8"],
            monkeypatch,
            capsys,
        )
        assert len(out.out.strip().splitlines()) == 1


# ===========================================================================
# 'generate' subcommand  –  file output
# ===========================================================================


class TestGenerateFileOutput:
    """generate writes to file when -o / --out is provided."""

    def test_bytes_to_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "random_bytes.txt"
        out = run_cli(
            ["generate", "bytes", "--from", "fake.arw", "-n", "2", "-l", "4", "-o", str(out_file)],
            monkeypatch,
            capsys,
        )
        # stdout should be empty
        assert out.out == ""
        lines = out_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_int_to_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "numbers.txt"
        run_cli(
            ["generate", "int", "--from", "fake.arw", "-n", "4", "-o", str(out_file)],
            monkeypatch,
            capsys,
        )
        lines = out_file.read_text().strip().splitlines()
        assert len(lines) == 4
        for line in lines:
            int(line)


# ===========================================================================
# 'generate' subcommand  –  validation errors
# ===========================================================================


class TestGenerateValidation:
    """generate raises argparse errors for missing required arguments."""

    def test_missing_type_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["generate", "--from", "fake.arw"], monkeypatch, capsys)
        assert "error" in out.err.lower()

    def test_int_range_without_min_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "int-range", "--from", "fake.arw", "--max", "6"],
            monkeypatch,
            capsys,
        )
        assert "error" in out.err.lower()

    def test_int_range_without_max_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(
            ["generate", "int-range", "--from", "fake.arw", "--min", "1"],
            monkeypatch,
            capsys,
        )
        assert "error" in out.err.lower()

    def test_missing_image_path_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli(["generate", "bytes"], monkeypatch, capsys)
        assert "error" in out.err.lower()


# ===========================================================================
# Top-level parser  –  routing
# ===========================================================================


class TestRouting:
    """Ensure the top-level parser routes to the correct subcommand."""

    def test_no_subcommand_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = run_cli([], monkeypatch, capsys)
        assert "error" in out.err.lower()

    def test_extract_routes_correctly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A successful extract run produces valid hex output
        out = run_cli(["extract", "hex", "--from", "fake.arw"], monkeypatch, capsys)
        assert out.out.strip() == MOCK_SEED.hex()

    def test_generate_routes_correctly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A successful generate run produces at least one output line
        out = run_cli(["generate", "bytes", "--from", "fake.arw", "-l", "8"], monkeypatch, capsys)
        assert len(out.out.strip()) > 0
