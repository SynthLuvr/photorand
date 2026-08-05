"""Integration tests for the photorand CLI with real images."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

from src.cli.main import main


def run_cli_integration(
    args_list: list[str], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> str:
    """Run the CLI and return captured stdout."""
    monkeypatch.setattr(sys, "argv", ["main.py", *args_list])
    with contextlib.suppress(SystemExit):
        main()
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize(
    ("image_filename", "expected_int", "expected_int_range"),
    [
        (
            "DSC02111.ARW",
            5441230399755021634661702627294439837643765743048098986473439861345281873618919255001375917882451696456632001720700828097273286841568547436452202507760768,  # noqa: E501
            20,
        ),
        (
            "IMG_6775.CR2",
            1983112000144019161378659574769935184153364571863522866163921903212591155076214460720178892638770579202958750617616524851798091481695319032837458803953969,  # noqa: E501
            20,
        ),
        (
            "DSC03088.ARW",
            3295623491421879308436711099041673549102516067157422785057540737909732005548384056302810108111505029508815460346658820938042570895769000942097100592489198,  # noqa: E501
            16,
        ),
        (
            "DSC03089.ARW",
            406985617972906688707974285432327461848063680588482669067211356871213446809847124085706365799856761437958785114466726225109296019541509202268552018876523,  # noqa: E501
            16,
        ),
    ],
)
def test_extract_integration(
    image_filename: str,
    expected_int: int,
    expected_int_range: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = Path(__file__).resolve().parent.parent
    image_file = base_dir / "data" / image_filename

    # Check if the file exists; if not, gracefully skip this specific iteration
    if not image_file.exists():
        pytest.skip(f"Test data file {image_filename} not found. Skipping test.")

    image_path = str(image_file)

    # Test hex format (default)
    out_hex = run_cli_integration(["extract", "hex", "--from", image_path], capsys, monkeypatch)
    assert len(out_hex) == 128  # 64 bytes * 2 chars/byte

    # Test int format
    out_int = run_cli_integration(["extract", "int", "--from", image_path], capsys, monkeypatch)
    assert int(out_int) == expected_int

    # Test int-range format
    out_int_range = run_cli_integration(
        ["extract", "int-range", "--from", image_path, "--min", "10", "--max", "20"],
        capsys,
        monkeypatch,
    )
    assert int(out_int_range) == expected_int_range

    # Test float-range format
    out_float_range = run_cli_integration(
        ["extract", "float-range", "--from", image_path, "--min", "10.0", "--max", "20.0"],
        capsys,
        monkeypatch,
    )
    assert 10.0 <= float(out_float_range) <= 20.0

    # Test negative float-range
    out_neg_float = run_cli_integration(
        ["extract", "float-range", "--from", image_path, "--min", "-2.0", "--max", "-1.0"],
        capsys,
        monkeypatch,
    )
    assert -2.0 <= float(out_neg_float) <= -1.0

    # Test bool format
    out_bool = run_cli_integration(["extract", "bool", "--from", image_path], capsys, monkeypatch)
    assert out_bool in ["True", "False"]

    # Test float format
    out_float = run_cli_integration(["extract", "float", "--from", image_path], capsys, monkeypatch)
    f_val = float(out_float)
    assert 0.0 <= f_val <= 1.0


def test_generate_integration(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the generate subcommand end-to-end with a real image."""
    base_dir = Path(__file__).resolve().parent.parent
    image_path = str(base_dir / "data" / "DSC02111.ARW")

    # Generate 5 integers
    out = run_cli_integration(
        ["generate", "int", "--from", image_path, "-n", "5"], capsys, monkeypatch
    )
    lines = out.splitlines()
    assert len(lines) == 5
    for line in lines:
        int(line)  # Valid integer

    # Generate integers with specific digits
    out_digits = run_cli_integration(
        ["generate", "int", "--from", image_path, "--digits", "100", "-n", "2"],
        capsys,
        monkeypatch,
    )
    lines_digits = out_digits.splitlines()
    assert len(lines_digits) == 2
    for line in lines_digits:
        assert len(line) == 100
        int(line)

    # Generate a specific string
    out_str = run_cli_integration(
        ["generate", "string", "--from", image_path, "-n", "1", "-l", "32", "--charset", "hex"],
        capsys,
        monkeypatch,
    )
    assert len(out_str) == 32
    assert all(c in "0123456789abcdef" for c in out_str)

    # Generate string with numeric-only
    out_num_str = run_cli_integration(
        ["generate", "string", "--from", image_path, "-n", "1", "-l", "32", "--numeric-only"],
        capsys,
        monkeypatch,
    )
    assert len(out_num_str) == 32
    assert out_num_str.isdigit()

    # Generate 3 bools
    out_bools = run_cli_integration(
        ["generate", "bool", "--from", image_path, "-n", "3"], capsys, monkeypatch
    )
    lines_bool = out_bools.splitlines()
    assert len(lines_bool) == 3
    for line in lines_bool:
        assert line in ["True", "False"]

    # Generate 3 floats
    out_floats = run_cli_integration(
        ["generate", "float", "--from", image_path, "-n", "3"], capsys, monkeypatch
    )
    lines_float = out_floats.splitlines()
    assert len(lines_float) == 3
    for line in lines_float:
        assert 0.0 <= float(line) <= 1.0

    # Generate 3 floats in range
    out_float_ranges = run_cli_integration(
        [
            "generate",
            "float-range",
            "--from",
            image_path,
            "--min",
            "50.5",
            "--max",
            "100.5",
            "-n",
            "3",
        ],
        capsys,
        monkeypatch,
    )
    lines_float_range = out_float_ranges.splitlines()
    assert len(lines_float_range) == 3
    for line in lines_float_range:
        assert 50.5 <= float(line) <= 100.5

    # Generate 3 integers in range
    out_int_ranges = run_cli_integration(
        ["generate", "int-range", "--from", image_path, "--min", "1", "--max", "6", "-n", "3"],
        capsys,
        monkeypatch,
    )
    lines_int_range = out_int_ranges.splitlines()
    assert len(lines_int_range) == 3
    for line in lines_int_range:
        assert 1 <= int(line) <= 6

    # Generate negative integers in range
    out_neg_int = run_cli_integration(
        ["generate", "int-range", "--from", image_path, "--min", "-10", "--max", "-5", "-n", "1"],
        capsys,
        monkeypatch,
    )
    assert -10 <= int(out_neg_int.strip()) <= -5


def test_generate_deterministic(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that --deterministic produces identical sequences across runs."""
    base_dir = Path(__file__).resolve().parent.parent
    image_path = str(base_dir / "data" / "DSC02111.ARW")

    # Run 1
    out1 = run_cli_integration(
        ["generate", "int", "--from", image_path, "-n", "10", "--deterministic"],
        capsys,
        monkeypatch,
    )

    # Run 2
    out2 = run_cli_integration(
        ["generate", "int", "--from", image_path, "-n", "10", "--deterministic"],
        capsys,
        monkeypatch,
    )

    assert out1 == out2
    assert len(out1.splitlines()) == 10


def test_generate_nondeterministic(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that default (salted) generation produces different sequences."""
    base_dir = Path(__file__).resolve().parent.parent
    image_path = str(base_dir / "data" / "DSC02111.ARW")

    # Run 1
    out1 = run_cli_integration(
        ["generate", "int", "--from", image_path, "-n", "10"], capsys, monkeypatch
    )

    # Run 2
    out2 = run_cli_integration(
        ["generate", "int", "--from", image_path, "-n", "10"], capsys, monkeypatch
    )

    # It is statistically impossible (1 in ~2^512) for these to match
    assert out1 != out2
