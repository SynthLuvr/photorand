"""Shared seed formatting/output helpers for the CLI.

Both the file-based ``extract`` command and the webcam ``capture`` command
produce a :class:`~src.high_level.seed.PhotoRandSeed` and emit it in the same
set of output formats.  These helpers keep that formatting in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from src.high_level.seed import PhotoRandSeed


def format_seed_value(
    seed: PhotoRandSeed, fmt: str, args: argparse.Namespace
) -> str | int | float | bool:
    """Format a seed according to the chosen output *fmt*.

    Args:
        seed: The seed to format.
        fmt: One of ``hex``, ``int``, ``int-range``, ``float-range``,
            ``bool``, ``float``.
        args: Parsed CLI arguments (``--min``/``--max`` are read for the range
            formats).

    Returns:
        The formatted value (type depends on *fmt*).
    """
    if fmt == "hex":
        return seed.to_hex_string()
    if fmt == "int":
        return seed.to_int()
    if fmt == "int-range":
        return seed.to_int_range(int(args.min), int(args.max))
    if fmt == "float-range":
        return seed.to_float_range(float(args.min), float(args.max))
    if fmt == "bool":
        return seed.to_bool()
    if fmt == "float":
        return seed.to_float()
    raise ValueError(f"Unknown format: {fmt}")


def emit_seed_output(
    seed: PhotoRandSeed,
    value: str | int | float | bool,
    out: str | None,
    binary: bool,
) -> None:
    """Write the formatted value to stdout or a file.

    Args:
        seed: The seed (used for ``--binary`` raw-byte output).
        value: The formatted value (ignored when *binary* is True).
        out: Destination file path, or ``None`` for stdout.
        binary: When True and *out* is set, write the raw seed bytes.
    """
    if out:
        if binary:
            with open(out, "wb") as f:
                f.write(seed.to_bytes())
        else:
            with open(out, "w") as f:
                f.write(str(value) + "\n")
    else:
        print(value)
