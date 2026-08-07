"""Shared seed formatting/output helpers for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from src.high_level.seed import PhotoRandSeed


def format_seed_value(
    seed: PhotoRandSeed, fmt: str, args: argparse.Namespace
) -> str | int | float | bool:
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
    if out is None:
        print(value)
        return

    data = seed.to_bytes() if binary else f"{value}\n"
    mode = "wb" if binary else "w"
    with open(out, mode) as f:
        f.write(data)
