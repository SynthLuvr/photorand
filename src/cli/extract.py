"""Handle the 'extract' subcommand — extract 64 bytes of true physical entropy."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.high_level.seed import PhotoRandSeed
from src.logger import logger

if TYPE_CHECKING:
    import argparse


def handle_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand: extract 64 bytes of true physical entropy.

    Args:
        args: Parsed CLI arguments.
    """
    image_path: str = args.image_path
    fmt: str = args.format
    out: str | None = getattr(args, "out", None)
    binary: bool = getattr(args, "binary", False)

    try:
        seed = PhotoRandSeed(image_path)
    except (FileNotFoundError, IsADirectoryError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error extracting entropy: %s", e)
        sys.exit(1)

    # Format the result using high-level methods
    if fmt == "hex":
        result: str | int | float | bool = seed.to_hex_string()
    elif fmt == "int":
        result = seed.to_int()
    elif fmt == "int-range":
        min_val: int = args.min
        max_val: int = args.max
        result = seed.to_int_range(min_val, max_val)
    elif fmt == "float-range":
        min_val_f: float = args.min
        max_val_f: float = args.max
        result = seed.to_float_range(min_val_f, max_val_f)
    elif fmt == "bool":
        result = seed.to_bool()
    elif fmt == "float":
        result = seed.to_float()
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Write to file or stdout
    if out:
        if binary:
            with open(out, "wb") as f:
                f.write(seed.to_bytes())
        else:
            with open(out, "w") as f:
                f.write(str(result) + "\n")
    else:
        print(result)
