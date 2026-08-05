"""Handle the 'generate' subcommand — expand a TRNG seed via ChaCha20."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.high_level.engine import PhotoRandEngine
from src.logger import logger

if TYPE_CHECKING:
    import argparse


def handle_generate(args: argparse.Namespace) -> None:
    """Handle the 'generate' subcommand: expand a TRNG seed via ChaCha20.

    Args:
        args: Parsed CLI arguments.
    """
    image_path: str = args.image_path
    deterministic: bool = getattr(args, "deterministic", False)
    gen_type: str = args.type
    count: int = getattr(args, "count", 1)
    out: str | None = getattr(args, "out", None)
    length: int | None = getattr(args, "length", None)
    digits: int | None = getattr(args, "digits", None)

    try:
        engine = PhotoRandEngine(image_path, salt=not deterministic)
    except (FileNotFoundError, IsADirectoryError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error during CSPRNG generation: %s", e)
        sys.exit(1)

    logger.info("[generate] type=%s, count=%d, length=%s", gen_type, count, length)

    results: list[str] = []

    if gen_type == "bytes":
        results = [engine.next_bytes(length if length else 32).hex() for _ in range(count)]

    elif gen_type == "int":
        original_limit = sys.get_int_max_str_digits()
        needed_digits = digits if digits else (length if length else 8) * 3

        if needed_digits > original_limit:
            sys.set_int_max_str_digits(needed_digits)

        try:
            if digits:
                results = [str(engine.next_int_digits(digits)) for _ in range(count)]
            else:
                results = [str(engine.next_int(length if length else 8)) for _ in range(count)]
        finally:
            sys.set_int_max_str_digits(original_limit)

    elif gen_type == "string":
        numeric_only: bool = getattr(args, "numeric_only", False)
        charset: str = getattr(args, "charset", "all")
        actual_charset = "numeric" if numeric_only else charset
        results = [
            engine.next_string(length if length else 16, actual_charset) for _ in range(count)
        ]

    elif gen_type == "int-range":
        min_val: int = args.min
        max_val: int = args.max
        results = [str(engine.next_int_range(min_val, max_val)) for _ in range(count)]

    elif gen_type == "float-range":
        min_val_f: float = args.min
        max_val_f: float = args.max
        results = [str(engine.next_float_range(min_val_f, max_val_f)) for _ in range(count)]

    elif gen_type == "bool":
        results = [str(engine.next_bool()) for _ in range(count)]

    elif gen_type == "float":
        results = [str(engine.next_float()) for _ in range(count)]

    output = "\n".join(results)

    if out:
        with open(out, "w") as f:
            f.write(output + "\n")
        logger.info("[generate] Output written to %s", out)
    else:
        print(output)
