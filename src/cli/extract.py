"""Handle the 'extract' subcommand — extract 64 bytes of true physical entropy."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.cli.seed_output import emit_seed_output, format_seed_value
from src.high_level.seed import PhotoRandSeed
from src.logger import logger
from src.low_level.entropy import EntropyHealthError, InsufficientEntropyError

if TYPE_CHECKING:
    import argparse


def handle_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand: extract true physical entropy from a RAW image.

    The entropy floor is enforced inside :class:`PhotoRandSeed`; ``--allow-weak``
    overrides it to emit a truncated seed.

    Args:
        args: Parsed CLI arguments.
    """
    allow_weak: bool = getattr(args, "allow_weak", False)

    try:
        seed = PhotoRandSeed(args.image_path, allow_weak=allow_weak)
    except (
        FileNotFoundError,
        IsADirectoryError,
        EntropyHealthError,
        InsufficientEntropyError,
    ) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error extracting entropy: %s", e)
        sys.exit(1)

    value = format_seed_value(seed, args.format, args)
    emit_seed_output(seed, value, getattr(args, "out", None), getattr(args, "binary", False))
