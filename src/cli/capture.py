"""Handle the 'capture' subcommand — extract entropy from a webcam."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.cli.seed_output import emit_seed_output, format_seed_value
from src.high_level.seed import PhotoRandSeed
from src.logger import logger
from src.low_level.capture import WebcamCaptureError
from src.low_level.entropy import EntropyHealthError, InsufficientEntropyError

if TYPE_CHECKING:
    import argparse

    from src.low_level.entropy import EntropyAssessment


def _print_capture_summary(
    assessment: EntropyAssessment, duration: float, camera_index: int
) -> None:
    print(
        f"[capture] camera {camera_index} · {duration:g}s · status "
        f"{assessment.overall_status} · {assessment.total_entropy_bits:,.1f} bits "
        f"(min-entropy {assessment.min_entropy:.3f} bits/symbol)",
        file=sys.stderr,
    )


def handle_capture(args: argparse.Namespace) -> None:
    """Extract entropy from a webcam and emit the seed in the chosen format.

    The entropy floor is enforced inside :meth:`PhotoRandSeed.from_webcam`; the
    library refuses (by default) to emit more bits than were measured.  ``--allow-weak``
    overrides that to emit a seed truncated to the measured entropy bound.
    """
    duration: float = args.duration
    camera_index: int = args.camera
    allow_weak: bool = getattr(args, "allow_weak", False)

    try:
        seed = PhotoRandSeed.from_webcam(
            duration=duration, camera_index=camera_index, allow_weak=allow_weak
        )
    except (WebcamCaptureError, EntropyHealthError, InsufficientEntropyError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error capturing webcam entropy: %s", e)
        sys.exit(1)

    assessment = seed.assessment
    _print_capture_summary(assessment, duration, camera_index)

    value = format_seed_value(seed, args.format, args)
    emit_seed_output(seed, value, getattr(args, "out", None), getattr(args, "binary", False))
