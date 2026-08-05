"""Handle the 'capture' subcommand — extract entropy from a webcam."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.cli.seed_output import emit_seed_output, format_seed_value
from src.high_level.seed import PhotoRandSeed
from src.logger import logger
from src.low_level.capture import WebcamCaptureError

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
    """Extract entropy from a webcam and emit the seed in the chosen format."""
    duration: float = args.duration
    camera_index: int = args.camera

    try:
        seed = PhotoRandSeed.from_webcam(duration=duration, camera_index=camera_index)
    except WebcamCaptureError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error capturing webcam entropy: %s", e)
        sys.exit(1)

    assessment = seed.assessment
    _print_capture_summary(assessment, duration, camera_index)

    if not assessment.passed_health_checks and not getattr(args, "allow_weak", False):
        logger.error(
            "Entropy health checks FAILED (status: FAIL). Refusing to emit a seed "
            "from a faulty source. Re-run with --allow-weak to emit anyway."
        )
        sys.exit(1)

    if not assessment.sufficient_for_seed:
        logger.warning(
            "Captured entropy (%.1f bits) is below the 512-bit seed target; "
            "the seed is still emitted but consider a longer --duration.",
            assessment.total_entropy_bits,
        )

    value = format_seed_value(seed, args.format, args)
    emit_seed_output(seed, value, getattr(args, "out", None), getattr(args, "binary", False))
