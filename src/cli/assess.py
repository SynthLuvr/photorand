"""Handle the 'assess' subcommand — measure entropy quality of a RAW image."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from src.logger import logger
from src.low_level.entropy import estimate_entropy
from src.low_level.ingest import ingest_raw_image
from src.low_level.sample import sample_entropy_grid

if TYPE_CHECKING:
    import argparse

    from src.low_level.entropy import EntropyAssessment


def _print_report(assessment: EntropyAssessment, image_path: str) -> None:
    """Print a human-readable entropy assessment report."""
    status = assessment.overall_status
    status_marker = {"GOOD": "✓", "WARN": "⚠", "LOW": "⚠", "FAIL": "✗"}.get(status, "?")

    print(f"\n{'=' * 60}")
    print(f"  photorand Entropy Assessment — {image_path}")
    print(f"{'=' * 60}")
    print()
    print(f"  Samples:         {assessment.sample_count:,d}")
    print(
        f"  Symbol width:    {assessment.bits_per_symbol} bits "
        f"(alphabet size {assessment.symbol_alphabet_size})"
    )
    print()
    print("  Min-Entropy Estimates (bits per symbol)")
    print(f"    Most Common Value:  {assessment.most_common_value_estimate:.3f}")
    print(f"    Collision:          {assessment.collision_estimate:.3f}")
    print(f"    Shannon:            {assessment.shannon_entropy:.3f}")
    print(f"    {'─' * 36}")
    print(f"    Conservative min:   {assessment.min_entropy:.3f} bits/symbol")
    print()
    total = assessment.total_entropy_bits
    print(f"  Total entropy:   {total:,.1f} bits")
    print(
        f"  Max safe seed:   {assessment.max_seed_bytes} bytes "
        f"({assessment.max_seed_bytes * 8} bits)"
    )
    if assessment.sufficient_for_seed:
        print("                   ✓ Sufficient for a 512-bit seed")
    else:
        print("                   ⚠ BELOW the 512-bit seed target")
    print()
    print("  Statistical Tests")
    chi_marker = "✓" if assessment.is_uniform else "✗"
    print(
        f"    Chi-square:         χ²={assessment.chi_square_statistic:.2f} "
        f"(p={assessment.chi_square_p_value:.4f})  {chi_marker} "
        f"{'PASS' if assessment.is_uniform else 'FAIL'} (α=0.01)"
    )

    rep_marker = "✓" if assessment.repetition_count_passed else "✗"
    print(
        f"    Repetition Count:   max run={assessment.repetition_count_max_run} "
        f"(cutoff={assessment.repetition_count_cutoff})  {rep_marker} "
        f"{'PASS' if assessment.repetition_count_passed else 'FAIL'}"
    )

    ap_marker = "✓" if assessment.adaptive_proportion_passed else "✗"
    print(
        f"    Adaptive Proportion: max count={assessment.adaptive_proportion_max_count} "
        f"(cutoff={assessment.adaptive_proportion_cutoff}, "
        f"{assessment.adaptive_proportion_windows} windows)  {ap_marker} "
        f"{'PASS' if assessment.adaptive_proportion_passed else 'FAIL'}"
    )
    print()
    print(f"  Overall: {status_marker} {status}")
    print()
    print("  Note: A hash (SHA3-512) distributes entropy but cannot create it.")
    print("        These measurements are on the RAW source, before conditioning,")
    print("        so they reflect the true entropy content of the sensor noise.")
    print("        For best results, use a dark-frame capture at high ISO.")
    print(f"{'=' * 60}\n")


def handle_assess(args: argparse.Namespace) -> None:
    """Handle the 'assess' subcommand: measure entropy quality of a RAW image.

    Args:
        args: Parsed CLI arguments.
    """
    image_path: str = args.image_path
    reduce_fpn: bool = not args.no_fpn
    json_output: bool = args.json

    try:
        raw_image_data = ingest_raw_image(image_path)
    except Exception as e:
        logger.error("Error reading image: %s", e)
        sys.exit(1)

    entropy_pool = sample_entropy_grid(raw_image_data, reduce_fpn=reduce_fpn)
    assessment = estimate_entropy(entropy_pool)

    if json_output:
        print(
            json.dumps(
                {
                    "image_path": image_path,
                    "sample_count": assessment.sample_count,
                    "bits_per_symbol": assessment.bits_per_symbol,
                    "most_common_value_estimate": round(assessment.most_common_value_estimate, 4),
                    "collision_estimate": round(assessment.collision_estimate, 4),
                    "shannon_entropy": round(assessment.shannon_entropy, 4),
                    "min_entropy": round(assessment.min_entropy, 4),
                    "total_entropy_bits": round(assessment.total_entropy_bits, 2),
                    "max_seed_bytes": assessment.max_seed_bytes,
                    "chi_square_statistic": round(assessment.chi_square_statistic, 4),
                    "chi_square_p_value": round(assessment.chi_square_p_value, 6),
                    "is_uniform": assessment.is_uniform,
                    "repetition_count_passed": assessment.repetition_count_passed,
                    "repetition_count_max_run": assessment.repetition_count_max_run,
                    "repetition_count_cutoff": assessment.repetition_count_cutoff,
                    "adaptive_proportion_passed": assessment.adaptive_proportion_passed,
                    "adaptive_proportion_max_count": assessment.adaptive_proportion_max_count,
                    "adaptive_proportion_cutoff": assessment.adaptive_proportion_cutoff,
                    "adaptive_proportion_windows": assessment.adaptive_proportion_windows,
                    "passed_health_checks": assessment.passed_health_checks,
                    "sufficient_for_seed": assessment.sufficient_for_seed,
                    "overall_status": assessment.overall_status,
                },
                indent=2,
            )
        )
    else:
        _print_report(assessment, image_path)
