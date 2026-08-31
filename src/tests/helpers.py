"""Shared test helpers: an ``EntropyAssessment`` factory reused across suites."""

from __future__ import annotations

from src.low_level.entropy import EntropyAssessment


def make_assessment(
    *,
    total_entropy_bits: float = 600.0,
    max_seed_bytes: int = 64,
    min_entropy: float = 2.4,
    is_uniform: bool = True,
    repetition_count_passed: bool = True,
    adaptive_proportion_passed: bool = True,
) -> EntropyAssessment:
    """Build an ``EntropyAssessment`` with controlled gate-relevant fields."""
    return EntropyAssessment(
        sample_count=256,
        bits_per_symbol=4,
        symbol_alphabet_size=16,
        most_common_value_estimate=2.4,
        collision_estimate=2.6,
        markov_estimate=2.5,
        shannon_entropy=2.9,
        min_entropy=min_entropy,
        total_entropy_bits=total_entropy_bits,
        max_seed_bytes=max_seed_bytes,
        chi_square_statistic=12.0,
        chi_square_p_value=0.6,
        is_uniform=is_uniform,
        repetition_count_passed=repetition_count_passed,
        repetition_count_max_run=4,
        repetition_count_cutoff=12,
        adaptive_proportion_passed=adaptive_proportion_passed,
        adaptive_proportion_max_count=40,
        adaptive_proportion_cutoff=59,
        adaptive_proportion_windows=1,
    )
