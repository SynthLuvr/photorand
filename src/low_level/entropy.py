"""Entropy estimation and statistical health testing for raw entropy pools.

Implements NIST SP 800-90B estimators and health checks to measure the
actual min-entropy of the sampled pool *before* SHA3-512 conditioning.
A hash distributes entropy but cannot create it: a 512-bit digest of a
50-bit source still contains ~50 bits.  Statistical tests pass on hash
output regardless, so the measurement must happen on the raw source.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.logger import logger

# One-sided z-score for a 2^-20 false-positive rate (NIST SP 800-90B).
_Z_99_9999 = 4.753

# Significance level for the chi-square uniformity test.
_CHI_ALPHA = 0.01

# scipy is not a dependency, so P(a, x) and Q(a, x) are implemented directly
# using the series / continued-fraction pair from Numerical Recipes (3rd ed.),
# used here for chi-square p-values.


def _gamma_series(a: float, x: float) -> float:
    """Series expansion for the lower regularised gamma P(a, x).

    Converges quickly when ``x < a + 1``.
    """
    gln = math.lgamma(a)
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gamma_cf(a: float, x: float) -> float:
    """Lentz's continued fraction for the upper regularised gamma Q(a, x).

    Converges quickly when ``x >= a + 1``.
    """
    tiny = 1e-30
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-x + a * math.log(x) - gln)


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``."""
    if x <= 0.0 or a <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


def _chi2_sf(statistic: float, df: int) -> float:
    """Survival function ``P(χ² > statistic)`` for *df* degrees of freedom."""
    if df <= 0:
        return 1.0
    return _gamma_q(df / 2.0, statistic / 2.0)


@dataclass(frozen=True)
class EntropyAssessment:
    """Entropy analysis of a raw entropy pool, in bits per symbol."""

    # --- Meta ---
    sample_count: int
    bits_per_symbol: int
    symbol_alphabet_size: int

    # --- Min-entropy estimators (NIST SP 800-90B) ---
    most_common_value_estimate: float
    collision_estimate: float
    shannon_entropy: float

    # --- Conservative summary ---
    min_entropy: float
    total_entropy_bits: float
    max_seed_bytes: int

    # --- Chi-square uniformity test ---
    chi_square_statistic: float
    chi_square_p_value: float
    is_uniform: bool

    # --- Health checks (NIST SP 800-90B §4.4) ---
    repetition_count_passed: bool
    repetition_count_max_run: int
    repetition_count_cutoff: int
    adaptive_proportion_passed: bool
    adaptive_proportion_max_count: int
    adaptive_proportion_cutoff: int
    adaptive_proportion_windows: int

    @property
    def passed_health_checks(self) -> bool:
        return self.repetition_count_passed and self.adaptive_proportion_passed

    @property
    def sufficient_for_seed(self) -> bool:
        return self.total_entropy_bits >= 512.0

    @property
    def overall_status(self) -> str:
        if not self.passed_health_checks:
            return "FAIL"
        if not self.sufficient_for_seed:
            return "LOW"
        if not self.is_uniform:
            return "WARN"
        return "GOOD"


def _extract_symbols(data: bytes, bits_per_symbol: int) -> list[int]:
    """Extract one symbol per byte (low *bits_per_symbol* bits)."""
    if bits_per_symbol <= 0 or bits_per_symbol > 8:
        raise ValueError("bits_per_symbol must be in [1, 8]")
    mask = (1 << bits_per_symbol) - 1
    return [b & mask for b in data]


def _mcv_estimate(symbols: list[int], bits_per_symbol: int) -> float:
    """NIST SP 800-90B Most Common Value estimator (§6.1).

    Bounds min-entropy by the frequency of the most common value, with a
    one-sided confidence correction.
    """
    n = len(symbols)
    if n == 0:
        return 0.0
    counter = Counter(symbols)
    _, max_count = counter.most_common(1)[0]
    p_max = max_count / n
    # Upper confidence bound on p_max (one-sided, 2^-20).
    p_bound = min(
        1.0,
        p_max + _Z_99_9999 * math.sqrt(p_max * (1.0 - p_max) / n),
    )
    estimate = float(bits_per_symbol) if p_bound < 1e-12 else -math.log2(p_bound)
    return max(0.0, min(estimate, float(bits_per_symbol)))


def _collision_estimate(symbols: list[int], bits_per_symbol: int) -> float:
    """NIST SP 800-90B Collision estimator (§6.3).

    Records the distance between consecutive occurrences of each symbol
    (not the birthday-problem "first collision"), converts the mean
    collision time to a probability, then to a min-entropy estimate.
    """
    last_pos: dict[int, int] = {}
    collision_times: list[int] = []
    for i, s in enumerate(symbols):
        if s in last_pos:
            collision_times.append(i - last_pos[s])
        last_pos[s] = i

    if not collision_times:
        return float(bits_per_symbol)

    c = len(collision_times)
    mean_t = sum(collision_times) / c
    if mean_t < 1.0:
        return 0.0
    if c > 1:
        variance = sum((t - mean_t) ** 2 for t in collision_times) / (c - 1)
        stderr = math.sqrt(variance / c)
    else:
        stderr = 0.0

    # Lower confidence bound on mean collision time.
    lower = max(mean_t - _Z_99_9999 * stderr, 0.1)

    p = 1.0 - math.exp(-1.0 / lower)
    if p <= 0.0:
        return 0.0
    return min(-math.log2(p), float(bits_per_symbol))


def _shannon_entropy(symbols: list[int], bits_per_symbol: int) -> float:
    """Shannon (average) entropy in bits per symbol."""
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = Counter(symbols)
    h = 0.0
    for count in counts.values():
        p = count / n
        h -= p * math.log2(p)
    return min(h, float(bits_per_symbol))


def _chi_square_test(symbols: list[int], alphabet_size: int) -> tuple[float, float, bool]:
    """Goodness-of-fit test against a discrete uniform distribution.

    Returns ``(statistic, p_value, is_uniform)`` where *is_uniform* is
    ``True`` when ``p_value >= 0.01``.
    """
    n = len(symbols)
    if n == 0 or alphabet_size <= 1:
        return 0.0, 1.0, True
    expected = n / alphabet_size
    counts = Counter(symbols)
    statistic = 0.0
    for i in range(alphabet_size):
        observed = counts.get(i, 0)
        diff = observed - expected
        statistic += diff * diff / expected
    df = alphabet_size - 1
    p_value = _chi2_sf(statistic, df)
    return statistic, p_value, p_value >= _CHI_ALPHA


def _repetition_count_test(symbols: list[int], bits_per_symbol: int) -> tuple[bool, int, int]:
    """NIST SP 800-90B Repetition Count Test (§4.4.1).

    Detects stuck-at faults by flagging runs of identical consecutive
    values that are too long to occur by chance.

    Returns ``(passed, max_run_length, cutoff)``.
    """
    p = 2.0 ** (-bits_per_symbol)
    cutoff = 1 + math.ceil(-1.0 / math.log2(1.0 - p))

    max_run = 1
    current = 1
    for i in range(1, len(symbols)):
        if symbols[i] == symbols[i - 1]:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 1

    return max_run < cutoff, max_run, cutoff


def _adaptive_proportion_test(
    symbols: list[int], alphabet_size: int, window: int = 512
) -> tuple[bool, int, int, int]:
    """NIST SP 800-90B Adaptive Proportion Test (§4.4.2).

    Uses non-overlapping windows.  For each window, counts how often the
    first sample repeats; a count above the cutoff indicates drift away
    from uniform.

    Returns ``(passed, max_count, cutoff, num_windows)``.
    """
    n = len(symbols)
    if n == 0:
        return True, 0, 0, 0

    w = min(window, n)
    p = 1.0 / alphabet_size
    mu = w * p
    sigma = math.sqrt(w * p * (1.0 - p))
    cutoff = int(math.ceil(mu + _Z_99_9999 * sigma))

    max_count = 0
    num_windows = 0
    idx = 0
    while idx + w <= n:
        target = symbols[idx]
        count = sum(1 for j in range(idx, idx + w) if symbols[j] == target)
        if count > max_count:
            max_count = count
        num_windows += 1
        idx += w

    return max_count <= cutoff, max_count, cutoff, num_windows


def estimate_entropy(data: bytes, bits_per_symbol: int = 4) -> EntropyAssessment:
    """Perform a full entropy assessment on a raw byte pool.

    Runs all NIST SP 800-90B min-entropy estimators, a chi-square
    uniformity test, and startup health checks.

    Args:
        data: Raw entropy bytes — one symbol per byte (the low
            *bits_per_symbol* bits of each byte are used).
        bits_per_symbol: Bit-width of each symbol.  ``4`` matches the
            default sampler (4-LSB extraction).  Must be in ``[1, 8]``.

    Returns:
        A frozen :class:`EntropyAssessment` with every estimate and test.
    """
    alphabet_size = 1 << bits_per_symbol
    symbols = _extract_symbols(data, bits_per_symbol)
    n = len(symbols)

    logger.info(
        "[entropy] Assessing %d %d-bit symbols (alphabet size %d).",
        n,
        bits_per_symbol,
        alphabet_size,
    )

    mcv_est = _mcv_estimate(symbols, bits_per_symbol)
    coll_est = _collision_estimate(symbols, bits_per_symbol)
    shannon = _shannon_entropy(symbols, bits_per_symbol)

    # Conservative min-entropy = minimum of the min-entropy estimators.
    min_entropy = min(mcv_est, coll_est)
    total_bits = min_entropy * n
    max_bytes = min(int(total_bits // 8), 64)

    chi_stat, chi_p, uniform = _chi_square_test(symbols, alphabet_size)
    rep_ok, rep_max, rep_cut = _repetition_count_test(symbols, bits_per_symbol)
    ap_ok, ap_max, ap_cut, ap_win = _adaptive_proportion_test(symbols, alphabet_size)

    assessment = EntropyAssessment(
        sample_count=n,
        bits_per_symbol=bits_per_symbol,
        symbol_alphabet_size=alphabet_size,
        most_common_value_estimate=mcv_est,
        collision_estimate=coll_est,
        shannon_entropy=shannon,
        min_entropy=min_entropy,
        total_entropy_bits=total_bits,
        max_seed_bytes=max_bytes,
        chi_square_statistic=chi_stat,
        chi_square_p_value=chi_p,
        is_uniform=uniform,
        repetition_count_passed=rep_ok,
        repetition_count_max_run=rep_max,
        repetition_count_cutoff=rep_cut,
        adaptive_proportion_passed=ap_ok,
        adaptive_proportion_max_count=ap_max,
        adaptive_proportion_cutoff=ap_cut,
        adaptive_proportion_windows=ap_win,
    )

    logger.info(
        "[entropy] Min-entropy %.3f bits/sym · total %.1f bits · χ² p=%.4f · health %s · status %s",
        min_entropy,
        total_bits,
        chi_p,
        "PASS" if assessment.passed_health_checks else "FAIL",
        assessment.overall_status,
    )

    if not assessment.sufficient_for_seed:
        logger.warning(
            "[entropy] Measured entropy (%.1f bits) is below the "
            "512-bit seed target.  Consider a dark-frame capture at high ISO.",
            total_bits,
        )

    return assessment
