# Entropy Estimation: Measuring the Source

This document explains **why** and **how** photorand measures the entropy of its
raw source *before* cryptographic conditioning — and why this matters for
anyone relying on the output for high-value keys.

## The Problem: A Hash Hides Bias

photorand's original design sampled the 4 least-significant bits of every
64th sensor pixel and assumed they were uniformly random. The sampled pool
was then passed through SHA3-512 to produce a 64-byte seed.

The problem is that **a cryptographic hash is a whitener, not an entropy
multiplier.** SHA3-512 distributes existing entropy uniformly across its
output, but a deterministic function can only conserve or reduce entropy —
never increase it. If the raw LSB pool genuinely holds 50 bits of
min-entropy, the 512-bit digest still holds only ~50 bits.

The digest will *pass statistical tests* regardless, because hashing hides
bias. This is exactly why you cannot use test-passing as evidence of
entropy: the test needs to run on the **raw source**, not the conditioned
output.

## The Solution: Measure First, Then Condition

Following the NIST SP 800-90B principle ("measure the source first, then
condition"), photorand now runs a full entropy assessment on the raw
sampled pool *before* SHA3-512 conditioning. The assessment is accessible
via the `assess` CLI command and the `PhotoRandSeed.assessment` property.

## Estimators Implemented

All entropy values are expressed in **bits per symbol** (4-bit symbols for
the default sampler).

### Most Common Value (MCV) Estimator — §6.1

The primary and most conservative min-entropy estimator. It bounds
min-entropy by the frequency of the single most common value, with a
one-sided confidence correction at the 2⁻²⁰ significance level:

```
p_max = max_count / n
p_bound = min(1, p_max + 4.753 × √(p_max × (1 − p_max) / n))
H_MCV = −log₂(p_bound)
```

For a perfectly uniform 4-bit source, H_MCV approaches 4.0 bits/symbol as
the sample count grows. For a biased source, it correctly identifies the
deficit.

### Collision Estimator — §6.3

Records the distance between *consecutive* occurrences of each symbol
(not the birthday-problem "first collision"), then converts the mean
collision time to a min-entropy estimate:

```
X̄ = mean collision distance
X̄_lower = X̄ − 4.753 × σ/√c    (lower confidence bound)
H_collision = −log₂(1 − e^(−1/X̄_lower))
```

For a uniform 4-bit source, the average distance between consecutive
occurrences of the same value is ~16, yielding ~4 bits/symbol.

### Shannon Entropy

The classic information-theoretic average entropy:

```
H = −Σ p_i × log₂(p_i)
```

Shannon entropy is always ≥ min-entropy. It provides an upper bound on the
information content but is **not** the right metric for worst-case
cryptographic security (min-entropy is).

### Conservative Min-Entropy

The final reported min-entropy is the **minimum** of the MCV and collision
estimates — the most conservative value. This is the number used to
calculate how many bytes of true entropy are present:

```
total_entropy_bits = min_entropy × sample_count
max_safe_seed_bytes = min(total_entropy_bits / 8, 64)
```

## Statistical Tests

### Chi-Square Uniformity Test

Tests whether the symbol distribution is consistent with a discrete
uniform distribution over all 2^k values:

```
χ² = Σ (observed_i − expected)² / expected
```

with `df = alphabet_size − 1` degrees of freedom. The test fails (flags
non-uniformity) when `p < 0.01`.

**Important:** For FPN-reduced data, the chi-square test may fail because
the *true stochastic noise* is not perfectly uniform — but the min-entropy
is still high. For raw (non-FPN-reduced) data, the test may pass because
the deterministic sensor fingerprint spreads values artificially. Always
look at the min-entropy estimate, not just the chi-square result.

### Repetition Count Test — §4.4.1

Detects stuck-at faults or catastrophic bias by flagging runs of identical
consecutive values that are too long to occur by chance. For 4-bit symbols,
the cutoff is 12 (i.e., 12+ identical values in a row triggers a failure).

### Adaptive Proportion Test — §4.4.2

Uses non-overlapping 512-sample windows. For each window, counts how often
the first value appears; a count above the cutoff indicates distributional
drift. For 4-bit symbols with 512-sample windows, the cutoff is ~59.

## Fixed-Pattern Noise Reduction

Real camera sensors exhibit **fixed-pattern noise (FPN)** — dark-signal
non-uniformity, row/column offsets, and pixel-response non-uniformity
(PRNU). This noise is a **deterministic per-sensor signature**: it
reproduces identically across captures of the same camera.

Without removal, these deterministic bits masquerade as entropy when in
fact they are predictable — a sensor fingerprint, not randomness.

photorand now subtracts row-wise and column-wise medians before extracting
LSBs, which removes the dominant additive FPN components. The residual
that remains is dominated by the stochastic noise floor — shot noise,
read noise, and thermal noise — which is genuine entropy.

This is enabled by default. Use `--no-fpn` to disable it for comparison.

### Real-World Comparison

On a Sony A7-series dark capture (DSC02111.ARW):

| Mode | Min-Entropy (bits/sym) | Total Bits | Chi-Square |
|------|----------------------|------------|------------|
| FPN-reduced (default) | 3.44 | 20,573 | FAIL (p=0.002) |
| Raw (no FPN reduction) | 3.60 | 21,515 | PASS (p=0.99) |

The raw mode appears more uniform, but part of that "entropy" is the
deterministic sensor fingerprint. The FPN-reduced mode is **more honest**:
it shows the true stochastic entropy. Both modes provide far more than the
512-bit seed target.

## Using the Assessment

### CLI

```bash
# Human-readable report
python -m src assess --from path/to/image.ARW

# JSON output for scripting
python -m src assess --from path/to/image.ARW --json

# Compare with and without FPN reduction
python -m src assess --from path/to/image.ARW --no-fpn
```

### Python API

```python
from src import PhotoRandSeed

seed = PhotoRandSeed("path/to/image.ARW")
assessment = seed.assessment

print(f"Min-entropy: {assessment.min_entropy:.3f} bits/symbol")
print(f"Total:       {assessment.total_entropy_bits:.1f} bits")
print(f"Status:      {assessment.overall_status}")
print(f"Sufficient:  {assessment.sufficient_for_seed}")
```

### Low-Level API

```python
from src.low_level import generate_with_assessment

seed, pool, assessment = generate_with_assessment("path/to/image.ARW")
```

## Recommendations

1. **Use dark-frame captures** (lens cap on, high ISO) for maximum
   stochastic noise and minimum signal contamination.
2. **Check the assessment** before trusting a seed for high-value keys.
3. **A "WARN" status is acceptable** if the min-entropy is well above the
   seed target — it just means the distribution isn't perfectly uniform.
4. **A "FAIL" status** means a health check detected a catastrophic
   problem (stuck values or extreme bias). Do not use the seed.
