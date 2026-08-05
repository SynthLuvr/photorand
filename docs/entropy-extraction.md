# The `photorand` Mechanism: Physical Entropy Extraction

This document outlines the scientific and technical architecture of `photorand`. It details how the system leverages raw camera sensor data to extract physical entropy, functioning as a True Random Number Generator (TRNG).

## Overview

The photorand extracts physical entropy from raw camera sensor data. Unlike typical pseudo-random number generators (PRNGs) that rely on mathematical algorithms starting from a seed, this photorand leverages the inherent physical noise present in digital image sensors.

## Entropy Source: Raw Sensor Noise

Digital image sensors (CMOS/CCD) are subject to various forms of physical noise, even when no light is present or when capturing a static scene. Key sources of this noise include:

*   **Shot Noise**: Random fluctuations in the number of photons hitting the sensor.
*   **Read Noise**: Electronic noise introduced during the digitization of the analog signal.
*   **Thermal Noise (Dark Current)**: Noise caused by the thermal agitation of electrons within the sensor.

By reading the **RAW** data (before any processing like denoising, demosaicing, or compression), we can isolate these tiny, unpredictable fluctuations.

## Implementation Details

### 1. Data Ingestion (`low_level/ingest.py`)
We use the `rawpy` library to access the unprocessed sensor grid. This ensures we are working with the "pure" values captured by the hardware, which contain the most entropy.

### 2. Sampling and Isolation (`low_level/sample.py`)
To extract the most random components and reduce spatial correlation (where neighboring pixels might share similar noise characteristics), we perform three steps:

*   **Grid Sampling**: We sample pixels at a fixed stride (e.g., every 64th pixel). This spreads the sampling across the entire sensor surface.
*   **Fixed-Pattern Noise Reduction**: We subtract row-wise and column-wise medians from the sampled grid to remove additive **fixed-pattern noise (FPN)** — dark-signal non-uniformity, row/column offsets, and similar deterministic per-sensor signatures. Without this step, the LSB pool is contaminated with predictable bits that reproduce across captures of the same camera (a fingerprint, not randomness). The residual that remains is dominated by the stochastic noise floor.
*   **LSB Extraction**: We isolate the 4 **Least Significant Bits (LSBs)** of each sampled pixel's noise residual. The higher-order bits represent the actual image content (which is predictable), while the LSBs are dominated by the physical noise floor.

> **Backward compatibility:** FPN reduction can be disabled with `reduce_fpn=False` to reproduce the legacy behaviour.

### 3. Entropy Estimation (`low_level/entropy.py`)
**This is the key addition.** Following the NIST SP 800-90B principle of *"measure first, then condition"*, photorand now estimates the actual min-entropy of the sampled pool *before* cryptographic conditioning.

A cryptographic hash is a **whitener**: it distributes existing entropy uniformly but can never *create* it. If the raw pool contains 50 bits of min-entropy, the 512-bit digest still contains only ~50 bits. Statistical tests will pass on the hash output regardless, which is why test-passing cannot serve as evidence of entropy.

The estimation module implements:

*   **Most Common Value (MCV) estimator** (NIST §6.1) — the primary, most conservative min-entropy bound.
*   **Collision estimator** (NIST §6.3) — based on the distance between consecutive symbol repetitions.
*   **Shannon entropy** — the information-theoretic average (always ≥ min-entropy).
*   **Chi-square goodness-of-fit test** — detects deviations from a discrete uniform distribution.
*   **Repetition Count Test** (NIST §4.4.1) — detects stuck-at faults.
*   **Adaptive Proportion Test** (NIST §4.4.2) — detects distributional drift.

The conservative min-entropy (minimum of the MCV and collision estimates) is used to compute the total entropy in the pool and the maximum number of bytes that can be safely trusted. See [Entropy Estimation](entropy-estimation.md) for full details.

### 4. Entropy Conditioning (`low_level/hash.py`)
The raw "lsb-pool" extracted from the sensor might still have small statistical biases. To produce a perfectly uniform output, we pass the extracted bytes through a **cryptographic hash function**.

*   **Algorithm**: SHA3-512
*   **Purpose**: SHA3-512 acts as an "entropy blender." It compresses the large pool of harvested noise into a 64-byte (512-bit) digest. The avalanche effect of the hash ensures that any small amount of entropy in the input is distributed uniformly across the entire output block.

> **Important:** The hash *distributes* entropy but does not *create* it. The entropy estimation in step 3 tells you exactly how much true entropy is present. If the measured entropy is below 512 bits, the 64-byte digest still only contains that many bits of entropy — it just looks uniform.

## Conclusion

The result is a 64-byte "seed" of true randomness, rooted in the physical reality of the camera sensor's environment. Unlike the original design, the system now **measures** how much entropy it actually extracted, so you can make an informed decision about whether to trust the output for your use case.

For best results, use a **dark-frame capture** (lens cap on, high ISO) where the stochastic noise floor dominates and the signal content is absent.
