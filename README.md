# photorand

[![PyPI version](https://img.shields.io/pypi/v/photorand)](https://pypi.org/project/photorand/)

A True Random Number Generator (TRNG) using raw camera sensor data to extract physical entropy.

![Architecture](https://res.cloudinary.com/dhnsmw569/image/upload/photorand_dtj6xn.webp)

## Tech Stack

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Package manager & virtual environment |
| [Python](https://www.python.org) | Language (≥ 3.14, managed by uv) |
| [Pyright](https://github.com/microsoft/pyright) | Static type checking (strict mode) |
| [Ruff](https://docs.astral.sh/ruff/) | Linter and formatter |
| [pytest](https://docs.pytest.org/) | Test runner |
| [Hatch](https://hatch.pypa.io/) | Build backend |

## Project Structure

```
├── src/
│   ├── __init__.py        # Package init (re-exports, version)
│   ├── __main__.py        # CLI entry point (`uv run photorand` / `python -m src`)
│   ├── logger.py          # Central logger
│   ├── low_level/         # Modular primitives (ingest, sample, entropy, hash, csprng)
│   ├── high_level/        # OO abstractions (PhotoRandSeed, PhotoRandEngine)
│   ├── cli/               # Command-line interface (parser, handlers)
│   └── tests/             # Test suite organized by layer
├── docs/                  # Technical documentation
├── examples/              # Code samples and useful scripts
├── pyproject.toml         # Project config, deps, tool settings
└── AGENTS.md              # AI agent instructions
```

## Quick Start

```bash
uv sync                      # install dependencies
uv sync --all-extras         # also install dev deps (pytest, ruff, pyright)
uv sync --extra capture      # also enable the optional webcam source (OpenCV)

uv run pytest                # run unit tests
```

## Commands

### Type Check

```bash
uv run pyright src/          # strict type checking
```

### Lint

```bash
uv run ruff check src/       # lint all files
```

### Format

```bash
uv run ruff format src/      # format all files (writes changes)
uv run ruff format --check src/  # check formatting without writing
uv run ruff check --fix src/ # auto-fix lint issues
```

### Test

```bash
uv run pytest                # run all tests
```

---

## Usage

### 1. High-Level Classes (Recommended)

The high-level classes provide a stateful and convenient interface for both TRNG and CSPRNG operations.

#### `PhotoRandSeed` (TRNG)

Encapsulates the process of extracting entropy from a RAW image. It is perfect for generating one-off secure seeds, keys, or dice rolls directly from physical noise.

```python
from src import PhotoRandSeed

# 1. Extract entropy from a RAW image
seed = PhotoRandSeed("path/to/image.raw")

# 2. Access as bytes, hex, or large integer
print(seed.to_hex_string())
print(seed.to_int())

# 3. Check the measured entropy quality (NIST SP 800-90B)
print(f"Min-entropy: {seed.assessment.min_entropy:.3f} bits/symbol")
print(f"Total:       {seed.assessment.total_entropy_bits:.1f} bits")
print(f"Status:      {seed.assessment.overall_status}")

# 4. Roll a D100 using physical entropy (Rejection Sampling)
luck = seed.to_int_range(1, 100)

# 5. Get a float in range
prob = seed.to_float_range(0.5, 1.5)
```

> **Webcam source (optional):** You can also derive a seed from a live webcam
> capture instead of a RAW file. It requires the optional `capture` extra
> (`uv sync --extra capture`) and runs through the exact same entropy pipeline:
>
> ```python
> from src import PhotoRandSeed
>
> seed = PhotoRandSeed.from_webcam(duration=5.0, camera_index=0)
> print(seed.to_hex_string())
> print(f"Status: {seed.assessment.overall_status}")
> ```

#### `PhotoRandEngine` (CSPRNG)

An infinite stream generator powered by ChaCha20, seeded by a `PhotoRandSeed`. It handles salting (Time + PID) automatically to ensure that even consecutive runs with the same image produce unique streams.

```python
from src import PhotoRandEngine

# 1. Initialize from image or existing Seed object
engine = PhotoRandEngine("path/to/image.raw")

# 2. Pull arbitrary amounts of secure data
key = engine.next_bytes(32)
pin = engine.next_string(length=6, charset='numeric')
dice = engine.next_int_range(1, 20)
float_luck = engine.next_float_range(10.5, 20.5)
coin_flip = engine.next_bool()
probability = engine.next_float()

# 3. Batch generation
multiple_passwords = engine.generate_batch(engine.next_string, n=5, length=16)
```

#### CLI (Command Line Interface)

The package includes a CLI to use these classes directly from your terminal. Because the project
uses [uv](https://docs.astral.sh/uv/), commands are run with `uv run`. The `photorand` entry
point is registered in `pyproject.toml` and installed automatically by `uv sync`.

> Every subcommand has its own `--help`, e.g. `uv run photorand extract --help`.

**`extract`** — extract a single value of true physical entropy (64 bytes) from a RAW image:

```bash
# Extract the seed as a hex string
uv run photorand extract hex --from path/to/raw_image.ARW

# Roll a D20
uv run photorand extract int-range --from path/to/raw_image.ARW --min 1 --max 20

# Save the seed to a file: --binary writes raw bytes, omit it to write hex text
uv run photorand extract hex --from path/to/raw_image.ARW -o seed.bin --binary
```

**`generate`** — expand the seed via ChaCha20 into an arbitrary amount of CSPRNG data:

```bash
# Generate 5 random 16-char alphanumeric passwords
uv run photorand generate string --from path/to/raw_image.ARW -n 5 -l 16 --charset alpha

# 32 random bytes (hex-encoded)
uv run photorand generate bytes --from path/to/raw_image.ARW -l 32
```

**`assess`** — measure the entropy quality of a RAW image's sensor noise:

```bash
# Human-readable entropy report
uv run photorand assess --from path/to/raw_image.ARW

# Machine-readable JSON output (great for scripting)
uv run photorand assess --from path/to/raw_image.ARW --json
```

**`capture`** — extract entropy from a webcam (requires the optional `capture` extra: `uv sync --extra capture`):

```bash
# Capture 5 seconds of webcam sensor noise and emit the 64-byte seed as hex
uv run photorand capture hex --duration 5

# Roll a D20 from a 3-second capture on camera index 1
uv run photorand capture int-range --duration 3 --camera 1 --min 1 --max 20

# Save the raw seed bytes to a file
uv run photorand capture hex -o seed.bin --binary
```

The captured data flows through the same NIST SP 800-90B assessment + SHA3-512
conditioning as RAW files, and a one-line quality summary is printed to stderr.
By default the command refuses to emit a seed when the entropy health checks
report `FAIL` (a faulty or silently-compressed source); pass `--allow-weak` to
override.

*For the full list of commands and options, run:* `uv run photorand --help`

---

### 2. Low-Level Modular Functions

For maximum control or research, you can use the modular primitives directly.

```python
from src.low_level import ingest_raw_image
from src.low_level import sample_entropy_grid
from src.low_level import hash_entropy_pool

# 1. Ingest raw sensor data
raw_data = ingest_raw_image("path/to/image.raw")

# 2. Extract raw LSB bits
entropy_pool = sample_entropy_grid(raw_data)

# 3. Condition entropy into uniform bytes
seed_bytes = hash_entropy_pool(entropy_pool)
```

Alternatively, use the functional pipeline. It accepts custom functions for each part of the algorithm (ingest_fn, sample_fn and hash_fn) although we already provide the values as default parameters:

```python
from src.low_level import generate_true_random_number
seed = generate_true_random_number("path/to/image.raw")  # returns bytes
```

---

## Resources

- **Blog Post**: [Physical Entropy with PhotoRand](https://www.daniel-ir.eu/blog/photorand)
- **PyPI Package**: [photorand on PyPI](https://pypi.org/project/photorand/)
- **Docs**: [Entropy Extraction](docs/entropy-extraction.md) · [Entropy Estimation](docs/entropy-estimation.md) · [Entropy Expansion](docs/entropy-expansion.md)
