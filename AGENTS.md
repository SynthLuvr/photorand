# AGENTS.md

Instructions for AI coding agents working in this repository.

## Quick Start

```bash
uv sync --all-extras   # install dependencies (canonist + poethepoet)
uv run poe lint        # full static pipeline (format check, lint, typecheck, lock, audit, dupes)
uv run poe test        # run tests
```

The Poe tasks invoke every tool as `python -m <module>` rather than through
generated console-script launchers; the equivalent module form also works
directly (`uv run python -m poethepoet check`).

## Required Workflow

Always run these before considering work complete:

```bash
uv run poe check   # the full lint pipeline plus the test suite
```

Everything must pass with zero errors. `poe check` runs `poe lint` (ruff format
check, ruff check incl. SAST, pyright strict, lockfile freshness, pip-audit,
duplication gate) followed by `poe test` (80% coverage gate).

## Toolchain

Lint, format, testing, and environment checks come from
[canonist](https://github.com/SynthLuvr/canonist) — one dev dependency that bundles
Ruff, Pyright (strict), pytest + pytest-cov, pip-audit, the lucidshark-duplo
duplication gate, and the canonical ruff/pyright/pytest presets.

- Run tooling through `uv run poe <task>` (`poe lint`, `poe format`, `poe test`,
  `poe doctor`), not by invoking tools directly.
- `python -m canonist lint` / `format` accept path arguments; `lint` also takes
  `--fast` (skips pip-audit and the duplication gate).
- `poe doctor` (`python -m canonist doctor`) diagnoses toolchain/environment problems.
- The duplication gate downloads a pinned prebuilt binary on first use. Where it
  cannot be downloaded or run it prints `Duplication gate SKIPPED` locally and exits 0,
  and fails in CI. `SKIPPED` means the gate did **not** run — do not read it as a pass,
  and do not move or rename the binary to get around the block.
- Tool, rule, threshold, and preset changes belong in canonist — bump its version in
  `pyproject.toml` to pick them up. Do not add per-step tool scripts or re-inline tool
  config blocks here; keep only true local deltas under `[tool.canonist.*]` (this
  repo has one: tests synthesize fixture data with `random.Random`, so `S311` is
  ignored alongside the preset's `S101` under `src/tests/*`).

## Coding Conventions (Enforced)

These are **not** preferences — the toolchain will fail if you violate them:

### Use `from __future__ import annotations`

Every module must start with this import for consistent type-hint semantics. Enforced
by ruff `I002` (`required-imports`), so a missing one fails `poe lint`.

```python
# ❌ Wrong
def foo() -> str: ...

# ✅ Right
from __future__ import annotations


def foo() -> str: ...
```

### Strict type checking

Pyright runs in `strict` mode. All function parameters and return types must
have explicit type annotations. Strict mode also rejects implicit `Any` types and
untyped `dict`/`list` literals where the type can't be inferred.

```python
# ❌ Wrong
def add(a, b):
    return a + b

# ✅ Right
def add(a: int, b: int) -> int:
    return a + b
```

### Formatting

- Line length: 100 characters
- Double quotes for strings
- Import sorting via `isort` (ruff's `I` rules)
- Modern Python idioms via `UP` (pyupgrade) rules

## Formatting

If the linter complains about formatting, run:

```bash
uv run poe format
```

This runs `python -m canonist format`:

1. `ruff format` — formats all files
2. `ruff check --fix` — applies lint auto-fixes (import sorting, etc.)

## Project Structure

- Source code lives in `src/`
- Tests live in `src/tests/` (filenames start with `test_`), and are excluded from
  coverage measurement so the 80% gate reflects real source coverage
- Python ≥ 3.14 — managed automatically by uv
- Dependencies are declared in `pyproject.toml`; the lockfile (`uv.lock`) is committed
  and its freshness is checked by `poe lint`

### Package Layout

```
src/
├── __init__.py          # Package init (re-exports, version)
├── __main__.py          # CLI entry point (python -m src)
├── logger.py            # Central logger
├── low_level/           # Modular primitives (ingest, sample, hash, csprng)
├── high_level/          # OO abstractions (PhotoRandSeed, PhotoRandEngine)
├── cli/                 # Command-line interface (parser, handlers)
└── tests/               # Test suite organized by layer
```

First-party imports use the `src.` prefix (e.g., `from src.low_level.hash import hash_entropy_pool`).
