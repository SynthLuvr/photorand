# Photorand Examples

This directory contains examples of how to use the `photorand` package in three different ways:
1.  **Low-level Functions**: Direct access to the cryptographic and entropy extraction primitives.
2.  **High-level Classes**: Using the `PhotoRandSeed` (TRNG) and `PhotoRandEngine` (CSPRNG) interfaces.
3.  **CLI Tool**: Using the `photorand` command-line interface.

## Quick Start: Running the Examples

To run these examples, you first need to install the `photorand` package in your virtual environment. The easiest way is to use an "editable" installation, which allows you to run the code directly from the source.

### 1. Install photorand locally

From the root of the project, run:

```bash
uv sync --all-extras  # was: pip install -e .
```

This will link the `src/` directory to your environment, making the `photorand` module available.

### 2. Run the Python examples

Once installed, you can run the examples from any directory:

```bash
# Run the low-level API example
python examples/low_level_example.py

# Run the high-level class-based example
python examples/high_level_example.py
```

### 3. Run the CLI example

The CLI example is a shell script that demonstrates various `photorand` commands:

```bash
bash examples/cli_example.sh
```

## Troubleshooting

If you encounter `ModuleNotFoundError: No module named 'photorand'`, ensure you have activated your virtual environment and run `uv sync --all-extras  # was: pip install -e .` as described above.
