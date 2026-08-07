"""Argument parser for the photorand CLI."""

from __future__ import annotations

import argparse


def _add_format_subparsers(
    command_parser: argparse.ArgumentParser,
    common: argparse.ArgumentParser,
) -> None:
    """Add the shared hex/int/bool/float/int-range/float-range output formats.

    Used by both ``extract`` and ``capture`` so their output set is identical.
    """
    format_subparsers = command_parser.add_subparsers(dest="format", required=True)

    format_subparsers.add_parser("hex", parents=[common], help="Output as hex string")
    format_subparsers.add_parser("int", parents=[common], help="Output as large integer")
    format_subparsers.add_parser("bool", parents=[common], help="Output as boolean")
    format_subparsers.add_parser("float", parents=[common], help="Output as float between 0 and 1")

    int_range = format_subparsers.add_parser(
        "int-range", parents=[common], help="Output integer in range"
    )
    int_range.add_argument("--min", type=int, required=True, help="Lower bound (inclusive)")
    int_range.add_argument("--max", type=int, required=True, help="Upper bound (inclusive)")

    float_range = format_subparsers.add_parser(
        "float-range", parents=[common], help="Output float in range"
    )
    float_range.add_argument("--min", type=float, required=True, help="Lower bound (inclusive)")
    float_range.add_argument("--max", type=float, required=True, help="Upper bound (exclusive)")


def create_parser() -> tuple[
    argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser
]:
    """Create and return the ArgumentParser for the photorand CLI."""
    parser = argparse.ArgumentParser(
        description="photorand: True Random & CSPRNG number generation from RAW images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # Common Arguments
    # ------------------------------------------------------------------
    extract_common = argparse.ArgumentParser(add_help=False)
    extract_common.add_argument(
        "--from",
        "--file",
        "-f",
        "--input",
        dest="image_path",
        required=True,
        help="Path to the RAW image file (e.g., .ARW, .CR2).",
    )
    extract_common.add_argument(
        "-o",
        "--out",
        "--to",
        dest="out",
        help="File path to save output. Omit to print to stdout.",
    )
    extract_common.add_argument(
        "--binary",
        action="store_true",
        help="Write raw binary bytes when saving to a file (requires --out).",
    )
    extract_common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    generate_common = argparse.ArgumentParser(add_help=False)
    generate_common.add_argument(
        "--from",
        "--file",
        "-f",
        "--input",
        dest="image_path",
        required=True,
        help="Path to the RAW image file (e.g., .ARW, .CR2).",
    )
    generate_common.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of items to generate (default: 1).",
    )
    generate_common.add_argument(
        "-o",
        "--out",
        "--to",
        dest="out",
        help="File path to save output. Omit to print to stdout.",
    )
    generate_common.add_argument(
        "--deterministic",
        action="store_true",
        help="Produce a reproducible sequence by skipping environmental salting.",
    )
    generate_common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # ------------------------------------------------------------------
    # 'extract' subcommand
    # ------------------------------------------------------------------
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract 64 bytes of true physical entropy from a RAW image.",
    )
    _add_format_subparsers(extract_parser, extract_common)

    # ------------------------------------------------------------------
    # 'capture' subcommand (webcam source)
    # ------------------------------------------------------------------
    capture_common = argparse.ArgumentParser(add_help=False)
    capture_common.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Capture duration in seconds (default: 5).",
    )
    capture_common.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    capture_common.add_argument(
        "-o",
        "--out",
        "--to",
        dest="out",
        help="File path to save output. Omit to print to stdout.",
    )
    capture_common.add_argument(
        "--binary",
        action="store_true",
        help="Write raw binary bytes when saving to a file (requires --out).",
    )
    capture_common.add_argument(
        "--allow-weak",
        action="store_true",
        help="Emit the seed even when entropy health checks report FAIL.",
    )
    capture_common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture entropy from a webcam.",
    )
    _add_format_subparsers(capture_parser, capture_common)

    # ------------------------------------------------------------------
    # 'assess' subcommand
    # ------------------------------------------------------------------
    assess_parser = subparsers.add_parser(
        "assess",
        help="Measure the entropy quality of a RAW image's sensor noise.",
    )
    assess_parser.add_argument(
        "--from",
        "--file",
        "-f",
        "--input",
        dest="image_path",
        required=True,
        help="Path to the RAW image file (e.g., .ARW, .CR2).",
    )
    assess_parser.add_argument(
        "--no-fpn",
        action="store_true",
        help="Disable fixed-pattern noise reduction (use raw LSBs without "
        "subtracting row/column bias).",
    )
    assess_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the assessment as JSON instead of a human-readable report.",
    )
    assess_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # ------------------------------------------------------------------
    # 'generate' subcommand
    # ------------------------------------------------------------------
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate arbitrary amounts of CSPRNG data seeded by a RAW image.",
    )
    gen_subparsers = generate_parser.add_subparsers(dest="type", required=True)

    gen_bytes = gen_subparsers.add_parser(
        "bytes", parents=[generate_common], help="Generate random bytes"
    )
    gen_bytes.add_argument("-l", "--length", type=int, help="Size of each item in bytes")

    gen_int = gen_subparsers.add_parser(
        "int", parents=[generate_common], help="Generate random integers"
    )
    gen_int.add_argument(
        "-l", "--length", type=int, help="Size of each item in bytes. Does not set digits for int."
    )
    gen_int.add_argument("--digits", type=int, help="Number of decimal digits")

    gen_string = gen_subparsers.add_parser(
        "string", parents=[generate_common], help="Generate random strings"
    )
    gen_string.add_argument("-l", "--length", type=int, help="Size of each item in characters")
    gen_string.add_argument(
        "--charset",
        choices=["ascii", "hex", "alpha", "all"],
        default="all",
        help="Character set to use (default: all)",
    )
    gen_string.add_argument(
        "--numeric-only", action="store_true", help="Only use numbers when generating strings"
    )

    gen_int_range = gen_subparsers.add_parser(
        "int-range", parents=[generate_common], help="Generate random integers in range"
    )
    gen_int_range.add_argument("--min", type=int, required=True, help="Lower bound (inclusive)")
    gen_int_range.add_argument("--max", type=int, required=True, help="Upper bound (inclusive)")

    gen_float_range = gen_subparsers.add_parser(
        "float-range", parents=[generate_common], help="Generate random floats in range"
    )
    gen_float_range.add_argument("--min", type=float, required=True, help="Lower bound (inclusive)")
    gen_float_range.add_argument("--max", type=float, required=True, help="Upper bound (exclusive)")

    gen_subparsers.add_parser("bool", parents=[generate_common], help="Generate random booleans")
    gen_subparsers.add_parser(
        "float", parents=[generate_common], help="Generate random floats between 0 and 1"
    )

    return parser, extract_parser, generate_parser