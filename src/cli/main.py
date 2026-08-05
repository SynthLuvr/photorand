"""Main entry point for the photorand CLI."""

from __future__ import annotations

from src.cli.assess import handle_assess
from src.cli.extract import handle_extract
from src.cli.generate import handle_generate
from src.cli.parser import create_parser
from src.logger import setup_logger


def main() -> None:
    """Main entry point for the photorand CLI."""
    parser, _extract_parser, _generate_parser = create_parser()

    args = parser.parse_args()
    setup_logger(getattr(args, "verbose", False))

    command = args.command
    if command == "extract":
        handle_extract(args)
    elif command == "generate":
        handle_generate(args)
    elif command == "assess":
        handle_assess(args)


if __name__ == "__main__":
    main()
