"""Central logger for the photorand package."""

from __future__ import annotations

import logging

# Level is NOTSET so that pytest's log_cli_level can control captured records.
# The StreamHandler enforces WARNING by default so the CLI stays quiet,
# unless setup_logger() is called with verbose=True.
logger: logging.Logger = logging.getLogger("photorand")

# Console handler (used by the CLI)
_ch: logging.Handler = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_ch)


def setup_logger(verbose: bool) -> None:
    """Configure the central logger for CLI use.

    Args:
        verbose: When True, lowers the log level to INFO so that progress
            messages are printed to the console.
    """
    if verbose:
        logger.setLevel(logging.INFO)
        _ch.setLevel(logging.INFO)
