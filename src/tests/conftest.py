"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"
_RAW_FILE = _DATA_DIR / "DSC02111.ARW"
_HAS_RAW_DATA = _RAW_FILE.exists() and _RAW_FILE.stat().st_size > 1024

requires_raw_data = pytest.mark.skipif(
    not _HAS_RAW_DATA, reason="RAW test data unavailable (Git LFS pointer or missing)"
)


@pytest.fixture
def example_image_path() -> str:
    """Provide absolute path to the realistic test image."""
    if not _HAS_RAW_DATA:
        pytest.skip("RAW test data unavailable (Git LFS pointer or missing)")
    return str(_RAW_FILE)


@pytest.fixture
def mock_image_data() -> np.ndarray:
    """Return a fake 100x100 grid of sequential integers representing pixels."""
    import numpy as np

    return np.arange(10000).reshape((100, 100)).astype(np.uint16)


@pytest.fixture
def mock_entropy_bytes() -> bytes:
    """Return exactly 32 bytes of deterministic pseudorandom data."""
    return bytes(range(32))
