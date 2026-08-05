"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np


@pytest.fixture
def example_image_path() -> str:
    """Provide absolute path to the realistic test image."""
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "data" / "DSC02111.ARW"
    if not path.exists() or path.stat().st_size < 1024:
        pytest.skip("RAW test data unavailable (Git LFS pointer or missing)")
    return str(path)


@pytest.fixture
def mock_image_data() -> np.ndarray:
    """Return a fake 100x100 grid of sequential integers representing pixels."""
    import numpy as np

    return np.arange(10000).reshape((100, 100)).astype(np.uint16)


@pytest.fixture
def mock_entropy_bytes() -> bytes:
    """Return exactly 32 bytes of deterministic pseudorandom data."""
    return bytes(range(32))
