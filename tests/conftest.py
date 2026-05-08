"""Shared test fixtures."""
from pathlib import Path

import pytest


@pytest.fixture
def working_dir(tmp_path: Path) -> Path:
    """A clean per-test working directory."""
    return tmp_path
