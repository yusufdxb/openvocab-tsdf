"""Shared pytest fixtures.

Keep this small — add project-specific fixtures here as needed. Marker behavior
is set in pyproject.toml.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU / dataset tests when the environment cannot satisfy them."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_dataset = pytest.mark.skip(reason="dataset not present (set OPENVOCAB_DATASETS_ROOT)")

    datasets_root = os.environ.get("OPENVOCAB_DATASETS_ROOT")
    dataset_ok = datasets_root is not None and Path(datasets_root).expanduser().is_dir()

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "dataset" in item.keywords and not dataset_ok:
            item.add_marker(skip_dataset)
