"""Tests for the heatmap PLY exporter (no CLIP model needed)."""

from __future__ import annotations

import numpy as np

from openvocab_tsdf.viz.heatmap import _viridis_like


def test_viridis_like_returns_uint8_rgb():
    x = np.linspace(0.0, 1.0, 7)
    c = _viridis_like(x)
    assert c.shape == (7, 3)
    assert c.dtype == np.uint8
    # first color should be dark purple-ish, last yellowish
    assert c[0, 0] < 100 and c[0, 2] > 50  # purple: low R, some B
    assert c[-1, 0] > 200 and c[-1, 1] > 200 and c[-1, 2] < 100  # yellow


def test_viridis_like_monotonic_in_brightness():
    x = np.linspace(0.0, 1.0, 50)
    c = _viridis_like(x).astype(np.float32).mean(axis=-1)
    # brightness should be roughly increasing across the colormap
    diffs = np.diff(c)
    assert (diffs >= -5).all(), "colormap brightness should be near-monotonic"
