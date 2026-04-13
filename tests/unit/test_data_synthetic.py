"""Tests for the synthetic scene generator."""

from __future__ import annotations

import numpy as np

from openvocab_tsdf.data.base import CameraIntrinsics
from openvocab_tsdf.data.synthetic import (
    Box,
    Sphere,
    make_ring_poses,
    make_synthetic_dataset,
    render_depth,
)


def test_ring_poses_lookat():
    """Cameras on a ring should have +Z axis pointing at the look-at target."""
    poses = make_ring_poses(8, radius=1.5, height=0.0, look_at=(0.0, 0.0, 0.0))
    for T in poses:
        cam_pos = T[:3, 3]
        forward = T[:3, 2]  # z column of rotation = camera forward in world
        to_origin = -cam_pos / np.linalg.norm(cam_pos)
        cos = float(np.dot(forward / np.linalg.norm(forward), to_origin))
        assert cos > 0.98, f"camera not looking inward: cos={cos}"


def test_render_sphere_center_depth():
    """A sphere of radius r centered at distance d should produce central depth ~ d - r."""
    intr = CameraIntrinsics(fx=80.0, fy=80.0, cx=80.0, cy=60.0, width=160, height=120)
    # camera at (0,0,-2) looking at origin along +Z
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 3] = [0.0, 0.0, -2.0]
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    depth, color = render_depth(primitives, intr, T_wc)

    center_depth = depth[60, 80]
    assert abs(center_depth - (2.0 - 0.3)) < 0.05, f"center depth off: {center_depth}"
    assert np.all(color[60, 80] == [200, 80, 80])
    # edges miss
    assert depth[0, 0] == 0.0


def test_make_synthetic_dataset_shapes():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=64, height=48)
    assert len(frames) == 4
    for f in frames:
        assert f.color.shape == (48, 64, 3)
        assert f.depth_m.shape == (48, 64)
        assert f.T_wc.shape == (4, 4)
        # sphere is visible from every ring camera — expect some non-zero depth
        assert (f.depth_m > 0).sum() > 50


def test_box_renders_nonzero_area():
    intr = CameraIntrinsics(fx=80.0, fy=80.0, cx=80.0, cy=60.0, width=160, height=120)
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 3] = [0.0, 0.0, -2.0]
    primitives = [Box(min_xyz=(-0.3, -0.3, -0.2), max_xyz=(0.3, 0.3, 0.2))]
    depth, _ = render_depth(primitives, intr, T_wc)
    hits = (depth > 0).sum()
    assert hits > 200, f"box too small to be visible? hits={hits}"
