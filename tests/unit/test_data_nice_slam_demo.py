"""Smoke tests for the NICE-SLAM demo loader (on a fabricated fixture)."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from openvocab_tsdf.data.nice_slam_demo import NiceSlamDemoDataset


def _make_fake_demo(root: Path, n: int = 3) -> None:
    frames = root / "FakeDemo" / "frames"
    for sub in ("color", "depth", "pose", "intrinsic"):
        (frames / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        color = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        depth_m = 1.5 + 0.1 * rng.random((48, 64)).astype(np.float32)
        depth_u = (depth_m * 1000).astype(np.uint16)
        iio.imwrite(frames / "color" / f"{i}.jpg", color, quality=95)
        iio.imwrite(frames / "depth" / f"{i}.png", depth_u)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.1 * i
        np.savetxt(frames / "pose" / f"{i}.txt", pose)

    for name, K in (
        (
            "intrinsic_depth.txt",
            np.array([[80, 0, 32, 0], [0, 80, 24, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        (
            "intrinsic_color.txt",
            np.array([[160, 0, 64, 0], [0, 160, 48, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        ("extrinsic_depth.txt", np.eye(4)),
        ("extrinsic_color.txt", np.eye(4)),
    ):
        np.savetxt(frames / "intrinsic" / name, K)


def test_demo_loader_reads_frames(tmp_path: Path):
    _make_fake_demo(tmp_path, n=4)
    ds = NiceSlamDemoDataset(root=tmp_path, scene="FakeDemo")
    assert len(ds) == 4

    f = ds[0]
    assert f.color.shape == (48, 64, 3)
    assert f.depth_m.shape == (48, 64)
    assert f.color.dtype == np.uint8
    assert f.depth_m.dtype == np.float32
    assert f.T_wc.shape == (4, 4)
    # intrinsics pulled from intrinsic_depth.txt
    assert abs(f.intrinsics.fx - 80) < 1e-3
    assert f.intrinsics.width == 64


def test_demo_loader_stride_and_max(tmp_path: Path):
    _make_fake_demo(tmp_path, n=6)
    ds = NiceSlamDemoDataset(root=tmp_path, scene="FakeDemo", stride=2, max_frames=2)
    assert len(ds) == 2


def test_demo_loader_missing_scene_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        NiceSlamDemoDataset(root=tmp_path, scene="NoSuchScene")
