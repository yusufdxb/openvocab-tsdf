"""Tests for the ScanNet v2 dataset loader using a fabricated on-disk fixture."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from openvocab_tsdf.data.scannet import ScanNetDataset


def _make_fake_scannet(root: Path, scene: str, n: int = 5) -> Path:
    """Create a minimal ScanNet-style directory layout."""
    scene_dir = root / "scans" / scene
    (scene_dir / "color").mkdir(parents=True)
    (scene_dir / "depth").mkdir(parents=True)
    (scene_dir / "pose").mkdir(parents=True)
    (scene_dir / "intrinsic").mkdir(parents=True)

    rng = np.random.default_rng(42)
    H, W = 48, 64

    for i in range(n):
        color = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        depth_mm = (1500 + 100 * rng.random((H, W))).astype(np.uint16)
        iio.imwrite(scene_dir / "color" / f"{i}.jpg", color, quality=95)
        iio.imwrite(scene_dir / "depth" / f"{i}.png", depth_mm)

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.1 * i
        np.savetxt(scene_dir / "pose" / f"{i}.txt", pose)

    K = np.array(
        [
            [500.0, 0.0, 31.5, 0.0],
            [0.0, 500.0, 23.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savetxt(scene_dir / "intrinsic" / "intrinsic_depth.txt", K)

    return scene_dir


def test_scannet_loads_frames(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=5)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00")
    assert len(ds) == 5

    frame = ds[0]
    assert frame.color.shape == (48, 64, 3)
    assert frame.depth_m.shape == (48, 64)
    assert frame.color.dtype == np.uint8
    assert frame.depth_m.dtype == np.float32
    assert frame.T_wc.shape == (4, 4)


def test_scannet_intrinsics_from_file(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=2)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00")
    assert ds.intrinsics.fx == 500.0
    assert ds.intrinsics.fy == 500.0
    assert ds.intrinsics.cx == 31.5
    assert ds.intrinsics.cy == 23.5
    assert ds.intrinsics.width == 64
    assert ds.intrinsics.height == 48


def test_scannet_respects_stride_and_max_frames(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=10)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00", stride=3, max_frames=2)
    assert len(ds) == 2


def test_scannet_skips_invalid_poses(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=5)
    pose_dir = tmp_path / "scans" / "scene0001_00" / "pose"
    # Corrupt frame 2 with inf
    bad_pose = np.full((4, 4), np.inf, dtype=np.float32)
    np.savetxt(pose_dir / "2.txt", bad_pose)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00")
    assert len(ds) == 4


def test_scannet_iter_yields_all(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=4)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00")
    frames = list(ds)
    assert len(frames) == 4


def test_scannet_depth_truncation(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=1)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00", depth_trunc_m=1.0)
    frame = ds[0]
    # Fake depth is ~1.5m, so everything should be truncated to 0.
    assert (frame.depth_m == 0.0).all()


def test_scannet_missing_dirs_raises(tmp_path: Path):
    scene_dir = tmp_path / "scans" / "scene0001_00"
    scene_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ScanNetDataset(root=tmp_path, scene="scene0001_00")


def test_scannet_negative_index(tmp_path: Path):
    _make_fake_scannet(tmp_path, "scene0001_00", n=3)
    ds = ScanNetDataset(root=tmp_path, scene="scene0001_00")
    frame = ds[-1]
    assert frame.frame_id == 2
