"""Tests for the Replica dataset loader using a fabricated on-disk fixture."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from openvocab_tsdf.data.replica import ReplicaDataset


def _make_fake_replica(root: Path, scene: str, n: int = 3) -> Path:
    scene_dir = root / scene
    (scene_dir / "results").mkdir(parents=True)

    rng = np.random.default_rng(0)
    for i in range(n):
        color = rng.integers(0, 255, size=(64, 80, 3), dtype=np.uint8)
        depth_m = 1.5 + 0.1 * rng.random((64, 80)).astype(np.float32)
        depth_u = (depth_m * 6553.5).astype(np.uint16)
        iio.imwrite(scene_dir / "results" / f"frame{i:06d}.jpg", color, quality=95)
        iio.imwrite(scene_dir / "results" / f"depth{i:06d}.png", depth_u)

    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        poses[i, 0, 3] = 0.1 * i  # slide along x to disambiguate
    np.savetxt(scene_dir / "traj.txt", poses.reshape(n, 16))

    return scene_dir


def test_replica_loads_frames(tmp_path: Path):
    _make_fake_replica(tmp_path, "test_scene", n=3)
    ds = ReplicaDataset(root=tmp_path, scene="test_scene")
    assert len(ds) == 3

    frame = ds[0]
    assert frame.color.shape == (64, 80, 3)
    assert frame.depth_m.shape == (64, 80)
    assert frame.color.dtype == np.uint8
    assert frame.depth_m.dtype == np.float32
    assert frame.T_wc.shape == (4, 4)


def test_replica_respects_stride_and_max_frames(tmp_path: Path):
    _make_fake_replica(tmp_path, "scene", n=6)
    ds = ReplicaDataset(root=tmp_path, scene="scene", stride=2, max_frames=2)
    assert len(ds) == 2


def test_replica_iter_yields_all(tmp_path: Path):
    _make_fake_replica(tmp_path, "scene", n=4)
    ds = ReplicaDataset(root=tmp_path, scene="scene")
    frames = list(ds)
    assert len(frames) == 4
    assert frames[0].frame_id == 0
    assert frames[-1].frame_id == 3


def test_replica_missing_traj_raises(tmp_path: Path):
    scene_dir = tmp_path / "scene"
    (scene_dir / "results").mkdir(parents=True)
    (scene_dir / "results" / "frame000000.jpg").write_bytes(b"")  # any file
    with pytest.raises((FileNotFoundError, ValueError)):
        ReplicaDataset(root=tmp_path, scene="scene")
