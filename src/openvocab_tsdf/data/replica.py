"""Replica RGB-D dataset loader.

Expected layout (matching the popular `semantic-nerf` / `nice-slam` dumps):

    <root>/<scene>/
        traj.txt              # (N x 16) rows, row-major 4x4 T_wc per frame
        results/
            frame000000.jpg   # or .png; RGB color
            depth000000.png   # uint16 depth, scale per dataset dump
            ...

If your dump uses a different scheme, subclass or adapt. Keep any scene-specific
intrinsics or depth scales in a YAML next to the scene folder, not hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np

from openvocab_tsdf.data.base import CameraIntrinsics, RGBDDataset, RGBDFrame


class ReplicaDataset(RGBDDataset):
    def __init__(
        self,
        root: str | Path,
        scene: str,
        *,
        intrinsics: CameraIntrinsics | None = None,
        depth_scale: float = 6553.5,
        depth_trunc_m: float = 6.0,
        max_frames: int | None = None,
        stride: int = 1,
        color_glob: str = "frame*.jpg",
        depth_glob: str = "depth*.png",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scene = scene
        self.scene_dir = self.root / scene
        self.depth_scale = float(depth_scale)
        self.depth_trunc_m = float(depth_trunc_m)
        self.stride = max(1, int(stride))

        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"Replica scene not found: {self.scene_dir}")

        # Try common layouts for frame dirs.
        for candidate in ("results", "frames", "."):
            frame_dir = self.scene_dir / candidate if candidate != "." else self.scene_dir
            colors = sorted(frame_dir.glob(color_glob))
            if colors:
                self.frame_dir = frame_dir
                break
        else:
            raise FileNotFoundError(f"no frames matching {color_glob} under {self.scene_dir}")

        self._color_paths = sorted(self.frame_dir.glob(color_glob))[:: self.stride]
        self._depth_paths = sorted(self.frame_dir.glob(depth_glob))[:: self.stride]
        if len(self._color_paths) != len(self._depth_paths):
            raise ValueError(
                f"color/depth count mismatch: {len(self._color_paths)} vs {len(self._depth_paths)}"
            )
        if max_frames is not None:
            self._color_paths = self._color_paths[:max_frames]
            self._depth_paths = self._depth_paths[:max_frames]

        traj_path = self.scene_dir / "traj.txt"
        if not traj_path.exists():
            raise FileNotFoundError(f"missing traj.txt at {traj_path}")
        traj = np.loadtxt(traj_path, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 16:
            raise ValueError(f"unexpected traj shape {traj.shape}")
        self._poses = traj.reshape(-1, 4, 4)[:: self.stride][: len(self._color_paths)]

        # Intrinsics: if not supplied, read the first color image and use a
        # reasonable default that matches the most common Replica dump (fx=600).
        if intrinsics is None:
            first = iio.imread(self._color_paths[0])
            H, W = first.shape[:2]
            # This matches the Replica renderer used by nice-slam; override via argument
            # when your dump differs.
            fx = fy = 600.0
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H)
        self.intrinsics = intrinsics

    def __len__(self) -> int:
        return len(self._color_paths)

    def __getitem__(self, idx: int) -> RGBDFrame:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        color = iio.imread(self._color_paths[idx])
        if color.ndim == 2:
            color = np.stack([color, color, color], axis=-1)
        color = color[..., :3].astype(np.uint8, copy=False)

        depth_u = iio.imread(self._depth_paths[idx])
        depth_m = (depth_u.astype(np.float32) / self.depth_scale).astype(np.float32)
        depth_m[depth_m > self.depth_trunc_m] = 0.0

        return RGBDFrame(
            color=color,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            T_wc=self._poses[idx].astype(np.float32),
            timestamp=float(idx) / 30.0,
            frame_id=int(idx),
        )

    def __iter__(self) -> Iterator[RGBDFrame]:
        for i in range(len(self)):
            yield self[i]
