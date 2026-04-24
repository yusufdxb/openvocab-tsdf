"""ScanNet v2 RGB-D dataset loader.

Expected layout (standard extracted format via SensReader):

    <root>/scans/<scene>/
        color/
            0.jpg, 1.jpg, ...
        depth/
            0.png, 1.png, ...
        pose/
            0.txt, 1.txt, ...       # 4x4 camera-to-world, one per frame
        intrinsic/
            intrinsic_depth.txt     # 4x4 intrinsic matrix

Depth images are uint16 in millimeters (depth_scale=1000).
Poses with any inf/nan values are treated as invalid and skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from openvocab_tsdf.data.base import CameraIntrinsics, RGBDDataset, RGBDFrame


class ScanNetDataset(RGBDDataset):
    def __init__(
        self,
        root: str | Path,
        scene: str,
        *,
        depth_scale: float = 1000.0,
        depth_trunc_m: float = 6.0,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scene = scene
        self.scene_dir = self.root / "scans" / scene
        self.depth_scale = float(depth_scale)
        self.depth_trunc_m = float(depth_trunc_m)
        self.stride = max(1, int(stride))

        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"ScanNet scene not found: {self.scene_dir}")

        color_dir = self.scene_dir / "color"
        depth_dir = self.scene_dir / "depth"
        pose_dir = self.scene_dir / "pose"
        if not color_dir.is_dir():
            raise FileNotFoundError(f"missing color/ directory: {color_dir}")
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"missing depth/ directory: {depth_dir}")
        if not pose_dir.is_dir():
            raise FileNotFoundError(f"missing pose/ directory: {pose_dir}")

        color_paths = sorted(color_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))
        pose_paths = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))

        frame_ids = sorted(
            set(int(p.stem) for p in color_paths)
            & set(int(p.stem) for p in depth_paths)
            & set(int(p.stem) for p in pose_paths)
        )

        self._valid_indices: list[int] = []
        self._poses: list[np.ndarray] = []
        for fid in frame_ids[:: self.stride]:
            pose = np.loadtxt(pose_dir / f"{fid}.txt", dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                continue
            self._valid_indices.append(fid)
            self._poses.append(pose)

        if max_frames is not None:
            self._valid_indices = self._valid_indices[:max_frames]
            self._poses = self._poses[:max_frames]

        if not self._valid_indices:
            raise ValueError(f"no valid frames found in {self.scene_dir}")

        self._color_dir = color_dir
        self._depth_dir = depth_dir

        intrinsic_path = self.scene_dir / "intrinsic" / "intrinsic_depth.txt"
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"missing intrinsic file: {intrinsic_path}")
        K = np.loadtxt(intrinsic_path, dtype=np.float32)
        if K.shape != (4, 4):
            raise ValueError(f"expected 4x4 intrinsic matrix, got {K.shape}")

        first_depth = iio.imread(self._depth_dir / f"{self._valid_indices[0]}.png")
        H, W = first_depth.shape[:2]
        self.intrinsics = CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=W,
            height=H,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> RGBDFrame:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        fid = self._valid_indices[idx]

        color = iio.imread(self._color_dir / f"{fid}.jpg")
        if color.ndim == 2:
            color = np.stack([color, color, color], axis=-1)
        color = color[..., :3].astype(np.uint8, copy=False)

        depth_u = iio.imread(self._depth_dir / f"{fid}.png")
        depth_m = (depth_u.astype(np.float32) / self.depth_scale).astype(np.float32)
        depth_m[depth_m > self.depth_trunc_m] = 0.0

        # ScanNet color and depth may differ in resolution; resize color to match depth.
        dh, dw = depth_m.shape[:2]
        ch, cw = color.shape[:2]
        if (ch, cw) != (dh, dw):
            from PIL import Image

            color = np.array(
                Image.fromarray(color).resize((dw, dh), Image.BILINEAR),
                dtype=np.uint8,
            )

        return RGBDFrame(
            color=color,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            T_wc=self._poses[idx],
            timestamp=float(fid) / 30.0,
            frame_id=fid,
        )

    def __iter__(self) -> Iterator[RGBDFrame]:
        for i in range(len(self)):
            yield self[i]
