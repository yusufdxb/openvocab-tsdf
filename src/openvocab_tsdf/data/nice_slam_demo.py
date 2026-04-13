"""NICE-SLAM demo-scene RGB-D loader.

Layout (as shipped in `Demo.zip` from `cvg-data.inf.ethz.ch/nice-slam/data/`):

    <root>/<scene>/
        frames/
            color/<i>.jpg                 # RGB
            depth/<i>.png                 # uint16 depth, scale = 1000 (mm → m)
            pose/<i>.txt                  # 4×4 camera-to-world
            intrinsic/
                intrinsic_color.txt       # 4×4 (3×3 K + I padding)
                intrinsic_depth.txt       # 4×4 — we use this one (depth-aligned)
                extrinsic_color.txt       # 4×4 (identity in this dump)
                extrinsic_depth.txt

Numbering is 0-based integer (no zero-pad), sortable by integer index.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from openvocab_tsdf.data.base import CameraIntrinsics, RGBDDataset, RGBDFrame


def _load_4x4(path: Path) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.float32)
    if a.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4, got {a.shape}")
    return a


def _numeric_sort(p: Path) -> int:
    stem = p.stem
    try:
        return int(stem)
    except ValueError:
        return 0


class NiceSlamDemoDataset(RGBDDataset):
    def __init__(
        self,
        root: str | Path,
        scene: str = "Demo",
        *,
        depth_scale: float = 1000.0,
        depth_trunc_m: float = 6.0,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scene = scene
        frames_dir = self.root / scene / "frames"
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"scene not found: {frames_dir}")

        color_dir = frames_dir / "color"
        depth_dir = frames_dir / "depth"
        pose_dir = frames_dir / "pose"
        intr_dir = frames_dir / "intrinsic"

        color_all = sorted(color_dir.glob("*.jpg"), key=_numeric_sort)
        depth_all = sorted(depth_dir.glob("*.png"), key=_numeric_sort)
        pose_all = sorted(pose_dir.glob("*.txt"), key=_numeric_sort)
        n = min(len(color_all), len(depth_all), len(pose_all))
        self._color = color_all[:n:stride]
        self._depth = depth_all[:n:stride]
        self._pose = pose_all[:n:stride]
        if max_frames is not None:
            self._color = self._color[:max_frames]
            self._depth = self._depth[:max_frames]
            self._pose = self._pose[:max_frames]

        K_d = _load_4x4(intr_dir / "intrinsic_depth.txt")
        # read depth image to get (H, W)
        first_d = iio.imread(self._depth[0])
        H, W = first_d.shape[:2]
        self.intrinsics = CameraIntrinsics.from_matrix(K_d[:3, :3], W, H)

        self.depth_scale = float(depth_scale)
        self.depth_trunc_m = float(depth_trunc_m)

    def __len__(self) -> int:
        return len(self._color)

    def __getitem__(self, idx: int) -> RGBDFrame:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        color = iio.imread(self._color[idx])
        if color.ndim == 2:
            color = np.stack([color, color, color], axis=-1)
        color = color[..., :3].astype(np.uint8, copy=False)

        depth_u = iio.imread(self._depth[idx])
        depth_m = (depth_u.astype(np.float32) / self.depth_scale).astype(np.float32)
        depth_m[depth_m > self.depth_trunc_m] = 0.0

        # color and depth may differ in resolution — resize color to depth shape
        if color.shape[:2] != depth_m.shape:
            from PIL import Image

            img = Image.fromarray(color).resize(
                (depth_m.shape[1], depth_m.shape[0]), Image.BILINEAR
            )
            color = np.asarray(img, dtype=np.uint8)

        T_wc = _load_4x4(self._pose[idx])

        return RGBDFrame(
            color=color,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            T_wc=T_wc,
            timestamp=float(idx) / 30.0,
            frame_id=int(idx),
        )

    def __iter__(self) -> Iterator[RGBDFrame]:
        for i in range(len(self)):
            yield self[i]
