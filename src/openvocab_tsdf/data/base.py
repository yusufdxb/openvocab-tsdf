"""Common RGB-D dataset interface.

Every dataset loader yields a uniform `RGBDFrame` stream so downstream code
(TSDF fusion, semantic encoding, evaluation) is dataset-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"expected 3x3 K, got {K.shape}")
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(width),
            height=int(height),
        )


@dataclass(frozen=True)
class RGBDFrame:
    """Single RGB-D observation with camera pose.

    Conventions:
      - `color` is uint8 H×W×3 in RGB order.
      - `depth_m` is float32 H×W in meters. Invalid pixels are 0.0 or NaN.
      - `T_wc` is a 4×4 float32 camera-to-world transform (world = T_wc @ camera).
      - `timestamp` is seconds since dataset start (not epoch); used for ordering only.
    """

    color: np.ndarray  # u8[H,W,3]
    depth_m: np.ndarray  # f32[H,W]
    intrinsics: CameraIntrinsics
    T_wc: np.ndarray  # f32[4,4]
    timestamp: float = 0.0
    frame_id: int = -1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.color.dtype != np.uint8:
            raise ValueError(f"color must be uint8, got {self.color.dtype}")
        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise ValueError(f"color must be H,W,3 uint8, got {self.color.shape}")
        if self.depth_m.dtype != np.float32:
            raise ValueError(f"depth_m must be float32, got {self.depth_m.dtype}")
        if self.depth_m.ndim != 2:
            raise ValueError(f"depth_m must be H,W float32, got {self.depth_m.shape}")
        if self.color.shape[:2] != self.depth_m.shape:
            raise ValueError(
                f"color {self.color.shape[:2]} and depth {self.depth_m.shape} mismatch"
            )
        if self.T_wc.shape != (4, 4):
            raise ValueError(f"T_wc must be 4x4, got {self.T_wc.shape}")


class RGBDDataset(Protocol):
    """A lazy stream of RGB-D frames with poses.

    Implementations must be iterable and indexable. Iteration is the primary
    access pattern — avoid loading all frames into memory in `__init__`.
    """

    root: Path
    scene: str
    intrinsics: CameraIntrinsics

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[RGBDFrame]: ...

    def __getitem__(self, idx: int) -> RGBDFrame: ...
