"""TSDF volume interface shared by reference and CUDA backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from openvocab_tsdf.data.base import RGBDFrame


@dataclass(frozen=True)
class VoxelQueryResult:
    """Per-point query result over the voxel map."""

    tsdf: torch.Tensor  # f32[N]
    weight: torch.Tensor  # f32[N]
    color: torch.Tensor  # u8[N,3]
    feat: torch.Tensor | None  # f32[N,D] or None if semantics disabled


@dataclass(frozen=True)
class Mesh:
    """Marching-cubes mesh in world coordinates."""

    vertices: np.ndarray  # f32[V,3]
    triangles: np.ndarray  # i32[T,3]
    vertex_colors: np.ndarray | None = None  # u8[V,3] or None


class TSDFVolume(Protocol):
    """Interface every TSDF backend must satisfy.

    Backends differ in storage (dense vs hashed) and execution (PyTorch vs
    custom CUDA). Callers should depend on this protocol, not concrete classes.
    """

    voxel_size_m: float
    truncation_distance_m: float
    device: torch.device

    def integrate(self, frame: RGBDFrame) -> None:
        """Fold a single RGB-D frame into the volume."""
        ...

    def extract_mesh(self) -> Mesh:
        """Marching-cubes mesh of the current zero-crossing surface."""
        ...

    def query(self, points_w: torch.Tensor) -> VoxelQueryResult:
        """Trilinear query of N world-frame points. Returns tsdf / weight / color / feat."""
        ...

    def reset(self) -> None:
        """Clear the volume."""
        ...
