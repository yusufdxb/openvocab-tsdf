"""Typed configuration loader.

A single `Config` object is built from a YAML file plus environment overrides.
Keep this file boring — add fields, not abstractions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    # Only loaders that exist in `data/` are listed here. ScanNet / TUM /
    # custom datasets are explicit follow-ups (see CLAUDE_CODE_NEXT.md);
    # exposing them in the literal would falsely advertise support.
    name: Literal["replica", "nice_slam_demo"] = "replica"
    root: Path
    scene: str
    max_frames: int | None = None
    stride: int = 1


class CameraConfig(BaseModel):
    depth_scale: float = 1000.0  # depth_px * (1 / depth_scale) = depth_m
    depth_trunc_m: float = 6.0


class MappingConfig(BaseModel):
    voxel_size_m: float = 0.02
    truncation_distance_m: float = 0.1
    hash_capacity: int = 1 << 20
    block_size: int = 8
    # `cuda` is intentionally not listed: there is no hand-written CUDA
    # backend (Triton fills that role — see decisions.md). The only mapping
    # backends that actually exist in `mapping/` are listed below.
    backend: Literal["reference", "sparse_feature", "block_hash", "triton"] = "reference"
    # sparse-feature backend only
    feat_update_backend: Literal["pytorch", "triton"] = "triton"
    initial_feat_capacity: int = 65_536
    max_feat_capacity: int = 8_000_000
    device: str = "cuda:0"
    store_color: bool = True
    store_features: bool = True
    feature_dim: int = 512
    # Dense-reference backend needs axis-aligned bounds. Leave None to auto-fit
    # from the dataset's first-pose + a configured radius.
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    auto_bounds_radius_m: float = 4.0
    max_weight: float = 32.0
    # Normalized-TSDF band for the per-frame feature gate. Features are only
    # written to voxels whose `|sdf / truncation_distance| <= near_surface_band`
    # for that frame. Default 0.5 keeps free-space voxels in front of a surface
    # from stealing the surface's features. Setting this to 1.0 disables the
    # gate (matches the pre-fix legacy behavior, retained as an escape hatch).
    near_surface_band: float = 0.5


class SemanticsConfig(BaseModel):
    model: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    # `mask` is not implemented; the only dense path that exists is
    # `sam_dense` (mask-features pipeline via MobileSAM + per-mask CLIP).
    mode: Literal["global", "patch", "sam_dense"] = "global"
    batch_size: int = 16
    device: str = "cuda:0"
    dtype: Literal["fp16", "fp32"] = "fp16"


class GroundingConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.22
    cluster_eps_vox: int = 2
    min_cluster_voxels: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rich_tracebacks: bool = True


class Config(BaseModel):
    dataset: DatasetConfig
    camera: CameraConfig = Field(default_factory=CameraConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    semantics: SemanticsConfig = Field(default_factory=SemanticsConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a typed `Config`."""
    with Path(path).expanduser().open() as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
