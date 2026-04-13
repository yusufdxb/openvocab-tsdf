"""High-level orchestration glue for the fuse / encode / ground commands.

This module is thin. Every helper here is a composition of the module-level
building blocks in `data/`, `mapping/`, `semantics/`, `grounding/`. Keep it
that way.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openvocab_tsdf.config import Config
from openvocab_tsdf.data.base import RGBDDataset
from openvocab_tsdf.data.replica import ReplicaDataset
from openvocab_tsdf.mapping.base import Mesh
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig

log = logging.getLogger(__name__)


def build_dataset(cfg: Config) -> RGBDDataset:
    d = cfg.dataset
    if d.name == "replica":
        return ReplicaDataset(
            root=d.root,
            scene=d.scene,
            depth_scale=cfg.camera.depth_scale,
            depth_trunc_m=cfg.camera.depth_trunc_m,
            max_frames=d.max_frames,
            stride=d.stride,
        )
    raise NotImplementedError(f"dataset '{d.name}' not implemented yet")


def _auto_bounds(
    dataset: RGBDDataset, radius: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit bounds as a cube of side 2*radius centered on the first camera origin."""
    first = dataset[0]
    c = first.T_wc[:3, 3].astype(np.float32)
    bmin = (float(c[0] - radius), float(c[1] - radius), float(c[2] - radius))
    bmax = (float(c[0] + radius), float(c[1] + radius), float(c[2] + radius))
    return bmin, bmax


def build_reference_tsdf(cfg: Config, dataset: RGBDDataset) -> ReferenceTSDF:
    m = cfg.mapping
    if m.bounds_min is None or m.bounds_max is None:
        bmin, bmax = _auto_bounds(dataset, m.auto_bounds_radius_m)
    else:
        bmin, bmax = tuple(m.bounds_min), tuple(m.bounds_max)
    tsdf_cfg = ReferenceTSDFConfig(
        voxel_size_m=m.voxel_size_m,
        truncation_distance_m=m.truncation_distance_m,
        bounds_min=bmin,
        bounds_max=bmax,
        max_weight=m.max_weight,
        store_color=m.store_color,
        store_features=m.store_features,
        feature_dim=m.feature_dim if m.store_features else 0,
        device=m.device,
    )
    return ReferenceTSDF(tsdf_cfg)


def fuse_dataset(cfg: Config, output_path: Path) -> Mesh:
    """Run dataset → TSDF → mesh end-to-end and save PLY. Returns the mesh."""
    from openvocab_tsdf.viz.mesh import save_ply

    dataset = build_dataset(cfg)
    vol = build_reference_tsdf(cfg, dataset)

    n = len(dataset)
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]fuse[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("fuse", total=n)
        for frame in dataset:
            vol.integrate(frame)
            prog.advance(task)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = vol.extract_mesh(min_weight=1.0)
    mc_s = time.perf_counter() - t0

    save_ply(mesh, output_path)
    log.info(
        "fused %d frames in %.2fs (%.1f FPS) -> mesh %d verts / %d tris (mc %.2fs) -> %s",
        n,
        fuse_s,
        n / fuse_s if fuse_s > 0 else 0,
        len(mesh.vertices),
        len(mesh.triangles),
        mc_s,
        output_path,
    )
    return mesh
