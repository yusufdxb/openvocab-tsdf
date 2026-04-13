"""Smoke script: fuse a synthetic multi-object scene and save a mesh.

Runs offline, no external datasets required. Useful as a sanity check and as a
reference for what a real dataset run should look like.

    python scripts/fuse_synthetic_demo.py --out outputs/synthetic_mesh.ply
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.viz.mesh import save_ply


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/synthetic_mesh.ply"))
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    primitives = [
        Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 50, 50)),
        Box(
            min_xyz=(-0.8, 0.25, -0.8), max_xyz=(0.8, 0.35, 0.8), color=(80, 140, 80)
        ),  # floor slab
        Box(
            min_xyz=(0.45, -0.35, -0.1), max_xyz=(0.60, 0.30, 0.1), color=(60, 60, 220)
        ),  # blue bar
    ]
    t0 = time.perf_counter()
    frames = make_synthetic_dataset(
        primitives, num_frames=args.frames, width=args.width, height=args.height, radius=1.6
    )
    t_render = time.perf_counter() - t0

    cfg = ReferenceTSDFConfig(
        voxel_size_m=args.voxel_size,
        truncation_distance_m=5 * args.voxel_size,
        bounds_min=(-1.0, -0.6, -1.0),
        bounds_max=(1.0, 0.6, 1.0),
        store_color=True,
        device=args.device,
    )
    vol = ReferenceTSDF(cfg)

    t0 = time.perf_counter()
    for f in frames:
        vol.integrate(f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_fuse = time.perf_counter() - t0
    fps = len(frames) / t_fuse if t_fuse > 0 else float("inf")

    t0 = time.perf_counter()
    mesh = vol.extract_mesh(min_weight=1.0)
    t_mc = time.perf_counter() - t0

    save_ply(mesh, args.out)

    observed_voxels = int((vol.weight > 0).sum().item())
    total_voxels = int(np.prod(vol.dims))
    print(f"render:     {t_render:7.2f} s  ({args.frames} frames @ {args.width}x{args.height})")
    print(f"fuse:       {t_fuse:7.2f} s  ({fps:6.1f} FPS)")
    print(f"marching:   {t_mc:7.2f} s")
    print(
        f"voxels obs: {observed_voxels} / {total_voxels} ({100 * observed_voxels / total_voxels:.1f}%)"
    )
    print(f"mesh:       {len(mesh.vertices)} verts, {len(mesh.triangles)} tris -> {args.out}")


if __name__ == "__main__":
    main()
