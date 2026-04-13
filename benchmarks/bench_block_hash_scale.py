"""Scale benchmark: block-hash sparse geometry vs dense reference on a volume
large enough that the dense backend OOMs.

Uses the synthetic scene generator with a ring of cameras looking at an empty
volume. We scale the bounds up to a 12 m cube and compare:
  - geometry memory footprint
  - integrate throughput
  - peak VRAM

    python benchmarks/bench_block_hash_scale.py --voxel 0.04 --side-m 12
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import BlockHashTSDF, BlockHashTSDFConfig
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


def _reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voxel", type=float, default=0.04)
    p.add_argument("--side-m", type=float, default=12.0, help="cube side length in meters")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    h = args.side_m / 2.0
    bounds_min = (-h, -h, -h)
    bounds_max = (h, h, h)

    primitives = [
        Sphere(center=(0.0, 0.0, 0.0), radius=0.8, color=(220, 40, 40)),
        Box(min_xyz=(-2.0, 0.7, -2.0), max_xyz=(2.0, 0.8, 2.0), color=(80, 170, 80)),
    ]
    frames = make_synthetic_dataset(
        primitives, num_frames=args.frames, width=320, height=240, radius=2.5
    )

    dims_est = (args.side_m / args.voxel) ** 3
    dense_bytes_est = int(dims_est * (4 + 4 + 12))  # tsdf + weight + color
    print(f"volume {args.side_m}^3 m at {args.voxel} m voxels → ~{dims_est:.1e} voxels")
    print(f"dense geometry footprint estimate: {dense_bytes_est/1e9:.2f} GB")

    rows = []

    # dense reference (may OOM on large volumes — catch and report)
    try:
        _reset_vram()
        ref = ReferenceTSDF(
            ReferenceTSDFConfig(
                voxel_size_m=args.voxel,
                truncation_distance_m=4 * args.voxel,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                store_color=True,
                store_features=False,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
        )
        t0 = time.perf_counter()
        for f in frames:
            ref.integrate(f)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_voxels = int(np.prod(ref.dims))
        observed = int((ref.weight > 0).sum().item())
        geom_bytes = total_voxels * (4 + 4 + 12)
        rows.append(
            {
                "backend": "dense_reference",
                "total_voxels": total_voxels,
                "observed_voxels": observed,
                "geom_bytes": geom_bytes,
                "geom_mb": round(geom_bytes / (1024**2), 1),
                "peak_vram_mb": round(_peak_vram_mb(), 1),
                "integrate_fps": float(len(frames) / dt),
            }
        )
        print(
            f"  dense: {total_voxels:.3e} voxels, {geom_bytes/1e9:.2f} GB, "
            f"peak VRAM {rows[-1]['peak_vram_mb']} MB, {rows[-1]['integrate_fps']:.1f} FPS"
        )
        del ref
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        rows.append({"backend": "dense_reference", "error": "OOM", "detail": str(e)[:200]})
        print(f"  dense: OOM ({str(e)[:80]})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # block-hash sparse
    _reset_vram()
    bh = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=args.voxel,
            truncation_distance_m=4 * args.voxel,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            store_color=True,
            initial_block_capacity=8192,
            max_block_capacity=131072,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    t0 = time.perf_counter()
    for f in frames:
        bh.integrate(f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    geom_bytes = bh.memory_bytes()
    total_blocks = int(np.prod(bh.block_dims))
    rows.append(
        {
            "backend": "block_hash",
            "total_blocks": total_blocks,
            "allocated_blocks": bh.num_allocated_blocks,
            "geom_bytes": geom_bytes,
            "geom_mb": round(geom_bytes / (1024**2), 1),
            "peak_vram_mb": round(_peak_vram_mb(), 1),
            "integrate_fps": float(len(frames) / dt),
            "sparsity_ratio": bh.num_allocated_blocks / total_blocks,
        }
    )
    print(
        f"  block_hash: {bh.num_allocated_blocks}/{total_blocks} blocks "
        f"({100*rows[-1]['sparsity_ratio']:.2f}%), "
        f"{geom_bytes/1e6:.1f} MB, peak VRAM {rows[-1]['peak_vram_mb']} MB, "
        f"{rows[-1]['integrate_fps']:.1f} FPS"
    )

    out = {
        "name": "block_hash_scale",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                if torch.cuda.is_available()
                else 0.0
            ),
            "host": platform.node(),
        },
        "params": {
            "side_m": args.side_m,
            "voxel_m": args.voxel,
            "frames": args.frames,
        },
        "runs": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outp = args.out_dir / f"{stamp}_block_hash_scale.json"
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
