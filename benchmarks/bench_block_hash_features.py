"""Combined sparse-geometry + sparse-features scale benchmark.

Demonstrates that `BlockHashTSDF(store_features=True)` — block-hash sparse
geometry composed with per-voxel sparse features — runs end-to-end at
warehouse scale (30 m³ cube) with per-voxel 512-d CLIP features on a
consumer GPU.

    python benchmarks/bench_block_hash_features.py --side-m 30 --voxel 0.04
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import BlockHashTSDF, BlockHashTSDFConfig


def _reset() -> None:
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
    p.add_argument("--side-m", type=float, default=30.0)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    h = args.side_m / 2.0
    primitives = [
        Sphere(center=(0.0, 0.0, 0.0), radius=0.8, color=(220, 40, 40)),
        Box(min_xyz=(-2.0, 0.7, -2.0), max_xyz=(2.0, 0.8, 2.0), color=(80, 170, 80)),
    ]
    frames = make_synthetic_dataset(
        primitives, num_frames=args.frames, width=320, height=240, radius=2.5
    )

    rows = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for D in (0, 256, 512):
        _reset()
        bh = BlockHashTSDF(
            BlockHashTSDFConfig(
                voxel_size_m=args.voxel,
                truncation_distance_m=4 * args.voxel,
                bounds_min=(-h, -h, -h),
                bounds_max=(h, h, h),
                store_color=True,
                store_features=(D > 0),
                feature_dim=max(D, 1),
                initial_block_capacity=4096,
                max_block_capacity=131_072,
                initial_feat_capacity=1 << 16,
                max_feat_capacity=1 << 24,
                device=device,
            )
        )
        feat = None
        if D > 0:
            feat = torch.randn(D, device=device)
            feat = feat / feat.norm()
        t0 = time.perf_counter()
        for f in frames:
            if feat is not None:
                bh.integrate(f, feature=feat)
            else:
                bh.integrate(f)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        rows.append(
            {
                "feature_dim": D,
                "blocks_allocated": bh.num_allocated_blocks,
                "feat_voxels_allocated": bh.num_allocated_feat_voxels,
                "geom_bytes": bh.memory_bytes() - bh.feat_memory_bytes(),
                "feat_bytes": bh.feat_memory_bytes(),
                "total_mb": round(bh.memory_bytes() / (1024**2), 1),
                "peak_vram_mb": round(_peak_vram_mb(), 1),
                "integrate_fps": float(len(frames) / dt),
            }
        )
        del bh

    print(
        f"\n### BlockHashTSDF(store_features=True) — {args.side_m} m cube @ {args.voxel} m, "
        f"{args.frames} synthetic frames"
    )
    print("| feat dim | blocks | feat voxels | geom | feat | total | peak VRAM | FPS |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        geom = r["geom_bytes"] / (1024**2)
        feat = r["feat_bytes"] / (1024**2)
        total = r["total_mb"]
        print(
            f"| {r['feature_dim']} | {r['blocks_allocated']} | "
            f"{r['feat_voxels_allocated']} | {geom:.1f} MB | {feat:.1f} MB | "
            f"{total:.1f} MB | {r['peak_vram_mb']} MB | {r['integrate_fps']:.1f} |"
        )

    out_doc = {
        "name": "block_hash_with_features_scale",
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
        "params": {"side_m": args.side_m, "voxel_m": args.voxel, "frames": args.frames},
        "runs": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_block_hash_with_features.json"
    out.write_text(json.dumps(out_doc, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
