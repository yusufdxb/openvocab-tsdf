"""Benchmark: dense vs sparse feature storage on Replica room0.

Runs the full encode-and-fuse pipeline twice — once with the dense reference
backend, once with the sparse-feature backend — and compares:
  - feature-memory footprint (bytes)
  - number of feature voxels allocated
  - wall-time integrate throughput
  - peak VRAM

    python benchmarks/bench_sparse_features.py --config configs/replica_room0.yaml \\
        --max-frames 250 --stride 8
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

from openvocab_tsdf.config import load_config
from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.mapping.sparse_reference import (
    SparseFeatureTSDF,
    SparseFeatureTSDFConfig,
)


def _bounds(cfg):  # type: ignore[no-untyped-def]
    m = cfg.mapping
    if m.bounds_min is None or m.bounds_max is None:
        raise SystemExit("this benchmark expects explicit bounds in the config")
    return tuple(m.bounds_min), tuple(m.bounds_max)


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def _reset_vram():  # type: ignore[no-untyped-def]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _run_backend(
    name: str,
    vol,  # TSDFVolume
    frames: list[RGBDFrame],
    feats: torch.Tensor,
) -> dict:
    _reset_vram()
    t0 = time.perf_counter()
    for f, fv in zip(frames, feats, strict=True):
        vol.integrate(f, feature=fv)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if isinstance(vol, ReferenceTSDF):
        feat_bytes = int(np.prod(vol.dims)) * vol.feat.shape[-1] * 4  # dense
        allocated = int(np.prod(vol.dims))
    else:
        feat_bytes = int(vol.feat_memory_bytes())
        allocated = int(vol.num_allocated_feat_voxels)

    return {
        "backend": name,
        "frames": len(frames),
        "integrate_s": dt,
        "fps": len(frames) / dt if dt > 0 else 0.0,
        "feat_bytes": feat_bytes,
        "feat_mb": round(feat_bytes / (1024**2), 1),
        "feat_voxels_allocated": allocated,
        "total_voxels": int(np.prod(vol.dims)),
        "peak_vram_mb": round(_peak_vram_mb(), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=250)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    cfg = load_config(args.config)
    # override for the bench
    cfg.dataset.max_frames = args.max_frames
    cfg.dataset.stride = args.stride

    from openvocab_tsdf.pipeline import build_dataset

    ds = build_dataset(cfg)
    frames: list[RGBDFrame] = [ds[i] for i in range(len(ds))]
    print(f"loaded {len(frames)} frames")

    bmin, bmax = _bounds(cfg)
    D = cfg.mapping.feature_dim

    # Fake per-frame feature vectors (size D). The backend math is the same
    # regardless of where features come from, so we skip the CLIP cost.
    feats = torch.nn.functional.normalize(torch.randn(len(frames), D), dim=-1).to(
        cfg.mapping.device
    )

    print("running dense reference backend ...")
    dense = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=cfg.mapping.voxel_size_m,
            truncation_distance_m=cfg.mapping.truncation_distance_m,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            store_features=True,
            feature_dim=D,
            device=cfg.mapping.device,
        )
    )
    dense_row = _run_backend("dense_reference", dense, frames, feats)
    del dense

    print("running sparse-feature backend (pytorch update) ...")
    sparse = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=cfg.mapping.voxel_size_m,
            truncation_distance_m=cfg.mapping.truncation_distance_m,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=1 << 16,
            max_feat_capacity=1 << 24,
            feat_update_backend="pytorch",
            device=cfg.mapping.device,
        )
    )
    sparse_row = _run_backend("sparse_pytorch", sparse, frames, feats)
    del sparse

    print("running sparse-feature backend (triton update) ...")
    sparse_trt = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=cfg.mapping.voxel_size_m,
            truncation_distance_m=cfg.mapping.truncation_distance_m,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=1 << 16,
            max_feat_capacity=1 << 24,
            feat_update_backend="triton",
            device=cfg.mapping.device,
        )
    )
    sparse_trt_row = _run_backend("sparse_triton", sparse_trt, frames, feats)
    del sparse_trt

    out_doc = {
        "name": "sparse_vs_dense",
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
            "config": str(args.config),
            "max_frames": args.max_frames,
            "stride": args.stride,
            "voxel_size_m": cfg.mapping.voxel_size_m,
            "bounds_min": list(bmin),
            "bounds_max": list(bmax),
            "feature_dim": D,
        },
        "runs": [dense_row, sparse_row, sparse_trt_row],
        "savings": {
            "feat_mb_dense": dense_row["feat_mb"],
            "feat_mb_sparse": sparse_row["feat_mb"],
            "feat_mem_ratio_dense_over_sparse": (
                dense_row["feat_bytes"] / max(1, sparse_row["feat_bytes"])
            ),
            "sparsity_ratio": (sparse_row["feat_voxels_allocated"] / dense_row["total_voxels"]),
            "triton_speedup_over_pytorch_sparse": (
                sparse_trt_row["fps"] / max(1e-9, sparse_row["fps"])
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_sparse_vs_dense.json"
    out.write_text(json.dumps(out_doc, indent=2) + "\n")

    # Console summary
    print("\n### dense vs sparse features")
    hdr = "| backend          | FPS | feat (MB) | allocated | sparsity | peak VRAM (MB) |"
    print(hdr)
    print("|---|---|---|---|---|---|")
    total = dense_row["total_voxels"]
    for r in (dense_row, sparse_row, sparse_trt_row):
        sparsity = 100 * r["feat_voxels_allocated"] / total
        print(
            f"| {r['backend']:16s} | {r['fps']:5.1f} | "
            f"{r['feat_mb']:8.1f} | {r['feat_voxels_allocated']:9d} | "
            f"{sparsity:5.2f}% | {r['peak_vram_mb']:7.1f} |"
        )
    print(
        f"\nfeature-memory ratio (dense/sparse): "
        f"{out_doc['savings']['feat_mem_ratio_dense_over_sparse']:.2f}x"
    )
    print(
        f"triton/pytorch sparse FPS ratio: "
        f"{out_doc['savings']['triton_speedup_over_pytorch_sparse']:.2f}x"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
