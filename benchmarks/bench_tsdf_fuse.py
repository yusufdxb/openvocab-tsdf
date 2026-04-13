"""Benchmark: reference TSDF fusion throughput on synthetic data.

Writes a JSON result to `benchmarks/results/<timestamp>_tsdf_fuse_<backend>.json`
so we can track performance across commits.

    python benchmarks/bench_tsdf_fuse.py --frames 64 --width 320 --height 240
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
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _gpu_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--backend", type=str, default="reference", choices=["reference", "triton"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    primitives = [
        Sphere(center=(0.0, 0.0, 0.0), radius=0.3),
        Box(min_xyz=(-0.8, 0.3, -0.8), max_xyz=(0.8, 0.4, 0.8)),
        Box(min_xyz=(0.5, -0.4, -0.1), max_xyz=(0.6, 0.3, 0.1)),
    ]
    frames = make_synthetic_dataset(
        primitives, num_frames=args.frames, width=args.width, height=args.height, radius=1.6
    )

    bounds_min = (-1.0, -0.6, -1.0)
    bounds_max = (1.0, 0.6, 1.0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.backend == "reference":
        cfg = ReferenceTSDFConfig(
            voxel_size_m=args.voxel_size,
            truncation_distance_m=5 * args.voxel_size,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            store_color=True,
            device=device,
        )
        vol = ReferenceTSDF(cfg)
    else:  # triton
        from openvocab_tsdf.mapping.triton_backend import TritonTSDF, TritonTSDFConfig

        if not torch.cuda.is_available():
            raise SystemExit("triton backend requires CUDA")
        cfg = TritonTSDFConfig(
            voxel_size_m=args.voxel_size,
            truncation_distance_m=5 * args.voxel_size,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            store_color=True,
            device=device,
        )
        vol = TritonTSDF(cfg)

    def _run_once() -> float:
        vol.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f in frames:
            vol.integrate(f)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # warmup
    for _ in range(args.warmup):
        _run_once()
    # timed
    times = [_run_once() for _ in range(args.repeats)]

    fps_per_run = [args.frames / t for t in times]
    total_voxels = int(np.prod(vol.dims))
    observed_voxels = int((vol.weight > 0).sum().item())
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )

    result = {
        "name": "tsdf_fuse",
        "backend": args.backend,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": {
            "gpu": _gpu_name(),
            "vram_gb": round(_gpu_vram_gb(), 2),
            "host": platform.node(),
        },
        "params": {
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "voxel_size_m": args.voxel_size,
            "volume_dims": list(vol.dims),
            "total_voxels": total_voxels,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "metrics": {
            "fps_mean": float(np.mean(fps_per_run)),
            "fps_min": float(np.min(fps_per_run)),
            "fps_max": float(np.max(fps_per_run)),
            "seconds_mean": float(np.mean(times)),
            "observed_voxels": observed_voxels,
            "peak_vram_mb": round(peak_vram_mb, 1),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_tsdf_fuse_{args.backend}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
