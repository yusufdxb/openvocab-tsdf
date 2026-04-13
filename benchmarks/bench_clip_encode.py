"""Benchmark: CLIP image-encode throughput — PyTorch fp16 vs TensorRT fp16.

    python benchmarks/bench_clip_encode.py --batch 16 --images 128 --trt

Writes a JSON result to `benchmarks/results/<stamp>_clip_encode_<backend>.json`.
TensorRT is optional — skips cleanly if the package is not installed or the
engine cannot be built.
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


def _random_rgb_batch(n: int, hw: int = 224) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(hw, hw, 3), dtype=np.uint8) for _ in range(n)]


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": "cpu"}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "host": platform.node(),
    }


def bench_pytorch(images: list[np.ndarray], batch: int, warmup: int, repeats: int) -> dict:
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    enc = OpenCLIPEncoder(
        OpenCLIPConfig(
            model="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda:0", dtype="fp16"
        )
    )

    def _run() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = enc.encode_images(images, batch_size=batch)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        _run()
    times = [_run() for _ in range(repeats)]
    return {
        "backend": "pytorch_fp16",
        "times_s": times,
        "fps_mean": float(len(images) / np.mean(times)),
        "fps_min": float(len(images) / np.max(times)),
        "fps_max": float(len(images) / np.min(times)),
    }


def bench_tensorrt(
    images: list[np.ndarray],
    batch: int,
    warmup: int,
    repeats: int,
    engine_path: Path,
) -> dict:
    from openvocab_tsdf.semantics.trt_encoder import (
        TRTImageEncoderConfig,
        build_engine_from_openclip,
    )

    cfg = TRTImageEncoderConfig(engine_path=engine_path, batch_size=batch)
    enc = build_engine_from_openclip(cfg)

    def _run() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = enc.encode_images(images)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        _run()
    times = [_run() for _ in range(repeats)]
    return {
        "backend": "tensorrt_fp16",
        "times_s": times,
        "fps_mean": float(len(images) / np.mean(times)),
        "fps_min": float(len(images) / np.max(times)),
        "fps_max": float(len(images) / np.min(times)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--images", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--trt", action="store_true", help="also benchmark TensorRT")
    p.add_argument("--engine", type=Path, default=Path("outputs/trt/clip_vit_b16_fp16.engine"))
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    images = _random_rgb_batch(args.images)

    result = {
        "name": "clip_encode",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": _gpu_info(),
        "params": {
            "batch": args.batch,
            "images": args.images,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "runs": [],
    }

    print("running pytorch fp16 ...")
    result["runs"].append(bench_pytorch(images, args.batch, args.warmup, args.repeats))
    print(f"  pytorch fp16: {result['runs'][-1]['fps_mean']:.1f} FPS")

    if args.trt:
        try:
            print("running tensorrt fp16 ...")
            result["runs"].append(
                bench_tensorrt(images, args.batch, args.warmup, args.repeats, args.engine)
            )
            print(f"  tensorrt fp16: {result['runs'][-1]['fps_mean']:.1f} FPS")
        except Exception as e:
            result["runs"].append({"backend": "tensorrt_fp16", "error": repr(e)})
            print(f"  tensorrt failed: {e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_clip_encode.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
