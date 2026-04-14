"""Benchmark: MobileSAM image-encoder throughput — PyTorch vs TensorRT.

Two levels of comparison:

  1. Raw image_encoder forward at the native 1024×1024 input. This is the
     apples-to-apples speedup claim for the TRT path — everything else
     (mask prompts, post-processing) is identical downstream.

  2. End-to-end `SAMDenseFeatureExtractor.extract()` at a user-configurable
     `sam_input_shortest_edge`. The image_encoder always runs at 1024
     (that shape is baked into the TinyViT weights); the shortest-edge
     knob just controls the upstream resize and therefore mask quality,
     not encoder cost. The end-to-end timing is still useful because it
     reflects actual pipeline wall-clock with SAM post-processing + CLIP
     per mask.

        python benchmarks/bench_sam_encode.py --encoder --e2e 384 --e2e 512 --trt
        # …optionally add --trt-fp16 to bench the lossy-but-fast engine

Writes a JSON result to `benchmarks/results/<stamp>_sam_encode.json`.
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


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": "cpu"}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "host": platform.node(),
    }


def _deterministic_images(n: int, hw: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(hw, hw, 3), dtype=np.uint8) for _ in range(n)]


def bench_encoder_pytorch(cfg, images: torch.Tensor, warmup: int, repeats: int) -> dict:
    """Raw image_encoder forward at 1024×1024, PyTorch fp16."""
    from mobile_sam import sam_model_registry

    sam = sam_model_registry[cfg["sam_model_type"]](checkpoint=str(cfg["sam_weights_path"]))
    sam = sam.to(cfg["device"]).eval().half()
    encoder = sam.image_encoder

    def _run() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(images.shape[0]):
                _ = encoder(images[i : i + 1].half())
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        _run()
    times = [_run() for _ in range(repeats)]
    n = int(images.shape[0])
    return _summarise("encoder_pytorch_fp16", times, n)


def bench_encoder_tensorrt(cfg, images: torch.Tensor, warmup: int, repeats: int) -> dict:
    """Raw image_encoder forward at 1024×1024, TRT engine (precision per cfg)."""
    from openvocab_tsdf.semantics.trt_sam import (
        TensorRTSamEncoder,
        TRTSamConfig,
        build_engine_from_mobile_sam,
    )

    trt_cfg = TRTSamConfig(
        sam_weights_path=cfg["sam_weights_path"],
        sam_model_type=cfg["sam_model_type"],
        engine_path=cfg["engine_path"],
        onnx_path=cfg["onnx_path"],
        fp16=cfg.get("fp16", False),
        device=cfg["device"],
    )
    if trt_cfg.engine_path.exists():
        encoder = TensorRTSamEncoder(trt_cfg)
    else:
        encoder = build_engine_from_mobile_sam(trt_cfg)

    def _run() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = encoder(images)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        _run()
    times = [_run() for _ in range(repeats)]
    n = int(images.shape[0])
    return _summarise(
        "encoder_tensorrt_" + ("fp16" if cfg.get("fp16", False) else "fp32"), times, n
    )


def bench_extract_backend(
    cfg, images_uint8: list[np.ndarray], shortest_edge: int, backend: str, warmup: int, repeats: int
) -> dict:
    """End-to-end SAMDenseFeatureExtractor.extract timing at a given
    `sam_input_shortest_edge`, for the chosen `image_encoder_backend`.

    Reports wall-clock per-frame time for the whole SAM + CLIP path, not
    just the encoder forward.
    """
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder
    from openvocab_tsdf.semantics.sam_dense import SAMDenseConfig, SAMDenseFeatureExtractor

    clip_enc = OpenCLIPEncoder(
        OpenCLIPConfig(
            model="ViT-B-16", pretrained="laion2b_s34b_b88k", device=cfg["device"], dtype="fp16"
        )
    )
    sam_cfg = SAMDenseConfig(
        sam_weights_path=cfg["sam_weights_path"],
        sam_model_type=cfg["sam_model_type"],
        sam_input_shortest_edge=shortest_edge,
        device=cfg["device"],
        image_encoder_backend=backend,
        trt_engine_path=cfg["engine_path"],
        trt_onnx_path=cfg["onnx_path"],
        trt_fp16=cfg.get("fp16", False),
    )
    extractor = SAMDenseFeatureExtractor(sam_cfg, clip_enc)

    def _run() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img in images_uint8:
            _ = extractor.extract(img)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        _run()
    times = [_run() for _ in range(repeats)]
    n = len(images_uint8)
    return _summarise(f"extract_{backend}_se{shortest_edge}", times, n)


def _summarise(name: str, times: list[float], n: int) -> dict:
    return {
        "backend": name,
        "times_s": times,
        "fps_mean": float(n / float(np.mean(times))),
        "fps_min": float(n / float(np.max(times))),
        "fps_max": float(n / float(np.min(times))),
        "ms_per_frame_mean": float(1000.0 * float(np.mean(times)) / n),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", action="store_true", help="bench raw image_encoder at 1024×1024")
    p.add_argument(
        "--e2e", type=int, action="append", default=None, help="end-to-end at this shortest_edge"
    )
    p.add_argument("--trt", action="store_true", help="also bench TRT variants")
    p.add_argument(
        "--trt-fp16",
        action="store_true",
        help="bench the fp16 TRT engine (default fp32). See docs/decisions.md for the accuracy trade-off.",
    )
    p.add_argument("--images", type=int, default=16)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("~/.cache/openvocab_tsdf/weights/mobile_sam.pt").expanduser(),
    )
    p.add_argument(
        "--engine",
        type=Path,
        default=Path("outputs/trt/mobile_sam_vit_t_fp32.engine"),
    )
    p.add_argument("--onnx", type=Path, default=Path("outputs/trt/mobile_sam_vit_t_fp32.onnx"))
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    if not args.encoder and not args.e2e:
        # by default, do both the apples-to-apples encoder timing and an
        # end-to-end sweep at the production default short edge.
        args.encoder = True
        args.e2e = [384]

    # Default TRT paths point at the fp32 engine. If the caller asks for fp16,
    # swap the filenames too so we don't clobber the fp32 cache.
    engine_path = args.engine
    onnx_path = args.onnx
    if args.trt_fp16 and engine_path.name == "mobile_sam_vit_t_fp32.engine":
        engine_path = engine_path.with_name("mobile_sam_vit_t_fp16.engine")
    if args.trt_fp16 and onnx_path.name == "mobile_sam_vit_t_fp32.onnx":
        onnx_path = onnx_path.with_name("mobile_sam_vit_t.onnx")

    cfg = {
        "sam_model_type": "vit_t",
        "sam_weights_path": args.weights,
        "engine_path": engine_path,
        "onnx_path": onnx_path,
        "fp16": args.trt_fp16,
        "device": args.device,
    }

    result = {
        "name": "sam_encode",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": _gpu_info(),
        "params": {
            "images": args.images,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "encoder": args.encoder,
            "e2e_shortest_edges": args.e2e or [],
        },
        "runs": [],
    }

    if args.encoder:
        # Raw encoder: 1024×1024 fp32 tensors (bench_encoder_pytorch will cast).
        torch.manual_seed(0)
        enc_inputs = torch.randn(
            args.images, 3, 1024, 1024, device=args.device, dtype=torch.float32
        )
        print("encoder pytorch fp16 @ 1024×1024 ...")
        result["runs"].append(bench_encoder_pytorch(cfg, enc_inputs, args.warmup, args.repeats))
        print(
            f"  pytorch: {result['runs'][-1]['fps_mean']:.2f} FPS "
            f"({result['runs'][-1]['ms_per_frame_mean']:.1f} ms/frame)"
        )
        if args.trt:
            try:
                print("encoder tensorrt fp16 @ 1024×1024 ...")
                result["runs"].append(
                    bench_encoder_tensorrt(cfg, enc_inputs, args.warmup, args.repeats)
                )
                print(
                    f"  tensorrt: {result['runs'][-1]['fps_mean']:.2f} FPS "
                    f"({result['runs'][-1]['ms_per_frame_mean']:.1f} ms/frame)"
                )
            except Exception as e:
                result["runs"].append({"backend": "encoder_tensorrt_fp16", "error": repr(e)})
                print(f"  tensorrt failed: {e}")

    for se in args.e2e or []:
        # End-to-end: bigger input images mean more SAM mask-proposal work.
        # We use a short-edge fixed at 480 to mimic Replica RGB frames.
        imgs = _deterministic_images(max(4, args.images // 4), hw=480)
        print(f"extract pytorch, shortest_edge={se} ...")
        result["runs"].append(
            bench_extract_backend(cfg, imgs, se, "pytorch", args.warmup, args.repeats)
        )
        print(
            f"  pytorch: {result['runs'][-1]['fps_mean']:.2f} FPS "
            f"({result['runs'][-1]['ms_per_frame_mean']:.1f} ms/frame)"
        )
        if args.trt:
            try:
                print(f"extract tensorrt, shortest_edge={se} ...")
                result["runs"].append(
                    bench_extract_backend(cfg, imgs, se, "tensorrt", args.warmup, args.repeats)
                )
                print(
                    f"  tensorrt: {result['runs'][-1]['fps_mean']:.2f} FPS "
                    f"({result['runs'][-1]['ms_per_frame_mean']:.1f} ms/frame)"
                )
            except Exception as e:
                result["runs"].append({"backend": f"extract_tensorrt_se{se}", "error": repr(e)})
                print(f"  tensorrt failed: {e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_sam_encode.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
