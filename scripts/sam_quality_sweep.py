"""SAM-dense quality sweep: vary input resolution + auto-mask grid density.

Reuses the office0 baseline pipeline; the only knobs that change are
`sam_input_shortest_edge` and `points_per_side`. Each variant produces a
fresh map.npz (overwrites by default) and is evaluated against the same
hand-annotated spec.

    python scripts/sam_quality_sweep.py                  # pytorch (default)
    python scripts/sam_quality_sweep.py --backend tensorrt  # TRT fp16

When `--backend tensorrt`, outputs go to `sweep_trt_*.npz` so the pytorch
baselines stay cached alongside for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from openvocab_tsdf.config import load_config
from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.pipeline import build_dataset
from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder
from openvocab_tsdf.semantics.sam_dense import SAMDenseConfig, SAMDenseFeatureExtractor


def _encode(
    cfg_path: Path,
    sam_short_edge: int,
    sam_pps: int,
    out_path: Path,
    *,
    backend: str = "pytorch",
) -> tuple[float, int]:
    cfg = load_config(cfg_path)
    cfg.dataset.max_frames = 100
    cfg.dataset.stride = 20
    ds = build_dataset(cfg)
    frames: list[RGBDFrame] = list(ds)
    enc = OpenCLIPEncoder(
        OpenCLIPConfig(
            model=cfg.semantics.model,
            pretrained=cfg.semantics.pretrained,
            device=cfg.semantics.device,
            dtype=cfg.semantics.dtype,
        )
    )
    sam = SAMDenseFeatureExtractor(
        SAMDenseConfig(
            sam_input_shortest_edge=sam_short_edge,
            points_per_side=sam_pps,
            device=cfg.semantics.device,
            image_encoder_backend=backend,
        ),
        enc,
    )
    m = cfg.mapping
    vol = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=m.voxel_size_m,
            truncation_distance_m=m.truncation_distance_m,
            bounds_min=tuple(m.bounds_min),
            bounds_max=tuple(m.bounds_max),
            store_color=True,
            store_features=True,
            feature_dim=m.feature_dim,
            device=m.device,
        )
    )
    t0 = time.perf_counter()
    for f in frames:
        dfm_np = sam.extract(f.color)
        dfm = torch.from_numpy(dfm_np).to(m.device)
        vol.integrate(f, dense_feature_map=dfm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        tsdf=vol.tsdf.cpu().numpy(),
        weight=vol.weight.cpu().numpy(),
        color=vol.color.cpu().numpy(),
        feat=vol.feat.cpu().numpy(),
        origin=vol.origin.cpu().numpy(),
        voxel_size=np.float32(vol.voxel_size_m),
        truncation=np.float32(vol.truncation_distance_m),
        dims=np.int64(vol.dims),
        feature_dim=np.int64(enc.feature_dim),
        model=np.array(cfg.semantics.model, dtype=object),
        pretrained=np.array(cfg.semantics.pretrained, dtype=object),
        mode=np.array("sam_dense", dtype=object),
        sparse=np.bool_(False),
    )
    return dt, len(frames)


def _eval(map_path: Path, spec_path: Path) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from eval_grounding import run_eval

    return run_eval(map_path, spec_path, Path("benchmarks/results"))["summary"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["pytorch", "tensorrt"], default="pytorch")
    p.add_argument("--force", action="store_true", help="re-encode even if cached")
    args = p.parse_args()
    rows = []
    targets = [
        ("office0", "configs/replica_office0_6cm_sam.yaml", "eval/specs/replica_office0.yaml"),
        ("room0", "configs/replica_room0_6cm_sam.yaml", "eval/specs/replica_room0.yaml"),
    ]
    variants = [
        ("vit_t", 384, 12),
        ("vit_t", 512, 16),
    ]
    tag = "trt" if args.backend == "tensorrt" else ""

    for scene, cfg_str, spec_str in targets:
        for model, sz, pps in variants:
            suffix = f"{tag}_" if tag else ""
            out = Path(f"outputs/sweep_{suffix}{scene}_{model}_{sz}_pps{pps}.npz")
            if out.exists() and not args.force:
                print(f"\n=== {scene} {model} @ {sz}/{pps} ({args.backend}, cached) ===")
                dt = -1.0
            else:
                print(
                    f"\n=== {scene} {model} @ short_edge={sz}, pps={pps} " f"({args.backend}) ==="
                )
                dt, n_frames = _encode(Path(cfg_str), sz, pps, out, backend=args.backend)
                print(f"  encode: {dt:.1f}s ({n_frames/dt:.2f} FPS)")
            summary = _eval(out, Path(spec_str))
            rows.append(
                {"scene": scene, "model": model, "edge": sz, "pps": pps, "encode_s": dt, **summary}
            )
            print(
                f"  hit@1 {100*summary['hit@1']:.1f}%  hit@5 {100*summary['hit@5']:.1f}%  "
                f"L2 {summary['mean_top1_l2_m_hit_only']:.2f}m  query {1000*summary['mean_latency_s']:.0f}ms"
            )

    print(f"\n### SAM-dense quality sweep ({args.backend}) — office0 & room0, 100 frames each")
    print("| scene | model | input | pps | encode (s) | hit@1 | hit@5 | hit-L2 (m) | query (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['scene']} | {r['model']} | {r['edge']} | {r['pps']} | "
            f"{r['encode_s']:.1f} | "
            f"{100*r['hit@1']:5.1f}% | {100*r['hit@5']:5.1f}% | "
            f"{r['mean_top1_l2_m_hit_only']:.2f} | {1000*r['mean_latency_s']:.0f} |"
        )

    out_doc = Path("benchmarks/results")
    out_doc.mkdir(parents=True, exist_ok=True)
    out_doc_path = out_doc / f"sam_quality_sweep_{args.backend}.json"
    out_doc_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"\nwrote {out_doc_path}")


if __name__ == "__main__":
    main()
