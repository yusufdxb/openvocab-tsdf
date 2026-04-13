"""Grounding evaluation harness.

Input: a precomputed map (npz) and a YAML spec listing test queries with
expected 3D regions. Reports:

  - top-1 / top-5 grounding success (hit rate inside expected bbox)
  - mean centroid L2 error to the expected center (hits only)
  - median rank of the expected target in the returned list
  - end-to-end per-query latency (text encode + rank + cluster)

The spec format is intentionally simple so it is easy to hand-author for small
synthetic scenes and to generate programmatically for ScanNet-style labels.

    python eval/eval_grounding.py \\
        --map outputs/demo_map.npz \\
        --spec eval/specs/synthetic_demo.yaml

The runner writes a machine-readable JSON alongside stdout so results land in
benchmarks/results/<timestamp>_eval_grounding.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from openvocab_tsdf.grounding.query import rank_query
from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder


def _load_spec(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _in_bbox(
    point: tuple[float, float, float],
    bmin: list[float],
    bmax: list[float],
    slack: float = 0.0,
) -> bool:
    return all(bmin[i] - slack <= point[i] <= bmax[i] + slack for i in range(3))


def run_eval(map_path: Path, spec_path: Path, out_dir: Path) -> dict:
    spec = _load_spec(spec_path)
    data = np.load(map_path, allow_pickle=True)

    device = spec.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = spec.get("dtype", "fp16" if torch.cuda.is_available() else "fp32")

    weight = torch.from_numpy(data["weight"]).to(device)
    origin = data["origin"]
    voxel_size = float(data["voxel_size"])

    is_sparse = bool(data["sparse"]) if "sparse" in data.files else False
    if is_sparse:
        dims = tuple(int(d) for d in data["dims"])
        slot = torch.from_numpy(data["voxel_slot"]).to(device)
        pool = torch.from_numpy(data["feat_pool"]).to(device)
        feat = None  # sparse path: scores computed via pool
    else:
        feat = torch.from_numpy(data["feat"]).to(device)
        slot = None
        pool = None
        dims = tuple(feat.shape[:3])

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(
            model=spec.get("model", str(data["model"])),
            pretrained=spec.get("pretrained", str(data["pretrained"])),
            device=device,
            dtype=dtype,
        )
    )

    cases = spec["cases"]
    per_case = []
    for case in cases:
        q_text = case["query"]
        bmin_exp = case["bbox_min"]
        bmax_exp = case["bbox_max"]

        t0 = time.perf_counter()
        neg_text = case.get("neg") or spec.get("neg")
        if neg_text:
            t_emb = encoder.encode_texts([q_text, neg_text])
            q = t_emb[0]
            neg_e = t_emb[1]
        else:
            q = encoder.encode_texts([q_text])[0]
            neg_e = None
        if is_sparse:
            # Sparse: score only the observed voxels via the pool, scatter into
            # a dense (Nx,Ny,Nz) scores tensor (22 MB at 5.6M voxels).
            scores_flat = torch.full(
                (dims[0] * dims[1] * dims[2],), -1e4, dtype=torch.float32, device=device
            )
            alloc = slot.view(-1) >= 0
            pool_scores = pool @ q
            if neg_e is not None:
                pool_scores = pool_scores - pool @ neg_e
            scores_flat[alloc] = pool_scores[slot.view(-1)[alloc].long()]
            scores_vol = scores_flat.view(*dims)
            results = rank_query(
                voxel_feats=None,
                voxel_weights=weight,
                text_embedding=None,
                precomputed_scores=scores_vol,
                origin=origin,
                voxel_size=voxel_size,
                min_weight=float(spec.get("min_weight", 1.0)),
                score_threshold=spec.get("score_threshold"),
                top_percentile=spec.get("top_percentile", 0.02),
                cluster_eps_vox=int(spec.get("cluster_eps_vox", 2)),
                min_cluster_voxels=int(spec.get("min_cluster_voxels", 8)),
                top_k=int(spec.get("top_k", 5)),
                scene_mean_subtract=bool(spec.get("scene_mean_subtract", False)),
            )
        else:
            results = rank_query(
                voxel_feats=feat,
                voxel_weights=weight,
                text_embedding=q,
                origin=origin,
                voxel_size=voxel_size,
                min_weight=float(spec.get("min_weight", 1.0)),
                score_threshold=spec.get("score_threshold"),
                top_percentile=spec.get("top_percentile", 0.02),
                cluster_eps_vox=int(spec.get("cluster_eps_vox", 2)),
                min_cluster_voxels=int(spec.get("min_cluster_voxels", 8)),
                top_k=int(spec.get("top_k", 5)),
                scene_mean_subtract=bool(spec.get("scene_mean_subtract", False)),
                neg_text_embedding=neg_e,
            )
        latency = time.perf_counter() - t0

        slack = float(spec.get("bbox_slack_m", 0.1))
        # hit rate: centroid within expected bbox (with slack)
        hit1 = bool(results and _in_bbox(results[0].center_m, bmin_exp, bmax_exp, slack))
        hit5 = any(_in_bbox(r.center_m, bmin_exp, bmax_exp, slack) for r in results)
        rank_of_hit = next(
            (
                i + 1
                for i, r in enumerate(results)
                if _in_bbox(r.center_m, bmin_exp, bmax_exp, slack)
            ),
            None,
        )

        # centroid L2 to expected center (center of expected bbox)
        exp_center = 0.5 * (np.array(bmin_exp) + np.array(bmax_exp))
        if results:
            l2 = float(np.linalg.norm(np.array(results[0].center_m) - exp_center))
        else:
            l2 = float("inf")

        per_case.append(
            {
                "query": q_text,
                "hit@1": bool(hit1),
                "hit@5": bool(hit5),
                "rank_of_hit": rank_of_hit,
                "top1_centroid_l2_m": l2,
                "top1_score": float(results[0].score) if results else None,
                "latency_s": latency,
                "num_results": len(results),
            }
        )

    def _mean(key: str) -> float:
        vals = [c[key] for c in per_case if isinstance(c[key], (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    # hit-only L2: mean distance conditioned on at least a top-5 hit — this is
    # the more informative localization number for cases the system can solve.
    hit_l2 = [c["top1_centroid_l2_m"] for c in per_case if c["hit@5"]]
    summary = {
        "num_cases": len(cases),
        "hit@1": float(np.mean([c["hit@1"] for c in per_case])),
        "hit@5": float(np.mean([c["hit@5"] for c in per_case])),
        "mean_top1_l2_m": _mean("top1_centroid_l2_m"),
        "mean_top1_l2_m_hit_only": float(np.mean(hit_l2)) if hit_l2 else float("nan"),
        "mean_latency_s": _mean("latency_s"),
    }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map": str(map_path),
        "spec": str(spec_path),
        "summary": summary,
        "per_case": per_case,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{stamp}_eval_grounding.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    result = run_eval(args.map, args.spec, args.out_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
