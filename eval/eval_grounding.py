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
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from openvocab_tsdf.grounding.map_bundle import MapBundle
from openvocab_tsdf.grounding.query import rank_query
from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

log = logging.getLogger(__name__)


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
    device = spec.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = spec.get("dtype", "fp16" if torch.cuda.is_available() else "fp32")

    # All three on-disk layouts go through MapBundle. The bundle materialises a
    # dense (Nx,Ny,Nz) score tensor per query without ever building the 4-D
    # feature volume — for voxel_slot/block_hash it scores the pool first then
    # scatters; for dense it does the standard `feat @ q` reshape.
    bundle = MapBundle(map_path, device=device)

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(
            model=spec.get("model", bundle.meta.model),
            pretrained=spec.get("pretrained", bundle.meta.pretrained),
            device=device,
            dtype=dtype,
        )
    )

    if encoder.cfg.model != bundle.meta.model and bundle.meta.model:
        log.warning(
            "model mismatch: map built with %r, eval spec uses %r — " "results may be invalid",
            bundle.meta.model,
            encoder.cfg.model,
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
        scores_vol = bundle.score_query(q, neg=neg_e)
        # Note: `voxel_tsdf` deliberately omitted so `surface_only` is a no-op,
        # matching the pre-refactor behavior of this script (neither the dense
        # nor the sparse branch used to pass tsdf in).
        results = rank_query(
            voxel_feats=None,
            voxel_weights=bundle.weight,
            text_embedding=None,
            precomputed_scores=scores_vol,
            origin=bundle.origin,
            voxel_size=bundle.voxel_size,
            min_weight=float(spec.get("min_weight", 1.0)),
            score_threshold=spec.get("score_threshold"),
            top_percentile=spec.get("top_percentile", 0.02),
            cluster_eps_vox=int(spec.get("cluster_eps_vox", 2)),
            min_cluster_voxels=int(spec.get("min_cluster_voxels", 8)),
            top_k=int(spec.get("top_k", 5)),
            scene_mean_subtract=bool(spec.get("scene_mean_subtract", False)),
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
