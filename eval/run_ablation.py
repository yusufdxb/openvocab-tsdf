"""Run an ablation sweep over {map, mean_sub, top_percentile, surface_only}.

Each row in the sweep runs the full eval against the given map + spec. Rows
that differ only in query-time settings reuse the same map file. Output is:
  - one JSON per row in `benchmarks/results/`
  - a consolidated markdown table on stdout

    python eval/run_ablation.py \\
        --spec eval/specs/replica_room0.yaml \\
        --map-patch outputs/replica_room0_map.npz \\
        --map-global outputs/replica_room0_global_map.npz
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# let the script be run with `python eval/run_ablation.py` from repo root
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))


def _write_spec_override(base_spec: dict, overrides: dict) -> Path:
    spec = copy.deepcopy(base_spec)
    spec.update(overrides)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, prefix="abl_", mode="w", encoding="utf-8"
    )
    yaml.safe_dump(spec, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--map-patch", type=Path, required=True)
    p.add_argument("--map-global", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    # Load CLIP once
    _ = OpenCLIPEncoder(
        OpenCLIPConfig(
            model="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda:0", dtype="fp16"
        )
    )

    base_spec = yaml.safe_load(args.spec.read_text())

    sweep_axes = {
        "map": [("patch", args.map_patch), ("global", args.map_global)],
        "mean_sub": [False, True],
        "top_pct": [0.005, 0.02],
    }

    rows: list[dict] = []
    for (m_label, m_path), ms, tp in itertools.product(
        sweep_axes["map"], sweep_axes["mean_sub"], sweep_axes["top_pct"]
    ):
        overrides = {
            "scene_mean_subtract": ms,
            "top_percentile": tp,
            "score_threshold": None,
        }
        spec_path = _write_spec_override(base_spec, overrides)

        # Inline runner to avoid process overhead (and share CLIP weights)
        from eval_grounding import run_eval

        t0 = time.perf_counter()
        result = run_eval(m_path, spec_path, args.out_dir)
        dt_total = time.perf_counter() - t0

        summary = result["summary"]
        per_kind: dict[str, dict] = {}
        for c in result["per_case"]:
            k = None
            # re-read the spec to get kind
            for case in base_spec["cases"]:
                if case["query"] == c["query"]:
                    k = case.get("kind", "object")
                    break
            per_kind.setdefault(k, {"n": 0, "hit1": 0, "hit5": 0})
            per_kind[k]["n"] += 1
            per_kind[k]["hit1"] += int(c["hit@1"])
            per_kind[k]["hit5"] += int(c["hit@5"])

        rows.append(
            {
                "map": m_label,
                "mean_sub": ms,
                "top_pct": tp,
                **summary,
                "per_kind": per_kind,
                "wall_time_s": round(dt_total, 2),
            }
        )

    # Emit markdown table on stdout
    print("\n### Ablation table — Replica room0, 500 frames @ 1200×680, 6 cm voxels")
    print("")
    print("| map | mean-sub | top% | hit@1 | hit@5 | hit-L2 (m) | struct h@1 | obj h@1 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["per_kind"].get("structural", {"n": 0, "hit1": 0})
        o = r["per_kind"].get("object", {"n": 0, "hit1": 0})
        struct_h1 = f"{s['hit1']}/{s['n']}"
        obj_h1 = f"{o['hit1']}/{o['n']}"
        hit_l2 = r.get("mean_top1_l2_m_hit_only", float("nan"))
        hit_l2_s = f"{hit_l2:5.2f}" if hit_l2 == hit_l2 else "  n/a"
        print(
            f"| {r['map']} | {r['mean_sub']!s:<5} | {r['top_pct']:.3f} | "
            f"{100*r['hit@1']:5.1f}% | {100*r['hit@5']:5.1f}% | "
            f"{hit_l2_s} | {struct_h1} | {obj_h1} |"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_ablation_replica_room0.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
