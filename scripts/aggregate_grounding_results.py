"""Aggregate per-scene eval_grounding JSONs into a single summary.

Inputs: a list of (scene_name, json_path) pairs.
Outputs:
  - a JSON dict with per-scene metrics + an unweighted aggregate.
  - a Markdown table on stdout, copy-pastable into README.md.

CLI:

    python scripts/aggregate_grounding_results.py \\
        --pair scene_a=benchmarks/results/<ts>_eval_grounding.json \\
        --pair scene_b=benchmarks/results/<ts>_eval_grounding.json \\
        --out benchmarks/results/<ts>_replica_aggregate.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def aggregate(scene_results: list[tuple[str, Path]]) -> dict:
    per_scene: dict[str, dict] = {}
    for scene, path in scene_results:
        data = json.loads(Path(path).read_text())
        s = data["summary"]
        per_scene[scene] = {
            "hit@1": float(s["hit@1"]),
            "hit@5": float(s["hit@5"]),
            "mean_top1_l2_m": float(s["mean_top1_l2_m"]),
            "n_queries": int(s["num_cases"]),
            "source": str(path),
        }

    n = len(per_scene)
    if n == 0:
        agg = {
            "n_scenes": 0,
            "total_queries": 0,
            "mean_hit@1": 0.0,
            "mean_hit@5": 0.0,
            "mean_top1_l2_m": 0.0,
        }
    else:
        agg = {
            "n_scenes": n,
            "total_queries": sum(v["n_queries"] for v in per_scene.values()),
            "mean_hit@1": sum(v["hit@1"] for v in per_scene.values()) / n,
            "mean_hit@5": sum(v["hit@5"] for v in per_scene.values()) / n,
            "mean_top1_l2_m": sum(v["mean_top1_l2_m"] for v in per_scene.values()) / n,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_scene": per_scene,
        "aggregate": agg,
    }


def format_markdown_table(result: dict) -> str:
    rows = [
        "| scene | n_queries | hit@1 | hit@5 | mean top-1 L2 (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene, v in result["per_scene"].items():
        rows.append(
            f"| {scene} | {v['n_queries']} | "
            f"{v['hit@1']:.3f} | {v['hit@5']:.3f} | {v['mean_top1_l2_m']:.3f} |"
        )
    a = result["aggregate"]
    rows.append(
        f"| **aggregate** | {a['total_queries']} | "
        f"{a['mean_hit@1']:.3f} | {a['mean_hit@5']:.3f} | {a['mean_top1_l2_m']:.3f} |"
    )
    return "\n".join(rows)


def _parse_pairs(raw: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for chunk in raw:
        if "=" not in chunk:
            raise SystemExit(f"--pair expects scene=path, got: {chunk}")
        name, p = chunk.split("=", 1)
        out.append((name, Path(p)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pair",
        action="append",
        default=[],
        help="scene=path/to/eval.json (repeatable)",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pairs = _parse_pairs(args.pair)
    if not pairs:
        raise SystemExit("at least one --pair is required")

    result = aggregate(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_markdown_table(result))


if __name__ == "__main__":
    main()
