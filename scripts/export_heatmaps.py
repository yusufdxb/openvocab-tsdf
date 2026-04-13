"""Write a heatmap PLY for each query against a saved map.

    python scripts/export_heatmaps.py \\
        --map outputs/demo_map.npz \\
        --query "a red ball" --query "a blue bar" --query "green grass floor" \\
        --out-dir outputs/heatmaps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openvocab_tsdf.viz.heatmap import save_query_heatmap_ply


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "q"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--query", action="append", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/heatmaps"))
    p.add_argument("--model", type=str, default="ViT-B-16")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="fp16")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for q in args.query:
        out = args.out_dir / f"{_slug(q)}.ply"
        info = save_query_heatmap_ply(
            args.map,
            q,
            out,
            model=args.model,
            pretrained=args.pretrained,
            device=args.device,
            dtype=args.dtype,
        )
        print(f"  {q!r:40s} -> {out}   (N={info['num_points']} p99={info['score_p99']:.3f})")
        summary.append(info)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
