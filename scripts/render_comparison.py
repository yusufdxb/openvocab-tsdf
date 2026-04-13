"""Side-by-side comparison figure: baseline vs treatment.

Renders the same queries from two heatmap directories (typically the same
scene with a different feature extractor — e.g. global vs sam_dense), in a
2-row × 3-view grid per query, so the accuracy lift is visually obvious.

    python scripts/render_comparison.py \\
        --mesh outputs/replica_room0_mesh.ply \\
        --baseline-dir outputs/heatmaps_replica_room0 --baseline-label "global CLIP" \\
        --treatment-dir outputs/heatmaps_room0_sam --treatment-label "SAM + CLIP" \\
        --spec eval/specs/replica_room0.yaml \\
        --queries "a sofa" "a chair" "a plant" \\
        --out-dir figures/comparison_room0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Re-use PLY loader from render_figures.py
from render_figures import _parse_ply  # type: ignore[import-not-found]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "q"


_PROJ = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _panel(
    ax,
    mesh_xyz: np.ndarray,
    mesh_rgb: np.ndarray,
    heat_xyz: np.ndarray,
    heat_rgb: np.ndarray,
    gt_min,
    gt_max,
    proj: str,
) -> None:
    i, j = _PROJ[proj]
    rng = np.random.default_rng(0)
    mesh_sub = rng.choice(len(mesh_xyz), size=min(120_000, len(mesh_xyz)), replace=False)
    mcols = (mesh_rgb[mesh_sub] / 255.0) * 0.35 + 0.6
    ax.scatter(
        mesh_xyz[mesh_sub, i],
        mesh_xyz[mesh_sub, j],
        c=mcols,
        s=0.5,
        alpha=0.35,
        linewidths=0,
    )
    heat_bright = heat_rgb.astype(np.float32).mean(axis=-1)
    cutoff = float(np.quantile(heat_bright, 0.80)) if len(heat_bright) else 0.0
    hot = heat_bright >= cutoff
    hx = heat_xyz[hot]
    hc = (heat_rgb[hot] / 255.0) if len(heat_rgb[hot]) else np.zeros((0, 3))
    hsub = rng.choice(len(hx), size=min(8_000, len(hx)), replace=False) if len(hx) else []
    if len(hsub):
        order = np.argsort(hc[hsub].mean(axis=-1))
        ax.scatter(
            hx[hsub[order], i],
            hx[hsub[order], j],
            c=hc[hsub[order]],
            s=4.0,
            alpha=0.9,
            linewidths=0,
        )
    if gt_min is not None and gt_max is not None:
        rect = mpatches.Rectangle(
            (gt_min[i], gt_min[j]),
            gt_max[i] - gt_min[i],
            gt_max[j] - gt_min[j],
            fill=False,
            edgecolor="red",
            linewidth=1.6,
            zorder=5,
        )
        ax.add_patch(rect)
    lo_x, hi_x = mesh_xyz[:, i].min(), mesh_xyz[:, i].max()
    lo_y, hi_y = mesh_xyz[:, j].min(), mesh_xyz[:, j].max()
    px = 0.05 * max(1e-3, hi_x - lo_x)
    py = 0.05 * max(1e-3, hi_y - lo_y)
    ax.set_xlim(lo_x - px, hi_x + px)
    ax.set_ylim(lo_y - py, hi_y + py)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--baseline-label", type=str, default="baseline")
    p.add_argument("--treatment-dir", type=Path, required=True)
    p.add_argument("--treatment-label", type=str, default="treatment")
    p.add_argument("--spec", type=Path, required=False)
    p.add_argument("--queries", type=str, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    print(f"loading mesh {args.mesh} ...")
    mesh_xyz, mesh_rgb = _parse_ply(args.mesh)

    spec_bboxes = {}
    if args.spec is not None:
        spec = yaml.safe_load(args.spec.read_text())
        for case in spec.get("cases", []):
            spec_bboxes[case["query"]] = (case["bbox_min"], case["bbox_max"])

    for q in args.queries:
        fname = f"{_slug(q)}.ply"
        bl_ply = args.baseline_dir / fname
        tr_ply = args.treatment_dir / fname
        if not bl_ply.exists() or not tr_ply.exists():
            print(f"  skip {q!r}: missing {bl_ply.name} or {tr_ply.name}")
            continue
        print(f"  rendering comparison for {q!r}")

        bl_xyz, bl_rgb = _parse_ply(bl_ply)
        tr_xyz, tr_rgb = _parse_ply(tr_ply)
        gt = spec_bboxes.get(q)
        gt_min = gt[0] if gt else None
        gt_max = gt[1] if gt else None

        fig, axes = plt.subplots(2, 3, figsize=(13, 8.3))
        fig.suptitle(
            f"query: {q!r}   —   {args.baseline_label}  vs  {args.treatment_label}", fontsize=12
        )
        for col, (name, proj) in enumerate(
            [("xy — top-down", "xy"), ("xz — front", "xz"), ("yz — side", "yz")]
        ):
            _panel(axes[0, col], mesh_xyz, mesh_rgb, bl_xyz, bl_rgb, gt_min, gt_max, proj)
            _panel(axes[1, col], mesh_xyz, mesh_rgb, tr_xyz, tr_rgb, gt_min, gt_max, proj)
            axes[0, col].set_title(name, fontsize=9)
        axes[0, 0].set_ylabel(args.baseline_label, fontsize=10)
        axes[1, 0].set_ylabel(args.treatment_label, fontsize=10)

        fig.tight_layout()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{_slug(q)}_compare.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> {out}")


if __name__ == "__main__":
    main()
