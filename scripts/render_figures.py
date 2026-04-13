"""Render publishable 3-view figures for the README / paper.

For each query, produces one PNG with:
  - top-down (xy) view — full mesh in light gray, heatmap overlaid, GT bbox outline
  - side (xz) view — same
  - perspective — same

Backed by matplotlib 3D so it runs headless, no OpenGL / EGL / display needed.

    python scripts/render_figures.py \\
        --mesh outputs/replica_room0_mesh.ply \\
        --heatmap-dir outputs/heatmaps_replica_room0 \\
        --spec eval/specs/replica_room0.yaml \\
        --out-dir figures/room0
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


def _parse_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices xyz, colors rgb u8). Handles ascii-header binary bodies."""
    with path.open("rb") as f:
        props: list[tuple[str, str]] = []
        n_verts = 0
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
                in_vertex = True
            elif line.startswith(b"element"):
                in_vertex = False
            elif line.startswith(b"property") and in_vertex:
                parts = line.decode("ascii").strip().split()
                # property <type> <name>
                props.append((parts[1], parts[2]))
            elif line.startswith(b"end_header"):
                break

        dt_fields: list[tuple[str, str]] = []
        for tname, pname in props:
            dt_map = {
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "uchar": "u1",
                "uint8": "u1",
                "int": "<i4",
                "int32": "<i4",
            }
            dt_fields.append((pname, dt_map[tname]))
        dt = np.dtype(dt_fields)
        raw = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt, count=n_verts)

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)
    has_rgb = all(k in raw.dtype.names for k in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.stack([raw["red"], raw["green"], raw["blue"]], axis=-1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    return xyz, rgb


def _subsample(a: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(a) <= cap:
        return a
    idx = rng.choice(len(a), size=cap, replace=False)
    return a[idx]


_PROJ_INDICES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _draw_bbox_2d(ax, bmin: list[float], bmax: list[float], proj: str, color: str = "red") -> None:
    """Outline the bbox projected onto a 2D plane."""
    i, j = _PROJ_INDICES[proj]
    x0, x1 = bmin[i], bmax[i]
    y0, y1 = bmin[j], bmax[j]
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.6, zorder=5
    )
    ax.add_patch(rect)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "q"


def _render_one_query(
    mesh_xyz: np.ndarray,
    mesh_rgb: np.ndarray,
    heat_xyz: np.ndarray,
    heat_rgb: np.ndarray,
    gt_min: list[float] | None,
    gt_max: list[float] | None,
    title: str,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    mesh_sub = _subsample(np.arange(len(mesh_xyz)), 120_000, rng)

    # Filter heatmap to the top-brightness quartile (bright = high score via
    # our viridis-like colormap) so the signal pops against the mesh.
    heat_bright = heat_rgb.astype(np.float32).mean(axis=-1)
    cutoff = float(np.quantile(heat_bright, 0.80))
    hot = heat_bright >= cutoff
    heat_xyz = heat_xyz[hot]
    heat_rgb = heat_rgb[hot]
    heat_sub = _subsample(np.arange(len(heat_xyz)), 8_000, rng)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=11)

    projections = [("xy — top-down", "xy"), ("xz — front", "xz"), ("yz — side", "yz")]
    labels = {"xy": ("x (m)", "y (m)"), "xz": ("x (m)", "z (m)"), "yz": ("y (m)", "z (m)")}
    for ax, (name, proj) in zip(axes, projections, strict=True):
        i, j = _PROJ_INDICES[proj]

        # Mesh wash (light gray-ish from vertex color)
        mcols = (mesh_rgb[mesh_sub] / 255.0) * 0.35 + 0.6
        ax.scatter(
            mesh_xyz[mesh_sub, i],
            mesh_xyz[mesh_sub, j],
            c=mcols,
            s=0.5,
            alpha=0.35,
            linewidths=0,
        )

        # Heatmap points (already viridis-colored)
        hcols = heat_rgb[heat_sub] / 255.0
        # sort so the hottest (yellow) end is drawn on top
        brightness = hcols.mean(axis=-1)
        order = np.argsort(brightness)
        ax.scatter(
            heat_xyz[heat_sub[order], i],
            heat_xyz[heat_sub[order], j],
            c=hcols[order],
            s=4.0,
            alpha=0.85,
            linewidths=0,
        )

        if gt_min is not None and gt_max is not None:
            _draw_bbox_2d(ax, gt_min, gt_max, proj, color="red")

        lo_x, hi_x = mesh_xyz[:, i].min(), mesh_xyz[:, i].max()
        lo_y, hi_y = mesh_xyz[:, j].min(), mesh_xyz[:, j].max()
        px = 0.05 * max(1e-3, hi_x - lo_x)
        py = 0.05 * max(1e-3, hi_y - lo_y)
        ax.set_xlim(lo_x - px, hi_x + px)
        ax.set_ylim(lo_y - py, hi_y + py)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=9)
        xl, yl = labels[proj]
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--heatmap-dir", type=Path, required=True)
    p.add_argument(
        "--spec", type=Path, required=False, help="optional eval spec for GT bbox overlay"
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    print(f"loading mesh {args.mesh} ...")
    mesh_xyz, mesh_rgb = _parse_ply(args.mesh)
    print(f"  mesh: {len(mesh_xyz)} verts")

    spec_bboxes: dict[str, tuple[list[float], list[float]]] = {}
    if args.spec is not None:
        spec = yaml.safe_load(args.spec.read_text())
        for case in spec.get("cases", []):
            spec_bboxes[case["query"]] = (case["bbox_min"], case["bbox_max"])

    heatmaps = sorted(args.heatmap_dir.glob("*.ply"))
    print(f"found {len(heatmaps)} heatmap PLYs")

    # Try to recover the original query from the filename slug; fall back to slug
    def _match_query(name: str) -> str:
        if not spec_bboxes:
            return name
        for q in spec_bboxes:
            if _slug(q) == name:
                return q
        return name

    for hm_path in heatmaps:
        stem = hm_path.stem
        q = _match_query(stem)
        print(f"  rendering {q!r} -> {args.out_dir / (stem + '.png')}")
        heat_xyz, heat_rgb = _parse_ply(hm_path)
        gt = spec_bboxes.get(q)
        _render_one_query(
            mesh_xyz=mesh_xyz,
            mesh_rgb=mesh_rgb,
            heat_xyz=heat_xyz,
            heat_rgb=heat_rgb,
            gt_min=gt[0] if gt else None,
            gt_max=gt[1] if gt else None,
            title=f"{args.mesh.stem}  —  query: {q!r}",
            out_path=args.out_dir / f"{stem}.png",
        )


if __name__ == "__main__":
    main()
