"""Render a headless 3-view PNG (top-down, front, side) of a colored PLY mesh.

No grounding heatmap required (unlike render_figures.py, which overlays a
query heatmap). Used for the real GO2 rosbag reconstruction figure, where the
point is showing the raw geometry + color, not a grounding result.

    python scripts/render_real_bag_mesh.py \\
        --mesh /path/to/go2_room_recon.ply \\
        --out figures/go2_real_bag_recon.png \\
        --title "GO2 real-bag reconstruction (session_20260331_1957)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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


_PROJ_INDICES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default="")
    args = p.parse_args()

    xyz, rgb = _parse_ply(args.mesh)
    print(f"loaded {len(xyz)} verts from {args.mesh}")
    cols = rgb.astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    if args.title:
        fig.suptitle(args.title, fontsize=11)

    projections = [
        ("top-down (xy)", "xy"),
        ("front (xz)", "xz"),
        ("side (yz)", "yz"),
    ]
    labels = {"xy": ("x (m)", "y (m)"), "xz": ("x (m)", "z (m)"), "yz": ("y (m)", "z (m)")}
    for ax, (name, proj) in zip(axes, projections, strict=True):
        i, j = _PROJ_INDICES[proj]
        ax.scatter(xyz[:, i], xyz[:, j], c=cols, s=1.2, alpha=0.9, linewidths=0)
        lo_x, hi_x = xyz[:, i].min(), xyz[:, i].max()
        lo_y, hi_y = xyz[:, j].min(), xyz[:, j].max()
        px = 0.05 * max(1e-3, hi_x - lo_x)
        py = 0.05 * max(1e-3, hi_y - lo_y)
        ax.set_xlim(lo_x - px, hi_x + px)
        ax.set_ylim(lo_y - py, hi_y + py)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#111111")
        ax.set_title(name, fontsize=9)
        xl, yl = labels[proj]
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
