"""Inspect a Replica mesh to help write an eval spec.

Prints:
  - scene bbox
  - floor / ceiling z-planes (from horizontal-surface z histogram)
  - a coarse xy floor-plan density map of object-height vertices

    python scripts/inspect_replica_mesh.py \\
        --mesh $HOME/data/replica/Replica/office0_mesh.ply \\
        --grid 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_replica_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replica meshes: (xyz + nxnynz + u8 rgb) + faces. Return (xyz, normals, rgb)."""
    with path.open("rb") as f:
        header_lines = []
        n_verts = None
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
        if n_verts is None:
            raise RuntimeError("could not parse vertex count")
        dt = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
        )
        verts = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt, count=n_verts)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float64)
    nrm = np.stack([verts["nx"], verts["ny"], verts["nz"]], axis=-1).astype(np.float64)
    rgb = np.stack([verts["r"], verts["g"], verts["b"]], axis=-1).astype(np.uint8)
    return xyz, nrm, rgb


def _print_density_map(pts: np.ndarray, x_lo: float, x_hi: float, y_lo: float, y_hi: float, grid: float) -> None:
    bins_x = np.arange(x_lo, x_hi + grid, grid)
    bins_y = np.arange(y_lo, y_hi + grid, grid)
    ix = np.digitize(pts[:, 0], bins_x) - 1
    iy = np.digitize(pts[:, 1], bins_y) - 1
    counts = np.zeros((len(bins_x) - 1, len(bins_y) - 1), dtype=np.int64)
    for a, b in zip(ix, iy):
        if 0 <= a < counts.shape[0] and 0 <= b < counts.shape[1]:
            counts[a, b] += 1

    total = max(1, counts.sum())
    print(f"\nobject-height density (# >0.5%, @ >1%, █ >2%, grid={grid}m):")
    for j in range(counts.shape[1] - 1, -1, -1):
        row = ""
        for i in range(counts.shape[0]):
            frac = counts[i, j] / total
            if frac > 0.02:
                row += "█"
            elif frac > 0.01:
                row += "@"
            elif frac > 0.005:
                row += "#"
            elif frac > 0.002:
                row += "."
            else:
                row += " "
        print(f"y={bins_y[j]:+5.2f} |{row}|")
    print(f"        x={bins_x[0]:+5.2f} ... x={bins_x[-1]:+5.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--grid", type=float, default=0.3)
    args = p.parse_args()

    xyz, nrm, rgb = _load_replica_ply(args.mesh)
    print(f"mesh: {args.mesh.name} — {len(xyz)} verts")
    for ax, name in enumerate("xyz"):
        lo, hi = xyz[:, ax].min(), xyz[:, ax].max()
        print(f"  {name}: [{lo:+.2f}, {hi:+.2f}]  extent={hi-lo:.2f}")

    horiz = np.abs(nrm[:, 2]) > 0.9
    hist, edges = np.histogram(xyz[horiz, 2], bins=50)
    peaks = sorted(enumerate(hist), key=lambda x: x[1], reverse=True)[:4]
    print("\nhorizontal-surface z peaks (floor/ceiling candidates):")
    for i, h in peaks:
        print(f"  z=[{edges[i]:+.2f}, {edges[i+1]:+.2f}]: {h} verts")
    top2 = sorted([(edges[i], edges[i + 1], h) for i, h in peaks[:2]])
    floor_z = (top2[0][0] + top2[0][1]) / 2
    ceil_z = (top2[1][0] + top2[1][1]) / 2
    print(f"\n-> floor ≈ z={floor_z:+.2f}, ceiling ≈ z={ceil_z:+.2f}, height={abs(ceil_z-floor_z):.2f}")

    obj_mask = (
        ~horiz
        & (xyz[:, 2] > min(floor_z, ceil_z) + 0.1)
        & (xyz[:, 2] < max(floor_z, ceil_z) - 0.1)
    )
    interior = (
        (xyz[:, 0] > xyz[:, 0].min() + 0.2)
        & (xyz[:, 0] < xyz[:, 0].max() - 0.2)
        & (xyz[:, 1] > xyz[:, 1].min() + 0.2)
        & (xyz[:, 1] < xyz[:, 1].max() - 0.2)
    )
    pts = xyz[obj_mask & interior]

    _print_density_map(
        pts,
        xyz[:, 0].min() + 0.2,
        xyz[:, 0].max() - 0.2,
        xyz[:, 1].min() + 0.2,
        xyz[:, 1].max() - 0.2,
        args.grid,
    )


if __name__ == "__main__":
    main()
