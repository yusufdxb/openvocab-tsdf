"""Mesh IO — write meshes as .ply without requiring Open3D at import time."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openvocab_tsdf.mapping.base import Mesh


def save_ply(mesh: Mesh, path: str | Path) -> None:
    """Write an ASCII-header binary-little-endian PLY. Supports optional vertex colors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    verts = mesh.vertices.astype(np.float32)
    tris = mesh.triangles.astype(np.int32)
    has_color = mesh.vertex_colors is not None

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header += [
        f"element face {len(tris)}",
        "property list uchar int vertex_indices",
        "end_header",
        "",
    ]
    header_bytes = ("\n".join(header)).encode("ascii")

    with path.open("wb") as f:
        f.write(header_bytes)
        if has_color:
            colors = mesh.vertex_colors.astype(np.uint8)
            vx = np.zeros(
                len(verts),
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("r", "u1"),
                    ("g", "u1"),
                    ("b", "u1"),
                ],
            )
            vx["x"] = verts[:, 0]
            vx["y"] = verts[:, 1]
            vx["z"] = verts[:, 2]
            vx["r"] = colors[:, 0]
            vx["g"] = colors[:, 1]
            vx["b"] = colors[:, 2]
            f.write(vx.tobytes())
        else:
            f.write(verts.astype("<f4").tobytes())
        faces = np.empty(len(tris), dtype=[("n", "u1"), ("i", "<i4", 3)])
        faces["n"] = 3
        faces["i"] = tris
        f.write(faces.tobytes())
