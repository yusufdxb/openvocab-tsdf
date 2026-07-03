"""Render a shaded PNG of a colored TSDF mesh with Open3D offscreen (headless).

Produces a single wide PNG with two lit views of the SAME mesh:
  - left:  vertex-color shading (true reconstructed color)
  - right: uniform-albedo shading (geometry-only, so the floor + wall read
           as a solid lit surface rather than washed-out color)

Runs headless via Filament's EGL surfaceless platform (no X display). If the
offscreen context cannot be created on this box, exits non-zero so the caller
falls back to another renderer.

    EGL_PLATFORM=surfaceless \\
    .venv/bin/python scripts/render_real_bag_mesh.py \\
        --mesh /path/to/go2_room_recon.ply \\
        --out figures/go2_real_bag_recon.png \\
        --title "GO2 real-bag reconstruction"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Filament needs a headless EGL platform when there is no X display.
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

rendering = o3d.visualization.rendering


def _shaded_view(
    mesh: o3d.geometry.TriangleMesh,
    width: int,
    height: int,
    use_vertex_color: bool,
) -> np.ndarray:
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    r.scene.scene.set_sun_light([-0.4, -0.6, -0.7], [1.0, 1.0, 1.0], 90000)
    r.scene.scene.enable_sun_light(True)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    if use_vertex_color:
        m = mesh
    else:
        m = o3d.geometry.TriangleMesh(mesh)
        m.paint_uniform_color([0.72, 0.74, 0.78])
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    r.scene.add_geometry("mesh", m, mat)

    # Frame the mesh: look down at an angle from the +x/-y corner so the floor
    # and the wall are both visible in one shot.
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent)) * 0.5
    eye = center + np.array([radius * 0.85, -radius * 1.0, radius * 0.7])
    r.scene.camera.look_at(center, eye, [0.0, 0.0, 1.0])

    img = r.render_to_image()
    arr = np.asarray(img).copy()
    del r
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default="")
    p.add_argument("--width", type=int, default=760)
    p.add_argument("--height", type=int, default=680)
    args = p.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_vertex_normals()
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"loaded mesh: {n_v} verts / {n_t} tris from {args.mesh}")

    left = _shaded_view(mesh, args.width, args.height, use_vertex_color=True)
    right = _shaded_view(mesh, args.width, args.height, use_vertex_color=False)
    print("rendered both views")

    gap = 8
    banner = 34 if args.title else 0
    canvas = np.full(
        (args.height + banner, args.width * 2 + gap, 3), 255, dtype=np.uint8
    )
    canvas[banner : banner + args.height, : args.width] = left[..., :3]
    canvas[banner : banner + args.height, args.width + gap :] = right[..., :3]

    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
        small = font
    if args.title:
        draw.text((10, 8), args.title, fill=(20, 20, 20), font=font)
    draw.text((12, banner + 8), "vertex color", fill=(30, 30, 30), font=small)
    draw.text(
        (args.width + gap + 12, banner + 8),
        "geometry (uniform albedo)",
        fill=(30, 30, 30),
        font=small,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"wrote {args.out} ({im.width}x{im.height})")


if __name__ == "__main__":
    main()
