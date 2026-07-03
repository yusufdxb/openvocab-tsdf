"""Fuse an extracted GO2 RGB-D+pose folder into a colored TSDF mesh."""
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

from openvocab_tsdf.data.base import RGBDFrame, CameraIntrinsics
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.viz.mesh import save_ply

FRAMES = Path(sys.argv[1])
OUT_PLY = Path(sys.argv[2])
VOXEL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04

fx, fy, cx, cy, W, H = (Path(FRAMES / "intrinsics.txt").read_text().split())
intr = CameraIntrinsics(float(fx), float(fy), float(cx), float(cy), int(W), int(H))
traj = np.loadtxt(FRAMES / "traj.txt").reshape(-1, 4, 4).astype(np.float32)
n = len(traj)

cfg = ReferenceTSDFConfig(
    voxel_size_m=VOXEL,
    truncation_distance_m=VOXEL * 3,
    bounds_min=(-4.0, -4.0, -0.5),
    bounds_max=(4.0, 4.0, 2.5),
    max_weight=100.0,
    store_color=True,
    store_features=False,
    feature_dim=0,
    near_surface_band=0.5,
    device="cuda",
)
vol = ReferenceTSDF(cfg)

t0 = time.perf_counter()
used = 0
for i in range(n):
    color = np.asarray(Image.open(FRAMES / "rgb" / f"{i:06d}.png").convert("RGB"), np.uint8)
    depth = np.load(FRAMES / "depth" / f"{i:06d}.npy").astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 8.0] = 0.0  # drop absurd far returns beyond room scale
    frame = RGBDFrame(color=color, depth_m=depth, intrinsics=intr,
                      T_wc=traj[i], frame_id=i)
    vol.integrate(frame)
    used += 1
import torch
torch.cuda.synchronize()
fuse_s = time.perf_counter() - t0

mesh = vol.extract_mesh(min_weight=1.0)
save_ply(mesh, OUT_PLY)
V = np.asarray(mesh.vertices)
print(f"fused {used} frames in {fuse_s:.1f}s ({used/fuse_s:.0f} FPS)")
print(f"mesh: {len(mesh.vertices)} verts / {len(mesh.triangles)} tris -> {OUT_PLY}")
if len(V):
    print(f"vertex extent min {V.min(0).round(2)} max {V.max(0).round(2)} "
          f"span {(V.max(0)-V.min(0)).round(2)}")
