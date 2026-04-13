"""Synthetic RGB-D scene generator for tests and smoke runs.

Produces deterministic RGB-D frames of known geometry so we can verify TSDF
fusion without shipping external datasets. Primitives: planes, boxes, spheres.

The camera convention matches the rest of the codebase:
- `T_wc` is camera-to-world.
- Camera looks down +Z in its own frame, +X right, +Y down (OpenCV).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openvocab_tsdf.data.base import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float
    color: tuple[int, int, int] = (200, 80, 80)


@dataclass(frozen=True)
class Box:
    """Axis-aligned box. `min_xyz` and `max_xyz` are world-frame corners."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    color: tuple[int, int, int] = (80, 180, 80)


Primitive = Sphere | Box


def make_ring_poses(
    num: int,
    radius: float = 1.5,
    height: float = 0.0,
    look_at: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """`num` camera-to-world poses on a ring around `look_at`, looking inward."""
    look_at_v = np.array(look_at, dtype=np.float32)
    poses = np.zeros((num, 4, 4), dtype=np.float32)
    for i in range(num):
        theta = 2 * np.pi * i / num
        cam_pos = np.array(
            [
                look_at_v[0] + radius * np.cos(theta),
                look_at_v[1] + height,
                look_at_v[2] + radius * np.sin(theta),
            ],
            dtype=np.float32,
        )
        # forward = look_at - cam_pos, normalized
        forward = look_at_v - cam_pos
        forward /= np.linalg.norm(forward) + 1e-8
        # world up is +Y down in OpenCV convention, so pick -Y as up
        up_world = np.array([0, -1, 0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=1)  # columns are x, y, z axes
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        poses[i] = T
    return poses


def render_depth(
    primitives: list[Primitive],
    intrinsics: CameraIntrinsics,
    T_wc: np.ndarray,
    z_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a depth + color image by ray-casting per pixel (slow but correct).

    Returns:
        depth_m  float32 H×W, 0.0 for misses
        color    uint8 H×W×3, black for misses
    """
    H, W = intrinsics.height, intrinsics.width
    K = intrinsics.K
    T_cw = np.linalg.inv(T_wc.astype(np.float32))
    cam_origin_w = T_wc[:3, 3].astype(np.float32)

    # build rays in camera frame
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_c = (uu - K[0, 2]) / K[0, 0]
    y_c = (vv - K[1, 2]) / K[1, 1]
    z_c = np.ones_like(x_c)
    dirs_c = np.stack([x_c, y_c, z_c], axis=-1)
    dirs_c = dirs_c / np.linalg.norm(dirs_c, axis=-1, keepdims=True)

    # rotate to world
    R_wc = T_wc[:3, :3].astype(np.float32)
    dirs_w = dirs_c @ R_wc.T  # [H,W,3]

    depth = np.zeros((H, W), dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.uint8)
    best_t = np.full((H, W), np.inf, dtype=np.float32)

    for prim in primitives:
        if isinstance(prim, Sphere):
            center = np.array(prim.center, dtype=np.float32)
            oc = cam_origin_w - center
            b = 2.0 * np.einsum("hwc,c->hw", dirs_w, oc)
            c = float(oc @ oc) - prim.radius**2
            disc = b * b - 4.0 * c
            hit = disc > 0
            t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0, np.inf)
            t = np.where((t > 0) & (t < z_max), t, np.inf)
            update = t < best_t
            best_t = np.where(update, t, best_t)
            col = np.array(prim.color, dtype=np.uint8)
            color = np.where(update[..., None], col, color)
        elif isinstance(prim, Box):
            bmin = np.array(prim.min_xyz, dtype=np.float32)
            bmax = np.array(prim.max_xyz, dtype=np.float32)
            inv_d = 1.0 / (dirs_w + 1e-8 * (dirs_w == 0))
            t0 = (bmin - cam_origin_w) * inv_d
            t1 = (bmax - cam_origin_w) * inv_d
            tmin = np.minimum(t0, t1).max(axis=-1)
            tmax = np.maximum(t0, t1).min(axis=-1)
            hit = (tmax > np.maximum(tmin, 0.0)) & (tmin < z_max)
            t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
            update = t < best_t
            best_t = np.where(update, t, best_t)
            col = np.array(prim.color, dtype=np.uint8)
            color = np.where(update[..., None], col, color)
        else:
            raise TypeError(f"unknown primitive: {type(prim)}")

    # Convert ray distance to camera-z depth (OpenCV depth convention).
    # depth = t * cos(angle between ray and principal axis) = t * (dir . z_c_axis)
    # In camera frame, z_c axis is (0,0,1), ray dir (x,y,1)/norm -> z = 1/norm.
    ray_z_cos = 1.0 / np.linalg.norm(np.stack([x_c, y_c, z_c], axis=-1), axis=-1)
    depth = np.where(np.isfinite(best_t), best_t * ray_z_cos, 0.0).astype(np.float32)

    return depth, color


def make_synthetic_dataset(
    primitives: list[Primitive],
    num_frames: int = 16,
    width: int = 160,
    height: int = 120,
    fov_deg: float = 60.0,
    radius: float = 1.5,
    z_max: float = 6.0,
) -> list[RGBDFrame]:
    """Return a list of RGBDFrames of `primitives` from cameras on a ring."""
    fx = fy = (width / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = width / 2, height / 2
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    poses = make_ring_poses(num_frames, radius=radius)
    frames: list[RGBDFrame] = []
    for i, T_wc in enumerate(poses):
        depth, color = render_depth(primitives, intr, T_wc, z_max=z_max)
        frames.append(
            RGBDFrame(
                color=color,
                depth_m=depth,
                intrinsics=intr,
                T_wc=T_wc.astype(np.float32),
                timestamp=float(i) / 30.0,
                frame_id=i,
            )
        )
    return frames
