"""Triton-based dense TSDF backend.

Same storage layout and public interface as `ReferenceTSDF`. Uses a single
1D Triton kernel that assigns one voxel per thread so no atomics are needed
and all writes are coalesced.

Why Triton and not hand-written CUDA yet: the system nvcc on mewtwo is 11.5
and cannot target the Blackwell sm_120 compute capability. Triton 3.6 ships
with PyTorch 2.11 and supports sm_120 via the bundled build, so we get a real
GPU kernel today with no toolchain install. See docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.base import Mesh, VoxelQueryResult


@dataclass
class TritonTSDFConfig:
    voxel_size_m: float = 0.02
    truncation_distance_m: float = 0.10
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_weight: float = 32.0
    store_color: bool = True
    device: str = "cuda:0"


@triton.jit
def _tsdf_integrate_kernel(
    tsdf_ptr,
    weight_ptr,
    color_ptr,
    depth_ptr,
    color_img_ptr,
    T_cw_ptr,
    K_ptr,
    origin_ptr,
    Nx,
    Ny,
    Nz,
    H,
    W,
    voxel_size,
    trunc,
    max_weight,
    store_color: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = Nx * Ny * Nz
    in_bounds = offs < total

    # decode (i, j, k)
    k = offs % Nz
    tmp = offs // Nz
    j = tmp % Ny
    i = tmp // Ny

    # voxel world coords (voxel center = origin + (idx + 0.5) * voxel_size)
    ox = tl.load(origin_ptr + 0)
    oy = tl.load(origin_ptr + 1)
    oz = tl.load(origin_ptr + 2)
    wx = ox + (i.to(tl.float32) + 0.5) * voxel_size
    wy = oy + (j.to(tl.float32) + 0.5) * voxel_size
    wz = oz + (k.to(tl.float32) + 0.5) * voxel_size

    # T_cw @ [w, 1]
    r00 = tl.load(T_cw_ptr + 0)
    r01 = tl.load(T_cw_ptr + 1)
    r02 = tl.load(T_cw_ptr + 2)
    t0 = tl.load(T_cw_ptr + 3)
    r10 = tl.load(T_cw_ptr + 4)
    r11 = tl.load(T_cw_ptr + 5)
    r12 = tl.load(T_cw_ptr + 6)
    t1 = tl.load(T_cw_ptr + 7)
    r20 = tl.load(T_cw_ptr + 8)
    r21 = tl.load(T_cw_ptr + 9)
    r22 = tl.load(T_cw_ptr + 10)
    t2 = tl.load(T_cw_ptr + 11)

    cx_w = r00 * wx + r01 * wy + r02 * wz + t0
    cy_w = r10 * wx + r11 * wy + r12 * wz + t1
    cz_w = r20 * wx + r21 * wy + r22 * wz + t2

    # intrinsics
    fx = tl.load(K_ptr + 0)
    cx_k = tl.load(K_ptr + 2)
    fy = tl.load(K_ptr + 4)
    cy_k = tl.load(K_ptr + 5)

    inv_z = 1.0 / (cz_w + 1e-8)
    u = fx * cx_w * inv_z + cx_k
    v = fy * cy_w * inv_z + cy_k
    # u, v can be negative when the voxel projects off-image; bias by +0.5 for
    # positive pixels and -0.5 for negative ones before truncation.
    u_i = tl.where(u >= 0, (u + 0.5).to(tl.int32), (u - 0.5).to(tl.int32))
    v_i = tl.where(v >= 0, (v + 0.5).to(tl.int32), (v - 0.5).to(tl.int32))

    in_img = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    pos_z = cz_w > 0
    projected = in_bounds & in_img & pos_z

    u_s = tl.where(projected, u_i, 0)
    v_s = tl.where(projected, v_i, 0)
    depth_sample = tl.load(depth_ptr + v_s * W + u_s, mask=projected, other=0.0)
    good_depth = depth_sample > 0

    sdf = depth_sample - cz_w
    in_truncation = sdf >= -trunc
    valid = projected & good_depth & in_truncation

    # normalize and clamp
    tsdf_new = tl.minimum(tl.maximum(sdf / trunc, -1.0), 1.0)

    w_old = tl.load(weight_ptr + offs, mask=in_bounds, other=0.0)
    t_old = tl.load(tsdf_ptr + offs, mask=in_bounds, other=1.0)

    w_sum = w_old + 1.0
    w_new = tl.minimum(w_sum, max_weight)
    t_new = (t_old * w_old + tsdf_new) / w_sum

    tl.store(tsdf_ptr + offs, tl.where(valid, t_new, t_old), mask=in_bounds)
    tl.store(weight_ptr + offs, tl.where(valid, w_new, w_old), mask=in_bounds)

    if store_color:
        c0_old = tl.load(color_ptr + offs * 3 + 0, mask=in_bounds, other=0.0)
        c1_old = tl.load(color_ptr + offs * 3 + 1, mask=in_bounds, other=0.0)
        c2_old = tl.load(color_ptr + offs * 3 + 2, mask=in_bounds, other=0.0)
        s0 = tl.load(color_img_ptr + v_s * W * 3 + u_s * 3 + 0, mask=valid, other=0.0)
        s1 = tl.load(color_img_ptr + v_s * W * 3 + u_s * 3 + 1, mask=valid, other=0.0)
        s2 = tl.load(color_img_ptr + v_s * W * 3 + u_s * 3 + 2, mask=valid, other=0.0)
        c0_new = (c0_old * w_old + s0) / w_sum
        c1_new = (c1_old * w_old + s1) / w_sum
        c2_new = (c2_old * w_old + s2) / w_sum
        tl.store(color_ptr + offs * 3 + 0, tl.where(valid, c0_new, c0_old), mask=in_bounds)
        tl.store(color_ptr + offs * 3 + 1, tl.where(valid, c1_new, c1_old), mask=in_bounds)
        tl.store(color_ptr + offs * 3 + 2, tl.where(valid, c2_new, c2_old), mask=in_bounds)


class TritonTSDF:
    """Dense Triton-based TSDF backend. Public interface matches `ReferenceTSDF`."""

    def __init__(self, cfg: TritonTSDFConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("TritonTSDF requires CUDA")
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        bmin = torch.tensor(cfg.bounds_min, dtype=torch.float32, device=self.device)
        bmax = torch.tensor(cfg.bounds_max, dtype=torch.float32, device=self.device)
        self.origin = bmin.contiguous()
        dims = ((bmax - bmin) / cfg.voxel_size_m).ceil().to(torch.int64)
        self.dims = tuple(int(d) for d in dims.tolist())
        Nx, Ny, Nz = self.dims

        self.tsdf = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        self.weight = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        if cfg.store_color:
            self.color = torch.zeros((Nx, Ny, Nz, 3), dtype=torch.float32, device=self.device)
        else:
            self.color = None

    @property
    def voxel_size_m(self) -> float:
        return self.cfg.voxel_size_m

    @property
    def truncation_distance_m(self) -> float:
        return self.cfg.truncation_distance_m

    def reset(self) -> None:
        self.tsdf.fill_(1.0)
        self.weight.zero_()
        if self.color is not None:
            self.color.zero_()

    @torch.no_grad()
    def integrate(self, frame: RGBDFrame) -> None:
        T_wc = torch.as_tensor(frame.T_wc, dtype=torch.float32, device=self.device)
        T_cw = torch.linalg.inv(T_wc).contiguous()
        K = torch.as_tensor(
            frame.intrinsics.K, dtype=torch.float32, device=self.device
        ).contiguous()
        depth = torch.as_tensor(frame.depth_m, dtype=torch.float32, device=self.device).contiguous()
        H, W = depth.shape

        if self.color is not None:
            color_img = torch.as_tensor(
                frame.color, dtype=torch.float32, device=self.device
            ).contiguous()
        else:
            color_img = torch.empty(0, dtype=torch.float32, device=self.device)

        Nx, Ny, Nz = self.dims
        total = Nx * Ny * Nz
        BLOCK = 1024
        grid = ((total + BLOCK - 1) // BLOCK,)

        _tsdf_integrate_kernel[grid](
            self.tsdf,
            self.weight,
            self.color if self.color is not None else torch.empty(0, device=self.device),
            depth,
            color_img,
            T_cw.reshape(-1),
            K.reshape(-1),
            self.origin,
            Nx,
            Ny,
            Nz,
            H,
            W,
            self.cfg.voxel_size_m,
            self.cfg.truncation_distance_m,
            self.cfg.max_weight,
            store_color=self.color is not None,
            BLOCK_SIZE=BLOCK,
        )

    @torch.no_grad()
    def extract_mesh(self, min_weight: float = 1.0) -> Mesh:
        from skimage.measure import marching_cubes

        tsdf_np = self.tsdf.detach().cpu().numpy()
        mask = (self.weight.detach().cpu().numpy() >= min_weight).astype(np.float32)
        tsdf_m = np.where(mask > 0, tsdf_np, 1.0)
        try:
            verts, faces, _, _ = marching_cubes(
                tsdf_m, level=0.0, spacing=(self.cfg.voxel_size_m,) * 3
            )
        except (ValueError, RuntimeError):
            return Mesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0, 3), dtype=np.int32),
                vertex_colors=None,
            )
        origin_np = self.origin.detach().cpu().numpy()
        verts = verts.astype(np.float32) + origin_np + 0.5 * self.cfg.voxel_size_m

        vcolors: np.ndarray | None = None
        if self.color is not None:
            vox_idx = ((verts - origin_np) / self.cfg.voxel_size_m).astype(np.int64)
            vox_idx = np.clip(vox_idx, [0, 0, 0], [d - 1 for d in self.dims])
            col = self.color.detach().cpu().numpy()
            vcolors = col[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].clip(0, 255).astype(np.uint8)

        return Mesh(vertices=verts, triangles=faces.astype(np.int32), vertex_colors=vcolors)

    @torch.no_grad()
    def query(self, points_w: torch.Tensor) -> VoxelQueryResult:
        """Trilinear query — same as reference; kernel-level fast query is Phase 3."""
        points_w = points_w.to(self.device, dtype=torch.float32)
        idx_f = (points_w - self.origin) / self.cfg.voxel_size_m - 0.5
        idx0 = idx_f.floor().long()
        frac = (idx_f - idx0.to(torch.float32)).clamp(0.0, 1.0)
        Nx, Ny, Nz = self.dims

        def _gather(
            tensor: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor
        ) -> torch.Tensor:
            i = i.clamp(0, Nx - 1)
            j = j.clamp(0, Ny - 1)
            k = k.clamp(0, Nz - 1)
            return tensor[i, j, k]

        i0, j0, k0 = idx0.unbind(-1)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        fx, fy, fz = frac.unbind(-1)

        def _trilin(tensor: torch.Tensor) -> torch.Tensor:
            w000 = (1 - fx) * (1 - fy) * (1 - fz)
            w001 = (1 - fx) * (1 - fy) * fz
            w010 = (1 - fx) * fy * (1 - fz)
            w011 = (1 - fx) * fy * fz
            w100 = fx * (1 - fy) * (1 - fz)
            w101 = fx * (1 - fy) * fz
            w110 = fx * fy * (1 - fz)
            w111 = fx * fy * fz

            def _scale(c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                if c.ndim == w.ndim:
                    return c * w
                return c * w.unsqueeze(-1)

            return (
                _scale(_gather(tensor, i0, j0, k0), w000)
                + _scale(_gather(tensor, i0, j0, k1), w001)
                + _scale(_gather(tensor, i0, j1, k0), w010)
                + _scale(_gather(tensor, i0, j1, k1), w011)
                + _scale(_gather(tensor, i1, j0, k0), w100)
                + _scale(_gather(tensor, i1, j0, k1), w101)
                + _scale(_gather(tensor, i1, j1, k0), w110)
                + _scale(_gather(tensor, i1, j1, k1), w111)
            )

        tsdf = _trilin(self.tsdf)
        weight = _trilin(self.weight)
        if self.color is not None:
            color = _trilin(self.color).clamp(0, 255).to(torch.uint8)
        else:
            color = torch.zeros((points_w.shape[0], 3), dtype=torch.uint8, device=self.device)
        return VoxelQueryResult(tsdf=tsdf, weight=weight, color=color, feat=None)
