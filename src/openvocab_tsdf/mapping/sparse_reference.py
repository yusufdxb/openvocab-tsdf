"""Sparse-feature reference TSDF.

Same correctness guarantees as `ReferenceTSDF` for geometry (tsdf, weight,
color stay dense — they are ~10 B per voxel, not the bottleneck). Features,
which are by far the dominant cost at 512-dim fp32 (2 KB per voxel), are
stored in a **sparse pool** allocated lazily on first voxel observation.

Layout
------
  - dense `voxel_slot[Nx, Ny, Nz]` int32: −1 = unobserved, else slot into
    the pool.
  - flat `feat_pool[capacity, D]` fp32: the actual features. Capacity is a
    ceiling; we grow by doubling when we hit it, up to `max_capacity`.
  - `num_allocated`: int, the high-water mark into the pool.

Expected savings on a typical indoor scene: observed surface voxels are
usually 5–15 % of the bounding volume, so feature-memory drop is ~6–20×
vs the dense backend. Geometry is unchanged.

This sparse backend satisfies the same `TSDFVolume` protocol as the dense
reference so the downstream query and mesh extraction paths are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.base import Mesh, VoxelQueryResult

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sparse_feat_update_kernel(
        pool_ptr,  # (CAPACITY, D) fp32
        slot_ptr,  # (N,) int32
        w_old_ptr,  # (N,) fp32
        feat_sample_ptr,  # (D,) fp32 — shared across all N
        N,
        D,
        BLOCK_D: tl.constexpr,
    ):
        voxel_id = tl.program_id(0)
        d_block = tl.program_id(1)
        # each program handles one (voxel, d-chunk)
        if voxel_id >= N:
            return
        slot = tl.load(slot_ptr + voxel_id)
        w = tl.load(w_old_ptr + voxel_id)
        d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        base = slot.to(tl.int64) * D
        f_old = tl.load(pool_ptr + base + d_offs, mask=d_mask, other=0.0)
        f_samp = tl.load(feat_sample_ptr + d_offs, mask=d_mask, other=0.0)
        f_new = (f_old * w + f_samp) / (w + 1.0)
        tl.store(pool_ptr + base + d_offs, f_new, mask=d_mask)

    def _triton_sparse_feat_update(
        pool: torch.Tensor,  # (CAPACITY, D) fp32, contiguous, cuda
        slot_ids: torch.Tensor,  # (N,) int32, cuda
        w_old: torch.Tensor,  # (N,) fp32, cuda
        feat_sample: torch.Tensor,  # (D,) fp32, cuda
    ) -> None:
        assert pool.is_cuda and pool.is_contiguous()
        N = int(slot_ids.numel())
        D = int(pool.shape[-1])
        if N == 0:
            return
        BLOCK_D = 64 if D >= 64 else D
        grid = (N, (D + BLOCK_D - 1) // BLOCK_D)
        _sparse_feat_update_kernel[grid](
            pool,
            slot_ids.contiguous(),
            w_old.contiguous(),
            feat_sample.contiguous(),
            N,
            D,
            BLOCK_D=BLOCK_D,
        )

else:

    def _triton_sparse_feat_update(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton not available; use feat_update_backend='pytorch'")


@dataclass
class SparseFeatureTSDFConfig:
    voxel_size_m: float = 0.02
    truncation_distance_m: float = 0.10
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_weight: float = 32.0
    store_color: bool = True
    feature_dim: int = 512
    initial_feat_capacity: int = 65_536  # ~128 MB at 512 fp32
    max_feat_capacity: int = 4_000_000  # cap the pool even in pathological scenes
    # Normalized-TSDF gate for feature accumulation. See `ReferenceTSDFConfig.
    # near_surface_band` for the full rationale; same default and meaning here.
    near_surface_band: float = 0.5
    device: str = "cuda:0"
    # Kernel backend for the feature-pool update. 'pytorch' uses
    # `index_copy_` + vectorized math (always available). 'triton' fuses
    # the gather-compute-scatter into one JIT kernel and wins a bit of
    # bandwidth back on larger voxel counts.
    feat_update_backend: str = "pytorch"  # or "triton"


class SparseFeatureTSDF:
    """Dense geometry + sparse per-voxel features."""

    def __init__(self, cfg: SparseFeatureTSDFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        bmin = torch.tensor(cfg.bounds_min, dtype=torch.float32, device=self.device)
        bmax = torch.tensor(cfg.bounds_max, dtype=torch.float32, device=self.device)
        self.origin = bmin
        dims = ((bmax - bmin) / cfg.voxel_size_m).ceil().to(torch.int64)
        self.dims = tuple(int(d) for d in dims.tolist())
        Nx, Ny, Nz = self.dims

        # Dense geometry — matches reference backend exactly
        self.tsdf = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        self.weight = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        self.color = (
            torch.zeros((Nx, Ny, Nz, 3), dtype=torch.float32, device=self.device)
            if cfg.store_color
            else None
        )

        # Sparse feature pool
        self._voxel_slot = torch.full((Nx, Ny, Nz), -1, dtype=torch.int32, device=self.device)
        self._feat_capacity = int(cfg.initial_feat_capacity)
        self._feat_pool = torch.zeros(
            (self._feat_capacity, cfg.feature_dim), dtype=torch.float32, device=self.device
        )
        self._num_allocated = 0

        # Cached voxel-center world coords (reused each integrate)
        ix = torch.arange(Nx, device=self.device, dtype=torch.float32)
        iy = torch.arange(Ny, device=self.device, dtype=torch.float32)
        iz = torch.arange(Nz, device=self.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
        centers = torch.stack([gx, gy, gz], dim=-1) + 0.5
        self._voxel_world = self.origin + centers * cfg.voxel_size_m

    # --------------------------------------------------------------- public
    @property
    def voxel_size_m(self) -> float:
        return self.cfg.voxel_size_m

    @property
    def truncation_distance_m(self) -> float:
        return self.cfg.truncation_distance_m

    @property
    def feat(self) -> torch.Tensor:
        """Dense feature view (Nx,Ny,Nz,D). Allocates lazily; rebuilt each call.

        Query / eval code that needs a dense tensor can use this. For the hot
        path, prefer `query()` or the sparse accessors.
        """
        Nx, Ny, Nz = self.dims
        D = self.cfg.feature_dim
        dense = torch.zeros((Nx * Ny * Nz, D), dtype=torch.float32, device=self.device)
        slot = self._voxel_slot.view(-1)
        observed = slot >= 0
        if observed.any():
            dense[observed] = self._feat_pool[slot[observed].long()]
        return dense.view(Nx, Ny, Nz, D)

    @property
    def num_allocated_feat_voxels(self) -> int:
        return int(self._num_allocated)

    def feat_memory_bytes(self) -> int:
        return int(self._num_allocated * self.cfg.feature_dim * 4)

    def reset(self) -> None:
        self.tsdf.fill_(1.0)
        self.weight.zero_()
        if self.color is not None:
            self.color.zero_()
        self._voxel_slot.fill_(-1)
        self._num_allocated = 0
        self._feat_pool.zero_()

    def _ensure_capacity(self, need: int) -> None:
        if need <= self._feat_capacity:
            return
        new_cap = self._feat_capacity
        while new_cap < need:
            new_cap *= 2
        new_cap = min(new_cap, self.cfg.max_feat_capacity)
        if new_cap < need:
            raise RuntimeError(
                f"sparse feat pool capacity exhausted: need {need}, cap={self.cfg.max_feat_capacity}"
            )
        new_pool = torch.zeros(
            (new_cap, self.cfg.feature_dim), dtype=torch.float32, device=self.device
        )
        new_pool[: self._num_allocated] = self._feat_pool[: self._num_allocated]
        self._feat_pool = new_pool
        self._feat_capacity = new_cap

    @torch.no_grad()
    def integrate(self, frame: RGBDFrame, feature: torch.Tensor | None = None) -> None:
        """Same math as `ReferenceTSDF.integrate`; sparse-feature path for the
        feature write. Only `feature` (global per-frame) is supported in this
        class for v1 — patch-mode sparse aggregation is a natural extension."""
        cfg = self.cfg

        T_wc = torch.as_tensor(frame.T_wc, dtype=torch.float32, device=self.device)
        T_cw = torch.linalg.inv(T_wc)
        K = torch.as_tensor(frame.intrinsics.K, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(frame.depth_m, dtype=torch.float32, device=self.device)
        H, W = depth.shape
        color_img = (
            torch.as_tensor(frame.color, dtype=torch.float32, device=self.device)
            if self.color is not None
            else None
        )

        pts_w = self._voxel_world.reshape(-1, 3)
        pts_c = (T_cw[:3, :3] @ pts_w.T).T + T_cw[:3, 3]
        z = pts_c[:, 2]
        valid_z = z > 0
        u = (K[0, 0] * pts_c[:, 0] / (z + 1e-8) + K[0, 2]).round().long()
        v = (K[1, 1] * pts_c[:, 1] / (z + 1e-8) + K[1, 2]).round().long()
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_z & in_img

        u_s = u.clamp(0, W - 1)
        v_s = v.clamp(0, H - 1)
        d_sample = depth[v_s, u_s]
        valid = valid & (d_sample > 0)

        sdf = d_sample - z
        valid = valid & (sdf >= -cfg.truncation_distance_m)
        tsdf_new = (sdf / cfg.truncation_distance_m).clamp(-1.0, 1.0)

        idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return

        w_old = self.weight.view(-1)[idx]
        t_old = self.tsdf.view(-1)[idx]
        t_sample = tsdf_new[idx]
        w_new = (w_old + 1.0).clamp(max=cfg.max_weight)
        t_merged = (t_old * w_old + t_sample) / (w_old + 1.0)
        self.tsdf.view(-1).index_copy_(0, idx, t_merged)
        self.weight.view(-1).index_copy_(0, idx, w_new)

        if self.color is not None and color_img is not None:
            u_valid = u_s[idx]
            v_valid = v_s[idx]
            c_sample = color_img[v_valid, u_valid]
            c_old = self.color.view(-1, 3)[idx]
            c_merged = (c_old * w_old.unsqueeze(-1) + c_sample) / (w_old + 1.0).unsqueeze(-1)
            self.color.view(-1, 3).index_copy_(0, idx, c_merged)

        if feature is None:
            return
        if feature.shape != (cfg.feature_dim,):
            raise ValueError(f"feature must be ({cfg.feature_dim},), got {tuple(feature.shape)}")

        # near-surface gate — same rationale as dense reference. `tsdf_new` is
        # clamped to [-1, 1], so a band of 1.0 would never exclude anything;
        # the default 0.5 keeps features off free-space voxels in front of the
        # surface.
        near = tsdf_new.abs() <= cfg.near_surface_band
        feat_mask = near[idx]
        sel = feat_mask.nonzero(as_tuple=False).squeeze(-1)
        if sel.numel() == 0:
            return
        idx_feat = idx[sel]
        w_sel = w_old[sel]

        slot_flat = self._voxel_slot.view(-1)
        old_slots = slot_flat[idx_feat]  # int32
        unalloc = old_slots < 0

        # Allocate slots for first-time-seen voxels
        n_new = int(unalloc.sum().item())
        if n_new > 0:
            self._ensure_capacity(self._num_allocated + n_new)
            new_slot_ids = torch.arange(
                self._num_allocated,
                self._num_allocated + n_new,
                device=self.device,
                dtype=torch.int32,
            )
            slot_flat.index_copy_(0, idx_feat[unalloc], new_slot_ids)
            self._num_allocated += n_new
            # freshly-allocated slots start at zero, so mean against (w_old=0) is fine
            old_slots = slot_flat[idx_feat]  # refresh

        # Aggregate
        slot_long = old_slots.long()
        feat_sample = feature.to(self._feat_pool.dtype).to(self.device)
        if self.cfg.feat_update_backend == "triton" and self._feat_pool.is_cuda:
            _triton_sparse_feat_update(
                self._feat_pool, slot_long.to(torch.int32), w_sel, feat_sample
            )
        else:
            f_old = self._feat_pool[slot_long]
            f_merged = (f_old * w_sel.unsqueeze(-1) + feat_sample) / (w_sel + 1.0).unsqueeze(-1)
            self._feat_pool.index_copy_(0, slot_long, f_merged)

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
        vcolors = None
        if self.color is not None:
            vox_idx = ((verts - origin_np) / self.cfg.voxel_size_m).astype(np.int64)
            vox_idx = np.clip(vox_idx, [0, 0, 0], [d - 1 for d in self.dims])
            col = self.color.detach().cpu().numpy()
            vcolors = col[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].clip(0, 255).astype(np.uint8)
        return Mesh(vertices=verts, triangles=faces.astype(np.int32), vertex_colors=vcolors)

    @torch.no_grad()
    def query(self, points_w: torch.Tensor) -> VoxelQueryResult:
        """Nearest-voxel query (trilinear interp not defined across holes)."""
        points_w = points_w.to(self.device, dtype=torch.float32)
        idx_f = (points_w - self.origin) / self.cfg.voxel_size_m
        idx = idx_f.floor().long()
        Nx, Ny, Nz = self.dims
        i = idx[:, 0].clamp(0, Nx - 1)
        j = idx[:, 1].clamp(0, Ny - 1)
        k = idx[:, 2].clamp(0, Nz - 1)

        tsdf = self.tsdf[i, j, k]
        weight = self.weight[i, j, k]
        if self.color is not None:
            color = self.color[i, j, k].clamp(0, 255).to(torch.uint8)
        else:
            color = torch.zeros((points_w.shape[0], 3), dtype=torch.uint8, device=self.device)

        slots = self._voxel_slot[i, j, k]
        feat = torch.zeros(
            (points_w.shape[0], self.cfg.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        alloc = slots >= 0
        if alloc.any():
            feat[alloc] = self._feat_pool[slots[alloc].long()]
        return VoxelQueryResult(tsdf=tsdf, weight=weight, color=color, feat=feat)
