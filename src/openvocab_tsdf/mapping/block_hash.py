"""Block-indexed sparse TSDF.

Storage layout:
  - scene is tiled into 8×8×8-voxel *blocks*
  - `block_slot[Nbx, Nby, Nbz]` int32 dense lookup — `-1` = unallocated, else
    the row index into the block pool. This costs 4 bytes per block
    regardless of observation, so the lookup table is always tiny
    (1 block : 512 voxels = 0.2 % of dense storage).
  - `tsdf_pool[CAPACITY, 512]`, `weight_pool[CAPACITY, 512]`,
    `color_pool[CAPACITY, 512, 3]` — flat arrays; rows allocated lazily
    on first-touch of each block. Capacity grows by doubling up to
    `max_capacity`.

Integrate is a 2-pass PyTorch implementation:
  1. Project all voxel centers through the camera, find which ones fall in
     the truncation band for this frame.
  2. Find the unique blocks among them, allocate any not yet seen, then
     scatter the per-voxel TSDF/weight/color updates into the block pool.

This is the minimum viable sparse-geometry design; swapping the Python
allocation loop for a Triton `atomic_cas`-based hash is a future extension
(see decisions.md). For the immediate goal — scenes > 100 m³ where dense
TSDF OOMs — this already delivers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.base import Mesh, VoxelQueryResult

BLOCK = 8
BLOCK3 = BLOCK * BLOCK * BLOCK


@dataclass
class BlockHashTSDFConfig:
    voxel_size_m: float = 0.04
    truncation_distance_m: float = 0.16
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_weight: float = 32.0
    store_color: bool = True
    initial_block_capacity: int = 4096  # 4096 * 512 voxels = 2M voxels
    max_block_capacity: int = 131_072  # ≈67M voxels ceiling
    device: str = "cuda:0"


class BlockHashTSDF:
    """Sparse-geometry TSDF with block-indexed lazy allocation.

    Public interface matches `ReferenceTSDF`: `integrate`, `extract_mesh`,
    `query`. Features are NOT stored here (would add 2 KB per voxel at
    D=512); pair with `SparseFeatureTSDF` for an open-vocab run.
    """

    def __init__(self, cfg: BlockHashTSDFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        bmin = torch.tensor(cfg.bounds_min, dtype=torch.float32, device=self.device)
        bmax = torch.tensor(cfg.bounds_max, dtype=torch.float32, device=self.device)
        self.origin = bmin
        voxel_dims_t = ((bmax - bmin) / cfg.voxel_size_m).ceil().to(torch.int64)
        # round voxel dims UP to a multiple of BLOCK
        voxel_dims_t = ((voxel_dims_t + BLOCK - 1) // BLOCK) * BLOCK
        self.dims = tuple(int(d) for d in voxel_dims_t.tolist())  # (Nx, Ny, Nz)
        self.block_dims = tuple(d // BLOCK for d in self.dims)  # (Nbx, Nby, Nbz)
        Nbx, Nby, Nbz = self.block_dims

        # Block lookup (dense int32)
        self._block_slot = torch.full((Nbx, Nby, Nbz), -1, dtype=torch.int32, device=self.device)

        # Block pools (lazy-allocated rows)
        self._capacity = int(cfg.initial_block_capacity)
        self._tsdf_pool = torch.ones(
            (self._capacity, BLOCK3), dtype=torch.float32, device=self.device
        )
        self._weight_pool = torch.zeros(
            (self._capacity, BLOCK3), dtype=torch.float32, device=self.device
        )
        if cfg.store_color:
            self._color_pool = torch.zeros(
                (self._capacity, BLOCK3, 3), dtype=torch.float32, device=self.device
            )
        else:
            self._color_pool = None
        self._num_blocks = 0

        # Cached BLOCK-center world coords (cheap; 1/512 of full-voxel storage).
        # Used for frustum culling at integrate time.
        ibx = torch.arange(Nbx, device=self.device, dtype=torch.float32)
        iby = torch.arange(Nby, device=self.device, dtype=torch.float32)
        ibz = torch.arange(Nbz, device=self.device, dtype=torch.float32)
        gbx, gby, gbz = torch.meshgrid(ibx, iby, ibz, indexing="ij")
        block_center_idx = torch.stack([gbx, gby, gbz], dim=-1)  # (Nbx, Nby, Nbz, 3)
        self._block_world_center = (
            self.origin + (block_center_idx + 0.5) * BLOCK * cfg.voxel_size_m
        )  # (Nbx, Nby, Nbz, 3)
        # block half-diagonal for frustum inclusion test
        self._block_half_diag = (0.5 * BLOCK * cfg.voxel_size_m) * (3**0.5)

        # Cached in-block local voxel center offsets (relative to block origin).
        lx = torch.arange(BLOCK, device=self.device, dtype=torch.float32)
        ly = torch.arange(BLOCK, device=self.device, dtype=torch.float32)
        lz = torch.arange(BLOCK, device=self.device, dtype=torch.float32)
        glx, gly, glz = torch.meshgrid(lx, ly, lz, indexing="ij")
        local_off = torch.stack([glx, gly, glz], dim=-1) + 0.5
        self._local_voxel_offset = (local_off * cfg.voxel_size_m).reshape(BLOCK3, 3)
        # also cache the local-flat index for each local (i, j, k)
        lfx, lfy, lfz = torch.meshgrid(lx, ly, lz, indexing="ij")
        self._local_flat_grid = (
            (lfx * BLOCK * BLOCK + lfy * BLOCK + lfz).to(torch.int64).reshape(BLOCK3)
        )

    # --------------------------------------------------------------- helpers
    @property
    def voxel_size_m(self) -> float:
        return self.cfg.voxel_size_m

    @property
    def truncation_distance_m(self) -> float:
        return self.cfg.truncation_distance_m

    @property
    def num_allocated_blocks(self) -> int:
        return int(self._num_blocks)

    def memory_bytes(self) -> int:
        n = self._num_blocks
        per_vox = 4 + 4 + (12 if self._color_pool is not None else 0)
        return int(n * BLOCK3 * per_vox)

    def _ensure_capacity(self, need: int) -> None:
        if need <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < need:
            new_cap *= 2
        new_cap = min(new_cap, self.cfg.max_block_capacity)
        if new_cap < need:
            raise RuntimeError(
                f"block pool capacity exhausted: need {need}, cap={self.cfg.max_block_capacity}"
            )
        n = self._num_blocks
        new_tsdf = torch.ones((new_cap, BLOCK3), dtype=torch.float32, device=self.device)
        new_tsdf[:n] = self._tsdf_pool[:n]
        self._tsdf_pool = new_tsdf
        new_w = torch.zeros((new_cap, BLOCK3), dtype=torch.float32, device=self.device)
        new_w[:n] = self._weight_pool[:n]
        self._weight_pool = new_w
        if self._color_pool is not None:
            new_c = torch.zeros((new_cap, BLOCK3, 3), dtype=torch.float32, device=self.device)
            new_c[:n] = self._color_pool[:n]
            self._color_pool = new_c
        self._capacity = new_cap

    def reset(self) -> None:
        self._block_slot.fill_(-1)
        self._num_blocks = 0
        self._tsdf_pool.fill_(1.0)
        self._weight_pool.zero_()
        if self._color_pool is not None:
            self._color_pool.zero_()

    # -------------------------------------------------------------- integrate
    @torch.no_grad()
    def integrate(self, frame: RGBDFrame) -> None:
        """Frustum-culled integrate.

        Stage 1: cheap per-BLOCK cull against the camera frustum (block
        centers transformed to camera space, conservative radius test on z
        and projected bounds). Skips the expensive 125 M-voxel projection
        that the naive path does at multi-meter scales.

        Stage 2: on the surviving blocks (typically 100–1000 out of millions
        for a typical indoor frame), expand their 8³ voxels and do the
        standard project/sdf/update. The per-voxel workload is now bounded
        by the frustum, not by the volume.
        """
        cfg = self.cfg
        T_wc = torch.as_tensor(frame.T_wc, dtype=torch.float32, device=self.device)
        T_cw = torch.linalg.inv(T_wc)
        K = torch.as_tensor(frame.intrinsics.K, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(frame.depth_m, dtype=torch.float32, device=self.device)
        H, W = depth.shape

        color_img = (
            torch.as_tensor(frame.color, dtype=torch.float32, device=self.device)
            if self._color_pool is not None
            else None
        )

        Nbx, Nby, Nbz = self.block_dims

        # Stage 1: block-level frustum cull.
        bw = self._block_world_center.view(-1, 3)  # (Nb_total, 3)
        bc = (T_cw[:3, :3] @ bw.T).T + T_cw[:3, 3]  # (Nb_total, 3) in camera frame
        z = bc[:, 2]
        r = self._block_half_diag  # conservative inclusion radius
        # keep blocks whose camera-z is within [0 - r, depth_trunc + r]
        trunc = cfg.truncation_distance_m
        # we'll also need an upper depth bound — use the frame's max depth or a cap
        max_depth = float(depth.max().item()) if depth.numel() else 6.0
        z_ok = (z + r > 0) & (z - r < max_depth + trunc)
        # projected center ± inclusion radius within image bounds
        u_c = K[0, 0] * bc[:, 0] / (z.abs().clamp_min(1e-3)) + K[0, 2]
        v_c = K[1, 1] * bc[:, 1] / (z.abs().clamp_min(1e-3)) + K[1, 2]
        proj_margin_u = K[0, 0] * r / (z.abs().clamp_min(1e-3))
        proj_margin_v = K[1, 1] * r / (z.abs().clamp_min(1e-3))
        u_ok = (u_c + proj_margin_u > 0) & (u_c - proj_margin_u < W)
        v_ok = (v_c + proj_margin_v > 0) & (v_c - proj_margin_v < H)
        block_keep = z_ok & u_ok & v_ok  # (Nb_total,)
        surviving = block_keep.nonzero(as_tuple=False).squeeze(-1)  # flat block idx
        if surviving.numel() == 0:
            return

        # Stage 2: expand surviving blocks into their 8³ voxels and run the
        # standard integrate math on those only. block coords in world:
        b_ix = surviving // (Nby * Nbz)
        tmp = surviving % (Nby * Nbz)
        b_iy = tmp // Nbz
        b_iz = tmp % Nbz
        block_origin_world = (
            self.origin
            + torch.stack([b_ix, b_iy, b_iz], dim=-1).to(torch.float32) * BLOCK * cfg.voxel_size_m
        )  # (n_blocks, 3)

        # expand to per-voxel world coords: (n_blocks, 512, 3)
        pts_w = block_origin_world.unsqueeze(1) + self._local_voxel_offset.unsqueeze(0)
        pts_w_flat = pts_w.reshape(-1, 3)  # (n_blocks*512, 3)

        pts_c = (T_cw[:3, :3] @ pts_w_flat.T).T + T_cw[:3, 3]
        z_v = pts_c[:, 2]
        valid_z = z_v > 0
        u = (K[0, 0] * pts_c[:, 0] / (z_v + 1e-8) + K[0, 2]).round().long()
        v = (K[1, 1] * pts_c[:, 1] / (z_v + 1e-8) + K[1, 2]).round().long()
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_z & in_img
        u_s = u.clamp(0, W - 1)
        v_s = v.clamp(0, H - 1)
        d_sample = depth[v_s, u_s]
        valid = valid & (d_sample > 0)
        sdf = d_sample - z_v
        valid = valid & (sdf >= -trunc)
        tsdf_new = (sdf / trunc).clamp(-1.0, 1.0)

        idx_flat = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if idx_flat.numel() == 0:
            return

        # Per-voxel: which surviving block it belongs to, and its local-flat
        # index within that block. Cached arrays take care of local indices.
        which_block = idx_flat // BLOCK3
        which_local = idx_flat % BLOCK3
        surviving_block_flat = surviving[which_block]  # global-flat block id

        # Allocate slots for newly-seen blocks among the surviving set
        unique_blocks, inverse = torch.unique(surviving_block_flat, return_inverse=True)
        ub_slots = self._block_slot.view(-1)[unique_blocks]
        need_alloc = ub_slots < 0
        n_new = int(need_alloc.sum().item())
        if n_new > 0:
            self._ensure_capacity(self._num_blocks + n_new)
            new_ids = torch.arange(
                self._num_blocks,
                self._num_blocks + n_new,
                device=self.device,
                dtype=torch.int32,
            )
            self._block_slot.view(-1).index_copy_(0, unique_blocks[need_alloc], new_ids)
            self._num_blocks += n_new
            ub_slots = self._block_slot.view(-1)[unique_blocks]

        voxel_slot = ub_slots[inverse].to(torch.int64)
        local_flat = which_local  # already int64

        w_old = self._weight_pool[voxel_slot, local_flat]
        t_old = self._tsdf_pool[voxel_slot, local_flat]
        t_samp = tsdf_new[idx_flat]
        w_new = (w_old + 1.0).clamp(max=cfg.max_weight)
        t_merge = (t_old * w_old + t_samp) / (w_old + 1.0)
        self._tsdf_pool[voxel_slot, local_flat] = t_merge
        self._weight_pool[voxel_slot, local_flat] = w_new

        if self._color_pool is not None and color_img is not None:
            u_v = u_s[idx_flat]
            v_v = v_s[idx_flat]
            c_samp = color_img[v_v, u_v]
            c_old = self._color_pool[voxel_slot, local_flat]
            c_new = (c_old * w_old.unsqueeze(-1) + c_samp) / (w_old + 1.0).unsqueeze(-1)
            self._color_pool[voxel_slot, local_flat] = c_new

    # ------------------------------------------------------------ densify
    @torch.no_grad()
    def _densify(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Materialise the dense (Nx, Ny, Nz) tsdf/weight/color tensors.

        Used by `extract_mesh` and `query`. Costs O(num_blocks) and is only
        viable when the total volume fits in VRAM. For really big scenes,
        use `query()` directly or the block pool.
        """
        Nx, Ny, Nz = self.dims
        tsdf = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        weight = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        color = (
            torch.zeros((Nx, Ny, Nz, 3), dtype=torch.float32, device=self.device)
            if self._color_pool is not None
            else None
        )
        Nbx, Nby, Nbz = self.block_dims
        slot = self._block_slot
        # gather each allocated block into dense positions
        alloc = (slot >= 0).nonzero(as_tuple=False)  # (K, 3) of block coords
        for bxyz in alloc:
            ibx, iby, ibz = int(bxyz[0]), int(bxyz[1]), int(bxyz[2])
            s = int(slot[ibx, iby, ibz])
            x0, y0, z0 = ibx * BLOCK, iby * BLOCK, ibz * BLOCK
            tsdf[x0 : x0 + BLOCK, y0 : y0 + BLOCK, z0 : z0 + BLOCK] = self._tsdf_pool[s].view(
                BLOCK, BLOCK, BLOCK
            )
            weight[x0 : x0 + BLOCK, y0 : y0 + BLOCK, z0 : z0 + BLOCK] = self._weight_pool[s].view(
                BLOCK, BLOCK, BLOCK
            )
            if color is not None:
                color[x0 : x0 + BLOCK, y0 : y0 + BLOCK, z0 : z0 + BLOCK, :] = self._color_pool[
                    s
                ].view(BLOCK, BLOCK, BLOCK, 3)
        return tsdf, weight, color

    @torch.no_grad()
    def extract_mesh(self, min_weight: float = 1.0) -> Mesh:
        from skimage.measure import marching_cubes

        tsdf_d, weight_d, color_d = self._densify()
        tsdf_np = tsdf_d.detach().cpu().numpy()
        mask = (weight_d.detach().cpu().numpy() >= min_weight).astype(np.float32)
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
        if color_d is not None:
            vox_idx = ((verts - origin_np) / self.cfg.voxel_size_m).astype(np.int64)
            vox_idx = np.clip(vox_idx, [0, 0, 0], [d - 1 for d in self.dims])
            col = color_d.detach().cpu().numpy()
            vcolors = col[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].clip(0, 255).astype(np.uint8)
        return Mesh(vertices=verts, triangles=faces.astype(np.int32), vertex_colors=vcolors)

    @torch.no_grad()
    def query(self, points_w: torch.Tensor) -> VoxelQueryResult:
        """Nearest-voxel query (no trilinear here — see reference backend for that)."""
        points_w = points_w.to(self.device, dtype=torch.float32)
        idx_f = (points_w - self.origin) / self.cfg.voxel_size_m
        idx = idx_f.floor().long()
        Nx, Ny, Nz = self.dims
        i = idx[:, 0].clamp(0, Nx - 1)
        j = idx[:, 1].clamp(0, Ny - 1)
        k = idx[:, 2].clamp(0, Nz - 1)
        bx = i // BLOCK
        by = j // BLOCK
        bz = k // BLOCK
        li = i % BLOCK
        lj = j % BLOCK
        lk = k % BLOCK
        local_flat = li * BLOCK * BLOCK + lj * BLOCK + lk
        slots = self._block_slot[bx, by, bz]

        tsdf = torch.ones_like(i, dtype=torch.float32)
        weight = torch.zeros_like(i, dtype=torch.float32)
        alloc = slots >= 0
        if alloc.any():
            s = slots[alloc].long()
            tsdf[alloc] = self._tsdf_pool[s, local_flat[alloc]]
            weight[alloc] = self._weight_pool[s, local_flat[alloc]]
        if self._color_pool is not None:
            color = torch.zeros((points_w.shape[0], 3), dtype=torch.uint8, device=self.device)
            if alloc.any():
                c = self._color_pool[s, local_flat[alloc], :]
                color[alloc] = c.clamp(0, 255).to(torch.uint8)
        else:
            color = torch.zeros((points_w.shape[0], 3), dtype=torch.uint8, device=self.device)

        return VoxelQueryResult(tsdf=tsdf, weight=weight, color=color, feat=None)
