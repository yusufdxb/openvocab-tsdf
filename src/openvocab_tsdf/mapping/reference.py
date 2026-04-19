"""Reference dense TSDF implementation in PyTorch.

Not fast. Not sparse. Used to validate the custom CUDA backend and to ground
unit tests against known-geometry synthetic scenes. Runs on GPU by default but
has no CUDA-specific code.

Dense volume layout: tensors of shape (Nx, Ny, Nz), stored in the xyz convention
where `tsdf[i, j, k]` is the voxel centered at
`origin + voxel_size * (i + 0.5, j + 0.5, k + 0.5)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openvocab_tsdf.data.base import RGBDFrame
from openvocab_tsdf.mapping.base import Mesh, VoxelQueryResult


@dataclass
class ReferenceTSDFConfig:
    voxel_size_m: float = 0.02
    truncation_distance_m: float = 0.10
    # Axis-aligned world-frame bounds. (min_xyz, max_xyz). Set to something
    # that comfortably contains your scene plus truncation margin.
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_weight: float = 32.0
    store_color: bool = True
    store_features: bool = False
    feature_dim: int = 0
    # Normalized-TSDF gate for feature accumulation. A voxel only receives a
    # feature on a frame where its per-frame `|sdf / truncation| <= band`.
    # 1.0 reproduces the old (broken) "always pass" behavior since `tsdf_new`
    # is clamped to [-1, 1]; 0.5 means features only land within half the
    # truncation distance of the surface, which is what excludes free-space
    # voxels in front of the surface from stealing whatever the ray hits.
    near_surface_band: float = 0.5
    device: str = "cuda:0"


class ReferenceTSDF:
    """Dense, per-voxel PyTorch TSDF. Correct, slow, easy to reason about."""

    def __init__(self, cfg: ReferenceTSDFConfig):
        if cfg.store_features and cfg.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0 when store_features is True")

        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        bmin = torch.tensor(cfg.bounds_min, dtype=torch.float32, device=self.device)
        bmax = torch.tensor(cfg.bounds_max, dtype=torch.float32, device=self.device)
        self.origin = bmin
        dims = ((bmax - bmin) / cfg.voxel_size_m).ceil().to(torch.int64)
        self.dims = tuple(int(d) for d in dims.tolist())  # (Nx, Ny, Nz)

        Nx, Ny, Nz = self.dims
        self.tsdf = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        self.weight = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=self.device)
        if cfg.store_color:
            self.color = torch.zeros((Nx, Ny, Nz, 3), dtype=torch.float32, device=self.device)
        else:
            self.color = None
        if cfg.store_features:
            self.feat = torch.zeros(
                (Nx, Ny, Nz, cfg.feature_dim), dtype=torch.float32, device=self.device
            )
        else:
            self.feat = None

        # Cached voxel center coords in world frame (Nx*Ny*Nz, 3)
        ix = torch.arange(Nx, device=self.device, dtype=torch.float32)
        iy = torch.arange(Ny, device=self.device, dtype=torch.float32)
        iz = torch.arange(Nz, device=self.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
        centers = torch.stack([gx, gy, gz], dim=-1) + 0.5  # voxel index + 0.5
        self._voxel_world = self.origin + centers * cfg.voxel_size_m  # (Nx,Ny,Nz,3)

    # ------------------------------------------------------------------ public
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
        if self.feat is not None:
            self.feat.zero_()

    @torch.no_grad()
    def integrate(
        self,
        frame: RGBDFrame,
        feature: torch.Tensor | None = None,
        feature_map: torch.Tensor | None = None,
        feature_map_input_size: int = 224,
        feature_map_patch_size: int = 16,
        dense_feature_map: torch.Tensor | None = None,
    ) -> None:
        """Fold one RGB-D frame into the volume.

        Feature modes (at most one may be provided):
          - `feature`: a (D,) global per-frame embedding. Every updated voxel
            accumulates the same vector.
          - `feature_map`: an (Hp, Wp, D) per-patch embedding produced by a
            ViT run on a resize-shortest-edge + center-crop square of the
            input image. Each voxel looks up its own patch via its projected
            pixel. `feature_map_input_size` is the square side after
            preprocess (typically 224), `feature_map_patch_size` is the ViT's
            patch stride in pixels (typically 16 for ViT-B/16).
          - `dense_feature_map`: a full-resolution (H, W, D) feature map whose
            (v, u) index matches the depth image directly. Used by the
            SAM-per-mask + CLIP dense path (`mode: sam_dense`).
        """
        n_modes = sum(x is not None for x in (feature, feature_map, dense_feature_map))
        if n_modes > 1:
            raise ValueError("pass at most one of `feature`, `feature_map`, or `dense_feature_map`")
        cfg = self.cfg

        T_wc = torch.as_tensor(frame.T_wc, dtype=torch.float32, device=self.device)
        T_cw = torch.linalg.inv(T_wc)
        K = torch.as_tensor(frame.intrinsics.K, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(frame.depth_m, dtype=torch.float32, device=self.device)
        H, W = depth.shape

        if cfg.store_color:
            color_img = torch.as_tensor(
                frame.color, dtype=torch.float32, device=self.device
            )  # (H,W,3)

        # world -> camera
        pts_w = self._voxel_world.reshape(-1, 3)  # (M,3)
        pts_c = (T_cw[:3, :3] @ pts_w.T).T + T_cw[:3, 3]  # (M,3)
        z = pts_c[:, 2]
        valid_z = z > 0
        # project
        u = (K[0, 0] * pts_c[:, 0] / (z + 1e-8) + K[0, 2]).round().long()
        v = (K[1, 1] * pts_c[:, 1] / (z + 1e-8) + K[1, 2]).round().long()
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_z & in_img

        # depth sample
        u_s = u.clamp(0, W - 1)
        v_s = v.clamp(0, H - 1)
        d_sample = depth[v_s, u_s]
        valid = valid & (d_sample > 0)

        sdf = d_sample - z
        valid = valid & (sdf >= -cfg.truncation_distance_m)

        # normalized tsdf in [-1, 1]
        tsdf_new = (sdf / cfg.truncation_distance_m).clamp(-1.0, 1.0)

        idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return

        w_old = self.weight.view(-1)[idx]
        t_old = self.tsdf.view(-1)[idx]
        t_sample = tsdf_new[idx]

        w_new = (w_old + 1.0).clamp(max=cfg.max_weight)
        t_merged = (t_old * w_old + t_sample * 1.0) / (w_old + 1.0)

        self.tsdf.view(-1).index_copy_(0, idx, t_merged)
        self.weight.view(-1).index_copy_(0, idx, w_new)

        if self.color is not None:
            u_valid = u_s[idx]
            v_valid = v_s[idx]
            c_sample = color_img[v_valid, u_valid]  # (Nv, 3)
            c_old = self.color.view(-1, 3)[idx]
            c_merged = (c_old * w_old.unsqueeze(-1) + c_sample) / (w_old + 1.0).unsqueeze(-1)
            self.color.view(-1, 3).index_copy_(0, idx, c_merged)

        # features are only credible on voxels near the integrated surface;
        # free-space voxels in front of a surface would otherwise steal the
        # features of whatever the ray eventually hits. `tsdf_new` is already
        # clamped to [-1, 1], so the band must be < 1.0 to actually exclude
        # anything — see `near_surface_band` on the config.
        near_surface_local = tsdf_new.abs() <= cfg.near_surface_band
        # in the surviving-idx space, build a mask
        near_surface_at_idx = near_surface_local[idx]

        if feature is not None and self.feat is not None:
            if feature.shape != (cfg.feature_dim,):
                raise ValueError(
                    f"feature must be ({cfg.feature_dim},), got {tuple(feature.shape)}"
                )
            feat_sample = feature.to(self.feat.dtype).to(self.device)
            sel = near_surface_at_idx.nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() > 0:
                idx_sel = idx[sel]
                w_sel = w_old[sel]
                f_old = self.feat.view(-1, cfg.feature_dim)[idx_sel]
                f_merged = (f_old * w_sel.unsqueeze(-1) + feat_sample) / (w_sel + 1.0).unsqueeze(-1)
                self.feat.view(-1, cfg.feature_dim).index_copy_(0, idx_sel, f_merged)

        if dense_feature_map is not None and self.feat is not None:
            # dense_feature_map: (H, W, D) indexed by image (v, u) directly —
            # no preprocess remapping needed.
            dfm = dense_feature_map.to(self.feat.dtype).to(self.device)
            if dfm.shape[:2] != (H, W):
                raise ValueError(
                    f"dense_feature_map shape {tuple(dfm.shape[:2])} must match depth {(H, W)}"
                )
            eligible = near_surface_at_idx
            sel = eligible.nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() > 0:
                idx_sel = idx[sel]
                w_sel = w_old[sel]
                feat_sample = dfm[v_s[idx_sel], u_s[idx_sel], :]
                f_old = self.feat.view(-1, cfg.feature_dim)[idx_sel]
                f_merged = (f_old * w_sel.unsqueeze(-1) + feat_sample) / (w_sel + 1.0).unsqueeze(-1)
                self.feat.view(-1, cfg.feature_dim).index_copy_(0, idx_sel, f_merged)

        if feature_map is not None and self.feat is not None:
            # feature_map: (Hp, Wp, D) where (Hp*patch_size, Wp*patch_size) is
            # the square the encoder saw after preprocess.
            fm = feature_map.to(self.feat.dtype).to(self.device)
            Hp, Wp, D = fm.shape
            if D != cfg.feature_dim:
                raise ValueError(f"feature_map D={D}, expected {cfg.feature_dim}")
            S = feature_map_input_size
            ps = feature_map_patch_size
            # preprocess: resize shortest edge to S, then center-crop to S×S.
            # Map image (u, v) → crop (x_c, y_c), then patch (u_p, v_p).
            # s = S / min(H, W)
            s = float(S) / float(min(H, W))
            H_r = float(H) * s
            W_r = float(W) * s
            dx = (W_r - S) * 0.5
            dy = (H_r - S) * 0.5
            u_valid = u_s[idx].to(torch.float32)
            v_valid = v_s[idx].to(torch.float32)
            x_c = u_valid * s - dx
            y_c = v_valid * s - dy
            in_crop = (x_c >= 0) & (x_c < S) & (y_c >= 0) & (y_c < S)
            u_p = (x_c / ps).clamp(0, Wp - 1).to(torch.int64)
            v_p = (y_c / ps).clamp(0, Hp - 1).to(torch.int64)
            # restrict to near-surface voxels (feature fidelity) AND inside the
            # center-cropped area of the preprocessed image (patch validity)
            eligible = in_crop & near_surface_at_idx
            if eligible.any():
                sel = eligible.nonzero(as_tuple=False).squeeze(-1)
                idx_sel = idx[sel]
                w_sel = w_old[sel]
                feat_sample = fm[v_p[sel], u_p[sel], :]
                f_old = self.feat.view(-1, cfg.feature_dim)[idx_sel]
                f_merged = (f_old * w_sel.unsqueeze(-1) + feat_sample) / (w_sel + 1.0).unsqueeze(-1)
                self.feat.view(-1, cfg.feature_dim).index_copy_(0, idx_sel, f_merged)

    @torch.no_grad()
    def extract_mesh(self, min_weight: float = 1.0) -> Mesh:
        """Marching cubes via scikit-image, in world coordinates."""
        from skimage.measure import marching_cubes

        tsdf_np = self.tsdf.detach().cpu().numpy()
        mask = (self.weight.detach().cpu().numpy() >= min_weight).astype(np.float32)
        # where unobserved, force +1 (free space) so marching cubes does not close through it
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
            # nearest-voxel color lookup for each vertex
            vox_idx = ((verts - origin_np) / self.cfg.voxel_size_m).astype(np.int64)
            vox_idx = np.clip(vox_idx, [0, 0, 0], [d - 1 for d in self.dims])
            col = self.color.detach().cpu().numpy()
            vcolors = col[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].clip(0, 255).astype(np.uint8)

        return Mesh(vertices=verts, triangles=faces.astype(np.int32), vertex_colors=vcolors)

    @torch.no_grad()
    def query(self, points_w: torch.Tensor) -> VoxelQueryResult:
        """Trilinear-interp query at N world points."""
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
            c000 = _gather(tensor, i0, j0, k0)
            c001 = _gather(tensor, i0, j0, k1)
            c010 = _gather(tensor, i0, j1, k0)
            c011 = _gather(tensor, i0, j1, k1)
            c100 = _gather(tensor, i1, j0, k0)
            c101 = _gather(tensor, i1, j0, k1)
            c110 = _gather(tensor, i1, j1, k0)
            c111 = _gather(tensor, i1, j1, k1)
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
                _scale(c000, w000)
                + _scale(c001, w001)
                + _scale(c010, w010)
                + _scale(c011, w011)
                + _scale(c100, w100)
                + _scale(c101, w101)
                + _scale(c110, w110)
                + _scale(c111, w111)
            )

        tsdf = _trilin(self.tsdf)
        weight = _trilin(self.weight)
        if self.color is not None:
            color = _trilin(self.color).clamp(0, 255).to(torch.uint8)
        else:
            color = torch.zeros((points_w.shape[0], 3), dtype=torch.uint8, device=self.device)
        feat = _trilin(self.feat) if self.feat is not None else None

        return VoxelQueryResult(tsdf=tsdf, weight=weight, color=color, feat=feat)
