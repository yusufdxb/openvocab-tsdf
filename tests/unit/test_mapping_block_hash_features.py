"""Feature-storage parity tests for the combined BlockHashTSDF backend.

The block backend with `store_features=True` must produce the same per-voxel
features as the dense ReferenceTSDF's feature path (up to fp noise), while
allocating memory only for observed surface voxels.
"""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import BLOCK3, BlockHashTSDF, BlockHashTSDFConfig
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


@pytest.mark.gpu
def test_block_hash_global_features_match_reference():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(primitives, num_frames=6, width=96, height=72, radius=1.2)
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    vs = 0.05
    trunc = 0.2
    D = 8
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ref = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            store_color=True,
            store_features=True,
            feature_dim=D,
            device=device,
        )
    )
    bh = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            store_color=True,
            store_features=True,
            feature_dim=D,
            initial_block_capacity=64,
            initial_feat_capacity=512,
            device=device,
        )
    )
    feats = torch.eye(D, device=device)[torch.arange(len(frames)) % D]
    for f, fv in zip(frames, feats, strict=True):
        ref.integrate(f, feature=fv)
        bh.integrate(f, feature=fv)

    # Reconstruct bh dense feat volume for comparison
    Nx, Ny, Nz = ref.dims
    feat_dense = torch.zeros((Nx, Ny, Nz, D), dtype=torch.float32, device=device)
    slot_blk = bh._block_slot  # (Nbx, Nby, Nbz)
    alloc_blocks = (slot_blk >= 0).nonzero(as_tuple=False)
    for bxyz in alloc_blocks:
        ibx, iby, ibz = int(bxyz[0]), int(bxyz[1]), int(bxyz[2])
        s = int(slot_blk[ibx, iby, ibz])
        x0, y0, z0 = ibx * 8, iby * 8, ibz * 8
        fslot = bh._feat_voxel_slot[s]  # (BLOCK3,)
        have_feat = fslot >= 0
        if have_feat.any():
            flat_vals = bh._feat_pool[fslot[have_feat].long()]  # (K, D)
            lindex = have_feat.nonzero(as_tuple=False).squeeze(-1)
            li = (lindex // 64).long()
            lj = ((lindex // 8) % 8).long()
            lk = (lindex % 8).long()
            # write inside the Nx,Ny,Nz region (skip dims rounded above ref)
            for a in range(flat_vals.shape[0]):
                ii = x0 + int(li[a])
                jj = y0 + int(lj[a])
                kk = z0 + int(lk[a])
                if ii < Nx and jj < Ny and kk < Nz:
                    feat_dense[ii, jj, kk] = flat_vals[a]

    observed = ref.weight > 0
    ref_f = ref.feat[observed]
    bh_f = feat_dense[observed]
    # allow fp noise; both paths do the same math in a different order
    diff = (ref_f - bh_f).abs()
    assert diff.max().item() < 1e-5, f"max diff {diff.max().item()}"


@pytest.mark.gpu
def test_block_hash_feature_pool_grows_lazily():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=80, height=60, radius=1.2)
    bh = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=0.05,
            truncation_distance_m=0.2,
            bounds_min=(-0.5, -0.5, -0.5),
            bounds_max=(0.5, 0.5, 0.5),
            store_color=True,
            store_features=True,
            feature_dim=4,
            initial_block_capacity=16,
            initial_feat_capacity=4,  # tiny on purpose
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    feat = torch.ones(4)
    for f in frames:
        bh.integrate(f, feature=feat)
    # we observed more than 4 voxels → pool must have grown
    assert bh.num_allocated_feat_voxels > 4
    # sparsity: allocated feature voxels much less than total volume
    total_vox = int(bh.dims[0] * bh.dims[1] * bh.dims[2])
    assert bh.num_allocated_feat_voxels < total_vox


_ = BLOCK3  # keep import-used
