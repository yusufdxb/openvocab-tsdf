"""Feature-storage parity tests for the combined BlockHashTSDF backend.

The block backend with `store_features=True` must produce the same per-voxel
features as the dense ReferenceTSDF's feature path (up to fp noise), while
allocating memory only for observed surface voxels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import (
    BLOCK3,
    BlockHashTSDF,
    BlockHashTSDFConfig,
    densify_block_pool,
    scatter_feat_pool_values,
)
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


@pytest.mark.gpu
def test_block_hash_save_load_round_trip(tmp_path: Path) -> None:
    """End-to-end round trip of the block_hash npz save format.

    This mirrors the save path in `pipeline.encode_and_fuse` with a synthetic
    dataset + deterministic per-frame features, then loads the npz and uses
    `densify_block_pool` + `scatter_feat_pool_values` to reconstruct the
    dense geometry and a dense per-query score volume. Asserts:

      1. Every advertised metadata key is present and well-typed.
      2. Loaded dense weight matches the in-memory dense view exactly.
      3. For each basis query, the scattered score volume agrees with the
         in-memory feature tensor contracted with the same query on every
         observed voxel (fp noise only).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(200, 80, 80))]
    frames = make_synthetic_dataset(primitives, num_frames=6, width=96, height=72, radius=1.2)
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    vs = 0.05
    trunc = 0.2
    D = 8

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
        bh.integrate(f, feature=fv)

    # Save using the same key set that pipeline.encode_and_fuse writes.
    nb = bh.num_allocated_blocks
    nfv = bh.num_allocated_feat_voxels
    map_path = tmp_path / "block_hash_round_trip.npz"
    np.savez_compressed(
        map_path,
        origin=bh.origin.cpu().numpy(),
        voxel_size=np.float32(bh.voxel_size_m),
        truncation=np.float32(bh.truncation_distance_m),
        dims=np.int64(bh.dims),
        block_dims=np.int64(bh.block_dims),
        feature_dim=np.int64(D),
        model=np.array("ViT-B-16", dtype=object),
        pretrained=np.array("laion2b_s34b_b88k", dtype=object),
        mode=np.array("global", dtype=object),
        sparse=np.bool_(True),
        sparse_kind=np.array("block_hash", dtype=object),
        block_slot=bh._block_slot.cpu().numpy(),
        tsdf_pool=bh._tsdf_pool[:nb].cpu().numpy(),
        weight_pool=bh._weight_pool[:nb].cpu().numpy(),
        color_pool=bh._color_pool[:nb].cpu().numpy(),
        feat_voxel_slot=bh._feat_voxel_slot[:nb].cpu().numpy(),
        feat_pool=bh._feat_pool[:nfv].cpu().numpy(),
    )

    data = np.load(map_path, allow_pickle=True)
    # 1. every key the downstream loaders look for is present
    required = {
        "origin",
        "voxel_size",
        "truncation",
        "dims",
        "block_dims",
        "feature_dim",
        "mode",
        "sparse",
        "sparse_kind",
        "block_slot",
        "tsdf_pool",
        "weight_pool",
        "feat_voxel_slot",
        "feat_pool",
    }
    missing = required - set(data.files)
    assert not missing, f"missing keys: {missing}"
    assert str(data["sparse_kind"]) == "block_hash"
    assert int(data["feature_dim"]) == D

    # 2. densified weight matches the in-memory densification exactly
    block_slot = torch.from_numpy(data["block_slot"]).to(device)
    weight_pool = torch.from_numpy(data["weight_pool"]).to(device)
    tsdf_pool = torch.from_numpy(data["tsdf_pool"]).to(device)
    dims = tuple(int(d) for d in data["dims"])
    block_dims = tuple(int(d) for d in data["block_dims"])
    w_loaded = densify_block_pool(block_slot, weight_pool, dims, block_dims, default=0.0)
    t_loaded = densify_block_pool(block_slot, tsdf_pool, dims, block_dims, default=1.0)
    # Compare against the instance's own private densify (same math — this
    # locks the module helper to the class method).
    t_mem, w_mem, _ = bh._densify()
    assert torch.allclose(w_loaded, w_mem)
    assert torch.allclose(t_loaded, t_mem)

    # 3. per-query score reconstruction matches in-memory feature contraction.
    # Compare on voxels that ACTUALLY received a feature (i.e. inside the
    # post-fix near-surface band). Far-band observed voxels diverge by
    # construction: scatter_feat_pool_values fills them with the unobserved
    # sentinel (-1e4), while the manual densification leaves them at 0.
    # Both filter out at the threshold step downstream.
    feat_voxel_slot = torch.from_numpy(data["feat_voxel_slot"]).to(device)
    feat_pool = torch.from_numpy(data["feat_pool"]).to(device)

    # build the in-memory dense feat view via the same scheme used by the
    # existing parity test above
    Nx, Ny, Nz = bh.dims
    feat_dense_mem = torch.zeros((Nx, Ny, Nz, D), dtype=torch.float32, device=device)
    slot_blk = bh._block_slot
    alloc_blocks = (slot_blk >= 0).nonzero(as_tuple=False)
    for bxyz in alloc_blocks:
        ibx, iby, ibz = int(bxyz[0]), int(bxyz[1]), int(bxyz[2])
        s = int(slot_blk[ibx, iby, ibz])
        x0, y0, z0 = ibx * 8, iby * 8, ibz * 8
        fslot = bh._feat_voxel_slot[s]
        have = fslot >= 0
        if have.any():
            flat_vals = bh._feat_pool[fslot[have].long()]
            lindex = have.nonzero(as_tuple=False).squeeze(-1)
            li = (lindex // 64).long()
            lj = ((lindex // 8) % 8).long()
            lk = (lindex % 8).long()
            for a in range(flat_vals.shape[0]):
                ii, jj, kk = x0 + int(li[a]), y0 + int(lj[a]), z0 + int(lk[a])
                if ii < Nx and jj < Ny and kk < Nz:
                    feat_dense_mem[ii, jj, kk] = flat_vals[a]

    has_feat_mem = feat_dense_mem.abs().sum(dim=-1) > 0
    for basis in range(D):
        q = torch.zeros(D, device=device)
        q[basis] = 1.0
        pool_scores = feat_pool @ q
        scores_loaded = scatter_feat_pool_values(
            block_slot, feat_voxel_slot, pool_scores, dims, default=-1e4
        )
        scores_mem = feat_dense_mem @ q
        diff = (scores_loaded[has_feat_mem] - scores_mem[has_feat_mem]).abs().max().item()
        assert diff < 1e-5, f"basis {basis}: max diff {diff}"
