"""Near-surface feature-gate tests across all three feature-storing backends.

Regression for the gating bug where the per-frame check `tsdf_new.abs() <= 1.0`
was tautological (because `tsdf_new` is clamped to `[-1, 1]` immediately
above), so features were written across the entire truncation band — including
free-space voxels in front of the surface, which then "stole" the features of
whatever the ray hit.

Each backend test integrates ONE synthetic frame so that the accumulated TSDF
on each observed voxel equals that frame's normalized signed distance. We then
assert two halves of the same property:

  * `default band (0.5)` → no observed voxel with `|tsdf| > band` carries a
    feature. This is the bug-fix invariant.
  * `band = 1.0` (legacy) → at least one observed voxel with `|tsdf| > 0.5`
    DOES carry a feature. This proves the gate is what made the difference,
    not some other change.
"""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import BLOCK, BlockHashTSDF, BlockHashTSDFConfig
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.mapping.sparse_reference import (
    SparseFeatureTSDF,
    SparseFeatureTSDFConfig,
)


def _scene() -> tuple[list, float, float, tuple, tuple, int, str]:
    """Single synthetic frame; chosen so the truncation band has a wide
    free-space half-band that the gate must exclude.
    """
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=1, width=96, height=72, radius=1.2)
    voxel_size = 0.05
    trunc = 0.4  # 8 voxels of truncation → wide free-space half-band per frame
    bmin = (-0.5, -0.5, -0.5)
    bmax = (0.5, 0.5, 0.5)
    D = 8
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return frames, voxel_size, trunc, bmin, bmax, D, device


@pytest.mark.parametrize("band", [0.5, 1.0])
def test_reference_near_surface_gate(band: float) -> None:
    frames, vs, trunc, bmin, bmax, D, device = _scene()
    cfg = ReferenceTSDFConfig(
        voxel_size_m=vs,
        truncation_distance_m=trunc,
        bounds_min=bmin,
        bounds_max=bmax,
        store_color=False,
        store_features=True,
        feature_dim=D,
        near_surface_band=band,
        device=device,
    )
    vol = ReferenceTSDF(cfg)
    feature = torch.ones(D, device=device)
    vol.integrate(frames[0], feature=feature)

    observed = vol.weight > 0
    has_feat = vol.feat.abs().sum(dim=-1) > 0
    far = observed & (vol.tsdf.abs() > 0.5)
    near = observed & (vol.tsdf.abs() <= 0.5)

    # sanity: there should be voxels in BOTH halves of the band on this scene
    assert int(far.sum().item()) > 0, "expected some far-band voxels for the test to be meaningful"
    assert (
        int(near.sum().item()) > 0
    ), "expected some near-band voxels for the test to be meaningful"

    if band <= 0.5:
        # default (fixed) behavior: no far-band voxel should have a feature
        leak = int((far & has_feat).sum().item())
        assert leak == 0, f"feature leaked onto {leak} far-band voxels at band={band}"
    else:
        # legacy band=1.0: the gate is a no-op, so far-band voxels DO get features
        leak = int((far & has_feat).sum().item())
        assert leak > 0, "regression check failed: legacy band=1.0 should leak"


@pytest.mark.parametrize("band", [0.5, 1.0])
def test_sparse_feature_near_surface_gate(band: float) -> None:
    frames, vs, trunc, bmin, bmax, D, device = _scene()
    vol = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=False,
            feature_dim=D,
            initial_feat_capacity=1024,
            near_surface_band=band,
            device=device,
        )
    )
    feature = torch.ones(D, device=device)
    vol.integrate(frames[0], feature=feature)

    observed = vol.weight > 0
    has_feat = vol._voxel_slot >= 0  # private, but the gate's whole point
    far = observed & (vol.tsdf.abs() > 0.5)
    near = observed & (vol.tsdf.abs() <= 0.5)

    assert int(far.sum().item()) > 0
    assert int(near.sum().item()) > 0

    if band <= 0.5:
        leak = int((far & has_feat).sum().item())
        assert leak == 0, f"sparse: feature slot allocated for {leak} far-band voxels"
    else:
        leak = int((far & has_feat).sum().item())
        assert leak > 0, "regression: legacy band=1.0 should allocate slots in far band"


@pytest.mark.parametrize("band", [0.5, 1.0])
def test_block_hash_near_surface_gate(band: float) -> None:
    frames, vs, trunc, bmin, bmax, D, device = _scene()
    vol = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=False,
            store_features=True,
            feature_dim=D,
            initial_block_capacity=64,
            initial_feat_capacity=1024,
            near_surface_band=band,
            device=device,
        )
    )
    feature = torch.ones(D, device=device)
    vol.integrate(frames[0], feature=feature)

    # Reconstruct a dense (Nx,Ny,Nz) view of `has_feat` from the double
    # indirection. A voxel has a feature iff its block's `_feat_voxel_slot`
    # entry at its local-flat index is >= 0.
    Nx, Ny, Nz = vol.dims
    tsdf_d, weight_d, _ = vol._densify()

    has_feat = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=device)
    slot_blk = vol._block_slot
    alloc_blocks = (slot_blk >= 0).nonzero(as_tuple=False)
    for bxyz in alloc_blocks:
        ibx, iby, ibz = int(bxyz[0]), int(bxyz[1]), int(bxyz[2])
        s = int(slot_blk[ibx, iby, ibz])
        x0, y0, z0 = ibx * BLOCK, iby * BLOCK, ibz * BLOCK
        fslot_block = vol._feat_voxel_slot[s].view(BLOCK, BLOCK, BLOCK)
        has_feat[x0 : x0 + BLOCK, y0 : y0 + BLOCK, z0 : z0 + BLOCK] = fslot_block >= 0

    observed = weight_d > 0
    far = observed & (tsdf_d.abs() > 0.5)
    near = observed & (tsdf_d.abs() <= 0.5)
    assert int(far.sum().item()) > 0
    assert int(near.sum().item()) > 0

    if band <= 0.5:
        leak = int((far & has_feat).sum().item())
        assert leak == 0, f"block_hash: feature slot allocated for {leak} far-band voxels"
    else:
        leak = int((far & has_feat).sum().item())
        assert leak > 0, "regression: legacy band=1.0 should allocate slots in far band"
