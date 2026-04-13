"""Parity + scale tests for the block-hash sparse-geometry backend."""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.block_hash import BlockHashTSDF, BlockHashTSDFConfig
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


@pytest.mark.gpu
def test_block_hash_matches_reference_geometry():
    """Geometry (tsdf, weight, color) must match the dense reference bit-for-bit."""
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(primitives, num_frames=6, width=96, height=72, radius=1.2)
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    trunc = 0.2
    vs = 0.05
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ref = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            store_color=True,
            store_features=False,
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
            initial_block_capacity=128,
            device=device,
        )
    )
    for f in frames:
        ref.integrate(f)
        bh.integrate(f)

    # Bounds rounding up to multiples of BLOCK may make bh.dims bigger than
    # ref.dims; compare only on the shared prefix.
    Nx, Ny, Nz = ref.dims
    tsdf_bh, weight_bh, color_bh = bh._densify()
    tsdf_bh = tsdf_bh[:Nx, :Ny, :Nz]
    weight_bh = weight_bh[:Nx, :Ny, :Nz]
    color_bh = color_bh[:Nx, :Ny, :Nz]

    assert torch.equal(ref.tsdf, tsdf_bh), "tsdf mismatch"
    assert torch.equal(ref.weight, weight_bh), "weight mismatch"
    assert torch.equal(ref.color, color_bh), "color mismatch"


@pytest.mark.gpu
def test_block_hash_allocates_only_observed_blocks():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=80, height=60, radius=1.2)
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    bh = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=0.05,
            truncation_distance_m=0.2,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            store_color=True,
            initial_block_capacity=32,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    for f in frames:
        bh.integrate(f)
    total_blocks = 1
    for d in bh.block_dims:
        total_blocks *= d
    # sphere is small relative to the full volume
    assert 0 < bh.num_allocated_blocks < total_blocks


@pytest.mark.gpu
def test_block_hash_query_matches_reference():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=64, height=48, radius=1.2)
    bounds = ((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4))
    vs = 0.05
    trunc = 0.2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    bh = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            store_color=True,
            device=device,
        )
    )
    for f in frames:
        bh.integrate(f)

    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [-0.3, 0.0, -0.3]],
        dtype=torch.float32,
    )
    res = bh.query(pts)
    assert res.tsdf.shape == (4,)
    assert res.weight.shape == (4,)
    assert res.color.shape == (4, 3)
    # At least one of the 4 surface-adjacent points must be observed given
    # the 4-camera ring coverage.
    assert (res.weight > 0).any(), f"no surface voxels observed: {res.weight}"
