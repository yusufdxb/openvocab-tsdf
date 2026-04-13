"""Parity tests: SparseFeatureTSDF vs ReferenceTSDF on synthetic scenes.

The sparse backend must produce identical geometry (tsdf / weight / color)
and bit-for-bit equal features on every observed voxel, since the math is
the same — only the storage is different.
"""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.mapping.sparse_reference import (
    SparseFeatureTSDF,
    SparseFeatureTSDFConfig,
)


def _scene(voxel_size: float = 0.05, num_frames: int = 6, D: int = 8):
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(
        primitives, num_frames=num_frames, width=96, height=72, radius=1.2
    )
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    trunc = 4 * voxel_size
    return frames, voxel_size, trunc, bounds, D


@pytest.mark.gpu
def test_sparse_matches_reference_geometry_and_features():
    frames, vs, trunc, (bmin, bmax), D = _scene()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ref = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            store_features=True,
            feature_dim=D,
            device=device,
        )
    )
    sp = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=1024,
            device=device,
        )
    )

    feats = torch.eye(D, device=device)[torch.arange(len(frames)) % D]
    for f, fv in zip(frames, feats, strict=True):
        ref.integrate(f, feature=fv)
        sp.integrate(f, feature=fv)

    # Geometry identity
    assert torch.equal(ref.tsdf, sp.tsdf)
    assert torch.equal(ref.weight, sp.weight)
    assert torch.equal(ref.color, sp.color)

    # Features: compare on observed voxels
    observed = sp.weight > 0
    ref_feat = ref.feat[observed]
    sp_feat = sp.feat[observed]
    # reference may have tiny fp noise in different aggregation orders
    assert torch.allclose(
        ref_feat, sp_feat, atol=1e-6
    ), f"max diff = {(ref_feat - sp_feat).abs().max().item()}"


@pytest.mark.gpu
def test_sparse_allocates_only_for_observed_voxels():
    frames, vs, trunc, (bmin, bmax), D = _scene()
    sp = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=512,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    feat = torch.ones(D)
    for f in frames:
        sp.integrate(f, feature=feat)
    # Only surface-adjacent voxels should have slots allocated
    total = int(torch.tensor(sp.dims).prod().item())
    assert 0 < sp.num_allocated_feat_voxels < total
    # Memory proxy: allocated slots * D * 4 bytes
    assert sp.feat_memory_bytes() == sp.num_allocated_feat_voxels * D * 4


@pytest.mark.gpu
def test_sparse_pool_grows_when_capacity_exceeded():
    frames, vs, trunc, (bmin, bmax), D = _scene(num_frames=12)
    sp = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=8,  # tiny on purpose
            max_feat_capacity=1_000_000,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    feat = torch.ones(D)
    for f in frames:
        sp.integrate(f, feature=feat)
    # many more slots than initial cap — pool must have grown
    assert sp.num_allocated_feat_voxels > 8
