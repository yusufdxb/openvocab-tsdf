"""Parity + perf sanity for the Triton-backed sparse feature update."""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.sparse_reference import (
    _HAS_TRITON,
    SparseFeatureTSDF,
    SparseFeatureTSDFConfig,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_TRITON, reason="triton not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _scene():
    prim = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(prim, num_frames=8, width=96, height=72, radius=1.2)
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    trunc = 0.2
    D = 8
    return frames, 0.05, trunc, bounds, D


def test_sparse_triton_matches_pytorch_feature_pool():
    frames, vs, trunc, (bmin, bmax), D = _scene()
    pyt = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=1024,
            feat_update_backend="pytorch",
            device="cuda:0",
        )
    )
    trt = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=vs,
            truncation_distance_m=trunc,
            bounds_min=bmin,
            bounds_max=bmax,
            store_color=True,
            feature_dim=D,
            initial_feat_capacity=1024,
            feat_update_backend="triton",
            device="cuda:0",
        )
    )
    feats = torch.eye(D, device="cuda:0")[torch.arange(len(frames)) % D]
    for f, fv in zip(frames, feats, strict=True):
        pyt.integrate(f, feature=fv)
        trt.integrate(f, feature=fv)

    # slot allocation order is deterministic (same integration order) →
    # pools should be identical bit-for-bit up to the allocated count.
    n_alloc = pyt.num_allocated_feat_voxels
    assert trt.num_allocated_feat_voxels == n_alloc
    diff = (pyt._feat_pool[:n_alloc] - trt._feat_pool[:n_alloc]).abs()
    assert diff.max().item() < 1e-5, f"max diff {diff.max().item()}"
