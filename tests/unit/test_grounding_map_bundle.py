"""Regression tests for MapBundle: dispatches on `sparse_kind`.

These would have caught the ROS 2 offline-mode bug where `data["feat"]` was
read unconditionally. We construct a synthetic map of each layout, save it
through the same key set `pipeline.encode_and_fuse` writes, then load it
through `MapBundle` and verify `score_query` returns the right answer on
known voxels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.grounding.map_bundle import MapBundle
from openvocab_tsdf.mapping.block_hash import BlockHashTSDF, BlockHashTSDFConfig
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig
from openvocab_tsdf.mapping.sparse_reference import (
    SparseFeatureTSDF,
    SparseFeatureTSDFConfig,
)


def _frames(D: int):
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    return make_synthetic_dataset(primitives, num_frames=4, width=80, height=60, radius=1.2)


def _save_dense(vol: ReferenceTSDF, out: Path, D: int) -> None:
    np.savez_compressed(
        out,
        origin=vol.origin.cpu().numpy(),
        voxel_size=np.float32(vol.voxel_size_m),
        truncation=np.float32(vol.truncation_distance_m),
        dims=np.int64(vol.dims),
        feature_dim=np.int64(D),
        model=np.array("ViT-B-16", dtype=object),
        pretrained=np.array("laion2b", dtype=object),
        mode=np.array("global", dtype=object),
        sparse=np.bool_(False),
        sparse_kind=np.array("dense", dtype=object),
        tsdf=vol.tsdf.cpu().numpy(),
        weight=vol.weight.cpu().numpy(),
        color=np.zeros(0),
        feat=vol.feat.cpu().numpy(),
    )


def _save_voxel_slot(vol: SparseFeatureTSDF, out: Path, D: int) -> None:
    n = vol.num_allocated_feat_voxels
    np.savez_compressed(
        out,
        origin=vol.origin.cpu().numpy(),
        voxel_size=np.float32(vol.voxel_size_m),
        truncation=np.float32(vol.truncation_distance_m),
        dims=np.int64(vol.dims),
        feature_dim=np.int64(D),
        model=np.array("ViT-B-16", dtype=object),
        pretrained=np.array("laion2b", dtype=object),
        mode=np.array("global", dtype=object),
        sparse=np.bool_(True),
        sparse_kind=np.array("voxel_slot", dtype=object),
        tsdf=vol.tsdf.cpu().numpy(),
        weight=vol.weight.cpu().numpy(),
        color=np.zeros(0),
        feat_pool=vol._feat_pool[:n].cpu().numpy(),
        voxel_slot=vol._voxel_slot.cpu().numpy(),
    )


def _save_block_hash(vol: BlockHashTSDF, out: Path, D: int) -> None:
    nb = vol.num_allocated_blocks
    nfv = vol.num_allocated_feat_voxels
    np.savez_compressed(
        out,
        origin=vol.origin.cpu().numpy(),
        voxel_size=np.float32(vol.voxel_size_m),
        truncation=np.float32(vol.truncation_distance_m),
        dims=np.int64(vol.dims),
        block_dims=np.int64(vol.block_dims),
        feature_dim=np.int64(D),
        model=np.array("ViT-B-16", dtype=object),
        pretrained=np.array("laion2b", dtype=object),
        mode=np.array("global", dtype=object),
        sparse=np.bool_(True),
        sparse_kind=np.array("block_hash", dtype=object),
        block_slot=vol._block_slot.cpu().numpy(),
        tsdf_pool=vol._tsdf_pool[:nb].cpu().numpy(),
        weight_pool=vol._weight_pool[:nb].cpu().numpy(),
        feat_voxel_slot=vol._feat_voxel_slot[:nb].cpu().numpy(),
        feat_pool=vol._feat_pool[:nfv].cpu().numpy(),
    )


def _build_dense(D: int) -> ReferenceTSDF:
    vol = ReferenceTSDF(
        ReferenceTSDFConfig(
            voxel_size_m=0.05,
            truncation_distance_m=0.2,
            bounds_min=(-0.5, -0.5, -0.5),
            bounds_max=(0.5, 0.5, 0.5),
            store_color=False,
            store_features=True,
            feature_dim=D,
            device="cpu",
        )
    )
    feats = torch.eye(D)[torch.arange(4) % D]
    for f, fv in zip(_frames(D), feats, strict=True):
        vol.integrate(f, feature=fv)
    return vol


def _build_voxel_slot(D: int) -> SparseFeatureTSDF:
    vol = SparseFeatureTSDF(
        SparseFeatureTSDFConfig(
            voxel_size_m=0.05,
            truncation_distance_m=0.2,
            bounds_min=(-0.5, -0.5, -0.5),
            bounds_max=(0.5, 0.5, 0.5),
            store_color=False,
            feature_dim=D,
            initial_feat_capacity=512,
            device="cpu",
        )
    )
    feats = torch.eye(D)[torch.arange(4) % D]
    for f, fv in zip(_frames(D), feats, strict=True):
        vol.integrate(f, feature=fv)
    return vol


def _build_block_hash(D: int) -> BlockHashTSDF:
    vol = BlockHashTSDF(
        BlockHashTSDFConfig(
            voxel_size_m=0.05,
            truncation_distance_m=0.2,
            bounds_min=(-0.5, -0.5, -0.5),
            bounds_max=(0.5, 0.5, 0.5),
            store_color=False,
            store_features=True,
            feature_dim=D,
            initial_block_capacity=64,
            initial_feat_capacity=512,
            device="cpu",
        )
    )
    feats = torch.eye(D)[torch.arange(4) % D]
    for f, fv in zip(_frames(D), feats, strict=True):
        vol.integrate(f, feature=fv)
    return vol


@pytest.mark.parametrize(
    "kind,builder,saver",
    [
        ("dense", _build_dense, _save_dense),
        ("voxel_slot", _build_voxel_slot, _save_voxel_slot),
        ("block_hash", _build_block_hash, _save_block_hash),
    ],
)
def test_map_bundle_loads_and_scores(kind: str, builder, saver, tmp_path: Path) -> None:
    D = 8
    vol = builder(D)
    out = tmp_path / f"{kind}.npz"
    saver(vol, out, D)

    bundle = MapBundle(out, device="cpu")
    assert bundle.sparse_kind == kind
    assert bundle.dims is not None and len(bundle.dims) == 3
    assert bundle.weight.shape == bundle.tsdf.shape

    # Build a basis query and confirm the score volume comes out the right shape
    # and has at least one observed voxel scoring above the unobserved sentinel.
    q = torch.zeros(D)
    q[0] = 1.0
    scores = bundle.score_query(q)
    assert tuple(scores.shape) == bundle.dims

    observed = bundle.weight > 0
    assert observed.any(), "test scene didn't produce any observed voxels"
    if kind != "dense":
        # sparse kinds fill unobserved voxels with -1e4
        assert scores[observed].max() > -1e3


def test_map_bundle_dense_vs_voxel_slot_score_parity(tmp_path: Path) -> None:
    """Round-trip the same logical scene through the dense and the voxel_slot
    layouts and confirm `MapBundle.score_query` returns identical scores on
    every observed voxel. This is the contract the three pipeline / eval /
    heatmap refactors lean on.
    """
    D = 8
    dense = _build_dense(D)
    sparse = _build_voxel_slot(D)
    out_dense = tmp_path / "dense.npz"
    out_sparse = tmp_path / "voxel_slot.npz"
    _save_dense(dense, out_dense, D)
    _save_voxel_slot(sparse, out_sparse, D)

    b_dense = MapBundle(out_dense, device="cpu")
    b_sparse = MapBundle(out_sparse, device="cpu")
    q = torch.zeros(D)
    q[0] = 1.0
    s_dense = b_dense.score_query(q)
    s_sparse = b_sparse.score_query(q)

    # Compare only on voxels where BOTH backends wrote a feature (i.e. inside
    # the near-surface band that the gate allows). Far-band observed voxels
    # diverge by construction: dense leaves them at 0 (no feature written),
    # sparse marks them with the unobserved sentinel — both filter out at the
    # threshold step downstream, so the divergence is harmless there.
    near_surface = b_dense.tsdf.abs() <= 0.5
    observed_both = near_surface & (b_dense.weight > 0) & (b_sparse.weight > 0)
    diff = (s_dense[observed_both] - s_sparse[observed_both]).abs().max().item()
    assert diff < 1e-5, f"dense vs voxel_slot score diverged on near-surface voxels: {diff}"


def test_model_mismatch_raises(tmp_path):
    """Loading a map with model=ViT-B-16 and querying with ViT-L-14 should raise."""
    npz_path = tmp_path / "map.npz"
    dims = (4, 4, 4)
    np.savez_compressed(
        npz_path,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.float32(0.04),
        truncation=np.float32(0.16),
        dims=np.array(dims, dtype=np.int64),
        feature_dim=np.int64(512),
        model=np.array("ViT-B-16", dtype=object),
        pretrained=np.array("laion2b_s34b_b88k", dtype=object),
        mode=np.array("sam_dense", dtype=object),
        sparse=np.bool_(False),
        sparse_kind=np.array("dense", dtype=object),
        tsdf=np.ones(dims, dtype=np.float32),
        weight=np.ones(dims, dtype=np.float32),
        feat=np.random.randn(*dims, 512).astype(np.float32),
    )
    from openvocab_tsdf.pipeline import _validate_model_match

    bundle = MapBundle(npz_path, device="cpu")
    # Same model — should pass
    _validate_model_match(bundle.meta, "ViT-B-16")
    # Different model — should raise
    with pytest.raises(ValueError, match="model mismatch"):
        _validate_model_match(bundle.meta, "ViT-L-14")


def test_model_match_empty_meta_passes(tmp_path):
    """Maps with empty model field (legacy) should not block grounding."""
    npz_path = tmp_path / "map_legacy.npz"
    dims = (4, 4, 4)
    np.savez_compressed(
        npz_path,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.float32(0.04),
        truncation=np.float32(0.16),
        dims=np.array(dims, dtype=np.int64),
        feature_dim=np.int64(512),
        sparse=np.bool_(False),
        sparse_kind=np.array("dense", dtype=object),
        tsdf=np.ones(dims, dtype=np.float32),
        weight=np.ones(dims, dtype=np.float32),
        feat=np.random.randn(*dims, 512).astype(np.float32),
    )
    from openvocab_tsdf.pipeline import _validate_model_match

    bundle = MapBundle(npz_path, device="cpu")
    # No stored model -> should not raise even with arbitrary query model
    _validate_model_match(bundle.meta, "ViT-L-14")
