"""Unit tests for scene-mean subtraction and negative-prompt refinements."""

from __future__ import annotations

import numpy as np
import torch

from openvocab_tsdf.grounding.query import rank_query


def _fake_map(D: int = 8, side: int = 6):
    """Small (6,6,6,D) map where a small blob has a distinctive feature."""
    feats = torch.zeros(side, side, side, D)
    feats[..., 0] = 1.0  # baseline: all voxels weakly match channel 0
    # blob in one corner with a channel-1 signal
    feats[1:3, 1:3, 1:3, :] = 0.0
    feats[1:3, 1:3, 1:3, 1] = 1.0
    weights = torch.ones(side, side, side)
    tsdf = torch.zeros(side, side, side)  # all on surface
    origin = np.zeros(3, dtype=np.float32)
    return feats, weights, tsdf, origin


def test_scene_mean_subtract_sharpens_blob():
    D = 8
    feats, weights, tsdf, origin = _fake_map(D=D)
    # query = channel-1 direction → blob should score highest
    q = torch.zeros(D)
    q[1] = 1.0

    # without subtraction the blob already wins, but so do many others tied
    r_plain = rank_query(
        voxel_feats=feats,
        voxel_weights=weights,
        voxel_tsdf=tsdf,
        text_embedding=q,
        origin=origin,
        voxel_size=0.1,
        top_percentile=0.1,
        cluster_eps_vox=1,
        min_cluster_voxels=1,
        score_threshold=None,
        top_k=5,
    )
    # with scene-mean subtract, the noise around the blob stays below the
    # cutoff and the blob becomes the only cluster
    r_sub = rank_query(
        voxel_feats=feats,
        voxel_weights=weights,
        voxel_tsdf=tsdf,
        text_embedding=q,
        origin=origin,
        voxel_size=0.1,
        top_percentile=0.1,
        cluster_eps_vox=1,
        min_cluster_voxels=1,
        score_threshold=None,
        scene_mean_subtract=True,
        top_k=5,
    )
    assert len(r_sub) >= 1
    # top cluster in subtracted mode must contain the blob centroid
    top = r_sub[0]
    cx, cy, cz = top.center_m
    # blob voxel indices are in [1, 3), world coords [0.15, 0.25] × same
    # Blob voxels are at index [1, 3) — centers 0.15..0.25. Allow dilation slack.
    assert 0.05 <= cx <= 0.35 and 0.05 <= cy <= 0.35 and 0.05 <= cz <= 0.35, top.center_m
    # at minimum, the subtract version did not return more clusters than plain
    assert len(r_sub) <= len(r_plain) or len(r_sub) == 1


def test_negative_prompt_subtract():
    """Providing a neg-text embedding should subtract that direction from the score."""
    D = 8
    feats, weights, tsdf, origin = _fake_map(D=D)
    q = torch.zeros(D)
    q[1] = 1.0
    neg = torch.zeros(D)
    neg[0] = 1.0  # subtract the baseline-channel affinity

    r = rank_query(
        voxel_feats=feats,
        voxel_weights=weights,
        voxel_tsdf=tsdf,
        text_embedding=q,
        origin=origin,
        voxel_size=0.1,
        top_percentile=0.15,
        cluster_eps_vox=1,
        min_cluster_voxels=1,
        score_threshold=None,
        neg_text_embedding=neg,
        top_k=5,
    )
    assert len(r) >= 1
    top = r[0]
    cx, cy, cz = top.center_m
    # Blob voxels are at index [1, 3) — centers 0.15..0.25. Allow dilation slack.
    assert 0.05 <= cx <= 0.35 and 0.05 <= cy <= 0.35 and 0.05 <= cz <= 0.35, top.center_m
