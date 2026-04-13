"""Tests for the tiny aggregation helpers (no model needed)."""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.semantics.aggregation import assert_normalized, cosine_score


def test_assert_normalized_passes_for_unit_vectors():
    f = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    assert_normalized(f)


def test_assert_normalized_raises_for_nonunit():
    f = torch.tensor([[2.0, 0.0]])
    with pytest.raises(ValueError):
        assert_normalized(f)


def test_cosine_score_identity_is_one():
    q = torch.tensor([0.6, 0.8])
    feats = torch.tensor([[0.6, 0.8], [-0.6, -0.8]])
    scores = cosine_score(q, feats)
    assert torch.isclose(scores[0], torch.tensor(1.0))
    assert torch.isclose(scores[1], torch.tensor(-1.0))


def test_cosine_score_over_3d_voxel_tensor():
    D = 4
    q = torch.ones(D) / D**0.5
    vox = torch.zeros(3, 3, 3, D)
    vox[1, 1, 1] = q  # one voxel perfectly aligned
    scores = cosine_score(q, vox)
    assert scores.shape == (3, 3, 3)
    assert torch.isclose(scores[1, 1, 1], torch.tensor(1.0))
