"""3D feature aggregation helpers.

The reference TSDF backend already supports a per-frame global feature via
`ReferenceTSDF.integrate(frame, feature=...)`. This module is the place to add
smarter aggregation (patch / mask features, per-pixel lifting, learned pooling)
when Phase 2b lands.

For v1 we only expose:
  - `assert_normalized(feats)` — cheap sanity check
  - `cosine_score(query, voxel_feats)` — inner product on L2-normalized inputs
"""

from __future__ import annotations

import torch


def assert_normalized(feats: torch.Tensor, atol: float = 1e-2) -> None:
    """Raise if any row is far from unit norm. Useful right before similarity."""
    n = feats.norm(dim=-1)
    mask = n > 0
    if not mask.any():
        return
    err = (n[mask] - 1.0).abs().max().item()
    if err > atol:
        raise ValueError(f"features not L2-normalized: max |‖f‖-1| = {err}")


def cosine_score(query: torch.Tensor, voxel_feats: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between a (D,) query and a (..., D) voxel feature tensor.

    Assumes both already L2-normalized. We do **not** renormalize — that is the
    caller's responsibility so this function stays hot and cheap.
    """
    if query.shape[-1] != voxel_feats.shape[-1]:
        raise ValueError(
            f"dim mismatch: query D={query.shape[-1]}, voxels D={voxel_feats.shape[-1]}"
        )
    flat = voxel_feats.reshape(-1, voxel_feats.shape[-1])
    scores = flat @ query
    return scores.view(voxel_feats.shape[:-1])
