"""Text-to-3D grounding query engine.

Input: a map with per-voxel CLIP features, a text query string.
Output: ranked 3D targets (centroid, bbox, score, #voxels).

Pipeline:
  1. Encode query text with CLIP text encoder → q ∈ ℝ^D, L2-normalized.
  2. Score voxel features against q by inner product (cosine, features assumed
     normalized).
  3. Mask by (weight >= min_weight) and (score >= score_threshold).
  4. Connected-component clustering in voxel space with a Chebyshev radius.
  5. For each cluster: centroid, bbox, mean score, voxel count.
  6. Rank clusters by (mean_score × log(count)) — this biases toward both
     confident AND non-trivially-sized regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from openvocab_tsdf.semantics.aggregation import cosine_score


@dataclass(frozen=True)
class GroundingResult:
    center_m: tuple[float, float, float]
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    score: float
    voxel_count: int


def _connected_components_3d(mask: np.ndarray, eps_vox: int = 1) -> np.ndarray:
    """3D connected components. `eps_vox > 1` dilates first to merge near-neighbors."""
    from scipy.ndimage import binary_dilation, label

    if eps_vox > 1:
        struct1 = np.ones((3, 3, 3), dtype=bool)
        mask = binary_dilation(mask, structure=struct1, iterations=eps_vox - 1)
    struct = np.ones((3, 3, 3), dtype=bool)  # 26-connected
    labels, _ = label(mask, structure=struct)
    return labels.astype(np.int32)


def rank_query(
    *,
    voxel_feats: torch.Tensor,  # (Nx, Ny, Nz, D)
    voxel_weights: torch.Tensor,  # same leading shape
    text_embedding: torch.Tensor,  # (D,), normalized
    origin: np.ndarray,  # (3,)
    voxel_size: float,
    voxel_tsdf: torch.Tensor | None = None,  # (Nx, Ny, Nz) — for surface filtering
    min_weight: float = 1.0,
    score_threshold: float | None = 0.22,
    top_percentile: float | None = None,
    surface_only: bool = True,
    surface_tsdf_abs_max: float = 0.5,
    cluster_eps_vox: int = 2,
    min_cluster_voxels: int = 8,
    top_k: int = 5,
) -> list[GroundingResult]:
    """Return the top-k clusters matching the text embedding.

    Scoring mode:
      - If `top_percentile` is set (e.g., 0.02 for top 2%), a dynamic threshold
        is picked per scene: only voxels in the top percentile of observed
        scores pass. This adapts to different queries / scenes where absolute
        CLIP similarity magnitudes vary.
      - Otherwise, `score_threshold` is used as a fixed cut-off.
    """
    if voxel_feats.ndim != 4:
        raise ValueError(f"expected (Nx,Ny,Nz,D), got {tuple(voxel_feats.shape)}")

    with torch.no_grad():
        scores = cosine_score(text_embedding, voxel_feats)  # (Nx,Ny,Nz)
        observed = voxel_weights >= min_weight
        if surface_only and voxel_tsdf is not None:
            observed = observed & (voxel_tsdf.abs() <= surface_tsdf_abs_max)
        if top_percentile is not None:
            observed_scores = scores[observed]
            if observed_scores.numel() == 0:
                return []
            q = 1.0 - float(top_percentile)
            thr = torch.quantile(observed_scores, q).item()
        elif score_threshold is not None:
            thr = float(score_threshold)
        else:
            raise ValueError("either score_threshold or top_percentile must be set")
        mask = observed & (scores >= thr)

    scores_np = scores.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    labels = _connected_components_3d(mask_np, eps_vox=cluster_eps_vox)

    Nx, Ny, Nz = scores_np.shape
    results: list[GroundingResult] = []
    max_label = int(labels.max())
    for lbl in range(1, max_label + 1):
        cluster_mask = labels == lbl
        count = int(cluster_mask.sum())
        if count < min_cluster_voxels:
            continue
        idx = np.argwhere(cluster_mask).astype(np.float32)  # (C, 3)
        centers = origin.reshape(1, 3) + (idx + 0.5) * voxel_size
        c_min = centers.min(axis=0)
        c_max = centers.max(axis=0)
        centroid = centers.mean(axis=0)
        mean_score = float(scores_np[cluster_mask].mean())
        results.append(
            GroundingResult(
                center_m=tuple(centroid.astype(float)),
                bbox_min_m=tuple(c_min.astype(float)),
                bbox_max_m=tuple(c_max.astype(float)),
                score=mean_score,
                voxel_count=count,
            )
        )

    results.sort(key=lambda r: r.score * np.log1p(r.voxel_count), reverse=True)
    return results[:top_k]
