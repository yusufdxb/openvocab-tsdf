"""Per-query voxel heatmap → colored PLY.

Loads a saved map, scores surface voxels against a text query, colors them by
score, and writes a point-cloud PLY ready to open in MeshLab / Open3D / RViz.
Use alongside the mesh PLY written by `fuse` to visually inspect grounding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from openvocab_tsdf.mapping.base import Mesh
from openvocab_tsdf.viz.mesh import save_ply


def _viridis_like(x: np.ndarray) -> np.ndarray:
    """Tiny 5-stop perceptual colormap, avoids a matplotlib hard dep.

    Input: (N,) float in [0, 1]. Output: (N, 3) uint8.
    """
    stops = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    x = np.clip(x, 0.0, 1.0)
    idx_f = x * (len(stops) - 1)
    lo = np.floor(idx_f).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (idx_f - lo)[:, None]
    return (stops[lo] * (1 - frac) + stops[hi] * frac).astype(np.uint8)


def save_query_heatmap_ply(
    map_path: str | Path,
    query: str,
    out_path: str | Path,
    *,
    model: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda:0",
    dtype: str = "fp16",
    min_weight: float = 1.0,
    surface_tsdf_abs_max: float = 0.5,
    percentile_low: float = 0.5,
    percentile_high: float = 0.99,
) -> dict:
    """Write a colored point-cloud PLY heatmap for `query` on the saved map.

    Scores below `percentile_low` of observed-surface scores are colored at
    the low end of the colormap; above `percentile_high` are saturated. This
    stretches the useful color range even when CLIP similarity magnitudes are
    narrow.
    """
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    data = np.load(map_path, allow_pickle=True)
    feat = torch.from_numpy(data["feat"]).to(device)
    weight = torch.from_numpy(data["weight"]).to(device)
    tsdf = torch.from_numpy(data["tsdf"]).to(device)
    origin = data["origin"].astype(np.float32)
    voxel_size = float(data["voxel_size"])

    enc = OpenCLIPEncoder(
        OpenCLIPConfig(model=model, pretrained=pretrained, device=device, dtype=dtype)
    )
    q = enc.encode_texts([query])[0]
    D = feat.shape[-1]
    scores = (feat.reshape(-1, D) @ q).reshape(feat.shape[:3])

    surface = (weight >= min_weight) & (tsdf.abs() <= surface_tsdf_abs_max)
    if not surface.any():
        raise RuntimeError("no surface voxels observed in this map")

    idx = torch.stack(torch.where(surface), dim=-1).cpu().numpy().astype(np.float32)  # (N, 3)
    centers = origin.reshape(1, 3) + (idx + 0.5) * voxel_size
    s = scores[surface].float().cpu().numpy()

    lo = float(np.quantile(s, percentile_low))
    hi = float(np.quantile(s, percentile_high))
    if hi <= lo:
        hi = lo + 1e-6
    s_norm = (s - lo) / (hi - lo)
    colors = _viridis_like(s_norm)

    # trim to surface voxels only; write as a "mesh" with zero triangles so
    # the existing save_ply handles point clouds via vertex-only output.
    mesh = Mesh(
        vertices=centers.astype(np.float32),
        triangles=np.zeros((0, 3), dtype=np.int32),
        vertex_colors=colors,
    )
    save_ply(mesh, out_path)

    return {
        "query": query,
        "num_points": int(centers.shape[0]),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "score_p50": float(np.quantile(s, 0.5)),
        "score_p99": float(np.quantile(s, 0.99)),
        "percentile_range": [lo, hi],
        "out_path": str(out_path),
    }
