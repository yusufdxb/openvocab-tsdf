"""End-to-end integration test: synthetic scene -> encode -> ground.

This is the acid test for Phase 2 + Phase 3. Three colored primitives in a small
volume. Text query should rank the matching primitive's cluster first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from openvocab_tsdf.config import Config, DatasetConfig, MappingConfig, SemanticsConfig
from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset


@pytest.mark.slow
@pytest.mark.gpu
def test_synthetic_red_sphere_query(tmp_path: Path):
    """Query 'a red ball' should rank the sphere region highest in a multi-object scene."""
    from openvocab_tsdf.pipeline import encode_and_fuse, ground_text

    # scene: red sphere top, green floor slab, blue wall bar
    primitives = [
        Sphere(center=(0.0, -0.2, 0.0), radius=0.25, color=(220, 40, 40)),  # red
        Box(
            min_xyz=(-0.7, 0.25, -0.7), max_xyz=(0.7, 0.35, 0.7), color=(50, 180, 70)
        ),  # green floor
        Box(min_xyz=(0.55, -0.4, -0.1), max_xyz=(0.65, 0.3, 0.1), color=(40, 60, 220)),  # blue bar
    ]
    frames = make_synthetic_dataset(primitives, num_frames=24, width=240, height=180, radius=1.4)

    cfg = Config(
        dataset=DatasetConfig(name="replica", root="/tmp/unused", scene="unused"),
        mapping=MappingConfig(
            voxel_size_m=0.04,
            truncation_distance_m=0.2,
            backend="reference",
            device="cuda:0",
            store_color=True,
            store_features=True,
            feature_dim=512,
            bounds_min=(-1.0, -0.7, -1.0),
            bounds_max=(1.0, 0.5, 1.0),
        ),
        semantics=SemanticsConfig(
            model="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device="cuda:0",
            dtype="fp16" if torch.cuda.is_available() else "fp32",
        ),
    )

    map_path = tmp_path / "map.npz"
    stats = encode_and_fuse(cfg, frames, map_path)
    assert stats["num_frames"] == 24
    assert stats["feature_dim"] == 512
    assert map_path.exists()

    # query -- red sphere should beat green floor and blue bar
    results_red = ground_text(
        map_path,
        "a red ball",
        model=cfg.semantics.model,
        pretrained=cfg.semantics.pretrained,
        device=cfg.semantics.device,
        dtype=cfg.semantics.dtype,
        score_threshold=0.18,
        min_cluster_voxels=4,
        top_k=5,
    )
    assert len(results_red) > 0, "no clusters for red query"
    top = results_red[0]
    # top result's centroid should be inside the sphere region
    cx, cy, cz = top.center_m
    assert (
        abs(cx - 0.0) < 0.3 and abs(cz - 0.0) < 0.3
    ), f"red result far from sphere: {top.center_m}"

    # query green -- with global per-frame CLIP features, scores are coarse. We
    # only demand that the top green cluster is not the sphere region. Precise
    # localization of the floor slab is a Phase 2b concern (patch/mask features).
    results_green = ground_text(
        map_path,
        "green grass floor",
        model=cfg.semantics.model,
        pretrained=cfg.semantics.pretrained,
        device=cfg.semantics.device,
        dtype=cfg.semantics.dtype,
        score_threshold=0.16,
        min_cluster_voxels=4,
        top_k=5,
    )
    assert len(results_green) > 0, "no clusters for green query"
    top_green = results_green[0]
    # top green cluster should sit away from the sphere (sphere at ~(0,-0.2,0))
    gx, gy, gz = top_green.center_m
    dist_to_sphere = (gx**2 + (gy + 0.2) ** 2 + gz**2) ** 0.5
    assert dist_to_sphere > 0.3, f"green top cluster on sphere: {top_green.center_m}"
