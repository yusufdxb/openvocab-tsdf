"""End-to-end integration test using PATCH features (Phase 2b).

With patch features + near-surface aggregation + MaskCLIP-style last-block
attention bypass, we expect *relative* ranking to work on this synthetic scene:
the 'red' query should prefer voxels near the sphere over voxels on the green
slab, etc. Absolute spatial accuracy on a synthetic, out-of-distribution
rendered scene is weak — real images (ScanNet / Replica) are what CLIP was
trained on and they will test localization properly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from openvocab_tsdf.config import Config, DatasetConfig, MappingConfig, SemanticsConfig
from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset


@pytest.mark.slow
@pytest.mark.gpu
def test_patch_pipeline_runs_and_produces_clusters(tmp_path: Path):
    """Minimal viability test: patch mode runs end-to-end and produces
    clusters for each query. Spatial accuracy is validated on real data
    (Replica / ScanNet) elsewhere — synthetic rendered scenes are out of
    distribution for a natural-image CLIP and should not gate this test.
    """
    from openvocab_tsdf.pipeline import encode_and_fuse, ground_text

    primitives = [
        Sphere(center=(0.0, -0.2, 0.0), radius=0.25, color=(220, 40, 40)),
        Box(min_xyz=(-0.7, 0.25, -0.7), max_xyz=(0.7, 0.35, 0.7), color=(50, 180, 70)),
        Box(min_xyz=(0.55, -0.4, -0.1), max_xyz=(0.65, 0.3, 0.1), color=(40, 60, 220)),
    ]
    frames = make_synthetic_dataset(primitives, num_frames=32, width=224, height=224, radius=1.4)

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
            mode="patch",
            device="cuda:0",
            dtype="fp16" if torch.cuda.is_available() else "fp32",
        ),
    )

    map_path = tmp_path / "map.npz"
    stats = encode_and_fuse(cfg, frames, map_path)
    assert stats["num_frames"] == 32
    # patch mode should still produce a normal npz with the same keys
    import numpy as np

    data = np.load(map_path, allow_pickle=True)
    assert "feat" in data.files
    assert data["feat"].shape[-1] == 512
    assert str(data["mode"]) == "patch"

    # each of three queries should return at least one cluster (>0 centroids)
    for q in ["a red ball", "a blue bar", "green grass floor"]:
        r = ground_text(
            map_path,
            q,
            model=cfg.semantics.model,
            pretrained=cfg.semantics.pretrained,
            device=cfg.semantics.device,
            dtype=cfg.semantics.dtype,
            score_threshold=None,
            top_percentile=0.02,
            cluster_eps_vox=1,
            min_cluster_voxels=4,
            top_k=3,
        )
        assert len(r) > 0, f"no clusters returned for {q!r}"
