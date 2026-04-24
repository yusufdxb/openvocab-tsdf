"""Tests for the ScanNet eval spec generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

try:
    from plyfile import PlyData, PlyElement

    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gen_scannet_eval_specs.py"


def _import_gen_spec():
    spec = importlib.util.spec_from_file_location("gen_scannet_eval_specs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fake_scannet_annotations(root: Path, scene: str) -> None:
    """Create minimal ScanNet annotations: segmented PLY + aggregation JSON + segs JSON."""
    scene_dir = root / "scans" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(99)
    n_verts = 500

    # Create vertices: floor cluster at z~0, ceiling at z~2.5, a chair cluster, a table cluster
    verts = np.zeros((n_verts, 3), dtype=np.float32)
    seg_ids = np.zeros(n_verts, dtype=np.int32)

    # Floor: z ~ 0, segments 0-2
    verts[:100] = rng.uniform([-2, -2, -0.05], [2, 2, 0.05], (100, 3)).astype(np.float32)
    seg_ids[:100] = rng.choice([0, 1, 2], 100)

    # Ceiling: z ~ 2.5, segments 3-5
    verts[100:200] = rng.uniform([-2, -2, 2.45], [2, 2, 2.55], (100, 3)).astype(np.float32)
    seg_ids[100:200] = rng.choice([3, 4, 5], 100)

    # Chair: cluster around (1, 0, 0.4), segments 6-8
    verts[200:350] = rng.uniform([0.5, -0.3, 0.0], [1.5, 0.3, 0.8], (150, 3)).astype(np.float32)
    seg_ids[200:350] = rng.choice([6, 7, 8], 150)

    # Table: cluster around (-1, 0, 0.35), segments 9-10
    verts[350:500] = rng.uniform([-1.5, -0.5, 0.0], [-0.5, 0.5, 0.7], (150, 3)).astype(np.float32)
    seg_ids[350:500] = rng.choice([9, 10], 150)

    # Write PLY
    if HAS_PLYFILE:
        vertex_data = np.array(
            [(v[0], v[1], v[2]) for v in verts],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el]).write(str(scene_dir / f"{scene}_vh_clean_2.ply"))

    # Write segmentation JSON
    seg_json = {"segIndices": seg_ids.tolist()}
    (scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(json.dumps(seg_json))

    # Write aggregation JSON
    agg_json = {
        "segGroups": [
            {"label": "chair", "label_id": 5, "segments": [6, 7, 8]},
            {"label": "table", "label_id": 7, "segments": [9, 10]},
        ]
    }
    (scene_dir / f"{scene}_vh_clean.aggregation.json").write_text(json.dumps(agg_json))


@pytest.mark.skipif(not HAS_PLYFILE, reason="plyfile not installed")
def test_gen_scannet_eval_spec(tmp_path: Path):
    mod = _import_gen_spec()
    generate_spec = mod.generate_spec

    scene = "scene_test_00"
    _make_fake_scannet_annotations(tmp_path, scene)

    spec = generate_spec(tmp_path, scene, bbox_slack_m=0.2, min_vertices=10)

    assert "cases" in spec
    assert len(spec["cases"]) >= 4  # floor, ceiling, chair, table

    queries = [c["query"] for c in spec["cases"]]
    assert "the floor" in queries
    assert "the ceiling" in queries
    assert "a chair" in queries
    assert "a table" in queries

    for case in spec["cases"]:
        assert "bbox_min" in case
        assert "bbox_max" in case
        assert len(case["bbox_min"]) == 3
        assert len(case["bbox_max"]) == 3
        for i in range(3):
            assert case["bbox_min"][i] < case["bbox_max"][i]


@pytest.mark.skipif(not HAS_PLYFILE, reason="plyfile not installed")
def test_gen_spec_filters_small_instances(tmp_path: Path):
    mod = _import_gen_spec()
    generate_spec = mod.generate_spec

    scene = "scene_small_00"
    _make_fake_scannet_annotations(tmp_path, scene)

    # With min_vertices=200, chair (150 verts) and table (150 verts) should be filtered.
    spec = generate_spec(tmp_path, scene, min_vertices=200)
    queries = [c["query"] for c in spec["cases"]]
    assert "a chair" not in queries
    assert "a table" not in queries
    # Structural queries should always be present.
    assert "the floor" in queries
    assert "the ceiling" in queries
