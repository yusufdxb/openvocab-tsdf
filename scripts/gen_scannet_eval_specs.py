"""Generate grounding eval specs from ScanNet v2 annotations.

Reads a scene's aggregation JSON + segmented PLY to extract per-instance
3D bounding boxes and NYU40 labels. Outputs a YAML eval spec compatible
with eval/eval_grounding.py.

Usage:

    python scripts/gen_scannet_eval_specs.py \
        --root ~/data/scannet \
        --scene scene0011_00 \
        --out eval/specs/scannet_scene0011_00.yaml

    # Batch: generate specs for multiple scenes
    python scripts/gen_scannet_eval_specs.py \
        --root ~/data/scannet \
        --scenes scene0011_00 scene0050_00 scene0231_00 \
        --out-dir eval/specs/

Requires: plyfile, numpy, pyyaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

# NYU40 label ID -> natural language query.
# Only grounding-friendly labels (objects you'd point at with language).
NYU40_QUERIES: dict[int, str] = {
    3: "a cabinet",
    4: "a bed",
    5: "a chair",
    6: "a sofa",
    7: "a table",
    8: "a door",
    9: "a window",
    10: "a bookshelf",
    11: "a picture",
    12: "a counter",
    14: "a desk",
    15: "a curtain",
    16: "a refrigerator",
    18: "a toilet",
    19: "a sink",
    20: "a bathtub",
    23: "a pillow",
    24: "a mirror",
    25: "a towel",
    27: "a television",
    33: "a lamp",
    34: "a bag",
    36: "a dresser",
    39: "a shelf",
}

# Structural queries derived from mesh z-histogram (added for every scene).
STRUCTURAL_QUERIES = ["the floor", "the ceiling"]


def _load_ply_vertices(ply_path: Path) -> np.ndarray:
    """Load vertex positions from a PLY file. Returns (N, 3) float32."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("pip install plyfile") from None

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)


def _load_segments(scene_dir: Path, scene: str) -> np.ndarray:
    """Load per-vertex segment IDs from the segmentation JSON."""
    seg_path = scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json"
    if not seg_path.exists():
        raise FileNotFoundError(f"missing segmentation file: {seg_path}")
    with seg_path.open() as f:
        seg_data = json.load(f)
    return np.array(seg_data["segIndices"], dtype=np.int64)


def _load_aggregation(scene_dir: Path, scene: str) -> list[dict]:
    """Load instance aggregation: segment groups with NYU40 labels."""
    agg_path = scene_dir / f"{scene}_vh_clean.aggregation.json"
    if not agg_path.exists():
        agg_path = scene_dir / f"{scene}.aggregation.json"
    if not agg_path.exists():
        raise FileNotFoundError(f"missing aggregation file in {scene_dir}")
    with agg_path.open() as f:
        agg_data = json.load(f)
    return agg_data["segGroups"]


def _nyu40_id_from_label(label: str) -> int | None:
    """Map a ScanNet raw label string to its NYU40 ID."""
    try:
        from openvocab_tsdf.data._scannet_labels import RAW_TO_NYU40
    except ImportError:
        pass
    else:
        return RAW_TO_NYU40.get(label.strip().lower())

    # Fallback: check the aggregation's own id field if available.
    return None


def generate_spec(
    root: Path,
    scene: str,
    *,
    bbox_slack_m: float = 0.2,
    min_vertices: int = 100,
) -> dict:
    """Build an eval spec dict for one ScanNet scene."""
    scene_dir = root / "scans" / scene

    mesh_path = scene_dir / f"{scene}_vh_clean_2.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"missing mesh: {mesh_path}")

    vertices = _load_ply_vertices(mesh_path)
    seg_ids = _load_segments(scene_dir, scene)
    seg_groups = _load_aggregation(scene_dir, scene)

    if len(seg_ids) != len(vertices):
        raise ValueError(f"segment count ({len(seg_ids)}) != vertex count ({len(vertices)})")

    cases: list[dict] = []
    seen_labels: set[str] = set()

    for group in seg_groups:
        nyu_id = group.get("label_id") or group.get("nyu40id")

        if nyu_id is not None:
            nyu_id = int(nyu_id)
        if nyu_id not in NYU40_QUERIES:
            continue

        query = NYU40_QUERIES[nyu_id]
        if query in seen_labels:
            # Multiple instances: disambiguate with instance index.
            inst_idx = sum(1 for q in seen_labels if q.startswith(query.split(" (")[0]))
            query_key = f"{query} (instance {inst_idx + 1})"
        else:
            query_key = query
        seen_labels.add(query_key)

        segments = set(group.get("segments", []))
        mask = np.isin(seg_ids, list(segments))
        instance_verts = vertices[mask]

        if len(instance_verts) < min_vertices:
            continue

        bbox_min = instance_verts.min(axis=0) - bbox_slack_m
        bbox_max = instance_verts.max(axis=0) + bbox_slack_m

        cases.append(
            {
                "query": query_key,
                "bbox_min": [round(float(x), 3) for x in bbox_min],
                "bbox_max": [round(float(x), 3) for x in bbox_max],
                "kind": "object",
                "nyu40_id": nyu_id,
                "n_vertices": int(len(instance_verts)),
            }
        )

    # Structural queries from z-histogram.
    z_vals = vertices[:, 2]
    xy_min = vertices[:, :2].min(axis=0)
    xy_max = vertices[:, :2].max(axis=0)

    floor_z = float(np.percentile(z_vals, 2))
    ceil_z = float(np.percentile(z_vals, 98))

    cases.insert(
        0,
        {
            "query": "the floor",
            "bbox_min": [
                round(float(xy_min[0]) - 0.1, 3),
                round(float(xy_min[1]) - 0.1, 3),
                round(floor_z - 0.15, 3),
            ],
            "bbox_max": [
                round(float(xy_max[0]) + 0.1, 3),
                round(float(xy_max[1]) + 0.1, 3),
                round(floor_z + 0.15, 3),
            ],
            "kind": "structural",
        },
    )
    cases.insert(
        1,
        {
            "query": "the ceiling",
            "bbox_min": [
                round(float(xy_min[0]) - 0.1, 3),
                round(float(xy_min[1]) - 0.1, 3),
                round(ceil_z - 0.15, 3),
            ],
            "bbox_max": [
                round(float(xy_max[0]) + 0.1, 3),
                round(float(xy_max[1]) + 0.1, 3),
                round(ceil_z + 0.15, 3),
            ],
            "kind": "structural",
        },
    )

    return {
        "model": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "device": "cuda:0",
        "dtype": "fp16",
        "top_percentile": 0.005,
        "cluster_eps_vox": 2,
        "min_cluster_voxels": 20,
        "top_k": 5,
        "bbox_slack_m": bbox_slack_m,
        "cases": cases,
    }


def _write_spec(spec: dict, out_path: Path, scene: str) -> None:
    header = (
        f"# ScanNet v2 {scene} — auto-generated ground-truth eval spec.\n"
        f"#\n"
        f"# Generated by scripts/gen_scannet_eval_specs.py from the scene's\n"
        f"# aggregation JSON + segmented PLY mesh. Bounding boxes are per-instance\n"
        f"# vertex extrema widened by bbox_slack_m.\n"
        f"#\n"
        f"# {len(spec['cases'])} cases ({sum(1 for c in spec['cases'] if c.get('kind') == 'structural')} structural + "
        f"{sum(1 for c in spec['cases'] if c.get('kind') == 'object')} object).\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(spec, f, default_flow_style=None, sort_keys=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ScanNet eval specs")
    p.add_argument("--root", type=Path, required=True, help="ScanNet root (parent of scans/)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", type=str, help="Single scene ID")
    g.add_argument("--scenes", nargs="+", type=str, help="Multiple scene IDs")
    p.add_argument("--out", type=Path, help="Output YAML path (single scene)")
    p.add_argument("--out-dir", type=Path, help="Output directory (batch mode)")
    p.add_argument("--bbox-slack-m", type=float, default=0.2)
    p.add_argument("--min-vertices", type=int, default=100)
    args = p.parse_args()

    scenes = [args.scene] if args.scene else args.scenes

    for scene in scenes:
        spec = generate_spec(
            args.root,
            scene,
            bbox_slack_m=args.bbox_slack_m,
            min_vertices=args.min_vertices,
        )
        if args.out and len(scenes) == 1:
            out_path = args.out
        elif args.out_dir:
            out_path = args.out_dir / f"scannet_{scene}.yaml"
        else:
            out_path = Path(f"eval/specs/scannet_{scene}.yaml")

        _write_spec(spec, out_path, scene)
        n_obj = sum(1 for c in spec["cases"] if c.get("kind") == "object")
        print(f"{scene}: {n_obj} object queries + 2 structural → {out_path}")


if __name__ == "__main__":
    main()
