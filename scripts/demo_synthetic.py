"""End-to-end synthetic demo: render → encode → query → print ranked targets.

Runs offline (no external datasets). Good showpiece and a runnable smoke test
for the full pipeline.

    python scripts/demo_synthetic.py
    python scripts/demo_synthetic.py --query "a blue bar"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from openvocab_tsdf.config import Config, DatasetConfig, MappingConfig, SemanticsConfig
from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset
from openvocab_tsdf.pipeline import encode_and_fuse, ground_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, default=Path("outputs/demo_map.npz"))
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--voxel-size", type=float, default=0.04)
    p.add_argument(
        "--query",
        action="append",
        help="query to run (may be passed multiple times)",
    )
    args = p.parse_args()

    if not args.query:
        args.query = ["a red ball", "a blue bar", "green grass floor"]

    primitives = [
        Sphere(center=(0.0, -0.2, 0.0), radius=0.25, color=(220, 40, 40)),
        Box(min_xyz=(-0.7, 0.25, -0.7), max_xyz=(0.7, 0.35, 0.7), color=(50, 180, 70)),
        Box(min_xyz=(0.55, -0.4, -0.1), max_xyz=(0.65, 0.3, 0.1), color=(40, 60, 220)),
    ]
    frames = make_synthetic_dataset(
        primitives, num_frames=args.frames, width=args.width, height=args.height, radius=1.4
    )

    cfg = Config(
        dataset=DatasetConfig(name="replica", root="/tmp/unused", scene="unused"),
        mapping=MappingConfig(
            voxel_size_m=args.voxel_size,
            truncation_distance_m=5 * args.voxel_size,
            backend="reference",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            store_features=True,
            feature_dim=512,
            bounds_min=(-1.0, -0.7, -1.0),
            bounds_max=(1.0, 0.5, 1.0),
        ),
        semantics=SemanticsConfig(
            model="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype="fp16" if torch.cuda.is_available() else "fp32",
        ),
    )

    args.map.parent.mkdir(parents=True, exist_ok=True)
    stats = encode_and_fuse(cfg, frames, args.map)
    console = Console()
    console.print(
        f"[green]encoded[/green] {stats['num_frames']} frames "
        f"(enc {stats['encode_s']:.2f}s, fuse {stats['fuse_s']:.2f}s) -> {args.map}"
    )

    for q in args.query:
        results = ground_text(
            args.map,
            q,
            model=cfg.semantics.model,
            pretrained=cfg.semantics.pretrained,
            device=cfg.semantics.device,
            dtype=cfg.semantics.dtype,
            score_threshold=None,
            top_percentile=0.02,  # top 2% of observed voxels per query
            cluster_eps_vox=1,
            min_cluster_voxels=8,
            top_k=3,
        )
        table = Table(title=f"query: {q!r}")
        table.add_column("#")
        table.add_column("center")
        table.add_column("score", justify="right")
        table.add_column("voxels", justify="right")
        for i, r in enumerate(results):
            cx, cy, cz = r.center_m
            table.add_row(
                str(i + 1),
                f"({cx:+.2f}, {cy:+.2f}, {cz:+.2f})",
                f"{r.score:.3f}",
                str(r.voxel_count),
            )
        console.print(table)


if __name__ == "__main__":
    main()
