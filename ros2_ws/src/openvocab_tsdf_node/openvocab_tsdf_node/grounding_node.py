"""ROS 2 grounding node.

Design summary:
  - Loads a pre-built feature map (npz) on startup. Live mapping is Phase 5b.
  - Holds the OpenCLIP text encoder in memory.
  - Exposes `/openvocab/ground` (GroundText.srv). Incoming queries trigger a
    text-embed + voxel-score + cluster pass, then return ranked targets.
  - Optional periodic publish of a coarse occupancy marker on
    `/openvocab/voxel_map_markers` for RViz debugging.

Intentionally thin. All heavy lifting lives in the pure-Python pipeline at
`openvocab_tsdf.pipeline`, so this file is easy to replace or mock in tests.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import rclpy
import torch
from geometry_msgs.msg import Point
from openvocab_tsdf_msgs.msg import GroundingTarget
from openvocab_tsdf_msgs.srv import GroundText
from rclpy.node import Node


class GroundingNode(Node):
    def __init__(self) -> None:
        super().__init__("openvocab_grounding")

        self.declare_parameter("map_path", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("model", "ViT-B-16")
        self.declare_parameter("pretrained", "laion2b_s34b_b88k")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("dtype", "fp16")
        self.declare_parameter("default_top_k", 5)
        self.declare_parameter("default_score_threshold", 0.22)
        self.declare_parameter("default_top_percentile", 0.02)

        map_path = self.get_parameter("map_path").value
        if not map_path or not Path(map_path).is_file():
            raise RuntimeError(f"map_path parameter must point to an existing npz: {map_path!r}")

        # Lazy import to keep import-time light for dry ROS 2 checks.
        from openvocab_tsdf.semantics.openclip_encoder import (
            OpenCLIPConfig,
            OpenCLIPEncoder,
        )

        self.get_logger().info(f"loading map: {map_path}")
        import numpy as np

        data = np.load(map_path, allow_pickle=True)
        self.feat = torch.from_numpy(data["feat"]).to(self.get_parameter("device").value)
        self.weight = torch.from_numpy(data["weight"]).to(self.get_parameter("device").value)
        self.origin = data["origin"]
        self.voxel_size = float(data["voxel_size"])

        self.encoder = OpenCLIPEncoder(
            OpenCLIPConfig(
                model=self.get_parameter("model").value,
                pretrained=self.get_parameter("pretrained").value,
                device=self.get_parameter("device").value,
                dtype=self.get_parameter("dtype").value,
            )
        )
        self.get_logger().info(
            f"encoder ready: {self.encoder.cfg.model} / {self.encoder.cfg.pretrained}"
        )

        self.srv = self.create_service(GroundText, "/openvocab/ground", self._on_ground)
        self.get_logger().info("service /openvocab/ground is up")

    def _on_ground(self, req: GroundText.Request, rsp: GroundText.Response) -> GroundText.Response:
        from openvocab_tsdf.grounding.query import rank_query

        score_threshold: float | None = (
            req.score_threshold if not math.isnan(req.score_threshold) else None
        )
        top_percentile: float | None = (
            req.top_percentile if not math.isnan(req.top_percentile) else None
        )
        top_k = int(req.top_k) if req.top_k > 0 else int(self.get_parameter("default_top_k").value)
        if score_threshold is None and top_percentile is None:
            # fall back to node defaults
            if self.get_parameter("default_top_percentile").value is not None:
                top_percentile = float(self.get_parameter("default_top_percentile").value)
            else:
                score_threshold = float(self.get_parameter("default_score_threshold").value)

        q = self.encoder.encode_texts([req.query])[0]
        results = rank_query(
            voxel_feats=self.feat,
            voxel_weights=self.weight,
            text_embedding=q,
            origin=self.origin,
            voxel_size=self.voxel_size,
            min_weight=1.0,
            score_threshold=score_threshold,
            top_percentile=top_percentile,
            cluster_eps_vox=2,
            min_cluster_voxels=8,
            top_k=top_k,
        )

        rsp.header.stamp = self.get_clock().now().to_msg()
        rsp.header.frame_id = str(self.get_parameter("map_frame").value)
        rsp.targets = []
        for r in results:
            t = GroundingTarget()
            t.center = Point(x=float(r.center_m[0]), y=float(r.center_m[1]), z=float(r.center_m[2]))
            t.bbox_min = Point(
                x=float(r.bbox_min_m[0]), y=float(r.bbox_min_m[1]), z=float(r.bbox_min_m[2])
            )
            t.bbox_max = Point(
                x=float(r.bbox_max_m[0]), y=float(r.bbox_max_m[1]), z=float(r.bbox_max_m[2])
            )
            t.score = float(r.score)
            t.voxel_count = int(r.voxel_count)
            rsp.targets.append(t)
        rsp.diagnostic = ""
        return rsp


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args if args is not None else os.sys.argv[1:])
    node = GroundingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
