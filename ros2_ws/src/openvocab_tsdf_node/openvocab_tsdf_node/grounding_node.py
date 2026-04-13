"""ROS 2 grounding node (offline + live).

Modes (set via the `live_mode` ROS 2 parameter):

  - `live_mode: false` (default) — loads a precomputed feature map from
    `map_path` on startup and serves `/openvocab/ground` against it.
  - `live_mode: true` — creates an empty `ReferenceTSDF` sized by the
    `bounds_min` / `bounds_max` / `voxel_size_m` params, subscribes to
    color + depth + camera_info, looks up `map -> camera_frame` through TF
    for each frame, and integrates geometry + CLIP features online.
    The grounding service returns up-to-the-last-frame results without any
    extra load on the main callback loop.

Dependencies: `rclpy`, `cv_bridge`, `tf2_ros`, `message_filters`. The live
mapping path is skipped in offline mode so the common case stays light.

Intentionally thin: heavy lifting lives in `openvocab_tsdf.pipeline`.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point
from openvocab_tsdf_msgs.msg import GroundingTarget
from openvocab_tsdf_msgs.srv import GroundText
from rclpy.node import Node


class GroundingNode(Node):
    def __init__(self) -> None:
        super().__init__("openvocab_grounding")

        # ---- parameters ----
        self.declare_parameter("live_mode", False)
        self.declare_parameter("map_path", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("model", "ViT-B-16")
        self.declare_parameter("pretrained", "laion2b_s34b_b88k")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("dtype", "fp16")
        self.declare_parameter("default_top_k", 5)
        self.declare_parameter("default_score_threshold", 0.22)
        self.declare_parameter("default_top_percentile", 0.02)
        # live-mode-only
        self.declare_parameter("voxel_size_m", 0.06)
        self.declare_parameter("truncation_distance_m", 0.24)
        self.declare_parameter("bounds_min", [-2.0, -3.0, -2.0])
        self.declare_parameter("bounds_max", [7.0, 5.0, 3.0])
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("depth_trunc_m", 6.0)
        self.declare_parameter("frame_stride", 1)  # integrate every Nth frame
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/camera_info")
        self.declare_parameter("tf_timeout_s", 0.1)

        self._live_mode = bool(self.get_parameter("live_mode").value)
        self._device = str(self.get_parameter("device").value)
        self._dtype = str(self.get_parameter("dtype").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._camera_frame = str(self.get_parameter("camera_frame").value)

        # ---- encoder (both modes need it) ----
        from openvocab_tsdf.semantics.openclip_encoder import (
            OpenCLIPConfig,
            OpenCLIPEncoder,
        )

        self.encoder = OpenCLIPEncoder(
            OpenCLIPConfig(
                model=str(self.get_parameter("model").value),
                pretrained=str(self.get_parameter("pretrained").value),
                device=self._device,
                dtype=self._dtype,
            )
        )
        self.get_logger().info(
            f"encoder ready: {self.encoder.cfg.model} / {self.encoder.cfg.pretrained}"
        )

        # lock protecting the feature volume (accessed by subscriber callbacks
        # AND the query service callback)
        self._map_lock = threading.Lock()

        if self._live_mode:
            self._init_live_mode()
        else:
            self._init_offline_mode()

        self.srv = self.create_service(GroundText, "/openvocab/ground", self._on_ground)
        self.get_logger().info("service /openvocab/ground is up")

    # ----------------------------------------------------------- offline mode
    def _init_offline_mode(self) -> None:
        map_path = self.get_parameter("map_path").value
        if not map_path or not Path(map_path).is_file():
            raise RuntimeError(f"map_path must point to an existing npz: {map_path!r}")
        self.get_logger().info(f"offline mode: loading map {map_path}")
        data = np.load(map_path, allow_pickle=True)
        self.feat = torch.from_numpy(data["feat"]).to(self._device)
        self.weight = torch.from_numpy(data["weight"]).to(self._device)
        self.tsdf = (
            torch.from_numpy(data["tsdf"]).to(self._device) if "tsdf" in data.files else None
        )
        self.origin = data["origin"]
        self.voxel_size = float(data["voxel_size"])

    # ---------------------------------------------------------------- live mode
    def _init_live_mode(self) -> None:
        from message_filters import ApproximateTimeSynchronizer, Subscriber  # type: ignore
        from sensor_msgs.msg import CameraInfo, Image  # type: ignore
        from tf2_ros import Buffer, TransformListener  # type: ignore

        from openvocab_tsdf.mapping.reference import (
            ReferenceTSDF,
            ReferenceTSDFConfig,
        )

        bmin = tuple(float(x) for x in self.get_parameter("bounds_min").value)
        bmax = tuple(float(x) for x in self.get_parameter("bounds_max").value)
        vsz = float(self.get_parameter("voxel_size_m").value)
        trunc = float(self.get_parameter("truncation_distance_m").value)

        self._live_vol = ReferenceTSDF(
            ReferenceTSDFConfig(
                voxel_size_m=vsz,
                truncation_distance_m=trunc,
                bounds_min=bmin,
                bounds_max=bmax,
                store_color=True,
                store_features=True,
                feature_dim=self.encoder.feature_dim,
                device=self._device,
            )
        )
        # mirror the attributes the service uses in offline mode
        self.feat = self._live_vol.feat
        self.weight = self._live_vol.weight
        self.tsdf = self._live_vol.tsdf
        self.origin = self._live_vol.origin.detach().cpu().numpy()
        self.voxel_size = vsz

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        color_sub = Subscriber(self, Image, str(self.get_parameter("color_topic").value))
        depth_sub = Subscriber(self, Image, str(self.get_parameter("depth_topic").value))
        info_sub = Subscriber(self, CameraInfo, str(self.get_parameter("info_topic").value))
        self._sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10, slop=0.05
        )
        self._sync.registerCallback(self._on_live_frame)

        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._depth_trunc = float(self.get_parameter("depth_trunc_m").value)
        self._stride = max(1, int(self.get_parameter("frame_stride").value))
        self._tf_timeout = float(self.get_parameter("tf_timeout_s").value)
        self._frame_count = 0
        self._integrated_count = 0
        self._last_log_t = time.monotonic()

        self.get_logger().info(
            f"live mode: volume {self._live_vol.dims}, voxel={vsz} m, bounds={bmin}->{bmax}, "
            f"stride={self._stride}"
        )

    def _on_live_frame(self, color_msg, depth_msg, info_msg):  # type: ignore[no-untyped-def]
        self._frame_count += 1
        if self._frame_count % self._stride != 0:
            return

        # TF lookup at the color timestamp
        from openvocab_tsdf.data.base import CameraIntrinsics, RGBDFrame

        stamp = color_msg.header.stamp
        src = self._camera_frame or color_msg.header.frame_id or "camera"
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame,
                src,
                stamp,
                timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
            )
        except Exception as e:  # tf2 LookupException / ExtrapolationException
            self.get_logger().warn(f"tf lookup failed ({src}->{self._map_frame}): {e}")
            return

        T_wc = _transform_to_matrix(tf.transform)

        color = _image_msg_to_rgb8(color_msg)
        depth_m = _depth_msg_to_meters(depth_msg, self._depth_scale)
        depth_m[depth_m > self._depth_trunc] = 0.0

        H, W = depth_m.shape
        K = np.asarray(info_msg.k, dtype=np.float32).reshape(3, 3)
        intr = CameraIntrinsics.from_matrix(K, W, H)

        if color.shape[:2] != (H, W):
            from PIL import Image as PILImage

            color = np.asarray(
                PILImage.fromarray(color).resize((W, H), PILImage.BILINEAR),
                dtype=np.uint8,
            )

        frame = RGBDFrame(
            color=color,
            depth_m=depth_m,
            intrinsics=intr,
            T_wc=T_wc.astype(np.float32),
            timestamp=float(stamp.sec) + 1e-9 * float(stamp.nanosec),
            frame_id=self._frame_count,
        )

        # encode CLIP global feature (single image)
        feat = self.encoder.encode_images([color])[0]  # (D,)

        with self._map_lock:
            self._live_vol.integrate(frame, feature=feat)
            self._integrated_count += 1

        now = time.monotonic()
        if now - self._last_log_t > 5.0:
            self.get_logger().info(
                f"live: integrated {self._integrated_count} frames " f"(recv {self._frame_count})"
            )
            self._last_log_t = now

    # --------------------------------------------------------------- service
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
            if self.get_parameter("default_top_percentile").value is not None:
                top_percentile = float(self.get_parameter("default_top_percentile").value)
            else:
                score_threshold = float(self.get_parameter("default_score_threshold").value)

        q = self.encoder.encode_texts([req.query])[0]
        with self._map_lock:
            results = rank_query(
                voxel_feats=self.feat,
                voxel_weights=self.weight,
                voxel_tsdf=self.tsdf,
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
        rsp.header.frame_id = self._map_frame
        rsp.targets = []
        for r in results:
            t = GroundingTarget()
            t.center = Point(x=float(r.center_m[0]), y=float(r.center_m[1]), z=float(r.center_m[2]))
            t.bbox_min = Point(
                x=float(r.bbox_min_m[0]),
                y=float(r.bbox_min_m[1]),
                z=float(r.bbox_min_m[2]),
            )
            t.bbox_max = Point(
                x=float(r.bbox_max_m[0]),
                y=float(r.bbox_max_m[1]),
                z=float(r.bbox_max_m[2]),
            )
            t.score = float(r.score)
            t.voxel_count = int(r.voxel_count)
            rsp.targets.append(t)
        mode = "live" if self._live_mode else "offline"
        extra = ""
        if self._live_mode:
            extra = f"; integrated={self._integrated_count}"
        rsp.diagnostic = f"mode={mode}{extra}"
        return rsp


def _image_msg_to_rgb8(msg) -> np.ndarray:  # type: ignore[no-untyped-def]
    """sensor_msgs/Image → (H, W, 3) uint8 RGB, bypassing cv_bridge."""
    H, W = int(msg.height), int(msg.width)
    enc = msg.encoding
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc == "rgb8":
        return buf.reshape(H, W, 3).copy()
    if enc == "bgr8":
        arr = buf.reshape(H, W, 3).copy()
        return arr[..., ::-1].copy()
    if enc == "mono8":
        g = buf.reshape(H, W)
        return np.stack([g, g, g], axis=-1)
    raise ValueError(f"unsupported color encoding: {enc!r}")


def _depth_msg_to_meters(msg, fallback_scale: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    """sensor_msgs/Image depth → (H, W) float32 meters."""
    H, W = int(msg.height), int(msg.width)
    enc = msg.encoding
    if enc == "32FC1":
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(H, W)
        return arr.astype(np.float32, copy=True)
    if enc == "16UC1":
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(H, W)
        return arr.astype(np.float32) / float(fallback_scale)
    raise ValueError(f"unsupported depth encoding: {enc!r}")


def _transform_to_matrix(t) -> np.ndarray:  # type: ignore[no-untyped-def]
    """geometry_msgs/Transform (translation + rotation quaternion) → 4×4."""
    q = t.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # Quaternion → 3×3 rotation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


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
