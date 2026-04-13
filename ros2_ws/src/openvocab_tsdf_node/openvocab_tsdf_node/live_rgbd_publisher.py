"""Replay an RGBDDataset onto ROS 2 topics at a fixed rate.

Publishes:
  - /camera/color/image_raw        sensor_msgs/Image (rgb8)
  - /camera/depth/image_raw        sensor_msgs/Image (32FC1, depth in meters)
  - /camera/camera_info            sensor_msgs/CameraInfo
  - /tf                            geometry_msgs/TransformStamped (map -> camera frame)

Used for the live-mode smoke test: `grounding_node` in `live_mode:=true` can be
fed this publisher's topics and build a full map from scratch without any
actual camera hardware.

    ros2 run openvocab_tsdf_node live_rgbd_publisher --ros-args \\
        -p dataset:=replica -p root:=$HOME/data/replica/Replica -p scene:=room0 \\
        -p rate_hz:=30.0 -p max_frames:=200
"""

from __future__ import annotations

import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node


def _matrix_to_transform(
    T: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """4×4 matrix → (translation, quaternion xyzw)."""
    R = T[:3, :3]
    t = T[:3, 3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / float(np.sqrt(tr + 1.0))
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (float(t[0]), float(t[1]), float(t[2])), (float(x), float(y), float(z), float(w))


class LiveRGBDPublisher(Node):
    def __init__(self) -> None:
        super().__init__("live_rgbd_publisher")

        self.declare_parameter("dataset", "nice_slam_demo")
        self.declare_parameter("root", os.path.expanduser("~/data/replica"))
        self.declare_parameter("scene", "Demo")
        self.declare_parameter("stride", 1)
        self.declare_parameter("max_frames", 500)
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("depth_scale_out", 1000.0)  # depth_scale the receiving node uses
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("depth_trunc_m", 6.0)

        from sensor_msgs.msg import CameraInfo, Image  # type: ignore
        from tf2_ros import TransformBroadcaster  # type: ignore

        from openvocab_tsdf.config import CameraConfig, Config, DatasetConfig
        from openvocab_tsdf.pipeline import build_dataset

        cfg = Config(
            dataset=DatasetConfig(
                name=str(self.get_parameter("dataset").value),
                root=str(self.get_parameter("root").value),
                scene=str(self.get_parameter("scene").value),
                max_frames=int(self.get_parameter("max_frames").value),
                stride=int(self.get_parameter("stride").value),
            ),
            camera=CameraConfig(
                depth_scale=1000.0,  # for dataset loader; we emit fresh depth_m
                depth_trunc_m=float(self.get_parameter("depth_trunc_m").value),
            ),
        )
        self._dataset = build_dataset(cfg)
        self._depth_scale_out = float(self.get_parameter("depth_scale_out").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._camera_frame = str(self.get_parameter("camera_frame").value)

        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self._tfb = TransformBroadcaster(self)

        rate = float(self.get_parameter("rate_hz").value)
        self._idx = 0
        self._total = len(self._dataset)
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"replaying {self._total} frames at {rate} Hz from "
            f"dataset={cfg.dataset.name!s} scene={cfg.dataset.scene}"
        )

    def _tick(self) -> None:
        if self._idx >= self._total:
            if self._idx == self._total:
                self.get_logger().info("replay complete; idling")
                self._idx += 1
            return

        from sensor_msgs.msg import CameraInfo, Image  # type: ignore

        frame = self._dataset[self._idx]
        stamp = self.get_clock().now().to_msg()
        H, W = frame.depth_m.shape

        # color (rgb8) — build Image message manually (no cv_bridge)
        color_msg = Image()
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = self._camera_frame
        color_msg.height = int(H)
        color_msg.width = int(W)
        color_msg.encoding = "rgb8"
        color_msg.is_bigendian = 0
        color_msg.step = int(W * 3)
        color_msg.data = frame.color.astype(np.uint8, copy=False).tobytes()
        self.pub_color.publish(color_msg)

        # depth 32FC1 meters — build Image message manually
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._camera_frame
        depth_msg.height = int(H)
        depth_msg.width = int(W)
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = int(W * 4)
        depth_msg.data = frame.depth_m.astype(np.float32, copy=False).tobytes()
        self.pub_depth.publish(depth_msg)

        # camera info
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._camera_frame
        info.width = int(frame.intrinsics.width)
        info.height = int(frame.intrinsics.height)
        K = frame.intrinsics.K.flatten().tolist()
        info.k = K
        info.p = [K[0], K[1], K[2], 0.0, K[3], K[4], K[5], 0.0, K[6], K[7], K[8], 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_info.publish(info)

        # tf map -> camera
        T_wc = frame.T_wc.astype(np.float64)
        (tx, ty, tz), (qx, qy, qz, qw) = _matrix_to_transform(T_wc)
        tfmsg = TransformStamped()
        tfmsg.header.stamp = stamp
        tfmsg.header.frame_id = self._map_frame
        tfmsg.child_frame_id = self._camera_frame
        tfmsg.transform.translation.x = tx
        tfmsg.transform.translation.y = ty
        tfmsg.transform.translation.z = tz
        tfmsg.transform.rotation.x = qx
        tfmsg.transform.rotation.y = qy
        tfmsg.transform.rotation.z = qz
        tfmsg.transform.rotation.w = qw
        self._tfb.sendTransform(tfmsg)

        self._idx += 1
        if self._idx % 50 == 0:
            self.get_logger().info(f"published {self._idx}/{self._total}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args if args is not None else os.sys.argv[1:])
    node = LiveRGBDPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
