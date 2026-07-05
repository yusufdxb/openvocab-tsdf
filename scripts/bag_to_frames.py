"""Extract a GO2 rosbag into a Replica-style RGB-D + pose folder.

Uses system ROS 2 python (rosbag2_py + tf2). Emits:
  out/rgb/<i>.png      uint8 RGB
  out/depth/<i>.npy    float32 meters (0 = invalid)
  out/traj.txt         one 4x4 row-major T_wc (cam->odom) per kept frame
  out/intrinsics.txt   fx fy cx cy width height
Pose = lookup odom -> camera_color_optical_frame at each color-image stamp.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rclpy.time import Time
from tf2_ros import BufferCore
from rclpy.duration import Duration


def quat2mat(w, x, y, z):
    """Unit-quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

BAG = sys.argv[1]
OUT = Path(sys.argv[2])
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
(OUT / "rgb").mkdir(parents=True, exist_ok=True)
(OUT / "depth").mkdir(parents=True, exist_ok=True)

COLOR_T = "/go2/camera/image_raw"
DEPTH_T = "/go2/camera/depth/image_raw"
INFO_T = "/go2/camera/camera_info"
TARGET, SOURCE = "odom", "camera_color_optical_frame"


def reader():
    r = SequentialReader()
    r.open(StorageOptions(uri=BAG, storage_id="sqlite3"),
           ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    return r, types


def stamp_ns(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


# --- pass 1: fill tf buffer + grab intrinsics + index depth by stamp ---
buf = BufferCore(Duration(seconds=200))
r, types = reader()
K = None
depth_index = {}  # stamp_ns -> (data bytes) decoded lazily later; store raw
depth_msgs = []   # (stamp_ns, np.float32 HxW)
color_msgs = []   # (stamp_ns, idx_in_bag)
while r.has_next():
    topic, data, t = r.read_next()
    if topic == "/tf":
        m = deserialize_message(data, get_message(types[topic]))
        for tr in m.transforms:
            buf.set_transform(tr, "bag")
    elif topic == "/tf_static":
        m = deserialize_message(data, get_message(types[topic]))
        for tr in m.transforms:
            buf.set_transform_static(tr, "bag")
    elif topic == INFO_T and K is None:
        m = deserialize_message(data, get_message(types[topic]))
        K = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]),
             int(m.width), int(m.height))
    elif topic == DEPTH_T:
        m = deserialize_message(data, get_message(types[topic]))
        d = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.width).copy()
        depth_msgs.append((stamp_ns(m.header), d))
    elif topic == COLOR_T:
        m = deserialize_message(data, get_message(types[topic]))
        rgb = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3).copy()
        color_msgs.append((stamp_ns(m.header), rgb))

print(f"intrinsics fx,fy,cx,cy,w,h = {K}")
print(f"color frames={len(color_msgs)} depth frames={len(depth_msgs)}")

# --- match depth to color by nearest stamp ---
depth_stamps = np.array([s for s, _ in depth_msgs])

fx, fy, cx, cy, W, H = K
(OUT / "intrinsics.txt").write_text(f"{fx} {fy} {cx} {cy} {W} {H}\n")

traj_rows = []
kept = 0
for i, (cs, rgb) in enumerate(color_msgs):
    if i % STRIDE != 0:
        continue
    # nearest depth
    j = int(np.argmin(np.abs(depth_stamps - cs)))
    ddt = abs(depth_stamps[j] - cs) / 1e9
    depth = depth_msgs[j][1]
    # pose lookup
    try:
        tf = buf.lookup_transform_core(TARGET, SOURCE, Time(nanoseconds=cs).to_msg())
    except Exception as e:  # noqa: BLE001
        if kept < 3:
            print(f"  frame {i}: pose lookup failed ({e}); skipping")
        continue
    tr = tf.transform.translation
    q = tf.transform.rotation
    Tm = np.eye(4, dtype=np.float32)
    Tm[:3, :3] = quat2mat(q.w, q.x, q.y, q.z)
    Tm[:3, 3] = [tr.x, tr.y, tr.z]
    Image.fromarray(rgb, "RGB").save(OUT / "rgb" / f"{kept:06d}.png")
    np.save(OUT / "depth" / f"{kept:06d}.npy", depth.astype(np.float32))
    traj_rows.append(" ".join(f"{v:.9f}" for v in Tm.reshape(-1)))
    if kept < 3:
        print(f"  kept {kept} (bag idx {i}) depth dt={ddt*1000:.1f}ms "
              f"pos=({tr.x:.3f},{tr.y:.3f},{tr.z:.3f})")
    kept += 1

(OUT / "traj.txt").write_text("\n".join(traj_rows) + "\n")
print(f"WROTE {kept} frames to {OUT} (stride {STRIDE})")
