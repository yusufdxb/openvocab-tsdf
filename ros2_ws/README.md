# ROS 2 workspace — openvocab-tsdf

Packages:
  - `openvocab_tsdf_msgs` — `GroundingTarget.msg`, `GroundText.srv`
  - `openvocab_tsdf_node` — Python node that wraps `openvocab_tsdf.pipeline`

## Build

```bash
# 1. source the ROS 2 distro
source /opt/ros/humble/setup.bash

# 2. make the pure-Python openvocab_tsdf package importable from the ROS 2 env
#    (replace with the absolute path to the project on your machine)
export PYTHONPATH="$(pwd)/../src:$PYTHONPATH"

# 3. build
cd ros2_ws
colcon build --symlink-install --packages-up-to openvocab_tsdf_node
source install/setup.bash
```

## Run

Precompute a map via the Python CLI, then launch the node:

```bash
openvocab-tsdf encode --config configs/replica_room0.yaml --output outputs/map.npz

ros2 launch openvocab_tsdf_node bringup.launch.py \
    map_path:=$(pwd)/../outputs/map.npz
```

## Query

```bash
ros2 service call /openvocab/ground openvocab_tsdf_msgs/srv/GroundText \
    "{query: 'red chair near the desk', top_k: 5, score_threshold: .nan, top_percentile: 0.02}"
```

## Notes

- The node uses the pure-Python pipeline at `../src/openvocab_tsdf/`. No
  direct dependency on any ROS 2-specific tensor transport.
- Live mapping (subscribe to camera topics, build the map online) is Phase 5b.
  Today the node expects a precomputed npz.
- Set `score_threshold` to NaN to use `top_percentile` instead — this lets the
  server adapt to scenes where absolute CLIP similarity magnitudes differ.
