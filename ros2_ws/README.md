# ROS 2 workspace — openvocab-tsdf

Packages:
  - `openvocab_tsdf_msgs` — `GroundingTarget.msg`, `GroundText.srv`
  - `openvocab_tsdf_node` — Python node that wraps `openvocab_tsdf.pipeline`

## Build

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Verified on mewtwo: `ros2 humble`, `colcon 0.5+`. Both packages build clean in ~3s.

## Run

The node depends on PyTorch + OpenCLIP + the pure-Python `openvocab_tsdf`
package, all installed in the project's `uv` venv. The `ros2 run` entry-point
uses the system `/usr/bin/python3`, so we layer PYTHONPATH to include both the
venv's site-packages and the project's `src/`. Order matters: source ROS 2
first, then the workspace, then activate the venv, then extend PYTHONPATH.

```bash
PROJECT=/home/yusuf/Projects/personal/openvocab-tsdf  # adjust

source /opt/ros/humble/setup.bash
source $PROJECT/ros2_ws/install/setup.bash
source $PROJECT/.venv/bin/activate
export PYTHONPATH="$PROJECT/src:$PROJECT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# launch
ros2 run openvocab_tsdf_node grounding_node --ros-args \
    -p map_path:=$PROJECT/outputs/demo_map.npz \
    -p model:=ViT-B-16 -p pretrained:=laion2b_s34b_b88k \
    -p device:=cuda:0 -p dtype:=fp16
```

Expect ~3s to load CLIP weights on a warm cache, then `service /openvocab/ground is up`.

## Query

```bash
ros2 service call /openvocab/ground openvocab_tsdf_msgs/srv/GroundText \
    "{query: 'a red ball', top_k: 3, score_threshold: .nan, top_percentile: 0.02}"
```

A successful response contains a header (stamp + `frame_id: map`) plus a list
of `GroundingTarget` entries with `center`, `bbox_min`, `bbox_max`, `score`,
`voxel_count`.

## Notes

- Use `score_threshold: .nan` and a real `top_percentile` to adapt the cut-off
  to per-query CLIP similarity magnitude. Or set `score_threshold` to a real
  number and `top_percentile: .nan`.
- The node does not subscribe to live camera topics yet. Live mapping is
  Phase 5b.
- `launch/bringup.launch.py` accepts all node parameters as launch args.
