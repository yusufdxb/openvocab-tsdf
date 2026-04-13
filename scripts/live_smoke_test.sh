#!/usr/bin/env bash
# End-to-end live smoke test:
#   1. start live_rgbd_publisher (replays NICE-SLAM Demo scene)
#   2. start grounding_node in live_mode:=true (builds map online)
#   3. wait for integration, then call /openvocab/ground
# All processes are killed on exit.

# no `set -u` — ROS 2 setup.bash touches unbound vars internally

PROJECT=/home/yusuf/Projects/personal/openvocab-tsdf
source /opt/ros/humble/setup.bash
source "$PROJECT/ros2_ws/install/setup.bash"
source "$PROJECT/.venv/bin/activate"
export PYTHONPATH="$PROJECT/src:$PROJECT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

cleanup() {
  kill $PUB_PID $NODE_PID 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

# 1. publisher
ros2 run openvocab_tsdf_node live_rgbd_publisher --ros-args \
  -p dataset:=nice_slam_demo \
  -p root:=$HOME/data/replica \
  -p scene:=Demo \
  -p stride:=2 -p max_frames:=120 -p rate_hz:=15.0 \
  >/tmp/live_pub.log 2>&1 &
PUB_PID=$!
echo "[smoke] publisher PID=$PUB_PID"

# 2. live grounding node
ros2 run openvocab_tsdf_node grounding_node --ros-args \
  -p live_mode:=true \
  -p voxel_size_m:=0.08 \
  -p bounds_min:=[-1.0,-1.0,-1.0] \
  -p bounds_max:=[7.0,6.0,4.0] \
  -p frame_stride:=2 \
  -p device:=cuda:0 -p dtype:=fp16 \
  >/tmp/live_node.log 2>&1 &
NODE_PID=$!
echo "[smoke] node PID=$NODE_PID"

# wait for encoder load + a chunk of frames
echo "[smoke] waiting 30 s for integration..."
sleep 30

# 3. call the service
echo "[smoke] querying /openvocab/ground..."
ros2 service call /openvocab/ground openvocab_tsdf_msgs/srv/GroundText \
  "{query: 'a chair', top_k: 3, score_threshold: .nan, top_percentile: 0.02}" \
  > /tmp/live_query.log 2>&1
RC=$?

echo
echo "=== live_node.log (last 20) ==="
tail -20 /tmp/live_node.log
echo
echo "=== live_pub.log (last 10) ==="
tail -10 /tmp/live_pub.log
echo
echo "=== service call result ==="
tail -30 /tmp/live_query.log

if [ $RC -eq 0 ] && grep -q "mode=live" /tmp/live_query.log; then
  echo "[smoke] PASS — live mode confirmed by service diagnostic"
  exit 0
else
  echo "[smoke] FAIL — service call did not return live diagnostic"
  exit 1
fi
