# Getting started

Short, practical path from a clean clone to a working grounding demo.

## 1. Install the env

```bash
# uv bootstraps both the Python env and the package install
make sync
```

This creates `.venv/` inheriting system site-packages (so the system PyTorch on mewtwo is reused and we don't re-download a 2 GB torch wheel). On a fresh machine, replace with:

```bash
uv venv .venv --python 3.10
uv pip install -e ".[dev]"
```

You may need to adjust the torch install depending on your GPU:

| GPU | Recommended |
|---|---|
| NVIDIA Blackwell (sm_120) | PyTorch nightly built against CUDA 12.8 (`+cu128`), needed for `sm_120` |
| RTX 40-series / 30-series | PyTorch 2.6+ stable with CUDA 12.x |
| No GPU | Reference backend runs on CPU; Triton backend does not |

## 2. Sanity-check

```bash
make info
```

You should see a table with Python, PyTorch, CUDA availability, device name, and VRAM.

```bash
make test
```

All 22 tests should pass. GPU-only tests auto-skip on CPU-only machines.

## 3. Run the synthetic demo

```bash
.venv/bin/python scripts/demo_synthetic.py
```

Renders a 3-object scene (red sphere, green floor, blue bar), encodes CLIP features, runs three text queries, and prints ranked 3D targets. No datasets required.

## 4. Benchmark

```bash
.venv/bin/python benchmarks/bench_tsdf_fuse.py --backend reference --frames 64
.venv/bin/python benchmarks/bench_tsdf_fuse.py --backend triton   --frames 64
```

Results land in `benchmarks/results/<stamp>_tsdf_fuse_<backend>.json`. These files are tracked, run again after any change to the kernel path.

## 5. Run on a real dataset

Download Replica (roughly 12 GB) to `~/data/replica/`:

```bash
./scripts/download_datasets.sh replica
```

Then fuse + encode + ground:

```bash
openvocab-tsdf fuse     --config configs/replica_room0.yaml --output outputs/mesh.ply
openvocab-tsdf encode   --config configs/replica_room0.yaml --output outputs/map.npz
openvocab-tsdf ground   --map outputs/map.npz --query "chair near the window" --config configs/replica_room0.yaml
```

## 6. Eval

```bash
.venv/bin/python eval/eval_grounding.py \
    --map outputs/demo_map.npz \
    --spec eval/specs/synthetic_demo.yaml
```

Summary prints to stdout; a machine-readable JSON is written to `benchmarks/results/`.

## 7. ROS 2 (Phase 5)

See `ros2_ws/README.md`. Requires a sourced ROS 2 Humble installation. Build with `colcon build --symlink-install` and query the node with `ros2 service call /openvocab/ground ...`.

## 8. Troubleshooting

**`libcudnn.so.9: cannot open shared object file` or `CUDNN_STATUS_NOT_INITIALIZED`.**
Another Python package (often one that lists `torch` as a dep without pinning the CUDA variant) may silently pull in `nvidia-cudnn-cu13` alongside the cu12 build PyTorch actually loads, leaving you with a mismatched or missing cuDNN. Fix:

```bash
# remove the cu13 variant (safe if torch was built against cu128)
python3 -m pip uninstall -y nvidia-cudnn-cu13
# re-install the cu12 variant torch wants (also safe)
python3 -m pip install --user --force-reinstall nvidia-cudnn-cu12
```

**`ros2 run` of the node raises `ModuleNotFoundError: No module named 'openvocab_tsdf'`.**
The installed entry-point is executed by the *system* `/usr/bin/python3`, which does not see the uv venv. Layer PYTHONPATH at launch:

```bash
source /opt/ros/humble/setup.bash
source $PROJECT/ros2_ws/install/setup.bash
source $PROJECT/.venv/bin/activate
export PYTHONPATH="$PROJECT/src:$PROJECT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
```
