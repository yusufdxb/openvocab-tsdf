# openvocab-tsdf

**GPU-accelerated open-vocabulary 3D mapping and language grounding for robotics.**

Ingest RGB-D and poses → fuse a GPU TSDF / sparse voxel map → attach open-vocab CLIP features per voxel → answer free-form language queries with ranked 3D targets and confidence. Offline-first. Clean path to ROS 2.

> *"chair near the window"* → `(x, y, z), bbox, score, supporting frames`

## Status

- Phase 0 (audit + architecture + scaffold) — **in progress**
- Phase 1 (RGB-D ingestion + reference TSDF) — pending
- Phase 1b (custom CUDA TSDF kernel) — pending
- Phase 2 (OpenCLIP features + 3D aggregation) — pending
- Phase 3 (query engine + eval harness) — pending
- Phase 4 (optimization) — pending
- Phase 5 (ROS 2 interface) — pending
- Phase 6 (polish) — pending

See [`docs/architecture.md`](docs/architecture.md) for the full plan, performance targets, and what is explicitly cut. See [`docs/decisions.md`](docs/decisions.md) for the architectural-decision log. See [`AGENTS.md`](AGENTS.md) for the rules that govern agent work in this repository.

## Why this project

- Modern 3D perception (TSDF / sparse voxel hashing) — not another 2D detector repo.
- Serious GPU-first systems work — custom CUDA kernels, profiled, budgeted.
- Open-vocabulary grounding instead of closed-set detection.
- Benchmark-first discipline: every perf claim has a JSON file behind it.
- Clean offline pipeline with a ROS 2 wrapper, not a live-only demo.

The sibling [`go2-semantic-nav`](../go2-semantic-nav) project owns the Jetson / GO2 side and intentionally bans custom CUDA there. This project owns it here.

## Hardware expectations

- Linux, x86_64
- NVIDIA GPU, compute capability ≥ 7.5 (developed on RTX 5070, sm_120)
- CUDA 12.x runtime (via PyTorch) plus a matching toolkit if you build the CUDA extension
- 12+ GB VRAM comfortable; 8 GB works with smaller voxel volumes and fp16

## Quick start

```bash
# 1. get the code
cd openvocab-tsdf

# 2. install the env (uv)
make sync

# 3. sanity check
make info              # prints Python, torch, CUDA, GPU

# 4. [phase 1] fuse a Replica scene
openvocab-tsdf fuse --config configs/replica_room0.yaml --output outputs/mesh.ply

# 5. [phase 2] encode CLIP features into voxels
openvocab-tsdf encode --config configs/replica_room0.yaml

# 6. [phase 3] text query
openvocab-tsdf ground --config configs/replica_room0.yaml --query "plant on the desk"
```

Subcommands past `info` return a non-zero exit until their phase lands.

## Repository layout

```
openvocab-tsdf/
├── AGENTS.md                 # rules for any LLM-driven contributor
├── README.md
├── pyproject.toml
├── Makefile
├── configs/                  # YAML configs
├── docs/                     # architecture, decisions, design notes
├── src/openvocab_tsdf/
│   ├── data/                 # dataset loaders (Replica, ScanNet, TUM)
│   ├── mapping/              # TSDF / voxel-hash backends
│   │   └── cuda/             # custom CUDA kernels + C++ bindings
│   ├── semantics/            # OpenCLIP / dense encoders + 3D aggregation
│   ├── grounding/            # text-to-3D query engine
│   ├── viz/                  # Open3D viewers
│   ├── ros2/                 # ROS 2 wrapper (loaded only when ROS 2 is present)
│   ├── config.py
│   └── cli.py
├── tests/                    # unit + integration; markers: gpu, dataset, slow, benchmark
├── eval/                     # evaluation scripts (grounding accuracy, localization err)
├── benchmarks/               # named benchmarks + JSON result history
└── scripts/                  # one-shot helpers (download datasets, etc.)
```

## Performance budgets (contracts)

See [`docs/architecture.md`](docs/architecture.md#performance-targets-desktop-rtx-5070-12gb). Breaking a budget requires an entry in [`docs/decisions.md`](docs/decisions.md).

## License

MIT. See `LICENSE`.
