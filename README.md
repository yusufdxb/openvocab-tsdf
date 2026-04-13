# openvocab-tsdf

**GPU-accelerated open-vocabulary 3D mapping and language grounding for robotics.**

Ingest RGB-D and poses → fuse a GPU TSDF / sparse voxel map → attach open-vocab CLIP features per voxel → answer free-form language queries with ranked 3D targets and confidence. Offline-first. Clean path to ROS 2.

> *"chair near the window"* → `(x, y, z), bbox, score, supporting frames`

## Status

| Phase | State | Highlights |
|---|---|---|
| 0. Audit + architecture + scaffold | ✅ done | `docs/architecture.md`, `docs/decisions.md`, `AGENTS.md`, env via `uv` |
| 1. RGB-D ingestion + reference TSDF | ✅ done | Replica loader, PyTorch dense backend (~1.5 kFPS @ 320×240), marching-cubes mesh |
| 1b. Custom GPU TSDF kernel | ✅ done | **Triton** (sm_120-compatible), 4423 FPS @ 320×240, 25 MB VRAM, parity-tested |
| 2. OpenCLIP features + 3D aggregation | ✅ done (global) | ViT-B/16 per-frame → per-voxel weighted mean. End-to-end grounding works on a synthetic multi-object scene. |
| 2b. Patch / mask features | ⏸ next | Needed to tighten spatial localization; spec'd in `docs/decisions.md` |
| 3. Query engine + eval harness | ✅ done | Cosine-sim, connected-component cluster, YAML-driven eval producing JSON |
| 4. Optimization | 🟡 partial | Triton already ≥ 100× the 30-FPS fuse budget; TensorRT for CLIP is next |
| 5. ROS 2 interface | 🟡 scaffolded | `openvocab_tsdf_msgs` + `openvocab_tsdf_node` + launch file; colcon build pending |
| 6. Polish + figures | 🟡 partial | README, decisions, synthetic demo, first eval baseline logged |

### Current numbers on a synthetic scene (RTX 5070, 12 GB)

- TSDF fuse, Triton backend: **4423 FPS**, 25 MB peak VRAM
- TSDF fuse, reference backend: 1486 FPS, 68 MB peak VRAM
- End-to-end grounding query (text encode + voxel scan + cluster): **~50 ms**
- First honest grounding baseline (global features, 3-case synthetic eval): hit@1 = 0 %, hit@5 = 33 %, mean top-1 centroid L2 = 0.61 m

Real-dataset numbers (ScanNet / Replica) and the Phase 2b patch-feature numbers land next.

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
