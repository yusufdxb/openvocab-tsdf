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
| 1c. Sparse-feature backend | ✅ done | Lazy per-voxel slot allocation → **3.08× less feature memory on Replica room0** (1070 vs 3298 MB), parity-tested, same throughput |
| 2. OpenCLIP features + 3D aggregation | ✅ done | ViT-B/16 per-frame global embedding → per-voxel weighted mean. End-to-end grounding pipeline works. |
| 2b. Patch / dense features | ✅ done | ViT patch tokens with the **MaskCLIP** last-block-attn-bypass, lifted into 3D with per-voxel patch lookup and a near-surface feature gate. |
| 3. Query engine + eval harness | ✅ done | Cosine-sim, connected-component cluster, YAML-driven eval producing JSON. Surface-only filter on queries. |
| 4. Optimization | 🟡 partial | Triton already ≥ 100× the 30-FPS fuse budget; TensorRT for CLIP is next |
| 5. ROS 2 interface | ✅ **built & smoke-tested** | `openvocab_tsdf_msgs` + `openvocab_tsdf_node` colcon-built, service `/openvocab/ground` returns ranked targets over DDS. |
| 5b. Live RGB-D mapping | ✅ **end-to-end** | `grounding_node live_mode:=true` subscribes to color/depth/camera_info/TF, builds the feature map online. `live_rgbd_publisher` replays any `RGBDDataset` on topics for a hardware-free demo. |
| 6. Polish + figures | 🟡 partial | README, decisions, synthetic demo, first eval baseline logged, heatmap PLY exporter done. Publishable real-data figures pending dataset download. |

### Current numbers (RTX 5070, 12 GB)

**Synthetic scene (3 primitives, 32 frames, 224×224):**
- TSDF fuse, Triton backend: **4423 FPS**, 25 MB peak VRAM
- TSDF fuse, reference backend: 1486 FPS, 68 MB peak VRAM
- Grounding (global features, 0.1 m bbox slack): **hit@1 = 33 %, hit@5 = 100 %**, mean top-1 L2 = 0.61 m, 50 ms per query
- End-to-end grounding query (text encode + voxel scan + cluster): **~50 ms**
- ROS 2 service `/openvocab/ground` — DDS roundtrip returns ranked targets in well under a second

**Real scene (NICE-SLAM demo — 500 RGB-D frames @ 640×480, 1.2 M voxels at 6 cm):**
- Encode + fuse (patch-mode CLIP ViT-B/16 + MaskCLIP lift): **≈1.8 s end-to-end** (CLIP image encode 0.93 s, feature fusion 0.87 s)
- Reference backend geometry fuse: 79.6 FPS at native 640×480, mesh 25.1 k vertices / 50.4 k triangles
- Query latency: <100 ms per free-form prompt against the 1.2 M-voxel feature map
- Per-query heatmap PLYs: 19.4 k surface points each (`outputs/heatmaps_demo/*.ply`)

**Replica `room0` (500 frames @ 1200×680, 1.67 M voxels at 6 cm):**
- CLIP patch encode: 1.91 s (262 FPS), feature fusion: 3.51 s (143 FPS at native 1200×680)
- Geometry fuse: 81.3 FPS, mesh 97.3 k vertices / 194 k triangles
- Per-query latency: **83–283 ms** (mean ~135 ms) for 10 realistic prompts ("a chair", "a sofa", "a plant", ...); scene-mean-subtract on; top-1 + top-3 clusters returned per query
- Per-query heatmap PLYs: 104 k surface points each (`outputs/heatmaps_replica_room0/*.ply`)

### Grounding accuracy — Replica `room0` (hand-annotated, 9 queries)

See `eval/specs/replica_room0.yaml` for the annotation protocol (structural queries derived from the mesh's horizontal-surface z histogram; object bboxes from a mesh-density map, conservatively widened to ±0.2 m slack). Ablation produced by `eval/run_ablation.py`:

| map | mean-sub | top% | hit@1 | hit@5 | hit-L2 (m) | struct h@1 | obj h@1 |
|---|---|---|---|---|---|---|---|
| patch   | False | 0.005 | 22.2 % | 44.4 % | 3.46 | 1/3 | 0/5 |
| patch   | False | 0.020 | 22.2 % | 44.4 % | 3.15 | 0/3 | 1/5 |
| patch   | True  | 0.005 | 11.1 % | 33.3 % | 3.59 | 0/3 | 0/5 |
| patch   | True  | 0.020 | 11.1 % | 33.3 % | 4.05 | 0/3 | 0/5 |
| **global** | **False** | **0.005** | **33.3 %** | **55.6 %** | **2.63** | **2/3** | **0/5** |
| global  | False | 0.020 | 33.3 % | 44.4 % | 2.25 | 2/3 | 0/5 |
| global  | True  | 0.005 | 11.1 % | 44.4 % | 3.68 | 1/3 | 0/5 |
| global  | True  | 0.020 |  0.0 % | 44.4 % | 3.62 | 0/3 | 0/5 |

**Honest takeaways.**
- **Structural queries work.** `"the floor"`, `"the ceiling"`, `"a window"` score 2/3 h@1 with global features. These are easy wins but they're the first real-data ground-truth numbers in the repo.
- **Object-level localization is still weak on 500 strided frames.** Best obj h@1 = 1/5. This is the real open problem — more frames, better dense features, or finer voxels should all help and are the next experiment.
- **Patch features do not beat global features on this scene.** Suspected causes: (a) sparse frame coverage (500 strided @ native 1200×680 is ~25 % of the full trajectory), (b) per-voxel patch lookup is single-patch per observation so aggregation is noisier than expected, (c) our hand-annotated object bboxes are loose (±0.2 m slack).
- **Scene-mean-subtract hurts here.** It helped on the synthetic demo scene but on real data it flattens the signal — confirms our own decision-log prediction that the flag is scene-dependent and must stay off by default.
- **Mean latency 337 ms per query** (includes one text-encode each; drops to ~135 ms with the encoder reused).

**CLIP image encode (ViT-B/16 @ 224×224 fp16, batch 16):**
- PyTorch: **1280 FPS**
- TensorRT: **1414 FPS** (+10 %, parity-tested vs PyTorch with cosine > 0.98). See `benchmarks/bench_clip_encode.py`.

**Sparse-feature backend on Replica room0 (1.67 M voxels, 512-dim features):**

| backend | FPS | feat (MB) | allocated voxels | sparsity | peak VRAM (MB) |
|---|---|---|---|---|---|
| dense reference | 120 | 3297.7 | 1 688 400 | 100.00 % | 4833 |
| **sparse feature** | **125** | **1070.2** | **547 947** | **32.45 %** | **3430** |

Feature-memory reduction: **3.08×** — identical integrate math, lazy per-voxel slot allocation keeps memory proportional to observed surface area rather than to the bounding box. `benchmarks/bench_sparse_features.py`.

**Live ROS 2 mapping (smoke test on synthetic publisher):**
- Publisher replays 120 NICE-SLAM frames @ 15 Hz on `/camera/color/image_raw` + `/camera/depth/image_raw` + `/camera/camera_info` + TF.
- `grounding_node live_mode:=true` synchronizes them through `message_filters`, looks up `map→camera` through TF, runs CLIP encode + `ReferenceTSDF.integrate` per frame.
- After 30 s integration (49 frames at stride=2), `ros2 service call /openvocab/ground "{query: 'a chair', top_percentile: 0.02}"` returns 3 ranked targets in well under a second. End-to-end hardware-free robotic demo.

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
openvocab-tsdf ground --map outputs/map.npz --query "plant on the desk" --config configs/replica_room0.yaml

# 7. [phase 6] export per-query heatmap PLYs alongside the mesh
python scripts/export_heatmaps.py \
    --map outputs/map.npz \
    --query "a red chair" --query "the couch" \
    --out-dir outputs/heatmaps
```

Or try the fully self-contained demo (no dataset required):

```bash
python scripts/demo_synthetic.py
python scripts/export_heatmaps.py --map outputs/demo_map.npz \
    --query "a red ball" --query "a blue bar" --query "green grass floor" \
    --out-dir outputs/heatmaps
```

The ROS 2 node exposing `/openvocab/ground` as a service lives at `ros2_ws/` — see its dedicated README.

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
