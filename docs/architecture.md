# Architecture

Status: **draft v1** — locks in the high-level choices. Per-component design docs live under `docs/design/` as they are written.

## System Overview

openvocab-tsdf is an offline-first GPU pipeline that turns RGB-D plus poses into a queryable semantic voxel map, then answers free-form language queries by returning ranked 3D targets with confidence. ROS 2 integration is a thin wrapper around the offline core, added in Phase 5.

```
                RGB-D + poses          text query
                     │                      │
                     ▼                      ▼
              ┌─────────────┐        ┌─────────────┐
              │   Data      │        │  Tokenize   │
              │  ingestion  │        │  & embed    │
              └──────┬──────┘        └──────┬──────┘
                     │                      │
                     ▼                      │
              ┌─────────────┐               │
              │ CLIP image  │               │
              │  encoder    │               │
              └──────┬──────┘               │
                     │                      │
                     ▼                      │
              ┌─────────────┐               │
              │  Per-frame  │               │
              │ 2D features │               │
              └──────┬──────┘               │
                     │                      │
                     ▼                      │
              ┌──────────────────┐          │
              │ GPU TSDF fusion  │          │
              │ + 3D feature     │          │
              │   aggregation    │          │
              └──────┬───────────┘          │
                     │                      │
                     ▼                      │
              ┌─────────────────┐           │
              │ Voxel map       │───┐       │
              │ (geom + feats)  │   │       │
              └─────────────────┘   │       │
                                    ▼       ▼
                              ┌────────────────┐
                              │ Query engine:  │
                              │ cos-sim, NMS,  │
                              │ spatial filter │
                              └──────┬─────────┘
                                     │
                                     ▼
                           ranked 3D targets + conf
```

## Components

### 1. Data ingestion (`src/openvocab_tsdf/data/`)
Dataset-agnostic loader producing a typed `RGBDFrame` stream: `(color: u8[H,W,3], depth_m: f32[H,W], K: f32[3,3], T_wc: f32[4,4], timestamp: f64)`. Implemented today: Replica, the NICE-SLAM demo scene, and a synthetic ray-traced scene generator for tests. ScanNet v2 and TUM RGB-D drivers were originally listed here as v1 targets but are explicit follow-ups (see `CLAUDE_CODE_NEXT.md`). All implemented drivers lazy-load and stream to avoid pinning full sequences in host RAM.

### 2. TSDF fusion core (`src/openvocab_tsdf/mapping/`)
GPU-resident voxel structure. Implemented backends:
1. **Reference** (`mapping/reference.py`) — PyTorch dense TSDF. Slow but simple, correctness ground truth. All other backends are parity-tested against it.
2. **Triton** (`mapping/triton_backend.py`) — sm_120-compatible Triton geometry kernel; geometry only (no per-voxel features yet).
3. **SparseFeatureTSDF** (`mapping/sparse_reference.py`) — dense geometry + lazy per-voxel feature pool; ~3× feature-memory reduction at room scale.
4. **BlockHashTSDF** (`mapping/block_hash.py`) — block-hash sparse *geometry* with optional per-voxel sparse features composed on top; frustum-culled integrate.

A native CUDA backend was originally planned and is logged in `decisions.md` as deferred — Triton fills that role today and is the documented "fast kernel" path. All backends satisfy the same `TSDFVolume` interface: `integrate(frame) -> None`, `extract_mesh() -> Mesh`, `query(points_w) -> VoxelQueryResult(tsdf, weight, color, feat)`.

### 3. Open-vocab semantics (`src/openvocab_tsdf/semantics/`)
OpenCLIP encoder running in fp16 on GPU. Three feature modes are implemented:
- **`global`** — one ViT-B/16 embedding per frame (fast, coarse). Baseline.
- **`patch`** — ViT patch tokens lifted into 3D via the MaskCLIP last-block-attention bypass plus per-voxel patch lookup through the encoder's preprocess mapping. `reference` backend only.
- **`sam_dense`** — MobileSAM auto-masks → CLIP per-mask crops → per-pixel dense feature map (with mask-IoU blending and a frame-global fallback for pixels outside every mask). `reference` and `block_hash` backends.

Features are pooled into voxels during fusion via a weighted running mean. The per-frame contribution is restricted to voxels whose normalized TSDF is within `near_surface_band` (default 0.5) so free-space voxels in front of a surface do not "steal" the features of whatever the ray eventually hits.

### 4. Query engine (`src/openvocab_tsdf/grounding/`)
Text → OpenCLIP text encoder → query vector `q`. Score all active voxels by cos-sim against per-voxel feature mean. Spatial post-processing: threshold, connected-component clustering in voxel space, rank clusters by score×size. Return top-K with centroid, bounding box, confidence, and supporting evidence (best observing frames).

### 5. Visualization (`src/openvocab_tsdf/viz/`)
Open3D-based viewer for meshes, voxels, and query heatmaps. Used in tests, notebooks, and the demo app. Not a runtime dependency of the core pipeline.

### 6. ROS 2 interface (`ros2_ws/`)
Single node `openvocab_tsdf_node` (with `openvocab_tsdf_msgs` for the service type). Two modes selected via the `live_mode` parameter:
- **offline** (default) — loads a saved feature map (`dense`, `voxel_slot`, or `block_hash` `sparse_kind`) through the shared `grounding.map_bundle.MapBundle` loader and serves `/openvocab/ground`.
- **live** — synchronizes color + depth + camera_info + TF, runs CLIP encode + `ReferenceTSDF.integrate` per frame, and publishes a CUBE_LIST RViz preview.

## Technology Choices

| Concern | Choice | Reason |
|---|---|---|
| Orchestration | Python 3.10 + uv | Fast env, standard ML stack |
| Deep learning | PyTorch 2.11 + CUDA 12.8 | sm_120 (Blackwell) wheels available |
| Open-vocab VLM | OpenCLIP ViT-B/16 (default), ViT-L/14 (quality mode) | Strong baseline, frozen weights, well-cited |
| Geometry kernels | Custom CUDA, C++17, CMake | Portfolio differentiator; full control |
| Geom validation | Open3D | Cross-check meshes and voxel queries |
| Config | pydantic-settings + YAML | Typed, tool-friendly, reproducible |
| Testing | pytest + hypothesis | Property-based tests for TSDF integration math |
| Lint / format | ruff + black | Fast, opinionated |
| Benchmarks | pytest-benchmark + custom runners | Machine-readable JSON output |
| Profiling | Nsight Systems, PyTorch profiler | Timeline traces, cross-checked |
| ROS 2 | Humble (foxy is EOL) | Matches sibling projects on this machine |

## Performance Targets (desktop, RTX 5070 12GB)

These are contracts. Breaking one requires a `docs/decisions.md` entry.

| Metric | Target | Notes |
|---|---|---|
| TSDF fusion throughput | ≥ 30 FPS @ 640×480 depth | Per-frame integrate + hash update |
| CLIP ViT-B/16 image encode | ≥ 30 FPS @ 224×224 fp16 | Batched across frames |
| Voxel map size | ≥ 10 m³ @ 2 cm voxels | Active voxel count bounded by hash capacity |
| Text query latency | ≤ 200 ms end-to-end | Text encode + full voxel scan |
| Peak VRAM | ≤ 8 GB | Leaves headroom on 12 GB card |
| Offline eval wall time | ≤ 15 min / ScanNet scene | 500-frame scene, default config |

## What We Cut and Why

| Cut | Why |
|---|---|
| 3DGS backbone | Rendering demo, weak for geometric queries and planner integration. Optional layer in Phase 6+ only. |
| NeRF | Same reasoning; worse throughput. |
| Learned feature aggregation | YAGNI. Weighted mean is a strong, interpretable v1. Revisit only if ablation shows it caps grounding accuracy. |
| Custom SLAM / pose estimation | We consume ground-truth or external poses. SLAM would double the project size. |
| Fine-tuning the VLM | Out of scope. Open-vocab grounding works with frozen encoders in the literature. |
| Jetson port | Owned by sibling projects (`go2-semantic-nav`, `GO2-Perception-Optimization`). |
| Dense per-pixel open-vocab (LSeg/OpenSeg) in v1 | Starts too complex. Global / patch features first, dense mask-feature aggregation as Phase 2b if evaluation demands it. |
| Multi-robot fusion | Stretches scope without strengthening the core claim. |
| Live GO2 integration in v1 | ROS 2 interface first validates against bags. Live runs in Phase 5+. |

## Phased Plan

| Phase | Done-criteria | Estimated time (realistic) |
|---|---|---|
| 0. Audit + arch | This doc + scaffolded repo + env installed | < 1 session |
| 1. Geometry | RGB-D → TSDF → mesh on Replica, parity between reference and CUDA impl | 2–3 sessions |
| 2. Semantics | CLIP features fused into voxels, qualitative text queries work on 1 scene | 2 sessions |
| 2b. Dense features (optional) | Patch or mask-based features; ablation vs global | 1–2 sessions |
| 3. Query engine | Ranked 3D targets + confidence, metrics harness | 2 sessions |
| 4. Optimization | Custom CUDA hits perf targets; TensorRT for VLM encode | 2–3 sessions |
| 5. ROS 2 | Node, service, bag replay demo | 2 sessions |
| 6. Polish | Docs, failure cases, ablations, publishable figures | 1–2 sessions |

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RTX 5070 sm_120 toolchain gaps (TensorRT, some CUDA libs) | Fall back to PyTorch compile / ONNX Runtime; defer TensorRT to Phase 4 after the perf bar is otherwise met |
| Hash collisions / capacity overflow in sparse voxel map | Bounded capacity with observable resize; unit tests that exercise collision paths |
| CLIP global features too coarse for localized queries | Patch/region mode is already planned as Phase 2b |
| 12 GB VRAM pressure with ViT-L + large maps | fp16 everywhere, chunked batch encode, enforced map-size caps with clear errors |
| Pose-quality dependency | Start with datasets with ground-truth poses; never promise SLAM |
