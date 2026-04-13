# CLAUDE_CODE_NEXT.md — next-session prompt

Continuation prompt for the next Claude Code session on `openvocab-tsdf`.
Companion to `CLAUDE_CODE_PROMPT.md` (original mission statement) and
`AGENTS.md` (hard rules — read that file first).

## Current state (as of 2026-04-13)

- **34 commits on `main`, 38 passing fast tests + 4 slow (CLIP / TRT) that also pass.**
- **Four mapping backends**, all sharing `TSDFVolume`:
  - `reference` — dense PyTorch TSDF (~1.5 kFPS @ 320×240). The correctness oracle.
  - `triton` — custom Triton dense TSDF on sm_120 (4423 FPS, 25 MB). Parity-tested.
  - `sparse_feature` — dense geometry + voxel-slot sparse features; Triton kernel for the pool update (1.70× over PyTorch sparse).
  - `block_hash` — 8³-voxel blocks + block-slot lookup, frustum-culled integrate, **optional per-voxel sparse features** composed onto the block pool. 50 m³ cube at 4.4 MB (geom) / 304 FPS. 30 m³ + 512-d features at 379 MB / 304 FPS.
- **Open-vocab grounding**: SAM-per-mask CLIP (`mode: sam_dense`) via MobileSAM + OpenCLIP. Real-data eval on Replica `room0` hit@1 55.6 % / hit@5 88.9 %; `office0` @ 512/16 hit@1 50.0 %.
- **Robotics**: ROS 2 (Humble) `openvocab_tsdf_node` colcon-built, `/openvocab/ground` service over DDS, live RGB-D mapping from synchronized topics, RViz CUBE_LIST voxel preview, smoke test passing end-to-end.
- **Figures**: 18 single-query 3-panel PNGs + 4 side-by-side baseline-vs-SAM comparisons.

See `README.md` for tables, `docs/decisions.md` for ADRs.

---

## Ground rules (repeat — these are load-bearing)

1. **Inspect before editing.** Read `pyproject.toml`, `configs/*.yaml`, `docs/architecture.md`, and relevant source before changing anything. Cite `file:line` for every claim.
2. **Ruthless realism.** No feature is "working" without a run log, a benchmark JSON, or a passing test. Distinguish *verified / inferred / unverified* in every status update.
3. **Parity tests are mandatory** for any new mapping backend or kernel — must agree with the dense reference within fp32 noise (`atol=1e-5`).
4. **Benchmark JSONs go in `benchmarks/results/`.** Perf claims without a result file do not count.
5. **Never ask the user to run commands.** Shell access is available.
6. **Never add `Co-Authored-By: Claude` to commits** (user's global rule).
7. **Auto-update the Obsidian vault** at `/home/yusuf/Documents/Obsidian Vault/Projects/openvocab-tsdf/` after meaningful work. Append to `Daily Claude Logs/<date>.md`.

Full rules: `AGENTS.md`.

---

## Priority 1 — close the `block_hash + encode_and_fuse` loop

Currently `build_tsdf` can instantiate a `BlockHashTSDF`, but `encode_and_fuse`'s save path (`src/openvocab_tsdf/pipeline.py`) only knows how to serialise `ReferenceTSDF` (dense) and `SparseFeatureTSDF` (sparse-feature npz). The `block_hash` save path is marked TODO in commit `6ce2b80`.

**Done when:**

1. `encode_and_fuse(cfg)` with `mapping.backend: block_hash` writes a readable `.npz` that contains:
   - `block_slot[Nbx, Nby, Nbz]` int32
   - `tsdf_pool[NumBlocks, 512]`, `weight_pool`, `color_pool`
   - `feat_voxel_slot[NumBlocks, 512]` int32 (if `store_features`)
   - `feat_pool[NumFeatVoxels, D]`
   - `origin`, `voxel_size`, `dims`, `block_dims`, `feature_dim`, `mode`, `sparse_kind: "block_hash"`
2. `eval/eval_grounding.py` and `viz/heatmap.py` both detect `sparse_kind: "block_hash"` and score queries directly against `feat_pool` without densifying the 4-D feature tensor.
3. End-to-end: `openvocab-tsdf encode --config configs/replica_room0_4cm_block_hash_sam.yaml` runs, `eval_grounding.py` on the resulting map matches the existing `room0_6cm_sam` numbers within ±5 pp hit@1 (finer voxels should trend up, not down).
4. `SparseFeatureTSDF` save path still works; no regression on the existing `room0_4cm_global.npz` or `office0_6cm_sam.npz` evals.

**Why it matters.** Today the warehouse-scale numbers are synthetic-bench only; wiring the save format is what makes the combined backend *runnable against Replica / ScanNet / real camera bags*. That's the step that converts "benchmark result" into "usable open-vocab mapping system at warehouse scale."

**Touchpoints:**

- `src/openvocab_tsdf/pipeline.py` — `encode_and_fuse` save branch, `ground_text` load branch
- `eval/eval_grounding.py` — sparse-score path (mirror the existing `if is_sparse` block)
- `src/openvocab_tsdf/viz/heatmap.py` — same
- new: `configs/replica_room0_4cm_block_hash_sam.yaml`
- tests: extend `tests/unit/test_mapping_block_hash_features.py` to cover the round-trip (encode → save → load → query)

---

## Priority 2 — TensorRT MobileSAM export

SAM-dense encode is the grounding accuracy lever but it's 1.6 s / frame because MobileSAM runs fp32 eagerly in PyTorch. TensorRT infra already exists (`src/openvocab_tsdf/semantics/trt_encoder.py` for CLIP; 1414 FPS verified). A similar export for MobileSAM's `ImageEncoderViT` → ONNX → TRT fp16 should bring per-frame cost under 300 ms, turning the SAM pipeline from "1 fps tolerable" to "real-time plausible."

**Done when:**

1. `src/openvocab_tsdf/semantics/trt_sam.py` exports MobileSAM's `image_encoder` to ONNX (dynamo=False — the dynamo exporter hit the CLIP-MHA view bug; same pattern likely).
2. TRT fp16 engine built; parity test against PyTorch MobileSAM within cosine > 0.98 on 8 fixed inputs.
3. `benchmarks/bench_sam_encode.py` records PyTorch FPS vs TRT FPS at 384×384 and 512×512 inputs. Expected: ≥ 3× speedup.
4. `SAMDenseFeatureExtractor` gains a `backend: {"pytorch", "tensorrt"}` knob; sam_dense pipeline uses TRT when configured.
5. Re-run the existing SAM quality sweep (`scripts/sam_quality_sweep.py`) with TRT backend; grounding numbers must match within ±1 pp hit@1.

**Why it matters.** With TRT-MobileSAM at 300 ms/frame, the live ROS 2 mapping node (`grounding_node live_mode:=true`) can actually run SAM-dense online at ~3 Hz — not great, but demo-able. Without this, SAM-dense is an offline-only tool.

---

## Priority 3 — Scannet / ScanNet++ cross-scene eval

Current real-data eval is Replica `room0` + `office0` only. The original prompt called for "ScanNet or Replica for offline benchmarking." Replica is shipped with no semantic labels; ScanNet ships per-frame semantic masks and proper 3D object bboxes.

**Done when:**

1. `src/openvocab_tsdf/data/scannet.py` loader that reads a standard ScanNet v2 scene (`scene0000_00` etc.) — color/depth/poses/intrinsics.
2. `eval/specs/scannet_<scene>.yaml` auto-generated from the `_vh_clean.aggregation.json` + `_vh_clean_2.labels.ply` labels (per-instance 3D bboxes).
3. One evaluation run across ≥5 ScanNet val scenes with ≥10 queries each. Publish `hit@1 / hit@5 / hit-L2` per scene + aggregate.
4. README table updated with ScanNet results next to the existing Replica rows.

**Why it matters.** Real semantic ground truth — not hand-annotated — is what takes this from "we think it works" to "it works, measured against standard benchmarks." The single biggest remaining credibility piece.

**Blocker.** ScanNet requires signing a terms-of-use form. The user will need to do that once, then `scripts/download_datasets.sh scannet` can be fleshed out to fetch.

---

## Priority 4 — polish + ship

1. **GitHub Actions CI** — `.github/workflows/ci.yml` that runs `ruff check`, `black --check`, and `pytest -m "not gpu and not dataset and not slow"` on pushes to `main`. Non-GPU subset (~4 tests currently — extend this coverage too).
2. **Dockerfile** — minimal reproducibility env based on `nvcr.io/nvidia/pytorch:25.09-py3` or similar, wrapping the `uv sync` + ROS 2 Humble install. Document the `PYTHONPATH` layering needed for the ROS 2 node.
3. **Push to GitHub.** Repo is still local-only.
4. **`CITATION.cff`** + LICENSE hygiene.
5. **`docs/paper_outline.md`** — 4-page workshop-style writeup (RSS Workshop on Open-Vocab Robotics or similar). Abstract + problem statement + system figure + table of results + related work. Use the existing figures.

---

## Environment gotchas that bit us (do not rediscover)

1. **`uv pip install open_clip_torch open3d` can silently pull in `torch==2.11+cu130` and `nvidia-cudnn-cu13`** alongside torch+cu128 on the system. Symptom: `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` or `libcudnn.so.9: cannot open shared object file`. Fix: `python3 -m pip uninstall -y nvidia-cudnn-cu13 && python3 -m pip install --user --force-reinstall nvidia-cudnn-cu12`. See `docs/getting-started.md#8`.
2. **ROS 2 Humble's `cv_bridge` is compiled against numpy 1.x**, which blows up with numpy ≥ 2.0 via `AttributeError: _ARRAY_API not found`. We bypassed `cv_bridge` entirely in `grounding_node` + `live_rgbd_publisher` with manual `sensor_msgs/Image.data` byte conversions. Don't reintroduce `cv_bridge` without downgrading numpy.
3. **`ros2 run`'s entry-point shebang is `#!/usr/bin/python3`** (system Python), which does not see the venv's torch / openvocab_tsdf. Fix via `PYTHONPATH` layering in the launch env, not via installing into system Python. See `ros2_ws/README.md`.
4. **PyTorch 2.11's dynamo-based ONNX exporter crashes on CLIP MHA** at the view-reshape step (`Cannot view a tensor with shape [197, s77, 12, 64]...`). Use `torch.onnx.export(..., dynamo=False)` instead. This will hit the SAM export in Priority 2 too.
5. **System nvcc is 11.5**, which cannot target sm_120 (Blackwell). Custom native CUDA is blocked unless a CUDA 12.8 toolkit is installed via apt (system change, get user approval first). Triton covers all current kernel needs.
6. **ROS 2 setup.bash + `set -u`**: setup.bash references unbound vars. Shell scripts sourcing it must not use `set -u`.
7. **matplotlib's 3D projection** is broken in this venv (system `mpl_toolkits.mplot3d` wins the namespace-package lookup and its matplotlib is too old). `scripts/render_figures.py` uses 2-D projections instead. Don't waste time fighting this unless you need true 3D rendering.
8. **Replica NICE-SLAM dump has no semantic labels.** The original Meta release does, via a form-gated download. Hand-annotated bboxes in `eval/specs/replica_*.yaml` are derived from `scripts/inspect_replica_mesh.py`.
9. **SAM at native Replica resolution (1200×680) OOMs** after ~5 min. Use `sam_input_shortest_edge: 384` or `512` — the mask output is then nearest-neighbour upsampled.

---

## Validation workflow (run after every substantial change)

```bash
cd /home/yusuf/Projects/personal/openvocab-tsdf
.venv/bin/ruff check src tests benchmarks scripts eval ros2_ws
.venv/bin/black --check --target-version py310 src tests benchmarks scripts eval
.venv/bin/pytest tests/ -q                       # expect 38+4 slow
.venv/bin/python scripts/demo_synthetic.py       # smoke (no dataset)
bash scripts/live_smoke_test.sh                  # live ROS 2 smoke
```

Replica room-scale smoke (requires `~/data/replica/Replica/` to exist):

```bash
.venv/bin/python -m openvocab_tsdf.cli encode \
    --config configs/replica_room0.yaml \
    --output /tmp/smoke_map.npz
.venv/bin/python eval/eval_grounding.py \
    --map /tmp/smoke_map.npz --spec eval/specs/replica_room0.yaml
```

Expected: hit@1 ≥ 20 %, hit@5 ≥ 40 %.

---

## Non-goals (do not redo these)

- **Do not reintroduce 3DGS as the backbone.** Explicitly rejected in `docs/decisions.md` 2026-04-12.
- **Do not add SLAM / pose estimation.** Poses come from the dataset / bag / external stack.
- **Do not train models.** All encoders are frozen; no fine-tuning, no adapters.
- **Do not port to Jetson.** Owned by the sibling `go2-semantic-nav` + `GO2-Perception-Optimization` projects.
- **Do not rewrite the dense reference backend.** It is the correctness oracle. Touch it only to fix bugs or extend its interface (and every change must keep all three parity tests passing).

---

## How to pick up

1. Read this file (you just did).
2. `git log --oneline | head -40` — recent history and commit tags.
3. `cat AGENTS.md` — rules.
4. `cat docs/decisions.md` — why things are the way they are.
5. `.venv/bin/pytest tests/ -q` — baseline.
6. Pick Priority 1, 2, 3, or 4 based on time budget. 1 is 1 session, 2 is 1–2 sessions, 3 depends on the ScanNet download, 4 is a half-session each.
7. Commit atomically; update `README.md` and `docs/decisions.md` when a design choice is made or reversed.
