# CLAUDE_CODE_NEXT.md — next-session prompt

Continuation prompt for the next Claude Code session on `openvocab-tsdf`.
Companion to `AGENTS.md` (hard rules — read that file first).

## Current state (as of 2026-04-13, post-P1+P2)

- **36+ commits on `main`, 44 passing tests** (42 fast + 2 slow TRT: CLIP, MobileSAM).
- **Four mapping backends**, all sharing `TSDFVolume`:
  - `reference` — dense PyTorch TSDF (~1.5 kFPS @ 320×240). Correctness oracle.
  - `triton` — custom Triton dense TSDF on sm_120 (4423 FPS, 25 MB). Parity-tested.
  - `sparse_feature` — dense geometry + voxel-slot sparse features; Triton kernel for the pool update (1.70× over PyTorch sparse).
  - `block_hash` — 8³-voxel blocks + block-slot lookup, frustum-culled integrate, **per-voxel sparse features** composed onto the block pool, **and a runnable save/load format** (`sparse_kind: block_hash`). Verified end-to-end on Replica room0 at 4 cm + SAM-dense: hit@1 55.6 % (matches the 6 cm baseline), 4055 blocks / 1.68 M feat voxels / 1.8 GB on disk.
- **Open-vocab grounding**: SAM-per-mask CLIP (`mode: sam_dense`) via MobileSAM + OpenCLIP. Real-data eval on Replica `room0` hit@1 55.6 % / hit@5 88.9 %; `office0` @ 512/16 hit@1 50.0 %.
- **TensorRT fast paths**:
  - CLIP ViT-B/16: 1280 → 1414 FPS (+10 %).
  - **MobileSAM TinyViT: 133 → 366 FPS (2.74×)**; end-to-end `extract` 1.39 → 4.05 FPS at shortest_edge=384. Toggle with `SAMDenseConfig.image_encoder_backend: tensorrt`. Parity: cosine > 0.98 vs PyTorch on 8 fixed inputs.
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

## Priority 1 — block_hash save/load format ✅ done (2026-04-13)

Wired `encode_and_fuse` to produce a `sparse_kind: block_hash` npz and taught all three downstream loaders to dispatch on it. End-to-end verified on Replica room0 at 4 cm + SAM-dense (hit@1 55.6 %, matching the 6 cm baseline exactly). Fixed an OOM in `BlockHashTSDF.integrate`'s feature merge by chunking at 32 768 voxels — same fix pattern is available for `SparseFeatureTSDF` if needed.

Touchpoints that actually changed: `pipeline.py` (save + ground_text load), `eval/eval_grounding.py`, `viz/heatmap.py`, `mapping/block_hash.py` (adds `densify_block_pool` + `scatter_feat_pool_values` module-level helpers), `configs/replica_room0_4cm_block_hash_sam.yaml`, `tests/unit/test_mapping_block_hash_features.py` (round-trip test). See `docs/decisions.md` 2026-04-13 "block_hash save format" and "Chunked per-voxel feature merge" ADRs.

---

## Priority 2 — TensorRT MobileSAM export ✅ done with a fp16 caveat (2026-04-13)

`src/openvocab_tsdf/semantics/trt_sam.py` exports MobileSAM's `image_encoder` (TinyViT) to ONNX + TRT. `TensorRTSamEncoder` inherits `nn.Module` so it can be swapped into `sam.image_encoder` past the child-module type check. `SAMDenseConfig` has `image_encoder_backend: {pytorch, tensorrt}` (default pytorch) and `trt_fp16: bool = False` (default fp32).

**Why fp32 default.** fp16 passes the cosine > 0.98 random-input parity test cleanly, BUT on real Replica frames the mask-gen downstream of the image embedding is sensitive enough to fp16 quantization that grounding hit@1 on `room0` 6cm SAM drops 55.6 % → 22.2 % (−33.3 pp), far past the plan's ±1 pp tolerance. TRT fp32 matches PyTorch fp32 at mean cos 0.9995 end-to-end; fp16 is kept as an explicit opt-in for speed-over-quality regimes. Full details in `docs/decisions.md` 2026-04-13 "TensorRT MobileSAM" ADR.

**Measured.** image_encoder alone: PyTorch fp16 133 FPS → TRT fp32 147 FPS (+10 %) → TRT fp16 366 FPS (+175 %). End-to-end extract at shortest_edge=384: 1.39 → 4.05 FPS with fp16. fp32 end-to-end FPS similar to PyTorch baseline (encoder isn't the dominant cost in the extract path).

**Follow-up if someone wants the fp16 speedup with parity.** Rewrite MobileSAM's layernorm / GELU to opset-17 `INormalizationLayer` / `Gelu` so TRT can keep them in fp16 cleanly. Or swap MobileSAM for a mask-free dense encoder (MaskCLIP + LSeg) where fp16 quantization doesn't route through mask boundaries.

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
