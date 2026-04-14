# Architectural Decisions (ADR log)

Append-only log. Each entry: date, decision, rationale, alternatives considered, reversal record if any.

---

## 2026-04-12 — TSDF + sparse voxel hashing as core, not 3DGS

**Decision.** The geometric backbone is a GPU-resident TSDF with sparse voxel hashing. 3DGS is explicitly out of the v1 scope. If a rendering layer is added later, it is additive, not a replacement for the voxel map.

**Rationale.** TSDF / voxel maps are queryable by point, box, and ray in O(1) to O(log n); they integrate cleanly with planners; their correctness is measurable against mesh ground truth. 3DGS is optimized for photorealism, not geometric queries or replan-time consumption, and every "robotics 3DGS" project we surveyed ends up reintroducing a voxel or mesh companion to do the actual robotics work. For a portfolio project arguing "I can build a real-time 3D perception system for robotics," starting from 3DGS would dilute the claim.

**Alternatives considered.**
- *Pure 3DGS.* Rejected: rendering-centric, not planner-friendly.
- *NeRF / neural field.* Rejected: training-time cost, slower queries, same planner mismatch.
- *Dense TSDF only (no hashing).* Rejected for v1 target of 10 m³ @ 2 cm — dense grid works at that scale but does not scale past it, and hashing is the harder, more differentiating implementation.

**Reversal triggers.** If patch/region-level CLIP aggregation proves insufficient even at dense voxel granularity, a neural radiance / feature field might re-enter the scope. Not before.

---

## 2026-04-12 — Custom CUDA is in scope and welcome

**Decision.** Hot-path integration and aggregation kernels are written in custom CUDA. A slower PyTorch reference implementation is maintained for parity testing.

**Rationale.** The sibling `go2-semantic-nav` repository explicitly bans custom kernels; they are owned by `GO2-Perception-Optimization`. openvocab-tsdf is the right project to demonstrate that work because (a) it is a desktop-class target where iteration is cheap, (b) custom CUDA is exactly the portfolio gap we are trying to fill, and (c) the reference implementation gives us a non-CUDA escape path for CI and for debugging.

**Alternatives considered.**
- *nvblox.* Mature, fast. Rejected because importing a black box undermines the "I built this" claim; may be referenced for comparison benchmarks.
- *Open3D TSDF.* Python-bound, not GPU-resident in a useful way for our pipeline. Used as a validation oracle, not as the runtime.
- *PyTorch-only.* Fast enough for a demo, not fast enough to earn the "serious GPU systems" framing.

---

## 2026-04-12 — OpenCLIP ViT-B/16 as baseline VLM

**Decision.** Default open-vocab encoder is OpenCLIP ViT-B/16, fp16, frozen weights. ViT-L/14 available as a quality-mode switch. Dense segmentation-style encoders (LSeg / OpenSeg) and SAM-based mask features are Phase 2b experiments.

**Rationale.** Reproducibility, broad benchmark coverage, predictable VRAM use. Starting dense would entangle semantic aggregation with mask-prompt engineering on day one; global features first lets us nail the 3D aggregation math before adding that complexity.

**Alternatives considered.** Same as above; noted as Phase 2b.

---

## 2026-04-12 — Python 3.10 + uv, PyTorch 2.11 + CUDA 12.8

**Decision.** Environment management via `uv`. Python 3.10 (system default; reproducible across mewtwo and CI). PyTorch 2.11.0+cu128 (already installed, works on RTX 5070 sm_120).

**Rationale.** RTX 5070 is Blackwell sm_120 — many libraries do not yet ship wheels for it. The installed PyTorch build works; changing Python or CUDA versions risks losing that. `uv` is faster and less surprising than conda or poetry for a pure-Python project that shells out to a CMake-built CUDA extension.

**Alternatives considered.** conda (slower, heavier), poetry (slower resolver), system pip (less reproducible).

---

## 2026-04-13 — Phase 4: TensorRT CLIP encoder as opt-in fast path

**Decision.** Add a TensorRT-backed image encoder that serves the same (N, D) L2-normalized embeddings as the PyTorch reference. ONNX is exported from the OpenCLIP visual tower with the legacy tracing exporter (static batch), and a TRT fp16 engine is built on first call. The PyTorch encoder remains the reference.

**Measured perf.** On RTX 5070 (12 GB), batch 16, 224×224, random inputs:
- PyTorch fp16: 1280 FPS
- TensorRT fp16: 1414 FPS (+10 %)
- Parity: cosine similarity vs PyTorch > 0.98 on 8 fixed inputs.

**Rationale.** Even a modest win matters when encoding thousands of frames in a real-dataset run, and TRT becomes a much bigger lever at ViT-L/14 or higher resolutions where the visual tower's forward dominates host-side work. The export pipeline is in place and the engine build is cached, so the next model upgrade is one command away.

**Why static-batch ONNX.** PyTorch 2.11's dynamo-based exporter fails on CLIP's multi-head attention when the batch axis is marked dynamic (`Cannot view a tensor with shape [197, s77, 12, 64]...`). The legacy tracing exporter (`dynamo=False`) works fine for static shapes; we pad the last batch on the Python side to make the engine's fixed batch size acceptable. When dynamic-batch TRT engines become important (variable-batch live inference), revisit with either a newer PyTorch ONNX stack or a `trtexec`-based build off a dynamic-axis ONNX we author by hand.

---

## 2026-04-13 — Scoring refinements: scene-mean subtract + negative prompt

**Decision.** `rank_query` grows two optional refinements that stack: (a) `scene_mean_subtract` subtracts the mean cosine score over observed surface voxels before thresholding, and (b) `neg_text_embedding` subtracts the cosine score of a negative-prompt vector (relative-prompt delta). Both default off.

**When these help.** CLIP has a broad, scene-dependent similarity bias: many voxels score around a per-scene baseline. Mean subtraction isolates voxels that are *unusually* similar to the query, and negative prompts isolate specific adjectives from their category (e.g., "a red chair" vs "a chair"). On cluttered real-data maps these corrections are usually net positive.

**Observed on synthetic.** On the 3-object rendered scene used in tests, with 0.1 m bbox slack: plain global features already hit@1 = 33 % / hit@5 = 100 %. Adding `scene_mean_subtract` drops to hit@1 = 0 % / hit@5 = 67 % because the three objects are approximately equi-baseline for CLIP — subtracting the mean hurts rather than helps. So both flags default off. Turn them on when CLIP's per-voxel score histogram is unimodal with a heavy mass near the query cosine, which is the typical real-data pattern.

---

## 2026-04-13 — Phase 2b: patch features land with MaskCLIP trick + near-surface feature gating

**Decision.** Patch-feature mode (`semantics.mode: patch`) now lifts CLIP per-patch tokens into voxels via (a) the MaskCLIP-style last-block attention bypass, (b) per-voxel patch lookup using the encoder's preprocess mapping (resize-shortest-edge + center-crop), and (c) a near-surface feature aggregation gate (features only accumulate on voxels whose normalized TSDF is in `[-1, 1]`). Rationale:

- **MaskCLIP trick.** Standard CLIP ViT mixes all tokens in its final self-attention, so patch tokens lose spatial locality. MaskCLIP (Zhou et al., 2022) observes that replacing the last block's attention with its value-only projection restores per-patch spatial meaning. We apply that only in the last transformer block and leave earlier blocks untouched.
- **Near-surface feature gate.** Before this change, features were accumulated on every voxel within the TSDF truncation band — including free-space voxels *in front of* a surface. Those voxels would steal the features of whatever eventually occluded the ray, producing spurious hotspots. The new gate restricts feature accumulation to voxels whose `|tsdf|` is within the truncation band (i.e., the object's surface shell).
- **Surface-only querying.** `rank_query` gains `surface_only: bool = True` + `surface_tsdf_abs_max: float = 0.5` so grounding is evaluated only on the surface shell. This matches the semantics of how features were written.

**Known limitation.** On the synthetic ray-traced scene used for tests (3 untextured primitives on a black background), patch localization is still weak: natural-image CLIP is out-of-distribution on that content, and the synthetic rendering does not exercise realistic patch statistics. Real-data validation (Replica, ScanNet) is the proper test. The integration test therefore only asserts that patch mode runs end-to-end and produces non-empty clusters for each query; spatial-accuracy gating moves to the real-dataset eval.

**Reversal triggers.** If on Replica the patch-MaskCLIP features fail to improve grounding hit-rate over global features by a meaningful margin (say, +20 pp hit@1), reconsider: try intermediate-layer features, learned 2D→3D projection, or drop to a dedicated dense encoder like LSeg or OpenSeg.

---

## 2026-04-12 — Global CLIP features in v1; spatial localization is coarse and that is honest

**Decision.** v1 uses a single global CLIP embedding per frame, pooled per voxel by weighted mean. This delivers a working open-vocab pipeline end-to-end but its spatial localization quality is bounded by how much information a single per-frame embedding can carry about *where in the frame* a concept sits.

**Observed behavior on synthetic 3-object scene (red sphere, green floor slab, blue bar).**
- Query-ordering on colors is correct: "a red ball" produces higher-scored clusters than "a blue bar" in a red-dominant view, and vice versa.
- Query-direction is partially correct: "a blue bar" localizes on the scene's blue-bar half-space; "a red ball" localizes near the sphere in some configurations.
- Localization is noisy at ≤ 4 cm voxel size: the top-percentile clusters drift to the "most observed" region rather than to the object's true bbox.

**Why this is acceptable for v1.** The goal of Phase 2 is to prove the full pipeline runs and produces a non-random signal. It does. The goal of Phase 2b / Phase 3 is to improve localization with either patch-token features, SAM-based mask features, or dense segmentation encoders (LSeg / OpenSeg). Those are known fixes and are scheduled.

**Reversal triggers.** If Phase 2b lands and dense patch/region features do not materially improve grounding accuracy on a real dataset (ScanNet val), revisit the aggregation strategy: learned pooling, observation-visibility weighting, or a background-subtracted score like "score_voxel minus scene-mean-score per query".

**Implementation note.** `rank_query` now supports a `top_percentile` mode (e.g., keep the top 2 % of observed voxels) so threshold tuning is adaptive across queries with different absolute CLIP similarity magnitudes.

---

## 2026-04-12 — Triton for the first "custom kernel" backend; native CUDA/CMake deferred

**Decision.** The first fast mapping backend is written in **Triton**, not hand-written CUDA+CMake. Native CUDA is deferred to Phase 4 (optional) when we either (a) install CUDA 12.8 toolkit or (b) justify the additional complexity with a measurable gap Triton cannot close.

**Rationale.** The system nvcc is 11.5 and cannot target the RTX 5070's Blackwell `sm_120` compute capability — native CUDA builds would require a toolkit install. Triton 3.6 ships with PyTorch 2.11 and already supports `sm_120` via the bundled build, so the kernel path works today with no system changes. Triton is also the idiomatic choice for new GPU work at this scale in 2025/2026; hand-written `.cu` is still valuable but is a second-order optimization unless Triton blocks us.

**What this does NOT change.**
- The reference PyTorch implementation remains the correctness oracle.
- Every Triton kernel must pass a parity test against the reference within tolerance.
- Every benchmark claim still requires a JSON file in `benchmarks/results/`.
- The "custom GPU kernel" framing for portfolio purposes is accurate: Triton kernels are real kernels with explicit memory layout, block structure, and masking.

**Reversal triggers.** (1) Install CUDA 12.8 toolkit and re-evaluate if we need handwritten kernels for sparse voxel hashing that Triton cannot express cleanly. (2) Triton generates poor code for a specific kernel — drop down to CUDA for that one kernel only.

---

## 2026-04-12 — No SLAM; consume external poses

**Decision.** The pipeline consumes camera poses from dataset metadata, recorded bags, or an external localization stack. We do not build or embed a SLAM component.

**Rationale.** SLAM is an entire project of its own. Grounding quality is bounded by pose quality; we keep that variable external and auditable.

**Alternatives considered.** Integrating ORB-SLAM3 or a learned VO — rejected for scope.

---

## 2026-04-13 — block_hash save format: separate dispatch via `sparse_kind`

**Decision.** `encode_and_fuse` now writes three distinct npz layouts, selected by a `sparse_kind` string field at load time: `dense` (ReferenceTSDF, legacy "sparse=False"), `voxel_slot` (SparseFeatureTSDF, legacy "sparse=True" with a dense `voxel_slot[Nx,Ny,Nz]` lookup), and `block_hash` (BlockHashTSDF, with a `block_slot[Nbx,Nby,Nbz]` + block-pool structure and a double indirection through `feat_voxel_slot[NumBlocks, 512]` → `feat_pool[NumFeatVoxels, D]`). All three downstream loaders (`pipeline.ground_text`, `eval/eval_grounding.py`, `viz/heatmap.py`) dispatch on `sparse_kind`.

The old `sparse: bool` key stays alongside for back-compat with maps saved before this commit; new code reads `sparse_kind` and falls back to `sparse` only when it is missing.

**Rationale.** Until this commit, `encode_and_fuse`'s save path only knew two layouts and the block_hash backend could not produce a runnable npz, so the combined backend (block-hash geometry + per-voxel sparse features) was benchmark-only: no Replica hit@1 numbers, no eval. Wiring the save format is the step that converts "benchmark result" into "usable system at warehouse scale." `sparse_kind` was chosen over bumping `sparse` to an enum-coded int because string keys read well in an npz's `.files` list during debugging, and npz is forgiving of unused keys so old readers keep working.

**Scatter helpers live next to the format.** `mapping/block_hash.py` exposes `densify_block_pool` (scatter a scalar or channel pool into a dense `(Nx, Ny, Nz)` at room scale) and `scatter_feat_pool_values` (scatter per-feat-voxel scalars — usually `feat_pool @ query` — into a dense `(Nx, Ny, Nz)` score tensor via the double indirection, without ever materialising the 4-D feature volume). The helpers are module-level so the three loaders can reuse them without holding a live backend instance.

**Verified.** `configs/replica_room0_4cm_block_hash_sam.yaml` end-to-end: 100 frames, 4055 blocks (39.6 MB geom), 1.68M feat voxels (3.3 GB features compressed to 1.8 GB on disk). Eval on the hand-annotated room0 spec: hit@1 = 55.6 %, matching the 6 cm SAM baseline within ±1 pp as the success criterion required. Round-trip test in `tests/unit/test_mapping_block_hash_features.py` asserts metadata completeness, densified-weight parity with the in-memory `_densify`, and per-query score parity across all basis queries on every observed voxel.

**Scale note.** At room scale the load-side densify of `tsdf` and `weight` materialises two `(Nx, Ny, Nz)` fp32 volumes (≈ 22 MB each at 4 cm on room0). At warehouse scale this will OOM and the grounding path must stay block-sparse the whole way. Not fixed now because P1's success criterion is room scale; the ≥ 50 m³ case is a follow-up tracked under the combined-backend benchmark.

---

## 2026-04-13 — Chunked per-voxel feature merge in BlockHashTSDF.integrate

**Decision.** `BlockHashTSDF.integrate`'s per-voxel feature update processes `idx_flat` in `CHUNK = 32_768`-sized slices instead of all at once.

**Observed incident.** At 4 cm voxels with `sam_dense` features on the 100-frame Replica room0 sweep, the single-shot path `(f_old * w + feat) / (w + 1)` allocated ≈ 3 × (N, 512) fp32 temporaries where N ≈ 465 k near-surface voxels in some frames — ~2.8 GiB of intermediates. OOM'd at frame 25/100 on a 12 GB card.

**Fix.** Slice the merge into CHUNK rows at a time after feature-slot allocation. Peak per-chunk alloc is ~200 MiB, well below the pool's steady-state footprint. No behavioural change: `index_copy_` is an in-place write per slice, and every slice touches a disjoint set of pool rows because `fslot_long` is per-voxel (no aliasing across chunks).

**Why not the same fix in SparseFeatureTSDF.** SparseFeatureTSDF already has two feature-update backends (`pytorch` and a Triton kernel). The Triton path is bandwidth-bound and does not create intermediates; the PyTorch path is slower but also less likely to be used at 4 cm + sam_dense (that's exactly the regime where block_hash takes over). If a SparseFeature user hits OOM with the PyTorch kernel, apply the same chunk pattern there.

---

## 2026-04-13 — TensorRT MobileSAM image encoder: fp32 default, fp16 opt-in

**Decision.** Add a TensorRT-backed MobileSAM image encoder (`src/openvocab_tsdf/semantics/trt_sam.py`), drop-in replacement for `sam.image_encoder(x)`, toggled via `SAMDenseConfig.image_encoder_backend: {pytorch, tensorrt}`. Default precision is **fp32**, not fp16. fp16 is a separate opt-in via `TRTSamConfig.fp16 = True` / `SAMDenseConfig.trt_fp16 = True`.

**Why fp32 default — the key finding.** The initial fp16 engine passed cosine > 0.98 on random-input image_encoder parity (the naive smoke test). On real Replica frames through the full `SAMDenseFeatureExtractor.extract` pipeline, fp16 output disagreed with PyTorch fp32 at mean cosine 0.44 (min 0.28). Running the quality sweep with fp16 TRT dropped `room0` grounding hit@1 from 55.6 % → 22.2 % (−33.3 pp) and hit@5 from 88.9 % → 44.4 %, far past the plan's ±1 pp tolerance. Root cause: MobileSAM is not a plain feature extractor — its embedding drives a prompt-conditioned mask decoder, and tiny fp16 perturbations in the embedding shift auto-generated mask boundaries, which cascades into different CLIP-per-mask crops and therefore different per-voxel features.

fp32 eliminates that: TRT fp32 vs PyTorch fp32 on the same real frame through the same pipeline gives mean cosine 0.9995 (end-to-end dense feature map, not just encoder output). Grounding matches PyTorch within the ±1 pp budget. Cost is that the fp16 speedup (2.74×) collapses to ~10 %; fp32 is what makes the TRT path correctness-preserving.

**Measured perf.** RTX 5070 (12 GB), static batch 1, 1024×1024 input (the only size MobileSAM's TinyViT accepts — positional-embedding shapes are baked in):

| path | PyTorch fp32 | TRT fp32 (default) | TRT fp16 (opt-in) |
|---|---|---|---|
| image_encoder forward alone | — | 147 FPS, 6.8 ms | 366 FPS, 2.7 ms |
| (baseline) PyTorch fp16 image_encoder | 133 FPS, 7.5 ms | | |
| `extract` @ shortest_edge=384, end-to-end | 1.39 FPS, 717 ms | ≈ 1.5 FPS (sweep) | 4.05 FPS, 247 ms |
| `extract` @ shortest_edge=512, end-to-end | 1.29 FPS, 778 ms | ≈ 1.4 FPS (sweep) | 4.04 FPS, 248 ms |
| room0 hit@1 (grounding, 6 cm sam-dense) | 55.6 % | ≈ 55.6 % (parity) | 22.2 % (broken) |

The fp16 column is kept available because the speedup is real and some downstream uses tolerate the quality drop (e.g., live-mapping preview at 3+ Hz where the grounding signal is a visual sanity check, not the authoritative ranking). Callers opt in by setting `trt_fp16=True`.

**`TensorRTSamEncoder` inherits `nn.Module`.** `nn.Module.__setattr__` enforces that attributes previously registered as child modules can only be reassigned to another `nn.Module` (or `None`); since we swap `sam.image_encoder` after construction, the wrapper must also be an `nn.Module`. The engine and IO tensors stay as plain attributes so `.to()` / `.train() / .eval()` are effectively no-ops — the engine is pinned to the device it was built on.

**Why static batch 1.** SAM is called once per image inside `SamAutomaticMaskGenerator` — there is no frame-level batching on the public API. A static-batch-1 engine uses the smallest kernel plans TRT can pick and avoids dynamic-shape dispatch overhead.

**Why the random-input parity test was misleading.** Random-noise inputs cause SAM's auto-mask generator to short-circuit (no meaningful segments), so the downstream mask-dependent path doesn't exercise the fp16 sensitivity. The encoder output's raw fp16↔fp32 cosine is still > 0.98, which is a legitimate bound on the raw embedding — just not on the end-to-end pipeline. The integration truth test is `scripts/sam_quality_sweep.py` on real data, which is now what is actually being used to gate the TRT default.

**Reversal triggers.** (1) Rewriting MobileSAM's layernorm / GELU to opset-17 `INormalizationLayer` / `Gelu` recovers fp16 parity on real images — if someone proves that, fp16 can become the default. (2) Someone builds a mask-free single-pass dense encoder (e.g., MaskCLIP + LSeg) where fp16 perturbations don't change auto-mask boundaries — then fp16 + TRT is lossless by construction. (3) TRT engine builds become a portability burden across sm targets.

---
