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
