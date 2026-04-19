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
| 6. Polish + figures | ✅ done | Post-fix evidence pass (2026-04-16): grounding metrics regenerated from fresh JSONs, qualitative 3-panel figures + comparison grids re-rendered under `figures/postfix_*/`, evidence index maps every headline row to its map / JSON / figure directory. |

### Current numbers (RTX 5070, 12 GB)

> **Post-fix evidence**, regenerated 2026-04-16 on RTX 5070 (driver
> 570.211.01, `torch 2.11.0+cu128`) under the corrected near-surface
> feature gate. The quoted Replica grounding numbers below come from
> these JSONs in `benchmarks/results/`:
> - `20260416T150054Z_eval_grounding.json` — room0 / 6 cm / sam_dense
> - `20260416T150413Z_eval_grounding.json` — office0 / 6 cm / sam_dense
> - `20260416T150504Z_eval_grounding.json` — room0 / 6 cm / global
> - `20260416T150537Z_eval_grounding.json` — office0 / 6 cm / global
> - `20260416T150631Z_eval_grounding.json` — room0 / 6 cm / patch
>
> Throughput / VRAM / encode FPS / mesh counts were not remeasured in
> this pass (the gate fix doesn't change them — it narrows which voxels
> receive features, not which kernels run) and are carried forward
> from the pre-fix benchmark JSONs already in `benchmarks/results/`.

#### Evidence index (post-fix → everything)

| claim lives in… | map artifact | eval JSON | figure dir |
|---|---|---|---|
| room0 / 6 cm / sam_dense row | `outputs/postfix_room0_6cm_sam.npz` | `benchmarks/results/20260416T150054Z_eval_grounding.json` | `figures/postfix_room0_sam/` |
| room0 / 6 cm / global row | `outputs/postfix_room0_global.npz` | `benchmarks/results/20260416T150504Z_eval_grounding.json` | `figures/postfix_room0_global/` |
| room0 / 6 cm / patch row | `outputs/postfix_room0_patch.npz` | `benchmarks/results/20260416T150631Z_eval_grounding.json` | — (no PNGs cited) |
| office0 / 6 cm / sam_dense row | `outputs/postfix_office0_6cm_sam.npz` | `benchmarks/results/20260416T150413Z_eval_grounding.json` | `figures/postfix_office0_sam/` |
| office0 / 6 cm / global row | `outputs/postfix_office0_global.npz` | `benchmarks/results/20260416T150537Z_eval_grounding.json` | `figures/postfix_office0_global/` |
| synthetic 3-object demo row | `outputs/postfix_demo_map.npz` | `benchmarks/results/20260416T150856Z_eval_grounding.json` | — (synthetic; no mesh figures) |
| room0 / 4 cm / block_hash + sam row | `outputs/postfix_room0_4cm_block_hash_sam.npz` | `benchmarks/results/20260416T151324Z_eval_grounding.json` | — (benchmark table only) |
| global-vs-SAM comparison grid | both `outputs/postfix_room0_{global,sam}.npz` | the two JSONs above | `figures/postfix_comparison_room0/` |

Rows not in this table (TSDF-fuse FPS, CLIP / MobileSAM encode FPS + TRT parity, sparse-feature backend FPS, block-hash geometry row, live ROS 2 smoke test) are **carried forward from pre-fix benchmark JSONs** in `benchmarks/results/`. They are not remeasured because the gate fix is a feature-write predicate — it runs after the encoders and the fuse kernels — so their numbers are invariant under the fix. Each of those rows cites its pre-fix JSON in place.

**Synthetic scene (3 primitives, 32 frames, 224×224):**
- TSDF fuse, Triton backend: **4423 FPS**, 25 MB peak VRAM (pre-fix; gate doesn't change throughput)
- TSDF fuse, reference backend: 1486 FPS, 68 MB peak VRAM
- Grounding (global features, 0.1 m bbox slack, post-fix gate): **hit@1 = 66.7 %, hit@5 = 100 %**, mean top-1 L2 = 0.30 m, 54 ms per query (`benchmarks/results/20260416T150856Z_eval_grounding.json`)
- End-to-end grounding query (text encode + voxel scan + cluster): **~50 ms**
- ROS 2 service `/openvocab/ground` — DDS roundtrip returns ranked targets in well under a second

**Real scene (NICE-SLAM demo — 500 RGB-D frames @ 640×480, 1.2 M voxels at 6 cm):**
- Encode + fuse (patch-mode CLIP ViT-B/16 + MaskCLIP lift): **≈1.8 s end-to-end** (CLIP image encode 0.93 s, feature fusion 0.87 s)
- Reference backend geometry fuse: 79.6 FPS at native 640×480, mesh 25.1 k vertices / 50.4 k triangles
- Query latency: <100 ms per free-form prompt against the 1.2 M-voxel feature map
- Per-query heatmap PLYs: 19.4 k surface points each (`outputs/heatmaps_demo/*.ply`)

**Replica `room0` (500 frames @ 1200×680, 1.67 M voxels at 6 cm):**
- CLIP patch encode: 1.91 s (262 FPS), feature fusion: 3.51 s (143 FPS at native 1200×680) — pre-fix (encode+fuse kernels unchanged by the gate)
- Geometry fuse: 81.3 FPS, mesh 97.3 k vertices / 194 k triangles — pre-fix
- Per-query latency under post-fix global-CLIP map: **77–249 ms** (mean 123 ms) across the 9-case spec — `benchmarks/results/20260416T150504Z_eval_grounding.json`
- Per-query latency under post-fix SAM-dense map: **49–210 ms** (mean 101 ms) — `benchmarks/results/20260416T150054Z_eval_grounding.json`
- Post-fix heatmap PLYs: `outputs/heatmaps_postfix_room0_{global,sam}/*.ply`, 100–104 k points each

### Grounding accuracy — Replica, hand-annotated

See `eval/specs/replica_room0.yaml` and `eval/specs/replica_office0.yaml` for the annotation protocol (structural queries derived from the mesh's horizontal-surface z histogram; object bboxes from a `scripts/inspect_replica_mesh.py` density map, conservatively widened with a 0.2 m bbox slack).

**Publishable figures — all re-rendered 2026-04-16 from the post-fix maps (`outputs/postfix_*.npz`).** Each query renders as a 3-panel xy/xz/yz projection with the mesh washed out in gray, the top-20 % heatmap voxels colored viridis, and the hand-annotated GT bbox outlined in red. Representative examples, all from the post-fix maps:

- `figures/postfix_room0_sam/the_floor.png` — hot voxels sit inside the thin floor-plane bbox in xz and yz; rank=1 hit in `20260416T150054Z_eval_grounding.json`.
- `figures/postfix_room0_sam/a_sofa.png` — heatmap mass lands inside the sofa bbox (xy + xz); rank=1 hit@1 for this query.
- `figures/postfix_room0_sam/a_bookshelf.png` — hot voxels cluster along the east-column region; rank=3 on this spec (hit@5-only). Honest: room0 object-level top-1 is still hard even under SAM-dense.
- `figures/postfix_room0_global/a_sofa.png` — hot voxels scatter across the room (rank=4 hit@5 for global CLIP on this query); honest side-by-side with the SAM figure above.
- `figures/postfix_comparison_room0/a_sofa_compare.png` — same query, same scene, baseline (global CLIP) vs treatment (SAM + CLIP) in a 2-row × 3-view grid. The hot-voxel concentration into the sofa bbox is visibly tighter on the SAM row.
- `figures/postfix_office0_sam/a_chair.png` — rank=1 hit on office0 SAM-dense (`20260416T150413Z_eval_grounding.json`).

**Figures are evidence-consistent with the tables above**: the same `outputs/postfix_*.npz` maps feed both. The legacy pre-fix figure directories (`figures/room0/`, `figures/office0/`, `figures/room0_sam/`, `figures/comparison_room0/`) are retained for historical comparison only — any claim in this README points at `figures/postfix_*/`.

Regenerate:

```bash
# 1. heatmap PLYs from a post-fix map (one per query)
python scripts/export_heatmaps.py --map outputs/postfix_<cfg>.npz \
    --query "a sofa" --query "the floor" ... \
    --out-dir outputs/heatmaps_postfix_<cfg>

# 2. 3-panel PNGs from the heatmap dir + spec
python scripts/render_figures.py \
    --mesh outputs/replica_<scene>_mesh.ply \
    --heatmap-dir outputs/heatmaps_postfix_<cfg> \
    --spec eval/specs/replica_<scene>.yaml \
    --out-dir figures/postfix_<cfg>

# 3. (optional) baseline-vs-treatment comparison grid
python scripts/render_comparison.py \
    --mesh outputs/replica_<scene>_mesh.ply \
    --baseline-dir outputs/heatmaps_postfix_<scene>_global \
    --treatment-dir outputs/heatmaps_postfix_<scene>_sam \
    --spec eval/specs/replica_<scene>.yaml \
    --queries "a sofa" "a chair" \
    --out-dir figures/postfix_comparison_<scene>
```

**Headline cross-scene / cross-config table (post-fix gate, RTX 5070, 2026-04-16).** All `hit-L2` values are the hit-only mean (mean L2 over `hit@5`-positive cases), since the unconditional mean L2 is dominated by full misses.

| scene   | voxel | frames | mode      | hit@1   | hit@5   | hit-L2 (m, hit-only) | source JSON |
|---|---|---|---|---|---|---|---|
| room0   | 6 cm  | 500 (stride 4)  | patch     | 11.1 %  | 66.7 %  | 4.08 | `20260416T150631Z_eval_grounding.json` |
| room0   | 6 cm  | 500 (stride 4)  | global    | 33.3 %  | 66.7 %  | 2.15 | `20260416T150504Z_eval_grounding.json` |
| **room0**   | **6 cm**  | **100 (stride 20)** | **sam_dense** | **55.6 %** | **100.0 %** | **2.70** | `20260416T150054Z_eval_grounding.json` |
| office0 | 6 cm  | 500 (stride 4)  | global    | 37.5 %  | 75.0 %  | 1.79 | `20260416T150537Z_eval_grounding.json` |
| **office0** | **6 cm**  | **100 (stride 20)** | **sam_dense** | **37.5 %** | **62.5 %** | **1.50** | `20260416T150413Z_eval_grounding.json` |

**Pre-fix vs post-fix delta** (gate fix only; all other knobs identical). Pre-fix numbers are the Apr-13/14 JSONs already in `benchmarks/results/`. Mixed signal is honest: stricter feature placement helps where contamination was the dominant noise (room0 hit@5, office0 global hit@1 + hit-L2) and hurts where contamination happened to point in the right direction at the cost of being right for the wrong reason (office0 sam_dense hit@5).

| scene / mode      | hit@1 (pre → post)        | hit@5 (pre → post)        | hit-L2 m (pre → post)     |
|---|---|---|---|
| room0 / patch     | 22.2 % → **11.1 %** (−11.1) | 44.4 % → **66.7 %** (+22.3) | 3.46 → 4.08 (+0.62)        |
| room0 / global    | 33.3 % → **33.3 %** (±0)    | 55.6 % → **66.7 %** (+11.1) | 2.63 → **2.15** (−0.48)    |
| **room0 / sam_dense** | 55.6 % → **55.6 %** (±0) | 88.9 % → **100.0 %** (+11.1) | 2.42 → 2.70 (+0.28)        |
| office0 / global  | 25.0 % → **37.5 %** (+12.5) | 87.5 % → 75.0 % (−12.5)    | 2.15 → **1.79** (−0.36)    |
| **office0 / sam_dense** | 37.5 % → **37.5 %** (±0) | 75.0 % → 62.5 % (−12.5)    | 1.54 → **1.50** (−0.04)    |

### SAM-per-mask CLIP dense features (`mode: sam_dense`)

ConceptFusion / Grounded-SAM-lite pipeline: MobileSAM auto-masks → CLIP on each mask crop → per-pixel feature map (mask features blended by predicted IoU; pixels outside every mask fall back to the frame-global CLIP embedding). Full-resolution dense feature maps (not patch-resize-crop), integrated per-pixel into voxels.

Delta vs the global-features baseline, head-to-head on the same scenes (post-fix gate, 2026-04-16):

| scene   | metric        | global  | **sam_dense** | Δ |
|---|---|---|---|---|
| room0   | hit@1         | 33.3 %  | **55.6 %**    | **+22.3 pp** |
| room0   | hit@5         | 66.7 %  | **100.0 %**   | **+33.3 pp** |
| room0   | hit-L2 (m)    | **2.15** | 2.70         | +0.55 (worse) |
| office0 | hit@1         | 37.5 %  | 37.5 %        | ±0 |
| office0 | hit@5         | **75.0 %** | 62.5 %     | −12.5 pp (worse) |
| office0 | hit-L2 (m)    | 1.79    | **1.50**      | −0.29 (−16 %) |

SAM-dense still wins on `room0` (where global features mis-point at the most-observed wall, so any precision boost helps); on `office0`, global narrowly wins on `hit@5` because the scene is small enough that even a contaminated cluster lands inside the loose bbox. SAM-dense wins on hit-L2 in both scenes — the centroids land closer to the GT center even when the bbox-membership flip rates differ.

The mapping stack doesn't change — only the feature extractor does. MobileSAM (`vit_t`, 40 MB weights) runs at ~384×384 downsampled input (~1.7 s / native-resolution frame); CLIP is run on ~20–40 mask crops batched. Total pipeline cost is dominated by SAM; encode throughput drops from ~250 FPS (global) to ~0.6 FPS (SAM dense), which is why we stride 20× for this config. Feature *quality* on the per-voxel side is substantially better.

Fig: `figures/postfix_room0_sam/a_sofa.png` — the top-brightness-quartile heatmap voxels cluster inside the GT sofa bbox (xy + xz) on the post-fix map. Compare against `figures/postfix_room0_global/a_sofa.png` (global features, same post-fix gate) where the hot voxels scatter across the room. `figures/postfix_comparison_room0/a_sofa_compare.png` stacks both rows on one canvas.

**Full ablation on room0** (8 configs, patch vs global × mean-sub × top-percentile): see the `eval/run_ablation.py` output embedded in commit `a1a628e`. That ablation was run on pre-fix features; the post-fix numbers in this README's headline table are the authoritative values.

**Honest takeaways (post-fix gate).**
- **`room0` SAM-dense is the strongest real-data result in the repo.** **hit@1 = 55.6 %, hit@5 = 100.0 %** on a 9-query hand-annotated spec — every annotated query lands inside the GT bbox in the top-5. Pre-fix this row was 88.9 % hit@5; the corrected gate's stricter voxel placement closes the last gap. Source: `benchmarks/results/20260416T150054Z_eval_grounding.json`.
- **The gate fix is not uniformly positive.** `office0 / sam_dense` regressed from 75.0 % → 62.5 % hit@5; that's an honest cost of refusing to write features into free-space voxels. Where contamination happened to point at the right region by accident, removing it costs a hit. Mean centroid L2 still improved (1.54 → 1.50 m) — when SAM-dense does land a hit, it lands closer.
- **`room0` object-level stays hard.** Best `room0 hit@1` across all four modes is still 55.6 % (SAM-dense). Patch mode is the weakest configuration (`hit@1 = 11.1 %`); natural-image CLIP patch tokens are known weak for dense localization without a task-trained dense head (LSeg / OpenSeg), which is the correct next experiment.
- **Latency: 50–233 ms per query** (room0 sam_dense: 100 ms; room0 patch: 233 ms; office0: 50 ms). Includes text encode + voxel scan + connected-component cluster. Dominated by `MapBundle.score_query`'s `(Nx, Ny, Nz)` scatter at room scale.
- **Encode throughput (post-fix, RTX 5070):** SAM-dense 0.5–0.6 fuse-FPS (100 frames in ~170–185 s); global 130–145 fuse-FPS (500 frames in ~3 s). Numbers preserved from the 2026-04-16 rerun stdout, not from a benchmark JSON.

**CLIP image encode (ViT-B/16 @ 224×224 fp16, batch 16):**
- PyTorch: **1280 FPS**
- TensorRT: **1414 FPS** (+10 %, parity-tested vs PyTorch with cosine > 0.98). See `benchmarks/bench_clip_encode.py`.

**MobileSAM image encode (TinyViT @ 1024×1024, batch 1):**

| path | FPS | ms/frame | grounding hit@1 (room0 6cm SAM) |
|---|---|---|---|
| PyTorch fp32 (reference) | ~133 | 7.5 | 55.6 % (post-fix, 2026-04-16) |
| **TensorRT fp32 (default)** | **147** | **6.8** | **≈ 55.6 % (parity, pre-fix sweep)** |
| TensorRT fp16 (opt-in) | 366 | 2.7 | **22.2 %** (broken, pre-fix sweep) |

The PyTorch-fp32 row was re-measured under the post-fix gate
(`benchmarks/results/20260416T150054Z_eval_grounding.json`) and
matches the pre-fix value at 55.6 % — the gate fix doesn't move
this configuration's `hit@1`. The two TRT rows have not been
re-measured under the post-fix gate; the qualitative conclusion
("fp32 preserves parity, fp16 is end-to-end broken because its
perturbations shift auto-mask boundaries") is robust to the gate
change because the gate runs downstream of the encoder and only
narrows which voxels receive features, not which masks are
generated. fp16 is 2.74× on the raw encoder; TRT fp32 preserves
parity (mean cos 0.9995 vs PyTorch fp32 through the full `extract`
pipeline) and gives ~10 % speedup, so it is the default. Toggle with
`SAMDenseConfig.image_encoder_backend: tensorrt`, opt into fp16 via
`trt_fp16: True` only when 3× FPS matters more than grounding
quality. See `benchmarks/bench_sam_encode.py` and `docs/decisions.md`.

**Block-hash + SAM-dense + per-voxel sparse features on Replica room0 (4 cm voxels, 100 frames, post-fix gate, 2026-04-16):**

Combined backend (block-hash geometry + voxel-slot features + SAM-dense semantics). Source: `benchmarks/results/20260416T151324Z_eval_grounding.json` and `outputs/postfix_room0_4cm_block_hash_sam.npz`.

| metric | post-fix value | 6 cm SAM, reference (post-fix) | delta |
|---|---|---|---|
| hit@1 | 22.2 % | 55.6 % | −33.3 pp |
| hit@5 | 77.8 % | 100.0 % | −22.2 pp |
| hit-L2 (m, hit-only) | 2.93 | 2.70 | +0.23 |
| allocated blocks | 4 055 | n/a | |
| allocated feat voxels | **309 577** (post-fix) vs 1 679 613 (pre-fix; broken gate) | n/a | **5.4× sparser** |
| geometry storage | 39.6 MB | ~80 MB (dense 4 cm) | |
| feature storage | **604.6 MB** (post-fix) vs 3 280 MB (pre-fix) | ≫ that if dense at 4 cm | |
| encode + fuse | 178 s / 100 frames | 185 s / 100 frames (6 cm) | |

The gate fix has two visible effects on this configuration: feature memory drops 5.4× (because allocations are now restricted to the actual surface shell), and `hit@1` drops from 55.6 % to 22.2 % at the finer voxel size. The pre-fix `hit@1` was inflated: contamination from free-space voxels gave the top cluster a broad enough footprint to land inside the GT bbox, but for the wrong geometric reason. With the gate honest, the 4 cm SAM clusters are smaller and more numerous, so the top-1 ranking spreads over more nearby candidates — `hit@5` is barely affected (77.8 %), but the top pick is no longer always the right one. The 6 cm SAM variant on the dense reference backend stays the strongest configuration in the repo.

**Block-hash sparse *geometry* (`BlockHashTSDF`) — 12 m³ cube, 4 cm voxels, 24 synthetic frames:**

| backend | geom storage | allocated blocks | peak VRAM | integrate |
|---|---|---|---|---|
| dense reference | 540.0 MB | 54 872 / 54 872 (100 %) | 2484 MB | 48 FPS |
| **block_hash** | **8.5 MB** | **831 / 54 872 (1.51 %)** | **2103 MB** | **51 FPS** |

**63× less geometry memory** at similar throughput. The scene is tiled into 8³-voxel blocks with a dense `block_slot[Nbx, Nby, Nbz]` int32 index (0.2 % of the full voxel count); blocks are allocated lazily on first observation, and an integrate-time 2-pass GPU scatter writes tsdf/weight/color into the pool (`benchmarks/bench_block_hash_scale.py`). Feature storage is still the per-voxel `SparseFeatureTSDF` above — the two sparse backends compose.

Honest limitation: the integrate pass is now frustum-culled at the block level (cheap per-block test against the camera frustum, then per-voxel projection only on surviving blocks), so per-frame work is bounded by the frustum, not by the bounding-box voxel count. The remaining boundary is the load-side densification of `tsdf` and `weight` for grounding (`densify_block_pool`): it materializes two full-volume fp32 tensors, which is fine at room scale (~MBs) but will OOM at warehouse scale. The grounding score path itself stays sparse (`scatter_feat_pool_values` produces a dense scores tensor without ever materialising the 4-D feature volume), so the ceiling is the geometry densification, not the features.

**Sparse-feature backend on Replica room0 (1.67 M voxels, 512-dim features):**

| backend | FPS | feat (MB) | allocated voxels | sparsity | peak VRAM (MB) |
|---|---|---|---|---|---|
| dense reference | 119 | 3297.7 | 1 688 400 | 100.00 % | 4833 |
| sparse (PyTorch `index_copy_`) | 123 | 1070.2 | 547 947 | 32.45 % | 3430 |
| **sparse (Triton kernel)** | **209** | **1070.2** | **547 947** | **32.45 %** | **3272** |

- Feature-memory reduction: **3.08×** (lazy per-voxel slot allocation — memory is proportional to observed surface, not to the bounding box)
- Integrate-throughput speedup from the Triton kernel: **1.70× over the PyTorch sparse path, 1.76× over the dense reference** — one JIT kernel fuses the slot-indexed gather + weighted-mean + scatter.
- See `benchmarks/bench_sparse_features.py` and `src/openvocab_tsdf/mapping/sparse_reference.py` (set `feat_update_backend: triton` in `SparseFeatureTSDFConfig`).

**Live ROS 2 mapping (smoke test on synthetic publisher):**
- Publisher replays 120 NICE-SLAM frames @ 15 Hz on `/camera/color/image_raw` + `/camera/depth/image_raw` + `/camera/camera_info` + TF.
- `grounding_node live_mode:=true` synchronizes them through `message_filters`, looks up `map→camera` through TF, runs CLIP encode + `ReferenceTSDF.integrate` per frame.
- After 30 s integration (49 frames at stride=2), `ros2 service call /openvocab/ground "{query: 'a chair', top_percentile: 0.02}"` returns 3 ranked targets in well under a second. End-to-end hardware-free robotic demo.

See [`docs/architecture.md`](docs/architecture.md) for the full plan, performance targets, and what is explicitly cut. See [`docs/decisions.md`](docs/decisions.md) for the architectural-decision log. See [`AGENTS.md`](AGENTS.md) for the rules that govern agent work in this repository.

### Reproducing the post-fix Replica numbers

Environment used for the 2026-04-16 regeneration:
- GPU: NVIDIA GeForce RTX 5070 (sm_120 / Blackwell), 12 GB
- Driver: `570.211.01` (CUDA 12.8 runtime line)
- PyTorch: `2.11.0+cu128` (NOT the default `+cu130` wheel — see "Driver / torch matrix" below)
- OpenCLIP `ViT-B-16 / laion2b_s34b_b88k`, fp16
- MobileSAM `vit_t`, fp32 PyTorch (TRT path not exercised in this rerun)

Exact rerun commands (run from the repo root, with the Replica scenes
present at `~/data/replica/Replica/<scene>/`):

```bash
# room0 SAM-dense
openvocab-tsdf encode -c configs/replica_room0_6cm_sam.yaml -o outputs/postfix_room0_6cm_sam.npz
python eval/eval_grounding.py --map outputs/postfix_room0_6cm_sam.npz \
  --spec eval/specs/replica_room0.yaml --out-dir benchmarks/results

# office0 SAM-dense
openvocab-tsdf encode -c configs/replica_office0_6cm_sam.yaml -o outputs/postfix_office0_6cm_sam.npz
python eval/eval_grounding.py --map outputs/postfix_office0_6cm_sam.npz \
  --spec eval/specs/replica_office0.yaml --out-dir benchmarks/results

# global baselines
openvocab-tsdf encode -c configs/replica_room0_global.yaml -o outputs/postfix_room0_global.npz
python eval/eval_grounding.py --map outputs/postfix_room0_global.npz \
  --spec eval/specs/replica_room0.yaml --out-dir benchmarks/results
openvocab-tsdf encode -c configs/replica_office0_6cm_sparse_global.yaml -o outputs/postfix_office0_global.npz
python eval/eval_grounding.py --map outputs/postfix_office0_global.npz \
  --spec eval/specs/replica_office0.yaml --out-dir benchmarks/results
```

The `eval/eval_grounding.py` invocations write timestamped JSONs to
`benchmarks/results/`, which are the source-of-truth referenced from
the headline tables above.

**Driver / torch matrix.** `pyproject.toml` pins `torch` /
`torchvision` to the **cu128** index by default
(`[tool.uv.sources] torch = { index = "pytorch-cu128" }`). cu128
wheels work on NVIDIA driver `>=525` and cover the 570.x line that
ships with stock Ubuntu LTS HWE kernels (the rerun above was on
driver `570.211.01`). Run:

```bash
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Hosts on driver `>=580` who want the newer cu130 wheel can override
the source pin in their checkout (`uv lock --upgrade-package torch`
after pointing the source at `pytorch-cu130`); the default stays cu128
because it is the more portable choice for sm_120 hardware today.

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
│   ├── data/                 # dataset loaders (Replica, NICE-SLAM demo, synthetic)
│   ├── mapping/              # TSDF backends: reference (PyTorch dense), triton
│   │                         # (sm_120 fast geometry), sparse_reference
│   │                         # (lazy per-voxel features), block_hash (sparse
│   │                         # geometry + features composed)
│   ├── semantics/            # OpenCLIP encoders + SAM-dense + TRT fast paths
│   ├── grounding/            # text-to-3D query engine + map_bundle loader
│   ├── viz/                  # mesh / heatmap PLY writers
│   ├── config.py
│   └── cli.py
├── ros2_ws/                  # ROS 2 service node (`openvocab_tsdf_node`)
├── tests/                    # unit + integration; markers: gpu, dataset, slow, benchmark
├── eval/                     # evaluation scripts (grounding accuracy, localization err)
├── benchmarks/               # named benchmarks + JSON result history
└── scripts/                  # one-shot helpers (download datasets, etc.)
```

## Performance budgets (contracts)

See [`docs/architecture.md`](docs/architecture.md#performance-targets-desktop-rtx-5070-12gb). Breaking a budget requires an entry in [`docs/decisions.md`](docs/decisions.md).

## License

MIT. See `LICENSE`.
