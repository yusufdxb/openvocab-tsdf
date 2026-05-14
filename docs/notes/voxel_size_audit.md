# Voxel Size Audit: LSeg 6 cm vs SAM-dense 4 cm

**Date:** 2026-05-14
**Author:** read-only diagnostic pass (no code changed)

---

## 1. Is voxel size a confounder? YES.

Verified from configs and pipeline code:

| Config | Encoder | voxel_size_m | truncation_distance_m | feature_dim |
|---|---|---|---|---|
| `configs/replica_room0_6cm_lseg.yaml` (line 18) | LSeg ViT-B/32 | **0.06** | 0.24 | 512 |
| `configs/replica_room0_4cm_block_hash_sam.yaml` (line 23) | SAM-dense ViT-B/16 | **0.04** | 0.16 | 512 |
| `configs/replica_room0_4cm_block_hash_sam_vitl14.yaml` (line 20) | SAM-dense ViT-L/14 | **0.04** | 0.16 | 768 |

The headline comparison in `docs/decisions.md` (2026-05-12 entry, lines 370-398) states LSeg at 11.1% room0 hit@1 against SAM-dense baselines at 22.2% (ViT-B/16) and 33.3% (ViT-L/14), but those baselines run at 4 cm. This is **not** a controlled comparison.

Three mechanisms make voxel size a direct confounder:

**a) Cluster size filter.** `rank_query` in `src/openvocab_tsdf/grounding/query.py` (line 133) rejects clusters with `count < min_cluster_voxels`. The eval spec `eval/specs/replica_room0.yaml` (line 27) sets `min_cluster_voxels: 20`. At 6 cm, one voxel covers 0.216 cm^3; at 4 cm it covers 0.064 cm^3. A physical object that spans 20 4-cm voxels spans only ~6 6-cm voxels; it may be dropped entirely by the cluster filter at 6 cm.

**b) `cluster_eps_vox` is in voxel units.** `eval/specs/replica_room0.yaml` (line 26) sets `cluster_eps_vox: 2`. Two voxels of dilation = 12 cm at 6 cm grids vs 8 cm at 4 cm. The spatial footprint of the dilation kernel is different, changing which peaks merge and which stay split.

**c) `top_percentile` draws from a different-sized observed set.** At 6 cm, room0 produced 137,486 feat voxels (verified from `docs/decisions.md` line 372); at 4 cm it produced 1.68M (line 147). The `top_percentile: 0.005` threshold (room0 spec line 25) selects top-0.5%, which is a very different absolute count between the two grids.

**d) Additional confounders not directly related to voxel size:** LSeg uses ViT-B/32 text space; SAM-dense uses ViT-B/16 or ViT-L/14. These are not cross-comparable text spaces, and model-name validation in `pipeline.ground_text` (enforced per `docs/decisions.md` 2026-04-23 entry) confirms this. This is a second independent confound on top of the voxel size mismatch.

---

## 2. Why did the configs diverge?

Source: `docs/decisions.md` (2026-04-23 LSeg entry, lines 338-346) and `docs/superpowers/plans/2026-04-23-dense-encoder-docker-paper.md` (Task 8, lines 795-860).

The 6 cm choice was **deliberate, not accidental drift**. Two reasons are documented:

1. **VRAM budget:** The LSeg `lseg_minimal_e200.ckpt` is 3.1 GB (noted in `src/openvocab_tsdf/semantics/lseg_encoder.py` docstring, line 29). SAM-dense MobileSAM is ~150 MB plus a CLIP model. At 4 cm the block_hash feature pool for room0 is 3.3 GB compressed / 1.8 GB on disk (decisions.md line 147). Putting the 3.1 GB LSeg backbone plus a 4 cm feature pool on a 12 GB RTX 5070 simultaneously was not tested and the plan author chose 6 cm to "keep VRAM headroom" (decisions.md 2026-05-12 entry, line 388).

2. **Described as "matching the original LSeg config"** (decisions.md 2026-05-12, line 387): the LSeg configs were authored in Task 8 of the plan and from the start used 6 cm.

The `docs/decisions.md` 2026-05-12 entry (lines 386-392) explicitly flags the mismatch as a caveat: "LSeg uses 6 cm voxels vs 4 cm for the SAM-dense configs... The hit@5 gap between LSeg and SAM-dense is partly attributable to the coarser voxel grid." So the caveat was noted by the author but not resolved before shipping the headline number.

---

## 3. Cost of a fair-comparison rerun

Two options:

**Option A: LSeg at 4 cm.**

ASSUMPTION: VRAM estimate is from static figures in the repo, not a live measurement.

- LSeg checkpoint: 3.1 GB resident (lseg_encoder.py docstring)
- block_hash geometry at 4 cm room0: ~39.6 MB (decisions.md line 147)
- block_hash feature pool at 4 cm, 512-d: at 4 cm room0 had 1.68M feat voxels * 512 * 4 bytes = ~3.4 GB uncompressed. At 6 cm (137K feat voxels) it is ~280 MB. Moving from 6 to 4 cm multiplies feat voxels by ~12x, putting the live pool well above 3 GB.
- Total at 4 cm: 3.1 GB (LSeg backbone) + 3.4 GB (feat pool) + ~0.5 GB PyTorch base = ~7 GB. This likely fits in 12 GB VRAM but sits right at the OOM boundary. The chunked-merge patch in `BlockHashTSDF.integrate` (decisions.md 2026-04-13 entry, lines 153-162) was required at 4 cm + SAM-dense for exactly this reason; LSeg at 4 cm would exercise the same path.
- Code change: set `voxel_size_m: 0.04` and `truncation_distance_m: 0.16` in the LSeg config. No pipeline changes. Runtime: similar to SAM-dense 4 cm encode (~180-265 s for 100 frames per decisions.md line 179; LSeg is faster per-frame since no mask generation but the pool is bigger). ASSUMPTION: total runtime ~120-180 s, not measured.

**Option B: SAM-dense at 6 cm.**

- Lower VRAM risk (no 3.1 GB backbone).
- Existing SAM-dense 6 cm baseline result is cited as 55.6% (the "6 cm SAM baseline" in the room0 4 cm config comment, line 9): `configs/replica_room0_4cm_block_hash_sam.yaml` header cites this as the target to match. That result was run with a different (earlier) pipeline version. A clean 6 cm SAM-dense eval at the current pipeline version would be needed to pair with the LSeg 6 cm number.
- Code change: change `voxel_size_m: 0.06` and `truncation_distance_m: 0.24` in the SAM-dense config.
- Net effort: lower VRAM risk, but requires a fresh SAM-dense encode + eval run to ensure the same pipeline version is compared.

**Preferred option:** Option B (SAM at 6 cm with current pipeline) is safer to run first because it avoids the VRAM cliff. If it reproduces close to 55.6% it establishes the 6 cm SAM baseline cleanly, and LSeg at 11.1% at the same voxel size becomes a meaningful (though encoder-space-mismatched) comparison. Option A (LSeg at 4 cm) would be the full controlled comparison but carries OOM risk without a test run.

---

## 4. Recommendation for the paper outline

**Retract the current LSeg row from Table 2 in `docs/paper_outline.md` (lines 104-112) until the voxel size is matched.**

The current paper_outline.md Table 2 (line 107-109) shows the LSeg row as "n/a" because the LSeg encoder had an architecture bug at outline-writing time (noted in the table note, lines 113-115). The decisions.md 2026-05-12 entry filled in real numbers after that bug was fixed, but those numbers are at a different voxel size from the SAM-dense rows.

Specific actions before publication:

1. Add a "voxel_size" column to Table 2. This alone makes the comparison honest even if the mismatch is retained, provided it is explained.
2. Run SAM-dense at 6 cm with the current pipeline to produce a matched baseline (Option B above). Update Table 2 with that number alongside the existing LSeg 6 cm result.
3. Optionally run LSeg at 4 cm (Option A) for the fully controlled comparison; gate on whether the OOM risk materializes.
4. The `docs/decisions.md` caveat (2026-05-12, lines 386-392) should be promoted into the paper's Section 3.2 as an explicit limitation note. The current paper_outline Limitations section (lines 152-157) does not mention the voxel size mismatch; it needs a bullet.

The 11.1% number is not wrong as a raw measurement. It is wrong as a term in the "three-way encoder comparison" table if the other rows use 4 cm grids. Do not cite it as evidence that LSeg underperforms SAM-dense without the matching baseline.
