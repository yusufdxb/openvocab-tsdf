Robust cross-scene grounding will not come from threshold tuning alone. The repo is telling you that pretty clearly: the current 8-scene Replica
  sweep lands at only 15.3% hit@1 / 44.9% hit@5 under the same 4 cm block_hash + sam_dense profile, with very high scene-to-scene variance
  (Projects/personal/openvocab-tsdf/README.md:167).

  What is verified from the current codebase:

  - The semantic stack is still a frozen-feature pipeline: global, patch, or sam_dense, all fused into voxels by a weighted running mean (Projects/
    personal/openvocab-tsdf/docs/architecture.md:67).
  - The query engine is still simple cosine scoring plus connected components, with final rank based on mean_score * log1p(voxel_count) (Projects/
    personal/openvocab-tsdf/src/openvocab_tsdf/grounding/query.py:6, Projects/personal/openvocab-tsdf/src/openvocab_tsdf/grounding/query.py:151).
  - The eval path currently does not pass TSDF into rank_query, so the surface-only filter is effectively disabled there, even though MapBundle
    already exposes tsdf and weight (Projects/personal/openvocab-tsdf/eval/eval_grounding.py:89, Projects/personal/openvocab-tsdf/src/
    openvocab_tsdf/grounding/map_bundle.py:45).
  - The current cross-scene benchmark is also weak as a benchmark: hand-authored Replica centroid-in-box specs, not labeled semantic ground truth
    (Projects/personal/openvocab-tsdf/README.md:188, Projects/personal/openvocab-tsdf/docs/decisions.md:316).

  So the shortest serious path is:

  1. Fix the retrieval path before changing models.
     Pass bundle.tsdf into rank_query in eval and service code so surface_only actually works. Then run a sweep over scene_mean_subtract,
     top_percentile, and negative prompts; those hooks already exist in the engine (Projects/personal/openvocab-tsdf/src/openvocab_tsdf/grounding/
     query.py:77, Projects/personal/openvocab-tsdf/eval/eval_grounding.py:81). This is the cheapest likely gain.
  2. Stop using weighted voxel means as the whole semantic story.
     Right now you fuse CLIP-like features into voxels and hope clustering recovers objects. For cross-scene robustness, you want a stronger dense
     semantic head, not just better fusion. The repo already says the next likely experiment is something like LSeg/OpenSeg or another task-trained
     dense encoder (Projects/personal/openvocab-tsdf/README.md:194). I agree. That is the highest-ROI model change.
  3. Upgrade the benchmark before claiming robustness.
     A robust claim needs real semantic labels. The proper move is still ScanNet once unblocked, because Replica hand boxes are enough for relative
     debugging, not enough for a serious generalization claim (Projects/personal/openvocab-tsdf/docs/architecture.md:55, Projects/personal/
     openvocab-tsdf/docs/decisions.md:298). At minimum, add per-kind metrics now: structural vs object vs loose queries.
  4. Rerun at fuller scene coverage.
     The aggregate sweep used the 100-frame / stride-20 smoke profile (Projects/personal/openvocab-tsdf/README.md:170). That is useful for
     screening, not for “robust cross-scene grounding.” Once the ranking and semantic head improve, rerun on denser coverage, like the stronger 500-
     frame profiles used elsewhere in the README.

  If I were doing this repo next, I would implement it in this order:

  1. Turn on true surface filtering in eval/service.
  2. Add a systematic sweep for ranking knobs and per-kind metrics.
  3. Integrate one stronger dense open-vocab encoder on the reference backend first.
  4. Re-run full-scene evaluations on ScanNet or, if still blocked, denser Replica scenes.
  5. Only if frozen dense heads still fail, consider training/adaptation.
