# Open-Vocabulary 3D Mapping via GPU-Accelerated Semantic TSDF Fusion

*4-page workshop paper (e.g., RSS Workshop on Open-Vocab Robotics)*

---

## Abstract (~150 words)

[Problem] Robotic systems need to reason about open-ended language queries
in 3D environments, but existing open-vocabulary mapping systems either
sacrifice spatial resolution (global per-frame features) or require
expensive per-pixel models trained on limited label sets.

[Approach] We present openvocab-tsdf, a GPU-accelerated pipeline that
fuses RGB-D streams into a semantic TSDF volume with per-voxel CLIP
features. We implement three dense feature extraction strategies —
SAM-mask + CLIP (ViT-B/16 and ViT-L/14), and LSeg — and compare them
on a common volumetric backbone with four TSDF backends (dense, Triton,
sparse-feature, block-hash).

[Results] On Replica, [best encoder] achieves [X]% hit@1 grounding
accuracy while maintaining real-time fusion rates ([Y] kFPS). The
block-hash backend scales to warehouse-size environments within 12 GB
VRAM.

[Contribution] Open-source system with reproducible Docker builds,
YAML-driven evaluation, and a ROS 2 interface.

---

## 1. Introduction (0.75 page, ~600 words)

### Motivation
- Robots need to ground free-form language ("find the red mug") in 3D
- Existing approaches: ConceptGraphs (object-centric graph), HOV-SG
  (hierarchical), OpenScene (distilled features)
- Gap: no open-source, GPU-accelerated volumetric system that directly
  compares dense feature strategies on the same backbone

### Contributions
1. Four TSDF backends sharing a common `TSDFVolume` protocol, including
   a block-hash backend that scales past 50 m³
2. Three dense feature modes (SAM-dense ViT-B/16, SAM-dense ViT-L/14,
   LSeg) with controlled comparison on Replica
3. A YAML-driven evaluation harness and Docker image for reproducibility
4. A ROS 2 service interface for online deployment

[Figure 1: System diagram — RGB-D input → TSDF fusion → feature
integration → text query → ranked 3D targets]

---

## 2. System (1.5 pages, ~1200 words)

### 2.1 TSDF Backends
- Dense reference (PyTorch) — correctness oracle, ~1.5 kFPS
- Triton kernel — 4.4 kFPS @ 320×240, 25 MB VRAM, parity-tested
- Sparse-feature — dense geometry + voxel-slot sparse features
- Block-hash — 8³ voxel blocks, frustum-culled integrate, per-voxel
  sparse features; only backend that scales past ~50 m³

[Table 1: Backend comparison — FPS, VRAM, max volume size]

### 2.2 Feature Extraction Modes
- **Global CLIP**: one embedding per frame, weighted-mean per voxel.
  Fast, poor spatial resolution.
- **SAM-dense + CLIP**: MobileSAM auto-masks → CLIP per crop → IoU-weighted
  blend → (H,W,D) dense features. Good resolution, bottlenecked by SAM
  (~0.6 FPS).
- **LSeg**: DPT backbone → per-pixel 512-d features aligned with CLIP
  ViT-B/32 text space. Single forward pass, no mask generation. Requires
  pre-trained LSeg checkpoint.

### 2.3 Near-Surface Feature Gate
- Only write features to voxels with |normalized TSDF| ≤ 0.5
- Prevents free-space contamination of surface features

### 2.4 Query Engine
- Text → CLIP embedding → per-voxel cosine score → connected-component
  clustering → top-k ranked targets
- Supports negative queries (score subtraction)

### 2.5 ROS 2 Interface
- `openvocab_tsdf_node`: online fusion from synchronized RGB-D topics
- `/openvocab/ground` service: DDS-based text grounding
- RViz CUBE_LIST voxel preview

[Figure 2: Feature mode comparison — same Replica frame, three
 feature maps visualized as PCA-reduced RGB overlays]

---

## 3. Experiments (1 page, ~800 words)

### 3.1 Setup
- Dataset: Replica (8 scenes, 72 queries)
- Metrics: hit@1, hit@5, mean top-1 centroid L2 (m), encode FPS
- Hardware: RTX 5070 (12 GB), PyTorch 2.11

### 3.2 Grounding Accuracy

[Table 2: Three-way encoder comparison on Replica room0 + office0]

| Encoder | room0 hit@1 | room0 hit@5 | office0 hit@1 | office0 hit@5 |
|---------|-------------|-------------|---------------|---------------|
| SAM-dense ViT-B/16 | X% | X% | X% | X% |
| SAM-dense ViT-L/14 | X% | X% | X% | X% |
| LSeg (DPT) | X% | X% | X% | X% |

### 3.3 Throughput

[Table 3: Encode FPS and VRAM by feature mode]

| Mode | Encode FPS | VRAM (GB) |
|------|-----------|-----------|
| Global CLIP | ~X | X |
| SAM-dense ViT-B/16 | ~X | X |
| SAM-dense ViT-L/14 | ~X | X |
| LSeg | ~X | X |

### 3.4 Scaling

[Table 4: VRAM by backend at room scale (room0, 4 cm)]

| Backend | Geom VRAM | Feat VRAM | Total |
|---------|-----------|-----------|-------|
| Reference (dense) | X MB | X MB | X MB |
| Block-hash | X MB | X MB | X MB |

---

## 4. Discussion + Conclusion (0.5 page, ~400 words)

### Limitations
- Hit@1 accuracy is bounded by CLIP's natural-image distribution —
  rendered Replica scenes underperform real-world data
- LSeg uses an archived checkpoint (Intel ISL); no fine-tuning on
  indoor scenes
- Near-surface gate at band=0.5 is a heuristic — optimal value is
  scene-dependent
- Block-hash query path densifies for scoring, which OOMs at warehouse
  scale

### Future Work
- SigLIP/DFN as drop-in CLIP replacements
- Block-sparse query path (avoid densification)
- ScanNet cross-scene evaluation (infrastructure shipped, data pending)
- TensorRT acceleration for LSeg DPT backbone

### Conclusion
openvocab-tsdf provides a reproducible, GPU-accelerated platform for
comparing open-vocabulary 3D feature strategies. The block-hash backend
and three dense feature modes give researchers a controlled testbed for
semantic TSDF experiments.

---

## References

1. Gu et al., "ConceptGraphs: Open-Vocabulary 3D Scene Graphs," 2024.
2. Werby et al., "Hierarchical Open-Vocabulary 3D Scene Graphs (HOV-SG)," 2024.
3. Peng et al., "OpenScene: 3D Scene Understanding with Open Vocabularies," CVPR 2023.
4. Li et al., "Language-driven Semantic Segmentation (LSeg)," ICLR 2022.
5. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," ICML 2021.
6. Zhang et al., "Faster Segment Anything (MobileSAM)," 2023.
7. Newcombe et al., "KinectFusion," ISMAR 2011.
8. Curless and Levoy, "A Volumetric Method for Building Complex Models from Range Images," SIGGRAPH 1996.
