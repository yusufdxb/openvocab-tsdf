"""Per-mask CLIP dense features via MobileSAM.

ConceptFusion-style dense feature extraction: for each frame, run MobileSAM's
auto-mask generator to get region proposals, run OpenCLIP on each mask crop to
get per-mask embeddings, then lift the per-mask features to a dense (H, W, D)
feature map. Each pixel is assigned the feature of the highest-confidence mask
it belongs to. Pixels not covered by any mask fall back to the frame's global
CLIP embedding.

This is the "LSeg-alternative" path: dense, task-informed features produced by
a strong off-the-shelf segmenter (MobileSAM) + CLIP, without training our own
dense head.

Cost model (RTX 5070, per frame):
  - MobileSAM auto-mask: ~70–150 ms (tiny ViT-T backbone, 40 MB)
  - CLIP on ~30 masks, batched: ~40 ms
  - Total: ~100–200 ms/frame — 5–10× slower than global CLIP alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SAMDenseConfig:
    sam_weights_path: Path = Path("~/.cache/openvocab_tsdf/weights/mobile_sam.pt").expanduser()
    sam_model_type: str = "vit_t"
    # Downsample RGB to this short-edge length before SAM. Runtime scales ~linearly
    # with pixel count so 480→384 is ~2× faster than native. Masks are then
    # upsampled back to the original image size for per-pixel lookup.
    sam_input_shortest_edge: int = 384
    points_per_side: int = 12  # auto-mask grid — 12x12 = 144 candidate prompts
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.90
    min_mask_region_area: int = 100
    clip_batch_size: int = 32
    clip_crop_margin_px: int = 8
    device: str = "cuda:0"
    # Which backend runs MobileSAM's `image_encoder` forward. "pytorch" uses
    # the stock TinyViT in fp16; "tensorrt" swaps in a pre-built fp16 TRT
    # engine (see `trt_sam.py`). The rest of SAM (prompt encoder, mask
    # decoder, auto-mask post-processing) stays unchanged either way.
    image_encoder_backend: str = "pytorch"  # or "tensorrt"
    # Default TRT engine path + precision match `TRTSamConfig` (fp32). fp16
    # is available as an opt-in (big speedup, real grounding regression);
    # see `trt_sam.py` for the details.
    trt_engine_path: Path = Path("outputs/trt/mobile_sam_vit_t_fp32.engine")
    trt_onnx_path: Path = Path("outputs/trt/mobile_sam_vit_t_fp32.onnx")
    trt_fp16: bool = False
    trt_build_on_missing: bool = True


class SAMDenseFeatureExtractor:
    """Build (H, W, D) dense feature maps per frame via SAM + CLIP."""

    def __init__(
        self,
        cfg: SAMDenseConfig,
        clip_encoder,  # OpenCLIPEncoder
    ) -> None:
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        self.cfg = cfg
        self.encoder = clip_encoder
        self.device = torch.device(cfg.device)

        sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_weights_path))
        sam = sam.to(self.device).eval()
        self.sam = sam
        if cfg.image_encoder_backend == "tensorrt":
            self._install_tensorrt_image_encoder()
        elif cfg.image_encoder_backend != "pytorch":
            raise ValueError(
                f"image_encoder_backend must be 'pytorch' or 'tensorrt', "
                f"got {cfg.image_encoder_backend!r}"
            )
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=cfg.points_per_side,
            pred_iou_thresh=cfg.pred_iou_thresh,
            stability_score_thresh=cfg.stability_score_thresh,
            min_mask_region_area=cfg.min_mask_region_area,
        )

    def _install_tensorrt_image_encoder(self) -> None:
        """Swap MobileSAM's image_encoder for a pre-built TRT engine.

        Downstream SAM code only touches `sam.image_encoder(x)`, so a callable
        with the same signature is a drop-in replacement. `TensorRTSamEncoder`
        also exposes `.img_size`, which `SamAutomaticMaskGenerator` reads.
        """
        from openvocab_tsdf.semantics.trt_sam import (
            TensorRTSamEncoder,
            TRTSamConfig,
            build_engine_from_mobile_sam,
        )

        trt_cfg = TRTSamConfig(
            sam_weights_path=self.cfg.sam_weights_path,
            sam_model_type=self.cfg.sam_model_type,
            engine_path=self.cfg.trt_engine_path,
            onnx_path=self.cfg.trt_onnx_path,
            fp16=self.cfg.trt_fp16,
            device=self.cfg.device,
        )
        if trt_cfg.engine_path.exists():
            trt_enc = TensorRTSamEncoder(trt_cfg)
        elif self.cfg.trt_build_on_missing:
            trt_enc = build_engine_from_mobile_sam(trt_cfg)
        else:
            raise FileNotFoundError(
                f"TRT engine missing at {trt_cfg.engine_path} and " "trt_build_on_missing=False"
            )
        # The mask generator caches `predictor.model.image_encoder` at call time,
        # so replacing the attribute on `sam` is sufficient.
        self.sam.image_encoder = trt_enc

    @torch.no_grad()
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) → per-pixel features (H, W, D) fp32 L2-normalized.

        Pixels outside every mask get the frame's global CLIP embedding as a
        fallback, so the output is fully defined. Mask features are blended
        by `predicted_iou` when multiple masks overlap (unusual with SAM's
        auto-generator).
        """
        from PIL import Image as PILImage

        H_orig, W_orig = rgb.shape[:2]
        D = self.encoder.feature_dim

        # 0) Downsample to the SAM input resolution for speed; we'll upsample
        #    the final feature map at the end.
        short = min(H_orig, W_orig)
        scale = self.cfg.sam_input_shortest_edge / short
        if scale < 1.0:
            H_sam = int(round(H_orig * scale))
            W_sam = int(round(W_orig * scale))
            rgb_sam = np.asarray(
                PILImage.fromarray(rgb).resize((W_sam, H_sam), PILImage.BILINEAR),
                dtype=np.uint8,
            )
        else:
            rgb_sam = rgb
            H_sam, W_sam = H_orig, W_orig
        H, W = H_sam, W_sam

        # 1) SAM auto-masks (on the small image)
        masks = self.mask_generator.generate(rgb_sam)
        if len(masks) == 0:
            # fall back to global feature everywhere
            global_feat = self.encoder.encode_images([rgb])[0]  # (D,)
            g_np = global_feat.cpu().numpy().astype(np.float32)
            return np.broadcast_to(g_np, (H_orig, W_orig, D)).copy()

        # 2) Crop each mask's bbox, run CLIP in a single batch
        margin = self.cfg.clip_crop_margin_px
        crops: list[np.ndarray] = []
        mask_arrs: list[np.ndarray] = []
        ious: list[float] = []
        for m in masks:
            bx, by, bw, bh = (int(v) for v in m["bbox"])
            x0 = max(0, bx - margin)
            y0 = max(0, by - margin)
            x1 = min(W, bx + bw + margin)
            y1 = min(H, by + bh + margin)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            # mask + crop are in the DOWNSAMPLED image space
            crop = rgb_sam[y0:y1, x0:x1].copy()
            mm = m["segmentation"][y0:y1, x0:x1].astype(bool)
            crop[~mm] = 0
            crops.append(crop)
            mask_arrs.append(m["segmentation"].astype(bool))
            ious.append(float(m["predicted_iou"]))

        if not crops:
            global_feat = self.encoder.encode_images([rgb])[0]
            g_np = global_feat.cpu().numpy().astype(np.float32)
            return np.broadcast_to(g_np, (H_orig, W_orig, D)).copy()

        feats = self.encoder.encode_images(crops, batch_size=self.cfg.clip_batch_size)  # (N, D)
        feats_np = feats.cpu().numpy().astype(np.float32)

        # 3) Lift to (H, W, D). Use accumulation weighted by IoU to average
        # overlaps, then renormalize.
        sum_feat = np.zeros((H, W, D), dtype=np.float32)
        sum_w = np.zeros((H, W), dtype=np.float32)
        for i, (m_arr, iou) in enumerate(zip(mask_arrs, ious, strict=True)):
            sum_feat[m_arr] += feats_np[i] * iou
            sum_w[m_arr] += iou

        # 4) For uncovered pixels, use the global frame feature
        global_feat = self.encoder.encode_images([rgb])[0].cpu().numpy().astype(np.float32)
        uncovered = sum_w == 0
        sum_feat[uncovered] = global_feat
        sum_w[uncovered] = 1.0

        out = sum_feat / np.maximum(sum_w[..., None], 1e-6)
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.clip(n, 1e-8, None)

        # Upsample the feature map back to the original image resolution via
        # nearest-neighbour (we want per-pixel mask labels, not blended feats).
        if (H_sam, W_sam) != (H_orig, W_orig):
            t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(self.device)
            t = torch.nn.functional.interpolate(t, size=(H_orig, W_orig), mode="nearest")
            out = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return out.astype(np.float32)
