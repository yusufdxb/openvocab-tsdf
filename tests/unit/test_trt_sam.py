"""TensorRT MobileSAM image-encoder parity test.

Marked slow + gpu; also skipped automatically if TensorRT or MobileSAM
weights are not installed.

**This tests the image_encoder's raw output only**, not the downstream
`SAMDenseFeatureExtractor.extract` pipeline. On random-noise inputs the
mask generator short-circuits (no meaningful segments), so the encoder
output is effectively what downstream code consumes. On real images, fp16
perturbs image embeddings enough that SAM's auto-mask boundaries shift
between backends, and that shift propagates into per-voxel feature
accumulation — hence the default `TRTSamConfig.fp16 = False`. See
`docs/decisions.md` 2026-04-13 "TensorRT MobileSAM image encoder" ADR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

TRT_AVAILABLE = True
try:
    import tensorrt  # noqa: F401
except ImportError:
    TRT_AVAILABLE = False

MOBILE_SAM_WEIGHTS = Path("~/.cache/openvocab_tsdf/weights/mobile_sam.pt").expanduser()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not TRT_AVAILABLE, reason="tensorrt not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not MOBILE_SAM_WEIGHTS.exists(),
        reason=f"MobileSAM weights missing at {MOBILE_SAM_WEIGHTS}",
    ),
]


def _encoder_parity(
    tmp_path: Path, *, fp16: bool, mean_threshold: float, min_threshold: float
) -> None:
    from mobile_sam import sam_model_registry

    from openvocab_tsdf.semantics.trt_sam import (
        TRTSamConfig,
        build_engine_from_mobile_sam,
    )

    tag = "fp16" if fp16 else "fp32"
    cfg = TRTSamConfig(
        engine_path=tmp_path / f"mobile_sam_{tag}.engine",
        onnx_path=tmp_path / f"mobile_sam_{tag}.onnx",
        batch_size=1,
        fp16=fp16,
    )
    trt_enc = build_engine_from_mobile_sam(cfg)

    sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_weights_path))
    sam = sam.to(cfg.device).eval()
    if cfg.fp16:
        sam = sam.half()
    image_encoder = sam.image_encoder

    torch.manual_seed(0)
    inputs = torch.randn(
        8, 3, cfg.input_size, cfg.input_size, device=cfg.device, dtype=torch.float32
    )

    with torch.no_grad():
        ref_outs = []
        for i in range(inputs.shape[0]):
            x = inputs[i : i + 1].to(torch.float16 if cfg.fp16 else torch.float32)
            ref_outs.append(image_encoder(x).float())
        ref = torch.cat(ref_outs, dim=0)

    out = trt_enc(inputs).float()

    assert out.shape == ref.shape, f"shape mismatch: TRT {out.shape} vs ref {ref.shape}"
    out_flat = out.reshape(out.shape[0], -1)
    ref_flat = ref.reshape(ref.shape[0], -1)
    cos = (ref_flat * out_flat).sum(dim=-1) / (ref_flat.norm(dim=-1) * out_flat.norm(dim=-1) + 1e-8)
    mean_cos = float(cos.mean().item())
    min_cos = float(cos.min().item())
    assert mean_cos > mean_threshold, f"mean cosine {mean_cos} < {mean_threshold}"
    assert min_cos > min_threshold, f"min cosine {min_cos} < {min_threshold}"


def test_tensorrt_sam_fp32_matches_pytorch(tmp_path: Path) -> None:
    """fp32 TRT engine ≡ PyTorch fp32 on the image_encoder forward.

    Tight threshold: every sample must agree to cosine ≥ 0.9995.
    """
    _encoder_parity(tmp_path, fp16=False, mean_threshold=0.9995, min_threshold=0.999)


def test_tensorrt_sam_fp16_approx_matches_pytorch(tmp_path: Path) -> None:
    """fp16 engine parity is loose — the tight integration check lives in
    `scripts/sam_quality_sweep.py --backend tensorrt`, not here. This test
    only ensures the engine builds and produces the right shape in the
    right ballpark; real-image mask-gen sensitivity is tracked elsewhere.
    """
    _encoder_parity(tmp_path, fp16=True, mean_threshold=0.98, min_threshold=0.95)
