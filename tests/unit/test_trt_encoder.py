"""TensorRT encoder parity test.

Marked slow + gpu; also skipped automatically if TensorRT is not installed.
Confirms that the TRT engine produces features that agree with the PyTorch
encoder to within a tight cosine-similarity tolerance on a fixed random image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

TRT_AVAILABLE = True
try:
    import tensorrt  # noqa: F401
except ImportError:
    TRT_AVAILABLE = False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not TRT_AVAILABLE, reason="tensorrt not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def test_tensorrt_matches_pytorch(tmp_path: Path):
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder
    from openvocab_tsdf.semantics.trt_encoder import (
        TRTImageEncoderConfig,
        build_engine_from_openclip,
    )

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8) for _ in range(8)]

    # PyTorch reference
    enc = OpenCLIPEncoder(
        OpenCLIPConfig(
            model="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda:0", dtype="fp16"
        )
    )
    ref = enc.encode_images(imgs, batch_size=8)

    # TensorRT engine (built once into tmp_path)
    trt_cfg = TRTImageEncoderConfig(
        engine_path=tmp_path / "clip.engine",
        onnx_path=tmp_path / "clip.onnx",
        batch_size=8,
    )
    trt_enc = build_engine_from_openclip(trt_cfg)
    out = trt_enc.encode_images(imgs)

    assert out.shape == ref.shape, f"shape mismatch: TRT {out.shape} vs ref {ref.shape}"
    # cosine similarity per-row between TRT and ref
    cos = (ref * out).sum(dim=-1) / (ref.norm(dim=-1) * out.norm(dim=-1) + 1e-8)
    mean_cos = float(cos.mean().item())
    assert mean_cos > 0.98, f"TRT/pytorch cosine similarity too low: {mean_cos}"
