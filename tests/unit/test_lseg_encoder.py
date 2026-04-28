"""Unit tests for the LSeg encoder.

The full model requires a ~400 MB checkpoint. These tests exercise:
  1. DPT head shape math (synthetic weights, no checkpoint)
  2. Residual / fusion block plumbing
  3. Text encoder normalization invariant (real CLIP, marked slow+gpu)
"""

from __future__ import annotations

import pytest
import torch


def test_dpt_head_output_shape():
    """DPT head produces (B, 512, H, W) from four feature tensors."""
    from openvocab_tsdf.semantics.lseg_encoder import _DPTHead

    head = _DPTHead(in_channels=768, out_channels=512)
    B, C, H, W = 2, 768, 7, 7
    features = [torch.randn(B, C, H, W) for _ in range(4)]
    out = head(features, input_size=(224, 224))
    assert out.shape == (2, 512, 224, 224)


def test_residual_conv_unit_identity():
    """ResidualConvUnit is a residual block — output shape matches input."""
    from openvocab_tsdf.semantics.lseg_encoder import _ResidualConvUnit

    rcu = _ResidualConvUnit(channels=64)
    x = torch.randn(1, 64, 16, 16)
    out = rcu(x)
    assert out.shape == x.shape


def test_feature_fusion_block_upsample():
    """FeatureFusionBlock doubles spatial dims."""
    from openvocab_tsdf.semantics.lseg_encoder import _FeatureFusionBlock

    ffb = _FeatureFusionBlock(channels=64)
    x = torch.randn(1, 64, 8, 8)
    out = ffb(x)
    assert out.shape == (1, 64, 16, 16)


def test_feature_fusion_block_with_residual():
    """FeatureFusionBlock with a residual of different spatial size."""
    from openvocab_tsdf.semantics.lseg_encoder import _FeatureFusionBlock

    ffb = _FeatureFusionBlock(channels=64)
    x = torch.randn(1, 64, 4, 4)
    residual = torch.randn(1, 64, 8, 8)
    out = ffb(x, residual)
    assert out.shape == (1, 64, 16, 16)


def test_lseg_encoder_constants():
    """LSegEncoder exposes the right constants for the pipeline to consume."""
    from openvocab_tsdf.semantics.lseg_encoder import LSegEncoder

    assert LSegEncoder.FEATURE_DIM == 512
    assert LSegEncoder.TEXT_MODEL == "ViT-B-32"
    assert LSegEncoder.TEXT_PRETRAINED == "openai"


def test_lseg_encoder_missing_weights_raises(tmp_path):
    """Constructor raises FileNotFoundError when checkpoint missing."""
    from openvocab_tsdf.semantics.lseg_encoder import LSegEncoder

    with pytest.raises(FileNotFoundError, match="LSeg weights not found"):
        LSegEncoder(weights_path=tmp_path / "missing.ckpt", device="cpu")


@pytest.mark.slow
@pytest.mark.gpu
def test_lseg_text_encoder_produces_normalized_512d():
    """CLIP ViT-B/32 text encoder (used by LSeg) produces 512-d normalized."""
    import open_clip

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(["a sofa", "the floor"]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    assert feat.shape == (2, 512)
    norms = feat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
