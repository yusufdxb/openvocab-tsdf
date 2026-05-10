"""Unit tests for the LSeg encoder.

The encoder targets the public ``lseg_minimal_e200.ckpt`` (~3.1 GB) from
the Intel ISL LSeg release. Architecture: ViT-L/16 384 backbone + DPT
reassembly (project-readout) + DPT decoder with 256 internal channels +
1x1 head to 512-d CLIP space.

Tests cover, in order:

  1. Decoder primitive shape math (BN-style RCU, FeatureFusion with
     ``out_conv``).
  2. DPT head end-to-end shape ((B, {256,512,1024,1024}, H, W) -> (B, 512, H, W)).
  3. ``LSegEncoder`` constants the pipeline depends on.
  4. ``LSegEncoder`` raises a clear error when weights are missing.
  5. ``LSegDPT`` strict checkpoint load against the real ckpt -- only runs
     when the checkpoint is present locally; otherwise skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_LSEG_CKPT = Path(
    os.environ.get(
        "OPENVOCAB_TSDF_LSEG_CKPT",
        Path.home() / ".cache/openvocab_tsdf/weights/lseg_minimal_e200.ckpt",
    )
)


def test_residual_conv_unit_identity():
    """BN-style ``_ResidualConvUnit`` preserves input shape."""
    from openvocab_tsdf.semantics.lseg_encoder import _ResidualConvUnit

    rcu = _ResidualConvUnit(channels=64)
    x = torch.randn(1, 64, 16, 16)
    out = rcu(x)
    assert out.shape == x.shape


def test_feature_fusion_block_upsample():
    """``_FeatureFusionBlock`` doubles spatial dims (no residual path)."""
    from openvocab_tsdf.semantics.lseg_encoder import _FeatureFusionBlock

    ffb = _FeatureFusionBlock(channels=64)
    x = torch.randn(1, 64, 8, 8)
    out = ffb(x)
    assert out.shape == (1, 64, 16, 16)


def test_feature_fusion_block_with_residual():
    """``_FeatureFusionBlock`` with a residual of different spatial size
    bilinearly resizes ``x`` to match the residual before fusion, then
    upsamples by 2 and applies the 1x1 ``out_conv``.
    """
    from openvocab_tsdf.semantics.lseg_encoder import _FeatureFusionBlock

    ffb = _FeatureFusionBlock(channels=64)
    x = torch.randn(1, 64, 4, 4)
    residual = torch.randn(1, 64, 8, 8)
    out = ffb(x, residual)
    assert out.shape == (1, 64, 16, 16)


def test_dpt_head_output_shape():
    """``_DPTHead`` consumes per-scale features and yields (B, 512, H, W).

    Inputs use the canonical DPT-Large channel widths (256, 512, 1024, 1024)
    after reassembly. This is the same configuration the real LSeg checkpoint
    expects.
    """
    from openvocab_tsdf.semantics.lseg_encoder import _DPTHead

    head = _DPTHead(
        hooks_out_channels=(256, 512, 1024, 1024),
        inner_channels=256,
        out_channels=512,
    )
    B, H, W = 2, 7, 7
    features = [
        torch.randn(B, 256, H, W),
        torch.randn(B, 512, H, W),
        torch.randn(B, 1024, H, W),
        torch.randn(B, 1024, H, W),
    ]
    out = head(features, input_size=(224, 224))
    assert out.shape == (2, 512, 224, 224)


def test_lseg_encoder_constants():
    """``LSegEncoder`` exposes the constants the pipeline keys off of."""
    from openvocab_tsdf.semantics.lseg_encoder import LSegEncoder

    assert LSegEncoder.FEATURE_DIM == 512
    assert LSegEncoder.TEXT_MODEL == "ViT-B-32"
    assert LSegEncoder.TEXT_PRETRAINED == "openai"
    assert LSegEncoder.INPUT_SIZE == 384


def test_lseg_encoder_missing_weights_raises(tmp_path):
    """Constructor raises ``FileNotFoundError`` with an actionable message
    when the checkpoint isn't present.
    """
    from openvocab_tsdf.semantics.lseg_encoder import LSegEncoder

    with pytest.raises(FileNotFoundError, match="LSeg weights not found"):
        LSegEncoder(weights_path=tmp_path / "missing.ckpt", device="cpu")


@pytest.mark.skipif(
    not _LSEG_CKPT.exists(),
    reason=f"LSeg checkpoint not at {_LSEG_CKPT}",
)
def test_lseg_dpt_loads_real_checkpoint_strict():
    """``LSegDPT`` loads the real ``lseg_minimal_e200.ckpt`` with strict=True.

    The remap routes ``net.pretrained.model.* -> backbone.*``,
    ``net.pretrained.act_postprocess{N}.* -> act_postprocess{N}.*``, and
    ``net.scratch.* -> scratch.*``. CLIP keys (``net.clip_pretrained.*``)
    are dropped because the text encoder is loaded separately via
    ``open_clip``. After remap there must be 0 missing and 0 unexpected
    model keys -- this is the smoke test that catches architecture drift.
    """
    from openvocab_tsdf.semantics.lseg_encoder import LSegDPT

    # strict=True raises if any model key is missing or any remapped ckpt
    # key is unexpected. No exception -> all keys aligned.
    LSegDPT(_LSEG_CKPT, strict=True)
