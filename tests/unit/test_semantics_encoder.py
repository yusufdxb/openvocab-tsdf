"""Smoke tests for the OpenCLIP encoder wrapper.

Running the real model is slow (model download, model load). These tests are
marked `slow` and `gpu` so they stay out of the default fast run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.gpu
def test_openclip_encodes_image_and_text_matching():
    """Text 'a photo of grass' should score higher on green images than red ones.

    This is a sanity check on our wrapper, not a benchmark of CLIP quality.
    """
    from openvocab_tsdf.semantics.aggregation import assert_normalized
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    cfg = OpenCLIPConfig(
        model="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="fp16" if torch.cuda.is_available() else "fp32",
    )
    enc = OpenCLIPEncoder(cfg)

    assert enc.feature_dim == 512, f"unexpected feature_dim {enc.feature_dim}"

    green = np.full((224, 224, 3), (40, 180, 60), dtype=np.uint8)
    red = np.full((224, 224, 3), (200, 40, 40), dtype=np.uint8)
    imgs = enc.encode_images([green, red])
    assert imgs.shape == (2, 512)
    assert_normalized(imgs)

    txts = enc.encode_texts(["a photo of grass", "a photo of a red brick wall"])
    assert txts.shape == (2, 512)
    assert_normalized(txts)

    # grass prompt should prefer green, brick prompt should prefer red
    sim = imgs @ txts.T  # (2,2) image x text
    grass_green = sim[0, 0].item()
    grass_red = sim[1, 0].item()
    red_green = sim[0, 1].item()
    red_red = sim[1, 1].item()
    assert grass_green > grass_red, f"grass↔green={grass_green}, grass↔red={grass_red}"
    assert red_red > red_green, f"red↔red={red_red}, red↔green={red_green}"
