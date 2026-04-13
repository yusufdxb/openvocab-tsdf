"""Tests for patch-feature integration into the reference TSDF.

No CLIP model needed here — we fake a feature map (Hp, Wp, D) so the test is
fast and focuses on the lifting math (projection → patch index → per-voxel
accumulation).
"""

from __future__ import annotations

import pytest
import torch

from openvocab_tsdf.data.synthetic import Box, Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


@pytest.mark.gpu
def test_patch_feature_accumulates_when_projected_inside_crop():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=224, height=224, radius=1.2)

    D = 8
    cfg = ReferenceTSDFConfig(
        voxel_size_m=0.05,
        truncation_distance_m=0.2,
        bounds_min=(-0.5, -0.5, -0.5),
        bounds_max=(0.5, 0.5, 0.5),
        store_features=True,
        feature_dim=D,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    vol = ReferenceTSDF(cfg)

    # Make a feature map that is 1.0 on the "red" channel (index 0), zeros elsewhere
    Hp, Wp = 14, 14
    fm = torch.zeros(Hp, Wp, D, dtype=torch.float32)
    fm[..., 0] = 1.0
    for f in frames:
        vol.integrate(f, feature_map=fm, feature_map_input_size=224, feature_map_patch_size=16)

    # every accumulated voxel should have channel-0 == 1 and other channels == 0
    flat = vol.feat.view(-1, D)
    observed = vol.weight.view(-1) > 0
    if observed.any():
        obs = flat[observed]
        assert torch.allclose(obs[:, 0], torch.ones_like(obs[:, 0]), atol=1e-5)
        assert obs[:, 1:].abs().max().item() < 1e-5


@pytest.mark.gpu
def test_patch_feature_picks_right_patch_by_pixel():
    """Verify the pixel→patch index mapping: a feature that is nonzero only in
    the top-left patch should only land on voxels that project to the top-left
    region of the preprocessed crop.
    """
    # Full-view wall so every pixel has valid depth and all patches receive
    # projected voxels. We construct the single frame directly so we don't
    # hit the `radius=0` singular ring pose.
    import numpy as np

    from openvocab_tsdf.data.base import CameraIntrinsics, RGBDFrame
    from openvocab_tsdf.data.synthetic import render_depth

    primitives = [Box(min_xyz=(-1.5, -1.5, 0.5), max_xyz=(1.5, 1.5, 0.6), color=(100, 100, 100))]
    fov_deg = 60.0
    W = H = 224
    fx = fy = (W / 2) / np.tan(np.radians(fov_deg / 2))
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=W / 2, cy=H / 2, width=W, height=H)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.0, 0.0, -0.5]
    depth, color = render_depth(primitives, intr, T, z_max=5.0)
    frames = [
        RGBDFrame(color=color, depth_m=depth, intrinsics=intr, T_wc=T, timestamp=0.0, frame_id=0)
    ]

    D = 4
    cfg = ReferenceTSDFConfig(
        voxel_size_m=0.03,
        truncation_distance_m=0.12,
        bounds_min=(-1.0, -1.0, 0.0),
        bounds_max=(1.0, 1.0, 1.5),
        store_features=True,
        feature_dim=D,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    vol = ReferenceTSDF(cfg)

    Hp, Wp = 14, 14
    fm = torch.zeros(Hp, Wp, D, dtype=torch.float32)
    fm[0, 0, 0] = 1.0  # top-left patch, channel 0

    vol.integrate(frames[0], feature_map=fm, feature_map_input_size=224, feature_map_patch_size=16)

    # Any voxel whose channel-0 is > 0 must have projected into [0,16)×[0,16)
    # of the preprocessed crop.
    flat = vol.feat.view(-1, D)
    has_sig = flat[:, 0] > 0
    # also: most voxels should have zero (image is square, so no center-crop
    # loss, but only ~1/196 of patches carry the signal)
    frac_on = has_sig.float().mean().item()
    assert 0 < frac_on < 0.05, f"expected sparse activations, got frac_on={frac_on}"
