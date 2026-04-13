"""Parity tests: Triton backend vs reference backend on synthetic scenes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig

try:
    from openvocab_tsdf.mapping.triton_backend import TritonTSDF, TritonTSDFConfig

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_TRITON, reason="triton not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _make_scene(voxel_size: float = 0.03, num_frames: int = 8):
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(
        primitives, num_frames=num_frames, width=96, height=72, radius=1.2
    )
    bounds = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    trunc = 4 * voxel_size
    return frames, voxel_size, trunc, bounds


def test_triton_matches_reference_tsdf():
    frames, voxel_size, trunc, (bmin, bmax) = _make_scene()

    ref_cfg = ReferenceTSDFConfig(
        voxel_size_m=voxel_size,
        truncation_distance_m=trunc,
        bounds_min=bmin,
        bounds_max=bmax,
        store_color=True,
        device="cuda:0",
    )
    ref = ReferenceTSDF(ref_cfg)
    for f in frames:
        ref.integrate(f)

    tri_cfg = TritonTSDFConfig(
        voxel_size_m=voxel_size,
        truncation_distance_m=trunc,
        bounds_min=bmin,
        bounds_max=bmax,
        store_color=True,
        device="cuda:0",
    )
    tri = TritonTSDF(tri_cfg)
    for f in frames:
        tri.integrate(f)

    # Volumes must have the same dims
    assert ref.dims == tri.dims

    # Weights should be identical: same geometry, same projection, same updates
    w_ref = ref.weight.cpu().numpy()
    w_tri = tri.weight.cpu().numpy()
    np.testing.assert_allclose(w_tri, w_ref, atol=0.0, rtol=0.0)

    # TSDF: floating-point tolerance acceptable
    t_ref = ref.tsdf.cpu().numpy()
    t_tri = tri.tsdf.cpu().numpy()
    # where weight > 0, compare
    observed = w_ref > 0
    diff = np.abs(t_ref[observed] - t_tri[observed])
    assert diff.max() < 1e-3, f"TSDF max diff {diff.max()}"
    assert diff.mean() < 1e-5, f"TSDF mean diff {diff.mean()}"

    # Color: tolerate small numerical divergence
    c_ref = ref.color.cpu().numpy()
    c_tri = tri.color.cpu().numpy()
    diff_c = np.abs(c_ref[observed] - c_tri[observed])
    assert diff_c.max() < 1e-2, f"color max diff {diff_c.max()}"


def test_triton_extract_mesh_nonempty():
    frames, voxel_size, trunc, (bmin, bmax) = _make_scene()
    tri_cfg = TritonTSDFConfig(
        voxel_size_m=voxel_size,
        truncation_distance_m=trunc,
        bounds_min=bmin,
        bounds_max=bmax,
        store_color=True,
        device="cuda:0",
    )
    tri = TritonTSDF(tri_cfg)
    for f in frames:
        tri.integrate(f)
    mesh = tri.extract_mesh(min_weight=1.0)
    assert len(mesh.vertices) > 50
    # vertices near sphere radius 0.3
    r = np.linalg.norm(mesh.vertices, axis=-1)
    assert abs(float(np.median(r)) - 0.3) < 0.1
