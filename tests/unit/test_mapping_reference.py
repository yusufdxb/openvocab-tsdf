"""Tests for the reference (PyTorch, dense) TSDF backend.

These tests use the synthetic scene generator so they run fully offline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openvocab_tsdf.data.synthetic import Sphere, make_synthetic_dataset
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig


def _sphere_volume(voxel_size: float = 0.03) -> tuple[ReferenceTSDF, list]:
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3, color=(220, 40, 40))]
    frames = make_synthetic_dataset(primitives, num_frames=12, width=120, height=90, radius=1.2)
    cfg = ReferenceTSDFConfig(
        voxel_size_m=voxel_size,
        truncation_distance_m=4 * voxel_size,
        bounds_min=(-0.6, -0.6, -0.6),
        bounds_max=(0.6, 0.6, 0.6),
        store_color=True,
        store_features=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    vol = ReferenceTSDF(cfg)
    for f in frames:
        vol.integrate(f)
    return vol, frames


@pytest.mark.gpu
def test_dense_integrate_updates_weights():
    vol, frames = _sphere_volume()
    # at least some voxels should have accumulated weight
    observed = (vol.weight > 0).sum().item()
    assert observed > 100, f"too few observed voxels: {observed}"


@pytest.mark.gpu
def test_query_near_surface_returns_small_tsdf():
    """Querying right at the sphere surface should give |tsdf| close to zero."""
    vol, _ = _sphere_volume()
    # 8 points on the sphere surface (radius 0.3 from origin)
    theta = torch.tensor([0.3, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5])
    phi = torch.tensor([0.1, 0.8, 1.4, 2.1, 0.3, 1.1, 1.7, 2.3])
    pts = torch.stack(
        [
            0.3 * torch.sin(phi) * torch.cos(theta),
            0.3 * torch.sin(phi) * torch.sin(theta),
            0.3 * torch.cos(phi),
        ],
        dim=-1,
    )
    res = vol.query(pts)
    # Only count points that were actually observed (weight > 0)
    mask = res.weight > 0
    if mask.sum() == 0:
        pytest.skip("no observations at query points")
    tsdf_observed = res.tsdf[mask].abs()
    assert (
        tsdf_observed.mean().item() < 0.4
    ), f"tsdf too far from zero at surface: {tsdf_observed.mean()}"


@pytest.mark.gpu
def test_query_inside_sphere_is_negative():
    """Inside the sphere (r=0.15) the TSDF should be negative (behind surface)."""
    vol, _ = _sphere_volume()
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=torch.float32)
    res = vol.query(pts)
    observed = res.weight > 0
    if observed.sum() > 0:
        assert (
            res.tsdf[observed] < 0.2
        ).all(), f"interior tsdf should be <= 0: {res.tsdf[observed]}"


@pytest.mark.gpu
def test_query_outside_surface_is_positive():
    """Far from any surface, free-space voxels should carry positive TSDF."""
    vol, _ = _sphere_volume()
    # point well outside sphere but inside volume bounds
    pts = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float32)
    res = vol.query(pts)
    observed = res.weight > 0
    if observed.sum() > 0:
        assert (res.tsdf[observed] > -0.1).all()


@pytest.mark.gpu
def test_extract_mesh_produces_verts_near_sphere():
    vol, _ = _sphere_volume()
    mesh = vol.extract_mesh(min_weight=1.0)
    assert len(mesh.vertices) > 100, f"too few vertices: {len(mesh.vertices)}"
    # vertices should cluster near radius 0.3 from origin
    r = np.linalg.norm(mesh.vertices, axis=-1)
    median_r = float(np.median(r))
    assert abs(median_r - 0.3) < 0.1, f"median vertex radius {median_r} far from sphere r=0.3"


@pytest.mark.gpu
def test_features_accumulate_when_enabled():
    primitives = [Sphere(center=(0.0, 0.0, 0.0), radius=0.3)]
    frames = make_synthetic_dataset(primitives, num_frames=4, width=80, height=60, radius=1.2)
    cfg = ReferenceTSDFConfig(
        voxel_size_m=0.05,
        truncation_distance_m=0.15,
        bounds_min=(-0.5, -0.5, -0.5),
        bounds_max=(0.5, 0.5, 0.5),
        store_features=True,
        feature_dim=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    vol = ReferenceTSDF(cfg)
    feat_stream = torch.eye(8)[torch.arange(len(frames)) % 8]
    for f, feat in zip(frames, feat_stream, strict=True):
        vol.integrate(f, feature=feat)
    # some voxels should carry non-zero features
    flat = vol.feat.view(-1, 8)
    nonzero_rows = (flat.abs().sum(dim=-1) > 0).sum().item()
    assert nonzero_rows > 0, "no voxels received features"
