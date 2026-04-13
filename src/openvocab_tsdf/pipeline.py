"""High-level orchestration glue for the fuse / encode / ground commands.

This module is thin. Every helper here is a composition of the module-level
building blocks in `data/`, `mapping/`, `semantics/`, `grounding/`. Keep it
that way.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openvocab_tsdf.config import Config
from openvocab_tsdf.data.base import RGBDDataset, RGBDFrame
from openvocab_tsdf.data.replica import ReplicaDataset
from openvocab_tsdf.mapping.base import Mesh, TSDFVolume
from openvocab_tsdf.mapping.reference import ReferenceTSDF, ReferenceTSDFConfig

log = logging.getLogger(__name__)


def build_dataset(cfg: Config) -> RGBDDataset:
    d = cfg.dataset
    if d.name == "replica":
        return ReplicaDataset(
            root=d.root,
            scene=d.scene,
            depth_scale=cfg.camera.depth_scale,
            depth_trunc_m=cfg.camera.depth_trunc_m,
            max_frames=d.max_frames,
            stride=d.stride,
        )
    raise NotImplementedError(f"dataset '{d.name}' not implemented yet")


def _auto_bounds(
    dataset: RGBDDataset, radius: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit bounds as a cube of side 2*radius centered on the first camera origin."""
    first = dataset[0]
    c = first.T_wc[:3, 3].astype(np.float32)
    bmin = (float(c[0] - radius), float(c[1] - radius), float(c[2] - radius))
    bmax = (float(c[0] + radius), float(c[1] + radius), float(c[2] + radius))
    return bmin, bmax


def build_tsdf(cfg: Config, dataset: RGBDDataset) -> TSDFVolume:
    m = cfg.mapping
    if m.bounds_min is None or m.bounds_max is None:
        bmin, bmax = _auto_bounds(dataset, m.auto_bounds_radius_m)
    else:
        bmin, bmax = tuple(m.bounds_min), tuple(m.bounds_max)

    if m.backend == "reference":
        ref_cfg = ReferenceTSDFConfig(
            voxel_size_m=m.voxel_size_m,
            truncation_distance_m=m.truncation_distance_m,
            bounds_min=bmin,
            bounds_max=bmax,
            max_weight=m.max_weight,
            store_color=m.store_color,
            store_features=m.store_features,
            feature_dim=m.feature_dim if m.store_features else 0,
            device=m.device,
        )
        return ReferenceTSDF(ref_cfg)
    if m.backend == "triton":
        from openvocab_tsdf.mapping.triton_backend import TritonTSDF, TritonTSDFConfig

        if m.store_features:
            log.warning("triton backend does not store features yet; semantics will use reference")
        tri_cfg = TritonTSDFConfig(
            voxel_size_m=m.voxel_size_m,
            truncation_distance_m=m.truncation_distance_m,
            bounds_min=bmin,
            bounds_max=bmax,
            max_weight=m.max_weight,
            store_color=m.store_color,
            device=m.device,
        )
        return TritonTSDF(tri_cfg)
    raise NotImplementedError(f"mapping backend '{m.backend}' not implemented")


def _frame_to_pil(frame: RGBDFrame):
    from PIL import Image

    return Image.fromarray(frame.color)


def encode_and_fuse(
    cfg: Config,
    frames: list[RGBDFrame] | RGBDDataset,
    map_out: Path,
) -> dict:
    """Encode each frame with OpenCLIP and fuse geometry + global features.

    Features are stored in the reference backend only (Phase 2 limitation). The
    Triton backend will grow a feature path in Phase 2b.
    """
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    if cfg.mapping.backend != "reference":
        log.warning("semantics require the reference backend; forcing reference for this run")
    m = cfg.mapping
    # build a reference volume regardless of configured backend
    if m.bounds_min is None or m.bounds_max is None:
        # caller passed a plain list of frames; synthesize a simple bounds fit
        if isinstance(frames, list):

            class _S:
                def __getitem__(self, i):
                    return frames[i]

            bmin, bmax = _auto_bounds(_S(), m.auto_bounds_radius_m)
        else:
            bmin, bmax = _auto_bounds(frames, m.auto_bounds_radius_m)
    else:
        bmin, bmax = tuple(m.bounds_min), tuple(m.bounds_max)

    # force features on, force reference backend
    ref_cfg = ReferenceTSDFConfig(
        voxel_size_m=m.voxel_size_m,
        truncation_distance_m=m.truncation_distance_m,
        bounds_min=bmin,
        bounds_max=bmax,
        max_weight=m.max_weight,
        store_color=m.store_color,
        store_features=True,
        feature_dim=m.feature_dim,
        device=m.device,
    )
    vol = ReferenceTSDF(ref_cfg)

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(
            model=cfg.semantics.model,
            pretrained=cfg.semantics.pretrained,
            device=cfg.semantics.device,
            dtype=cfg.semantics.dtype,
        )
    )
    if encoder.feature_dim != m.feature_dim:
        raise ValueError(
            f"feature_dim mismatch: encoder={encoder.feature_dim}, mapping.feature_dim={m.feature_dim}"
        )

    # materialize frames list when needed (we need to batch colors for CLIP)
    frame_list = list(frames) if not isinstance(frames, list) else frames
    colors = [f.color for f in frame_list]

    mode = cfg.semantics.mode
    t0 = time.perf_counter()
    if mode == "global":
        img_feats = encoder.encode_images(colors, batch_size=cfg.semantics.batch_size)
    elif mode == "patch":
        img_feats = encoder.encode_images_patches(colors, batch_size=cfg.semantics.batch_size)
    else:
        raise NotImplementedError(f"semantics.mode={mode!r} not implemented")
    enc_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if mode == "global":
        for frame, feat in zip(frame_list, img_feats, strict=True):
            vol.integrate(frame, feature=feat)
    else:  # patch
        for frame, fm in zip(frame_list, img_feats, strict=True):
            vol.integrate(
                frame,
                feature_map=fm,
                feature_map_input_size=encoder.input_size,
                feature_map_patch_size=encoder.patch_size,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0

    # save map
    map_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        map_out,
        tsdf=vol.tsdf.cpu().numpy(),
        weight=vol.weight.cpu().numpy(),
        color=vol.color.cpu().numpy() if vol.color is not None else np.zeros(0),
        feat=vol.feat.cpu().numpy(),
        origin=vol.origin.cpu().numpy(),
        voxel_size=np.float32(vol.voxel_size_m),
        truncation=np.float32(vol.truncation_distance_m),
        dims=np.int64(vol.dims),
        feature_dim=np.int64(encoder.feature_dim),
        model=np.array(cfg.semantics.model, dtype=object),
        pretrained=np.array(cfg.semantics.pretrained, dtype=object),
        mode=np.array(mode, dtype=object),
    )

    log.info(
        "encoded %d frames in %.2fs (%.1f FPS); fused features in %.2fs -> %s",
        len(frame_list),
        enc_s,
        len(frame_list) / enc_s if enc_s > 0 else 0,
        fuse_s,
        map_out,
    )
    return {
        "num_frames": len(frame_list),
        "encode_s": enc_s,
        "fuse_s": fuse_s,
        "map_path": str(map_out),
        "dims": list(vol.dims),
        "feature_dim": int(encoder.feature_dim),
    }


def ground_text(
    map_path: Path,
    query: str,
    *,
    model: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda:0",
    dtype: str = "fp16",
    min_weight: float = 1.0,
    score_threshold: float | None = 0.22,
    top_percentile: float | None = None,
    cluster_eps_vox: int = 2,
    min_cluster_voxels: int = 8,
    top_k: int = 5,
) -> list:
    """Load a saved map, embed a query, return ranked 3D targets."""
    from openvocab_tsdf.grounding.query import rank_query
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    data = np.load(map_path, allow_pickle=True)
    feat = torch.from_numpy(data["feat"]).to(device)
    weight = torch.from_numpy(data["weight"]).to(device)
    tsdf = torch.from_numpy(data["tsdf"]).to(device) if "tsdf" in data.files else None
    origin = data["origin"]
    voxel_size = float(data["voxel_size"])

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(model=model, pretrained=pretrained, device=device, dtype=dtype)
    )
    q = encoder.encode_texts([query])[0]

    return rank_query(
        voxel_feats=feat,
        voxel_weights=weight,
        voxel_tsdf=tsdf,
        text_embedding=q,
        origin=origin,
        voxel_size=voxel_size,
        min_weight=min_weight,
        score_threshold=score_threshold,
        top_percentile=top_percentile,
        cluster_eps_vox=cluster_eps_vox,
        min_cluster_voxels=min_cluster_voxels,
        top_k=top_k,
    )


def fuse_dataset(cfg: Config, output_path: Path) -> Mesh:
    """Run dataset → TSDF → mesh end-to-end and save PLY. Returns the mesh."""
    from openvocab_tsdf.viz.mesh import save_ply

    dataset = build_dataset(cfg)
    vol = build_tsdf(cfg, dataset)

    n = len(dataset)
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]fuse[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("fuse", total=n)
        for frame in dataset:
            vol.integrate(frame)
            prog.advance(task)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = vol.extract_mesh(min_weight=1.0)
    mc_s = time.perf_counter() - t0

    save_ply(mesh, output_path)
    log.info(
        "fused %d frames in %.2fs (%.1f FPS) -> mesh %d verts / %d tris (mc %.2fs) -> %s",
        n,
        fuse_s,
        n / fuse_s if fuse_s > 0 else 0,
        len(mesh.vertices),
        len(mesh.triangles),
        mc_s,
        output_path,
    )
    return mesh
