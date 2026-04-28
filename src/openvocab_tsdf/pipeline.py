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
    if d.name == "nice_slam_demo":
        from openvocab_tsdf.data.nice_slam_demo import NiceSlamDemoDataset

        return NiceSlamDemoDataset(
            root=d.root,
            scene=d.scene,
            depth_scale=cfg.camera.depth_scale,
            depth_trunc_m=cfg.camera.depth_trunc_m,
            max_frames=d.max_frames,
            stride=d.stride,
        )
    if d.name == "scannet":
        from openvocab_tsdf.data.scannet import ScanNetDataset

        return ScanNetDataset(
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
            near_surface_band=m.near_surface_band,
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
    if m.backend == "sparse_feature":
        from openvocab_tsdf.mapping.sparse_reference import (
            SparseFeatureTSDF,
            SparseFeatureTSDFConfig,
        )

        return SparseFeatureTSDF(
            SparseFeatureTSDFConfig(
                voxel_size_m=m.voxel_size_m,
                truncation_distance_m=m.truncation_distance_m,
                bounds_min=bmin,
                bounds_max=bmax,
                max_weight=m.max_weight,
                store_color=m.store_color,
                feature_dim=m.feature_dim,
                initial_feat_capacity=m.initial_feat_capacity,
                max_feat_capacity=m.max_feat_capacity,
                feat_update_backend=m.feat_update_backend,
                near_surface_band=m.near_surface_band,
                device=m.device,
            )
        )
    if m.backend == "block_hash":
        from openvocab_tsdf.mapping.block_hash import BlockHashTSDF, BlockHashTSDFConfig

        return BlockHashTSDF(
            BlockHashTSDFConfig(
                voxel_size_m=m.voxel_size_m,
                truncation_distance_m=m.truncation_distance_m,
                bounds_min=bmin,
                bounds_max=bmax,
                max_weight=m.max_weight,
                store_color=m.store_color,
                store_features=m.store_features,
                feature_dim=m.feature_dim if m.store_features else 0,
                initial_feat_capacity=m.initial_feat_capacity,
                max_feat_capacity=m.max_feat_capacity,
                near_surface_band=m.near_surface_band,
                device=m.device,
            )
        )
    raise NotImplementedError(f"mapping backend '{m.backend}' not implemented")


def _validate_model_match(meta, query_model: str) -> None:
    """Raise if the text encoder model doesn't match what was used to build the map.

    Maps store the encoder model in their metadata (`model` key). With three
    encoder modes producing features in different CLIP text spaces (ViT-B/16,
    ViT-L/14, ViT-B/32 for LSeg), silently using the wrong text encoder yields
    garbage cosine scores. Fail-fast.

    Empty `meta.model` (legacy maps from before model-name persistence) is
    treated as a no-op so older maps remain queryable.
    """
    if meta.model and meta.model != query_model:
        raise ValueError(
            f"model mismatch: map was built with {meta.model!r} but "
            f"query uses {query_model!r}. Cosine scores will be meaningless."
        )


def _frame_to_pil(frame: RGBDFrame):
    from PIL import Image

    return Image.fromarray(frame.color)


def encode_and_fuse(
    cfg: Config,
    frames: list[RGBDFrame] | RGBDDataset,
    map_out: Path,
) -> dict:
    """Encode each frame with OpenCLIP and fuse geometry + features.

    Global-mode features work with all three feature-storing backends
    (`reference`, `sparse_feature`, `block_hash`). SAM-dense works with
    `reference` and `block_hash`. Patch mode only works with `reference` today
    (sparse patch aggregation is future work).
    """
    from openvocab_tsdf.mapping.block_hash import BlockHashTSDF
    from openvocab_tsdf.mapping.sparse_reference import SparseFeatureTSDF
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    m = cfg.mapping
    mode = cfg.semantics.mode
    if m.backend not in ("reference", "sparse_feature", "block_hash"):
        log.warning("backend %r does not support feature storage; forcing reference", m.backend)
        m.backend = "reference"  # fall through
    if mode in ("patch", "sam_dense", "lseg") and m.backend == "sparse_feature":
        log.warning("sparse_feature backend does not support %r mode yet; forcing global", mode)
        mode = "global"
    if mode == "patch" and m.backend == "block_hash":
        log.warning("block_hash backend does not support patch mode yet; forcing global")
        mode = "global"
    if mode == "lseg" and m.backend == "block_hash" and m.feature_dim != 512:
        log.warning(
            "lseg produces 512-d features; overriding feature_dim from %d to 512",
            m.feature_dim,
        )
        m.feature_dim = 512

    # Build the volume using `build_tsdf` (shared path with fuse)
    # but force store_features on.
    m.store_features = True
    if m.bounds_min is None or m.bounds_max is None:
        if isinstance(frames, list):

            class _S:
                def __getitem__(self, i):
                    return frames[i]

            bmin, bmax = _auto_bounds(_S(), m.auto_bounds_radius_m)
        else:
            bmin, bmax = _auto_bounds(frames, m.auto_bounds_radius_m)
        m.bounds_min = bmin
        m.bounds_max = bmax

    # build the volume using the selected backend (reference,
    # sparse_feature, or block_hash reach this path)
    if m.backend == "sparse_feature":
        from openvocab_tsdf.mapping.sparse_reference import (
            SparseFeatureTSDFConfig,
        )

        vol = SparseFeatureTSDF(
            SparseFeatureTSDFConfig(
                voxel_size_m=m.voxel_size_m,
                truncation_distance_m=m.truncation_distance_m,
                bounds_min=tuple(m.bounds_min),
                bounds_max=tuple(m.bounds_max),
                max_weight=m.max_weight,
                store_color=m.store_color,
                feature_dim=m.feature_dim,
                initial_feat_capacity=m.initial_feat_capacity,
                max_feat_capacity=m.max_feat_capacity,
                feat_update_backend=m.feat_update_backend,
                near_surface_band=m.near_surface_band,
                device=m.device,
            )
        )
    elif m.backend == "block_hash":
        from openvocab_tsdf.mapping.block_hash import BlockHashTSDFConfig

        vol = BlockHashTSDF(
            BlockHashTSDFConfig(
                voxel_size_m=m.voxel_size_m,
                truncation_distance_m=m.truncation_distance_m,
                bounds_min=tuple(m.bounds_min),
                bounds_max=tuple(m.bounds_max),
                max_weight=m.max_weight,
                store_color=m.store_color,
                store_features=True,
                feature_dim=m.feature_dim,
                initial_feat_capacity=m.initial_feat_capacity,
                max_feat_capacity=m.max_feat_capacity,
                near_surface_band=m.near_surface_band,
                device=m.device,
            )
        )
    else:
        ref_cfg = ReferenceTSDFConfig(
            voxel_size_m=m.voxel_size_m,
            truncation_distance_m=m.truncation_distance_m,
            bounds_min=tuple(m.bounds_min),
            bounds_max=tuple(m.bounds_max),
            max_weight=m.max_weight,
            store_color=m.store_color,
            store_features=True,
            feature_dim=m.feature_dim,
            near_surface_band=m.near_surface_band,
            device=m.device,
        )
        vol = ReferenceTSDF(ref_cfg)

    if mode == "lseg":
        from openvocab_tsdf.semantics.lseg_encoder import LSegEncoder

        lseg_enc = LSegEncoder(
            weights_path=cfg.semantics.lseg_weights_path,
            device=cfg.semantics.device,
        )
        feature_dim = lseg_enc.FEATURE_DIM
        encoder = None
    else:
        encoder = OpenCLIPEncoder(
            OpenCLIPConfig(
                model=cfg.semantics.model,
                pretrained=cfg.semantics.pretrained,
                device=cfg.semantics.device,
                dtype=cfg.semantics.dtype,
            )
        )
        feature_dim = encoder.feature_dim
        lseg_enc = None
    if feature_dim != m.feature_dim:
        raise ValueError(
            f"feature_dim mismatch: encoder={feature_dim}, mapping.feature_dim={m.feature_dim}"
        )

    # materialize frames list when needed (we need to batch colors for CLIP)
    frame_list = list(frames) if not isinstance(frames, list) else frames
    colors = [f.color for f in frame_list]

    t0 = time.perf_counter()
    sam_extractor = None
    if mode == "global":
        img_feats = encoder.encode_images(colors, batch_size=cfg.semantics.batch_size)
    elif mode == "patch":
        img_feats = encoder.encode_images_patches(colors, batch_size=cfg.semantics.batch_size)
    elif mode == "sam_dense":
        from openvocab_tsdf.semantics.sam_dense import (
            SAMDenseConfig,
            SAMDenseFeatureExtractor,
        )

        sam_extractor = SAMDenseFeatureExtractor(
            SAMDenseConfig(device=cfg.semantics.device), encoder
        )
        img_feats = None  # features computed lazily per-frame (too big to keep all)
    elif mode == "lseg":
        img_feats = None  # dense features computed per-frame, no batch encode step
    else:
        raise NotImplementedError(f"semantics.mode={mode!r} not implemented")
    enc_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if mode == "global":
        for frame, feat in zip(frame_list, img_feats, strict=True):
            vol.integrate(frame, feature=feat)
    elif mode == "patch":
        for frame, fm in zip(frame_list, img_feats, strict=True):
            vol.integrate(
                frame,
                feature_map=fm,
                feature_map_input_size=encoder.input_size,
                feature_map_patch_size=encoder.patch_size,
            )
    elif mode == "sam_dense":
        for i, frame in enumerate(frame_list):
            dfm_np = sam_extractor.extract(frame.color)
            dfm = torch.from_numpy(dfm_np).to(cfg.semantics.device)
            vol.integrate(frame, dense_feature_map=dfm)
            if (i + 1) % 25 == 0:
                log.info("sam_dense: integrated %d / %d", i + 1, len(frame_list))
    elif mode == "lseg":
        for i, frame in enumerate(frame_list):
            dfm_np = lseg_enc.extract(frame.color)
            dfm = torch.from_numpy(dfm_np).to(cfg.semantics.device)
            vol.integrate(frame, dense_feature_map=dfm)
            if (i + 1) % 25 == 0:
                log.info("lseg: integrated %d / %d", i + 1, len(frame_list))
    else:
        raise NotImplementedError(f"semantics.mode={mode!r} not implemented")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0

    # save map — each backend serialises the subset of fields it owns.
    # `sparse_kind` ("dense" / "voxel_slot" / "block_hash") selects the load path.
    # `sparse` stays as a bool for back-compat with maps saved before sparse_kind existed.
    map_out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "lseg":
        saved_model = lseg_enc.TEXT_MODEL
        saved_pretrained = lseg_enc.TEXT_PRETRAINED
    else:
        saved_model = cfg.semantics.model
        saved_pretrained = cfg.semantics.pretrained
    save_kwargs = {
        "origin": vol.origin.cpu().numpy(),
        "voxel_size": np.float32(vol.voxel_size_m),
        "truncation": np.float32(vol.truncation_distance_m),
        "dims": np.int64(vol.dims),
        "feature_dim": np.int64(feature_dim),
        "model": np.array(saved_model, dtype=object),
        "pretrained": np.array(saved_pretrained, dtype=object),
        "mode": np.array(mode, dtype=object),
    }
    if isinstance(vol, BlockHashTSDF):
        from openvocab_tsdf.mapping.block_hash import BLOCK3

        nb = vol.num_allocated_blocks
        nfv = vol.num_allocated_feat_voxels
        save_kwargs["sparse"] = np.bool_(True)
        save_kwargs["sparse_kind"] = np.array("block_hash", dtype=object)
        save_kwargs["block_dims"] = np.int64(vol.block_dims)
        save_kwargs["block_slot"] = vol._block_slot.cpu().numpy()
        save_kwargs["tsdf_pool"] = vol._tsdf_pool[:nb].cpu().numpy()
        save_kwargs["weight_pool"] = vol._weight_pool[:nb].cpu().numpy()
        if vol._color_pool is not None:
            save_kwargs["color_pool"] = vol._color_pool[:nb].cpu().numpy()
        if vol._store_features:
            save_kwargs["feat_voxel_slot"] = vol._feat_voxel_slot[:nb].cpu().numpy()
            save_kwargs["feat_pool"] = vol._feat_pool[:nfv].cpu().numpy()
        geom_mb = nb * BLOCK3 * (4 + 4 + (12 if vol._color_pool is not None else 0)) / (1024**2)
        feat_mb = nfv * feature_dim * 4 / (1024**2) if vol._store_features else 0.0
        log.info(
            "saving block_hash map: %d blocks / %d feat voxels, %.1f MB geom + %.1f MB feat",
            nb,
            nfv,
            geom_mb,
            feat_mb,
        )
    elif isinstance(vol, SparseFeatureTSDF):
        save_kwargs["tsdf"] = vol.tsdf.cpu().numpy()
        save_kwargs["weight"] = vol.weight.cpu().numpy()
        save_kwargs["color"] = vol.color.cpu().numpy() if vol.color is not None else np.zeros(0)
        n = vol.num_allocated_feat_voxels
        save_kwargs["sparse"] = np.bool_(True)
        save_kwargs["sparse_kind"] = np.array("voxel_slot", dtype=object)
        save_kwargs["feat_pool"] = vol._feat_pool[:n].cpu().numpy()
        save_kwargs["voxel_slot"] = vol._voxel_slot.cpu().numpy()
        log.info(
            "saving sparse map: %d allocated voxels, %.1f MB pool",
            n,
            n * feature_dim * 4 / (1024**2),
        )
    else:
        save_kwargs["tsdf"] = vol.tsdf.cpu().numpy()
        save_kwargs["weight"] = vol.weight.cpu().numpy()
        save_kwargs["color"] = vol.color.cpu().numpy() if vol.color is not None else np.zeros(0)
        save_kwargs["sparse"] = np.bool_(False)
        save_kwargs["sparse_kind"] = np.array("dense", dtype=object)
        save_kwargs["feat"] = vol.feat.cpu().numpy()
    np.savez_compressed(map_out, **save_kwargs)

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
        "feature_dim": int(feature_dim),
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
    scene_mean_subtract: bool = False,
    negative_query: str | None = None,
) -> list:
    """Load a saved map, embed a query, return ranked 3D targets.

    All three on-disk layouts (`dense`, `voxel_slot`, `block_hash`) go through
    `MapBundle`, which materialises a dense `(Nx, Ny, Nz)` score volume per
    query without ever building the 4-D feature volume. `negative_query` is
    folded into the bundle's `score_query`, so `rank_query` always sees a
    precomputed score tensor — its own `neg_text_embedding` knob is
    unnecessary on this path.
    """
    from openvocab_tsdf.grounding.map_bundle import MapBundle
    from openvocab_tsdf.grounding.query import rank_query
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    bundle = MapBundle(map_path, device=device)
    _validate_model_match(bundle.meta, model)
    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(model=model, pretrained=pretrained, device=device, dtype=dtype)
    )
    texts = [query] + ([negative_query] if negative_query else [])
    emb = encoder.encode_texts(texts)
    q = emb[0]
    neg = emb[1] if negative_query else None

    scores = bundle.score_query(q, neg=neg)
    return rank_query(
        voxel_feats=None,
        voxel_weights=bundle.weight,
        voxel_tsdf=bundle.tsdf,
        text_embedding=None,
        precomputed_scores=scores,
        origin=bundle.origin,
        voxel_size=bundle.voxel_size,
        min_weight=min_weight,
        score_threshold=score_threshold,
        top_percentile=top_percentile,
        cluster_eps_vox=cluster_eps_vox,
        min_cluster_voxels=min_cluster_voxels,
        top_k=top_k,
        scene_mean_subtract=scene_mean_subtract,
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
