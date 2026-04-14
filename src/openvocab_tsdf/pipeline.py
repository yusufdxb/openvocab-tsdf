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
                device=m.device,
            )
        )
    raise NotImplementedError(f"mapping backend '{m.backend}' not implemented")


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
    if mode in ("patch", "sam_dense") and m.backend == "sparse_feature":
        log.warning("sparse_feature backend does not support %r mode yet; forcing global", mode)
        mode = "global"
    if mode == "patch" and m.backend == "block_hash":
        log.warning("block_hash backend does not support patch mode yet; forcing global")
        mode = "global"

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
    else:  # sam_dense
        for i, frame in enumerate(frame_list):
            dfm_np = sam_extractor.extract(frame.color)
            dfm = torch.from_numpy(dfm_np).to(cfg.semantics.device)
            vol.integrate(frame, dense_feature_map=dfm)
            if (i + 1) % 25 == 0:
                log.info("sam_dense: integrated %d / %d", i + 1, len(frame_list))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0

    # save map — each backend serialises the subset of fields it owns.
    # `sparse_kind` ("dense" / "voxel_slot" / "block_hash") selects the load path.
    # `sparse` stays as a bool for back-compat with maps saved before sparse_kind existed.
    map_out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "origin": vol.origin.cpu().numpy(),
        "voxel_size": np.float32(vol.voxel_size_m),
        "truncation": np.float32(vol.truncation_distance_m),
        "dims": np.int64(vol.dims),
        "feature_dim": np.int64(encoder.feature_dim),
        "model": np.array(cfg.semantics.model, dtype=object),
        "pretrained": np.array(cfg.semantics.pretrained, dtype=object),
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
        feat_mb = nfv * encoder.feature_dim * 4 / (1024**2) if vol._store_features else 0.0
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
            n * encoder.feature_dim * 4 / (1024**2),
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
    scene_mean_subtract: bool = False,
    negative_query: str | None = None,
) -> list:
    """Load a saved map, embed a query, return ranked 3D targets."""
    from openvocab_tsdf.grounding.query import rank_query
    from openvocab_tsdf.semantics.openclip_encoder import OpenCLIPConfig, OpenCLIPEncoder

    data = np.load(map_path, allow_pickle=True)
    origin = data["origin"]
    voxel_size = float(data["voxel_size"])
    sparse_kind = (
        str(data["sparse_kind"])
        if "sparse_kind" in data.files
        else ("voxel_slot" if ("sparse" in data.files and bool(data["sparse"])) else "dense")
    )

    encoder = OpenCLIPEncoder(
        OpenCLIPConfig(model=model, pretrained=pretrained, device=device, dtype=dtype)
    )
    texts = [query] + ([negative_query] if negative_query else [])
    emb = encoder.encode_texts(texts)
    q = emb[0]
    neg = emb[1] if negative_query else None

    if sparse_kind == "block_hash":
        from openvocab_tsdf.mapping.block_hash import (
            densify_block_pool,
            scatter_feat_pool_values,
        )

        dims = tuple(int(d) for d in data["dims"])
        block_dims = tuple(int(d) for d in data["block_dims"])
        block_slot = torch.from_numpy(data["block_slot"]).to(device)
        tsdf_pool = torch.from_numpy(data["tsdf_pool"]).to(device)
        weight_pool = torch.from_numpy(data["weight_pool"]).to(device)
        tsdf = densify_block_pool(block_slot, tsdf_pool, dims, block_dims, default=1.0)
        weight = densify_block_pool(block_slot, weight_pool, dims, block_dims, default=0.0)
        # Score features via the feat pool and scatter back through the double
        # indirection into a dense (Nx, Ny, Nz) scores tensor. Avoids ever
        # materialising the 4-D feature volume.
        feat_voxel_slot = torch.from_numpy(data["feat_voxel_slot"]).to(device)
        feat_pool = torch.from_numpy(data["feat_pool"]).to(device)
        pool_scores = feat_pool @ q
        if neg is not None:
            pool_scores = pool_scores - feat_pool @ neg
        scores_vol = scatter_feat_pool_values(
            block_slot, feat_voxel_slot, pool_scores, dims, default=-1e4
        )
        return rank_query(
            voxel_feats=None,
            voxel_weights=weight,
            voxel_tsdf=tsdf,
            text_embedding=None,
            precomputed_scores=scores_vol,
            origin=origin,
            voxel_size=voxel_size,
            min_weight=min_weight,
            score_threshold=score_threshold,
            top_percentile=top_percentile,
            cluster_eps_vox=cluster_eps_vox,
            min_cluster_voxels=min_cluster_voxels,
            top_k=top_k,
            scene_mean_subtract=scene_mean_subtract,
        )

    # "voxel_slot" (sparse_feature) or "dense" (reference).
    weight = torch.from_numpy(data["weight"]).to(device)
    tsdf = torch.from_numpy(data["tsdf"]).to(device) if "tsdf" in data.files else None

    if sparse_kind == "voxel_slot":
        dims = tuple(int(d) for d in data["dims"])
        D = int(data["feature_dim"])
        slot = torch.from_numpy(data["voxel_slot"]).to(device)
        pool = torch.from_numpy(data["feat_pool"]).to(device)
        feat = torch.zeros((dims[0] * dims[1] * dims[2], D), dtype=torch.float32, device=device)
        alloc = slot.view(-1) >= 0
        feat[alloc] = pool[slot.view(-1)[alloc].long()]
        feat = feat.view(*dims, D)
    else:
        feat = torch.from_numpy(data["feat"]).to(device)

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
        scene_mean_subtract=scene_mean_subtract,
        neg_text_embedding=neg,
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
