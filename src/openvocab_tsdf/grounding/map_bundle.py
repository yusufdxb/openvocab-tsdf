"""Loaded feature-map bundle, dispatched on `sparse_kind`.

A `MapBundle` wraps a saved feature-map npz so callers do not have to know
which of the three on-disk layouts (`dense`, `voxel_slot`, `block_hash`) is
in use. It exposes the per-backend tensors needed by `rank_query` plus a
`score_query(q, neg=None)` helper that returns a dense `(Nx, Ny, Nz)`
score volume.

This was extracted so the ROS 2 grounding node could serve sparse maps
without re-implementing the per-`sparse_kind` dispatch that already lives in
`pipeline.ground_text`, `eval/eval_grounding.py`, and `viz/heatmap.py`.

Memory note. `score_query` materializes a dense `(Nx, Ny, Nz)` fp32 score
tensor on every call. For block_hash maps it also keeps the densified
`tsdf` and `weight` volumes in VRAM. At room scale (a few tens of MB per
volume) this is fine; at warehouse scale it will OOM and the path must
stay block-sparse the whole way — see `densify_block_pool` for the
documented boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MapMetadata:
    origin: np.ndarray  # (3,) float32 world coords of voxel (0,0,0)
    voxel_size: float
    dims: tuple[int, int, int]
    feature_dim: int
    model: str
    pretrained: str
    mode: str
    sparse_kind: str  # "dense" | "voxel_slot" | "block_hash"


class MapBundle:
    """A loaded feature map.

    Attributes (always present):
      meta:         metadata
      weight:       (Nx, Ny, Nz) fp32
      tsdf:         (Nx, Ny, Nz) fp32 — accumulated normalized TSDF
      device:       torch device

    Backend-specific tensors are kept private; callers should use
    `score_query(q, neg=None)` to get a dense `(Nx, Ny, Nz)` score tensor.
    """

    def __init__(self, map_path: str | Path, device: str | torch.device = "cuda:0") -> None:
        self.map_path = Path(map_path)
        d = torch.device(device) if isinstance(device, str) else device
        if d.type == "cuda" and not torch.cuda.is_available():
            d = torch.device("cpu")
        self.device = d

        data = np.load(self.map_path, allow_pickle=True)
        sparse_kind = (
            str(data["sparse_kind"])
            if "sparse_kind" in data.files
            else ("voxel_slot" if ("sparse" in data.files and bool(data["sparse"])) else "dense")
        )
        self.meta = MapMetadata(
            origin=data["origin"].astype(np.float32),
            voxel_size=float(data["voxel_size"]),
            dims=tuple(int(d) for d in data["dims"]) if "dims" in data.files else None,  # type: ignore[arg-type]
            feature_dim=int(data["feature_dim"]) if "feature_dim" in data.files else 0,
            model=str(data["model"]) if "model" in data.files else "",
            pretrained=str(data["pretrained"]) if "pretrained" in data.files else "",
            mode=str(data["mode"]) if "mode" in data.files else "",
            sparse_kind=sparse_kind,
        )

        if sparse_kind == "block_hash":
            from openvocab_tsdf.mapping.block_hash import densify_block_pool

            block_dims = tuple(int(d) for d in data["block_dims"])
            block_slot = torch.from_numpy(data["block_slot"]).to(self.device)
            tsdf_pool = torch.from_numpy(data["tsdf_pool"]).to(self.device)
            weight_pool = torch.from_numpy(data["weight_pool"]).to(self.device)
            dims = self.meta.dims
            self.tsdf = densify_block_pool(block_slot, tsdf_pool, dims, block_dims, default=1.0)
            self.weight = densify_block_pool(block_slot, weight_pool, dims, block_dims, default=0.0)
            self._block_slot = block_slot
            self._block_dims = block_dims
            self._feat_voxel_slot = torch.from_numpy(data["feat_voxel_slot"]).to(self.device)
            self._feat_pool = torch.from_numpy(data["feat_pool"]).to(self.device)
            self._dense_feat = None
            self._slot = None
            self._pool = None
        elif sparse_kind == "voxel_slot":
            self.tsdf = torch.from_numpy(data["tsdf"]).to(self.device)
            self.weight = torch.from_numpy(data["weight"]).to(self.device)
            self._slot = torch.from_numpy(data["voxel_slot"]).to(self.device)
            self._pool = torch.from_numpy(data["feat_pool"]).to(self.device)
            self._dense_feat = None
            self._block_slot = None
            self._feat_voxel_slot = None
            self._feat_pool = None
        elif sparse_kind == "dense":
            self.tsdf = torch.from_numpy(data["tsdf"]).to(self.device)
            self.weight = torch.from_numpy(data["weight"]).to(self.device)
            self._dense_feat = torch.from_numpy(data["feat"]).to(self.device)
            self._slot = None
            self._pool = None
            self._block_slot = None
            self._feat_voxel_slot = None
            self._feat_pool = None
        else:
            raise ValueError(f"unknown sparse_kind {sparse_kind!r} in {self.map_path}")

        # Fill in dims for legacy dense maps that didn't write the key
        if self.meta.dims is None:
            if self._dense_feat is not None:
                self.meta = MapMetadata(  # type: ignore[assignment]
                    **{**self.meta.__dict__, "dims": tuple(self._dense_feat.shape[:3])}
                )
            else:
                self.meta = MapMetadata(  # type: ignore[assignment]
                    **{**self.meta.__dict__, "dims": tuple(self.tsdf.shape)}
                )

    @property
    def origin(self) -> np.ndarray:
        return self.meta.origin

    @property
    def voxel_size(self) -> float:
        return self.meta.voxel_size

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.meta.dims

    @property
    def sparse_kind(self) -> str:
        return self.meta.sparse_kind

    def score_query(self, q: torch.Tensor, neg: torch.Tensor | None = None) -> torch.Tensor:
        """Return a dense `(Nx, Ny, Nz)` per-voxel score volume.

        Scores are the raw inner product `feat @ q`, not true cosine
        similarity: per-voxel features are a weighted mean of unit vectors
        and are not renormalized (see `semantics.aggregation.cosine_score`).
        Values are not bounded in [-1, 1].

        For sparse backends, unobserved voxels are filled with `-1e4` so they
        sort below any real score. The volume is materialized fresh on every
        call and is the caller's to free.
        """
        q = q.to(self.device)
        if neg is not None:
            neg = neg.to(self.device)

        if self.sparse_kind == "dense":
            assert self._dense_feat is not None
            feat = self._dense_feat
            scores = (feat.reshape(-1, feat.shape[-1]) @ q).reshape(feat.shape[:3])
            if neg is not None:
                scores = scores - (feat.reshape(-1, feat.shape[-1]) @ neg).reshape(feat.shape[:3])
            return scores

        if self.sparse_kind == "voxel_slot":
            assert self._slot is not None and self._pool is not None
            dims = self.dims
            pool_scores = self._pool @ q
            if neg is not None:
                pool_scores = pool_scores - self._pool @ neg
            scores_flat = torch.full(
                (dims[0] * dims[1] * dims[2],), -1e4, dtype=torch.float32, device=self.device
            )
            slot_flat = self._slot.view(-1)
            alloc = slot_flat >= 0
            scores_flat[alloc] = pool_scores[slot_flat[alloc].long()]
            return scores_flat.view(*dims)

        # block_hash
        from openvocab_tsdf.mapping.block_hash import scatter_feat_pool_values

        assert self._feat_pool is not None and self._feat_voxel_slot is not None
        assert self._block_slot is not None
        pool_scores = self._feat_pool @ q
        if neg is not None:
            pool_scores = pool_scores - self._feat_pool @ neg
        return scatter_feat_pool_values(
            self._block_slot,
            self._feat_voxel_slot,
            pool_scores,
            self.dims,
            default=-1e4,
        )
