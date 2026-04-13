"""OpenCLIP image and text encoder wrapper.

Frozen encoders. fp16 on GPU. No training. Returns L2-normalized embeddings so
downstream consumers can apply cosine similarity via inner product.

Two image-encoding modes:
  - `encode_images`       → one global (N, D) embedding per image (CLS token).
  - `encode_images_patches` → per-patch (N, Hp, Wp, D) embeddings for
                              dense 2D→3D lifting (Phase 2b). Hp=Wp=14 for
                              ViT-B/16 at 224×224 input (16-px patch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OpenCLIPConfig:
    model: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    device: str = "cuda:0"
    dtype: str = "fp16"  # or "fp32"


class OpenCLIPEncoder:
    """Wraps `open_clip` for image and text encoding at a stable interface."""

    def __init__(self, cfg: OpenCLIPConfig) -> None:
        import open_clip

        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if cfg.dtype == "fp16" else torch.float32

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model, pretrained=cfg.pretrained, device=self.device
        )
        model = model.eval()
        if self.torch_dtype == torch.float16:
            model = model.half()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(cfg.model)

        # expose feature dim + patch geometry
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.torch_dtype)
            self.feature_dim = int(model.encode_image(dummy).shape[-1])

        # Infer patch grid from the visual transformer's conv stem.
        conv1 = getattr(self.model.visual, "conv1", None)
        if conv1 is None:
            raise RuntimeError("unsupported visual backbone: no conv1 found for patch inference")
        kh, kw = conv1.kernel_size
        sh, sw = conv1.stride
        if (kh, kw) != (sh, sw):
            raise RuntimeError(
                f"unexpected non-patchwise conv stride: kernel {kh}x{kw}, stride {sh}x{sw}"
            )
        self.patch_size = int(kh)
        # input resolution comes from the preprocess size
        self.input_size = 224
        self.grid_hw = self.input_size // self.patch_size

    @torch.no_grad()
    def encode_images(self, images: list[np.ndarray], batch_size: int = 16) -> torch.Tensor:
        """Encode H×W×3 uint8 RGB images. Returns (N, D) float32 normalized on GPU."""
        from PIL import Image

        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(Image.fromarray(img)) for img in batch])
            tensors = tensors.to(self.device, dtype=self.torch_dtype, non_blocking=True)
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out.append(feats.float())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """CLIP-ViT forward returning per-patch features with the MaskCLIP trick.

        Standard CLIP ViT mixes all patches in the final self-attention, which
        collapses local information into a few globally-informative slots.
        MaskCLIP (Zhou et al. 2022) observes that a much better per-patch
        feature can be obtained by bypassing the final self-attention and
        using only its value+output projections. This preserves spatial
        locality and gives patches that correlate with what is physically in
        that image region — the semantic analogue of per-pixel features.

        Returns (B, N, D) — L2-normalization is left to the caller.
        """
        import torch.nn.functional as F

        vis = self.model.visual

        # stem: conv1 → (B, width, Hp, Wp)
        x = vis.conv1(x)
        B, C, Hp, Wp = x.shape
        x = x.reshape(B, C, Hp * Wp).permute(0, 2, 1)  # (B, N, C)

        # CLS + position embeddings
        cls = vis.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + vis.positional_embedding.to(x.dtype)

        if hasattr(vis, "patch_dropout"):
            x = vis.patch_dropout(x)
        if hasattr(vis, "ln_pre"):
            x = vis.ln_pre(x)

        # transformer: run all but the LAST block normally, then apply the
        # MaskCLIP substitute for the last block's attention.
        x = x.permute(1, 0, 2)  # NLD -> LND
        blocks = vis.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        last = blocks[-1]

        # --- MaskCLIP attn replacement for the last block ---
        ln_x = last.ln_1(x)  # (L, N_tok, C)
        qkv_weight = last.attn.in_proj_weight
        qkv_bias = last.attn.in_proj_bias
        qkv = F.linear(ln_x, qkv_weight, qkv_bias)
        _, _, v = qkv.chunk(3, dim=-1)
        attn_out = F.linear(v, last.attn.out_proj.weight, last.attn.out_proj.bias)
        x = x + attn_out
        # keep the MLP / LayerScale branch as-is
        if hasattr(last, "ls_2"):
            x = x + last.ls_2(last.mlp(last.ln_2(x)))
        else:
            x = x + last.mlp(last.ln_2(x))
        # ----------------------------------------------------

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = vis.ln_post(x)

        # drop CLS, project to output dim
        patch_tokens = x[:, 1:, :]
        if getattr(vis, "proj", None) is not None:
            patch_tokens = patch_tokens @ vis.proj
        return patch_tokens  # (B, N, D)

    @torch.no_grad()
    def encode_images_patches(self, images: list[np.ndarray], batch_size: int = 8) -> torch.Tensor:
        """Encode H×W×3 uint8 RGB → (N, Hp, Wp, D) L2-normalized patch features on GPU.

        Images are resized+center-cropped via the same `preprocess` as
        `encode_images` so the patch grid corresponds to a square 224×224 crop
        of the input, not the full original frame.
        """
        from PIL import Image

        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(Image.fromarray(img)) for img in batch])
            tensors = tensors.to(self.device, dtype=self.torch_dtype, non_blocking=True)
            tokens = self._forward_patches(tensors)  # (B, N, D)
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            B, N, D = tokens.shape
            hp = int(N**0.5)
            if hp * hp != N:
                raise RuntimeError(f"unexpected non-square patch grid: N={N}")
            out.append(tokens.reshape(B, hp, hp, D).float())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts. Returns (N, D) float32 normalized on GPU."""
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.float()
