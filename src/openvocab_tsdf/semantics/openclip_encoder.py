"""OpenCLIP image and text encoder wrapper.

Frozen encoders. fp16 on GPU. No training. Returns L2-normalized embeddings so
downstream consumers can apply cosine similarity via inner product.
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

        # expose feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.torch_dtype)
            self.feature_dim = int(model.encode_image(dummy).shape[-1])

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
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts. Returns (N, D) float32 normalized on GPU."""
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.float()
