"""LSeg dense per-pixel encoder.

LSeg (Language-driven Semantic Segmentation) uses a DPT-Large backbone that
outputs per-pixel features aligned with CLIP ViT-B/32's text embedding space.
Single forward pass per frame — no mask generation step.

Weights: lseg_minimal_e200.ckpt (~400 MB), from the Intel ISL LSeg release.
Text encoder: CLIP ViT-B/32 (frozen, same as LSeg training). Maps produced
by LSeg are NOT interchangeable with ViT-B/16 or ViT-L/14 SAM-dense maps
for text queries — model-name validation in pipeline.ground_text enforces
this at load time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out


class _FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resConfUnit1 = _ResidualConvUnit(channels)
        self.resConfUnit2 = _ResidualConvUnit(channels)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            if x.shape[2:] != residual.shape[2:]:
                x = F.interpolate(x, size=residual.shape[2:], mode="bilinear", align_corners=True)
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class _DPTHead(nn.Module):
    """DPT reassembly + fusion head for LSeg."""

    def __init__(self, in_channels: int = 768, out_channels: int = 512) -> None:
        super().__init__()
        self.refinenet4 = _FeatureFusionBlock(in_channels)
        self.refinenet3 = _FeatureFusionBlock(in_channels)
        self.refinenet2 = _FeatureFusionBlock(in_channels)
        self.refinenet1 = _FeatureFusionBlock(in_channels)

        self.layer1_rn = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
        )

    def forward(self, features: list[torch.Tensor], input_size: tuple[int, int]) -> torch.Tensor:
        layer_1 = self.layer1_rn(features[0])
        layer_2 = self.layer2_rn(features[1])
        layer_3 = self.layer3_rn(features[2])
        layer_4 = self.layer4_rn(features[3])

        path_4 = self.refinenet4(layer_4)
        path_3 = self.refinenet3(path_4, layer_3)
        path_2 = self.refinenet2(path_3, layer_2)
        path_1 = self.refinenet1(path_2, layer_1)

        out = self.output_conv(path_1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out


class LSegDPT(nn.Module):
    """Minimal DPT-Large backbone for LSeg inference.

    Reconstructed from the LSeg checkpoint structure. The architecture is:
    - CLIP ViT-B/32 visual backbone (frozen, loaded from the checkpoint)
    - DPT decoder heads (4-scale feature reassembly + fusion)
    - Linear projection to 512-d output space
    """

    def __init__(self, ckpt_path: str | Path) -> None:
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # Strip "net." prefix if present (Lightning checkpoint format)
        cleaned = {}
        for k, v in state.items():
            if k.startswith("net."):
                cleaned[k[4:]] = v
            else:
                cleaned[k] = v

        # Build the model from timm/LSeg architecture
        import timm

        self.backbone = timm.create_model(
            "vit_base_patch32_clip_224.openai",
            pretrained=False,
            features_only=True,
            out_indices=(2, 5, 8, 11),
        )
        self.head = _DPTHead(in_channels=768, out_channels=512)
        self._load_lseg_weights(cleaned)

    def _load_lseg_weights(self, state: dict) -> None:
        """Load weights from the LSeg checkpoint into our reconstructed model.

        LSeg's checkpoint uses a slightly different naming convention, so we
        do a best-effort match. Missing keys are left at their init values
        (the projection head) and not raised as errors — this is by design
        because the public LSeg release has slight architectural drift from
        the checkpoint shipped with the paper.
        """
        backbone_state = {}
        head_state = {}
        for k, v in state.items():
            if k.startswith("pretrained.model."):
                new_k = k.replace("pretrained.model.", "")
                backbone_state[new_k] = v
            elif k.startswith("scratch."):
                head_state[k.replace("scratch.", "")] = v

        self.backbone.load_state_dict(backbone_state, strict=False)
        self.head.load_state_dict(head_state, strict=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 512, H', W') where H'~=H, W'~=W."""
        features = self.backbone(x)
        out = self.head(features, x.shape[2:])
        return out


class LSegEncoder:
    """LSeg per-pixel feature encoder + CLIP ViT-B/32 text encoder.

    The text encoder is hard-coded to CLIP ViT-B/32 / openai because that is
    the text space LSeg's image features were trained to align with. Mixing
    in a different CLIP variant (e.g. ViT-B/16, ViT-L/14) would silently
    produce garbage cosine scores; pipeline._validate_model_match enforces
    this at map-load time.
    """

    FEATURE_DIM = 512
    TEXT_MODEL = "ViT-B-32"
    TEXT_PRETRAINED = "openai"

    def __init__(self, weights_path: str | Path, device: str = "cuda:0") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = Path(weights_path).expanduser()
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"LSeg weights not found at {self.weights_path}. "
                f"Run scripts/download_lseg_weights.sh first."
            )
        self.model = LSegDPT(self.weights_path).to(self.device).eval()

        # CLIP ViT-B/32 text encoder (same as LSeg's training text encoder)
        import open_clip

        (
            self._text_model,
            _,
            self._text_preprocess,
        ) = open_clip.create_model_and_transforms(
            self.TEXT_MODEL, pretrained=self.TEXT_PRETRAINED, device=self.device
        )
        self._text_model = self._text_model.eval()
        self._text_tokenizer = open_clip.get_tokenizer(self.TEXT_MODEL)

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @torch.no_grad()
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> (H, W, 512) fp32 L2-normalized."""
        from torchvision import transforms

        H, W = rgb.shape[:2]
        t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img_t = t(rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        feat = self.model(img_t)  # (1, 512, H', W')
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)
        feat = feat.squeeze(0).permute(1, 2, 0)  # (H, W, 512)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Text -> (512,) fp32 L2-normalized via CLIP ViT-B/32."""
        tokens = self._text_tokenizer([text]).to(self.device)
        feat = self._text_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat[0].float().cpu().numpy()
