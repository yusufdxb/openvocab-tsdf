"""LSeg dense per-pixel encoder.

LSeg (Language-driven Semantic Segmentation) uses a DPT-Large image branch
(ViT-L/16 384, 24 transformer blocks, embed_dim=1024) with a DPT decoder
that outputs per-pixel features aligned with CLIP ViT-B/32's text embedding
space. Single forward pass per frame -- no mask generation step.

Architecture, derived directly from the public ``lseg_minimal_e200.ckpt``
shipped with `isl-org/lang-seg <https://github.com/isl-org/lang-seg>`_:

* ``net.pretrained.model``: timm ``vit_large_patch16_384`` backbone (24 blocks,
  embed_dim=1024, patch=16, pos_embed (1, 577, 1024) for 384x384 input).
* ``net.pretrained.act_postprocess{1..4}``: DPT reassembly hooks on tokens
  from blocks 5, 11, 17, 23 with project-readout fusion (concat cls into x,
  Linear(2048, 1024) + GELU), 1x1 channel adjust to {256, 512, 1024, 1024},
  then ConvTranspose2d (x4, x2), identity, and stride-2 Conv (down) for the
  four scales respectively.
* ``net.scratch``: DPT decoder with internal channel width 256. Per-scale
  ``layer{1..4}_rn`` 3x3 convs map {256, 512, 1024, 1024} -> 256, then
  ``refinenet{4..1}`` (each = two BN-style ``ResidualConvUnit`` blocks +
  1x1 ``out_conv``) progressively fuse and upsample. Final ``head1`` 1x1
  conv projects 256 -> 512 (CLIP feature dim).
* ``net.clip_pretrained``: full CLIP ViT-B/32 from the OpenAI release. We
  do NOT load this from the LSeg ckpt -- the text encoder is loaded
  independently via ``open_clip`` to keep the dependency surface small,
  and the CLIP visual branch is unused at inference (LSeg replaces it
  with the DPT-Large image branch above).

Weights: ``lseg_minimal_e200.ckpt`` (~3.1 GB, full Lightning checkpoint
including optimizer state), from the Intel ISL LSeg release. Maps produced
by LSeg are NOT interchangeable with ViT-B/16 or ViT-L/14 SAM-dense maps
for text queries -- model-name validation in ``pipeline.ground_text``
enforces this at load time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DPT decoder primitives (BN-style ResidualConvUnit, FeatureFusion)
# ---------------------------------------------------------------------------


class _ResidualConvUnit(nn.Module):
    """BN-style residual conv unit used by the DPT-Large decoder.

    Order matches the upstream ISL DPT/LSeg implementation:
    ``relu -> bn1 -> conv1 -> bn2 -> conv2 -> + residual``.
    Conv layers have ``bias=False`` because BN follows.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        return x + out


class _FeatureFusionBlock(nn.Module):
    """DPT FeatureFusion: two RCUs + 1x1 out_conv + 2x bilinear upsample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resConfUnit1 = _ResidualConvUnit(channels)
        self.resConfUnit2 = _ResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        if residual is not None:
            if x.shape[2:] != residual.shape[2:]:
                x = F.interpolate(
                    x, size=residual.shape[2:], mode="bilinear", align_corners=True
                )
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        return x


# ---------------------------------------------------------------------------
# DPT-Large reassembly: project-readout + channel adjust + per-scale resize
# ---------------------------------------------------------------------------


class _ProjectReadout(nn.Module):
    """Project-readout fusion: cls_token is broadcast and concatenated to
    each spatial token, then projected back to embed_dim via Linear+GELU.

    Implements the ISL DPT-Large readout="project" mode. The Linear layer
    is wrapped in a ``Sequential`` to match the checkpoint key
    ``act_postprocessN.0.project.0.{weight, bias}``.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.GELU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 1+N, C) where index 0 is cls
        cls = tokens[:, 0:1, :].expand(-1, tokens.size(1) - 1, -1)
        x = tokens[:, 1:, :]
        return self.project(torch.cat([x, cls], dim=-1))


class _Reassemble(nn.Module):
    """One DPT reassembly stage: project-readout, transpose to (B,C,H,W),
    1x1 channel adjust, and per-scale resize.

    Args:
        embed_dim: backbone token dim (1024 for ViT-L).
        out_channels: target channel count for this scale.
        resize: ``"up4" | "up2" | "id" | "down2"`` -- DPT-Large per-scale resize.

    Module structure matches the checkpoint keys:
        ``[0] = _ProjectReadout``
        ``[1] = Transpose+Unflatten`` (no parameters; folded into forward)
        ``[2] = Identity`` (placeholder; no parameters)
        ``[3] = Conv2d(embed_dim, out_channels, kernel_size=1)``
        ``[4] = ConvTranspose2d / Identity / Conv2d`` per ``resize``.
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        resize: str,
        patches_per_side: int,
    ) -> None:
        super().__init__()
        self.patches_per_side = patches_per_side

        # Submodules registered with integer string indices so that
        # state_dict keys are act_postprocessN.0.*, .3.*, .4.*. Indices 1
        # (transpose) and 2 (identity) are parameter-free in upstream.
        self._modules["0"] = _ProjectReadout(embed_dim)
        self._modules["1"] = nn.Identity()  # transpose handled in forward
        self._modules["2"] = nn.Identity()
        self._modules["3"] = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        if resize == "up4":
            # Scale 1: x4 upsample
            self._modules["4"] = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=4, padding=0
            )
        elif resize == "up2":
            # Scale 2: x2 upsample
            self._modules["4"] = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2, padding=0
            )
        elif resize == "id":
            # Scale 3: identity (no resize, no parameters)
            self._modules["4"] = nn.Identity()
        elif resize == "down2":
            # Scale 4: stride-2 3x3 conv (downsample)
            self._modules["4"] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            raise ValueError(f"unknown resize mode: {resize!r}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._modules["0"](tokens)  # (B, N, C)
        B, N, C = x.shape
        H = W = self.patches_per_side
        assert N == H * W, f"token count {N} != {H}*{W}"
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        x = self._modules["3"](x)  # 1x1 channel adjust
        x = self._modules["4"](x)  # per-scale resize
        return x


# ---------------------------------------------------------------------------
# DPT-Large image branch + scratch decoder = LSeg image encoder
# ---------------------------------------------------------------------------


class _DPTHead(nn.Module):
    """LSeg DPT decoder ("scratch"): per-scale 3x3 channel-adjust convs,
    four FeatureFusion blocks, and the final 1x1 ``head1`` projection
    from 256 internal channels to the 512-d CLIP feature space.
    """

    def __init__(self, hooks_out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
                 inner_channels: int = 256, out_channels: int = 512) -> None:
        super().__init__()
        c1, c2, c3, c4 = hooks_out_channels
        self.layer1_rn = nn.Conv2d(c1, inner_channels, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(c2, inner_channels, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(c3, inner_channels, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(c4, inner_channels, 3, padding=1, bias=False)

        self.refinenet1 = _FeatureFusionBlock(inner_channels)
        self.refinenet2 = _FeatureFusionBlock(inner_channels)
        self.refinenet3 = _FeatureFusionBlock(inner_channels)
        self.refinenet4 = _FeatureFusionBlock(inner_channels)

        self.head1 = nn.Conv2d(inner_channels, out_channels, kernel_size=1)

    def forward(
        self, features: list[torch.Tensor], input_size: tuple[int, int]
    ) -> torch.Tensor:
        layer_1 = self.layer1_rn(features[0])
        layer_2 = self.layer2_rn(features[1])
        layer_3 = self.layer3_rn(features[2])
        layer_4 = self.layer4_rn(features[3])

        path_4 = self.refinenet4(layer_4)
        path_3 = self.refinenet3(path_4, layer_3)
        path_2 = self.refinenet2(path_3, layer_2)
        path_1 = self.refinenet1(path_2, layer_1)

        out = self.head1(path_1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out


class LSegDPT(nn.Module):
    """Faithful port of the LSeg image branch matching ``lseg_minimal_e200.ckpt``.

    Backbone: timm ``vit_large_patch16_384`` (24 blocks, embed_dim=1024,
    patch=16, 384x384 nominal input). Token activations from blocks
    ``HOOK_BLOCKS = (5, 11, 17, 23)`` feed four DPT reassembly stages, then
    the DPT decoder with 256 internal channels and a 1x1 head projecting
    to 512-d CLIP space.

    Inputs to ``forward`` should be 384x384 RGB tensors (H and W must be
    multiples of 16; 384 matches the trained pos_embed). For arbitrary
    input sizes, ``LSegEncoder.extract`` resizes to 384x384 before the
    forward pass and bilinearly upsamples the output back to the original
    HxW.
    """

    HOOK_BLOCKS: tuple[int, int, int, int] = (5, 11, 17, 23)
    EMBED_DIM: int = 1024
    INPUT_SIZE: int = 384
    PATCH_SIZE: int = 16
    HOOKS_OUT_CHANNELS: tuple[int, int, int, int] = (256, 512, 1024, 1024)

    def __init__(
        self,
        ckpt_path: str | Path,
        strict: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_large_patch16_384",
            pretrained=False,
            num_classes=0,  # drop classifier head; we only want tokens
        )

        patches_per_side = self.INPUT_SIZE // self.PATCH_SIZE  # 24

        # The four DPT reassembly stages -- registered as named submodules
        # whose state_dict keys read ``act_postprocess{1..4}.*`` to match
        # the checkpoint exactly.
        self.act_postprocess1 = _Reassemble(
            self.EMBED_DIM, self.HOOKS_OUT_CHANNELS[0], "up4", patches_per_side
        )
        self.act_postprocess2 = _Reassemble(
            self.EMBED_DIM, self.HOOKS_OUT_CHANNELS[1], "up2", patches_per_side
        )
        self.act_postprocess3 = _Reassemble(
            self.EMBED_DIM, self.HOOKS_OUT_CHANNELS[2], "id", patches_per_side
        )
        self.act_postprocess4 = _Reassemble(
            self.EMBED_DIM, self.HOOKS_OUT_CHANNELS[3], "down2", patches_per_side
        )

        self.scratch = _DPTHead(
            hooks_out_channels=self.HOOKS_OUT_CHANNELS,
            inner_channels=256,
            out_channels=512,
        )

        # Hook storage for the four backbone block activations
        self._hook_outputs: list[torch.Tensor] = [None] * 4  # type: ignore[list-item]
        for slot, block_idx in enumerate(self.HOOK_BLOCKS):
            self.backbone.blocks[block_idx].register_forward_hook(
                self._make_hook(slot)
            )

        if ckpt_path is not None:
            self.load_lseg_checkpoint(ckpt_path, strict=strict)

    def _make_hook(self, slot: int):
        def hook(_mod, _inp, out):
            self._hook_outputs[slot] = out
        return hook

    def load_lseg_checkpoint(self, ckpt_path: str | Path, strict: bool = True) -> None:
        """Load an ``lseg_minimal_e200.ckpt`` Lightning checkpoint.

        The ckpt is structured as ``{epoch, state_dict, optimizer_states,
        ...}`` with state_dict keys prefixed ``net.{pretrained,scratch,
        clip_pretrained}.*``. We strip the ``net.`` prefix, then route:

        * ``pretrained.model.*`` -> ``backbone.*``
        * ``pretrained.act_postprocess{N}.*`` -> ``act_postprocess{N}.*``
        * ``scratch.*`` -> ``scratch.*``
        * ``clip_pretrained.*`` -> dropped (text encoder loaded separately)

        With ``strict=True``, raises if any expected model key is missing
        from the checkpoint, or if any checkpoint key (other than the
        documented "expected extras") is unrecognized.
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        remapped: dict[str, torch.Tensor] = {}
        # Backbone classifier head was discarded by num_classes=0; the timm
        # ``norm.*`` layer is still present and matches.
        ckpt_extras_known = {
            "net.pretrained.model.head.weight",
            "net.pretrained.model.head.bias",
        }
        clip_extras: list[str] = []
        ignored_extras: list[str] = []

        for k, v in state.items():
            if not k.startswith("net."):
                ignored_extras.append(k)
                continue
            rest = k[len("net.") :]

            if rest.startswith("clip_pretrained."):
                clip_extras.append(k)  # text encoder loaded via open_clip
                continue
            if rest.startswith("pretrained.model."):
                new_k = "backbone." + rest[len("pretrained.model.") :]
                if k in ckpt_extras_known:
                    continue  # head dropped by num_classes=0
                remapped[new_k] = v
                continue
            if rest.startswith("pretrained.act_postprocess"):
                # pretrained.act_postprocessN.X.* -> act_postprocessN.X.*
                new_k = rest[len("pretrained.") :]
                remapped[new_k] = v
                continue
            if rest.startswith("scratch."):
                remapped[rest] = v
                continue
            ignored_extras.append(k)

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if strict:
            if missing:
                raise RuntimeError(
                    f"LSeg ckpt missing {len(missing)} model keys, e.g. {missing[:5]}"
                )
            if unexpected:
                raise RuntimeError(
                    f"LSeg ckpt has {len(unexpected)} unexpected keys after remap, "
                    f"e.g. {unexpected[:5]}"
                )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, 512, H, W)``.

        ``H`` and ``W`` must equal ``INPUT_SIZE`` (384). Upstream callers
        in :class:`LSegEncoder` handle the resize before/after.
        """
        # Run backbone forward; hooks populate self._hook_outputs.
        # timm ViT returns the pre-classifier features; we don't use them
        # directly -- we use the per-block activations captured by hooks.
        _ = self.backbone.forward_features(x)

        # Each captured tensor is (B, 1+N, C) with cls at index 0. Pass to
        # reassembly modules, which strip cls via project-readout.
        features: list[torch.Tensor] = []
        for slot in range(4):
            tokens = self._hook_outputs[slot]
            if tokens is None:
                raise RuntimeError(
                    f"Backbone hook slot {slot} not populated -- "
                    "did you forget to call backbone.forward_features?"
                )
            reassemble = getattr(self, f"act_postprocess{slot + 1}")
            features.append(reassemble(tokens))

        out = self.scratch(features, input_size=x.shape[2:])
        return out


# ---------------------------------------------------------------------------
# Public entry point used by pipeline.encode_and_fuse
# ---------------------------------------------------------------------------


class LSegEncoder:
    """LSeg per-pixel feature encoder + CLIP ViT-B/32 text encoder.

    The text encoder is hard-coded to CLIP ViT-B/32 / openai because that is
    the text space LSeg's image features were trained to align with. Mixing
    in a different CLIP variant (e.g. ViT-B/16, ViT-L/14) would silently
    produce garbage cosine scores; ``pipeline._validate_model_match`` enforces
    this at map-load time.
    """

    FEATURE_DIM = 512
    TEXT_MODEL = "ViT-B-32"
    TEXT_PRETRAINED = "openai"
    INPUT_SIZE = LSegDPT.INPUT_SIZE

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
        """RGB uint8 ``(H, W, 3)`` -> ``(H, W, 512)`` fp32 L2-normalized.

        The image is resized to 384x384 for the DPT-Large forward pass
        (matching the trained pos_embed), and the resulting feature map is
        bilinearly upsampled back to the original HxW before normalization.
        """
        from torchvision import transforms

        H, W = rgb.shape[:2]
        t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.INPUT_SIZE, self.INPUT_SIZE), antialias=True
                ),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img_t = t(rgb).unsqueeze(0).to(self.device)  # (1, 3, 384, 384)
        feat = self.model(img_t)  # (1, 512, 384, 384)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)
        feat = feat.squeeze(0).permute(1, 2, 0)  # (H, W, 512)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Text -> ``(512,)`` fp32 L2-normalized via CLIP ViT-B/32."""
        tokens = self._text_tokenizer([text]).to(self.device)
        feat = self._text_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat[0].float().cpu().numpy()
