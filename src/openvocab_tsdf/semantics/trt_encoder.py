"""TensorRT-accelerated CLIP image encoder.

Build flow:
  1. Export the OpenCLIP visual tower to ONNX (fp16 input, static batch).
  2. Build a TRT engine from the ONNX with fp16 precision.
  3. Serve encoded features from the TRT engine via pycuda-free host↔device copies
     (we use PyTorch tensors as the memory manager since they already live on GPU).

The wrapper class `TensorRTImageEncoder` mirrors the subset of
`OpenCLIPEncoder.encode_images` needed by the offline pipeline: it takes a list
of uint8 RGB numpy images, runs the same preprocess, and returns an (N, D)
float32 L2-normalized tensor on GPU. Text encoding stays on PyTorch — it is
already fast and would complicate the engine build.

This is strictly an inference-optimization path. The PyTorch encoder remains the
reference — TRT must agree with it to within ~5e-3 cosine similarity on a fixed
image for the build to be considered healthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TRTImageEncoderConfig:
    model: str = "ViT-B-16"
    pretrained: str = "laion2b_s34b_b88k"
    engine_path: Path = Path("outputs/trt/clip_vit_b16_fp16.engine")
    onnx_path: Path = Path("outputs/trt/clip_vit_b16.onnx")
    batch_size: int = 16
    input_size: int = 224
    fp16: bool = True
    device: str = "cuda:0"
    workspace_gb: int = 4


def _export_onnx(cfg: TRTImageEncoderConfig, pretrained_model) -> None:
    """Export the visual tower (CLS-pooled → proj) to ONNX for TRT consumption."""
    import torch.nn as nn

    class _VisualHead(nn.Module):
        def __init__(self, visual) -> None:
            super().__init__()
            self.visual = visual

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            f = self.visual(x)
            return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    net = _VisualHead(pretrained_model.visual).to(cfg.device).eval()
    dtype = next(pretrained_model.parameters()).dtype
    dummy = torch.zeros(
        cfg.batch_size, 3, cfg.input_size, cfg.input_size, device=cfg.device, dtype=dtype
    )
    cfg.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Static batch — dynamic-axis export through the new torch.export-based
    # dynamo exporter fails on CLIP's multihead attention at the view step.
    # Static-batch export via the legacy tracing exporter works cleanly and
    # is fine because we already pad to `cfg.batch_size` in `encode_images`.
    torch.onnx.export(
        net,
        (dummy,),
        str(cfg.onnx_path),
        input_names=["pixels"],
        output_names=["feat"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _build_engine(cfg: TRTImageEncoderConfig) -> None:
    """Build a TRT engine from the exported ONNX. Writes to `cfg.engine_path`."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(cfg.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_gb * (1 << 30))
    if cfg.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    shape = (cfg.batch_size, 3, cfg.input_size, cfg.input_size)
    # fixed batch for maximum kernel efficiency at this size
    profile.set_shape("pixels", shape, shape, shape)
    config.add_optimization_profile(profile)

    cfg.engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")
    cfg.engine_path.write_bytes(serialized)


class TensorRTImageEncoder:
    """fp16 TRT engine serving normalized CLIP image embeddings."""

    def __init__(
        self,
        cfg: TRTImageEncoderConfig,
        preprocess,
        feature_dim: int,
    ) -> None:
        import tensorrt as trt

        self.cfg = cfg
        self.preprocess = preprocess
        self.feature_dim = feature_dim
        self.device = torch.device(cfg.device)

        if not cfg.engine_path.exists():
            raise FileNotFoundError(
                f"TRT engine missing at {cfg.engine_path}. Build it with "
                f"`build_engine_from_openclip(cfg)`."
            )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(cfg.engine_path.read_bytes())
        self.context = self.engine.create_execution_context()

        shape = (cfg.batch_size, 3, cfg.input_size, cfg.input_size)
        self.context.set_input_shape("pixels", shape)
        self._input = torch.empty(
            shape, dtype=torch.float16 if cfg.fp16 else torch.float32, device=self.device
        )
        self._output = torch.empty(
            (cfg.batch_size, feature_dim),
            dtype=torch.float16 if cfg.fp16 else torch.float32,
            device=self.device,
        )
        self.context.set_tensor_address("pixels", int(self._input.data_ptr()))
        self.context.set_tensor_address("feat", int(self._output.data_ptr()))

    @torch.no_grad()
    def encode_images(self, images: list[np.ndarray]) -> torch.Tensor:
        """(N, D) float32 L2-normalized features on GPU."""
        from PIL import Image

        B = self.cfg.batch_size
        out = []
        for i in range(0, len(images), B):
            batch = images[i : i + B]
            n = len(batch)
            if n < B:
                # pad the trailing batch
                batch = batch + [batch[-1]] * (B - n)
            tensors = torch.stack([self.preprocess(Image.fromarray(img)) for img in batch])
            tensors = tensors.to(self.device, dtype=self._input.dtype, non_blocking=True)
            self._input.copy_(tensors)
            ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 returned False")
            torch.cuda.current_stream().synchronize()
            feats = self._output.float().clone()
            out.append(feats[:n])
        return torch.cat(out, dim=0)


def build_engine_from_openclip(cfg: TRTImageEncoderConfig):
    """One-shot helper: load OpenCLIP, export ONNX, build TRT engine.

    Returns a `TensorRTImageEncoder` ready to serve. Subsequent runs can skip
    the build by calling the encoder constructor directly.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model, pretrained=cfg.pretrained, device=cfg.device
    )
    model = model.eval()
    if cfg.fp16:
        model = model.half()

    # feature_dim from a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            3,
            cfg.input_size,
            cfg.input_size,
            device=cfg.device,
            dtype=next(model.parameters()).dtype,
        )
        feature_dim = int(model.encode_image(dummy).shape[-1])

    _export_onnx(cfg, model)
    _build_engine(cfg)
    return TensorRTImageEncoder(cfg, preprocess, feature_dim)
