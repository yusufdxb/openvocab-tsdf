"""TensorRT-accelerated MobileSAM image encoder.

MobileSAM's `image_encoder` is a TinyViT baked to a fixed 1024×1024 input
(the positional embedding and window-attention shapes are static). It is
the dominant cost inside `SAMDenseFeatureExtractor` — ~1.0–1.5 s / frame
in fp32 PyTorch on an NVIDIA Blackwell consumer GPU, which is what makes `sam_dense` pipelines
feel offline-only. An fp16 TRT engine for the same forward drops that to
~150–300 ms, putting the real-time live-mapping path within reach.

Build flow mirrors `trt_encoder.py` (CLIP visual tower):
  1. Export the TinyViT image_encoder to ONNX (fp16, static (1, 3, 1024, 1024)
     input, legacy tracing exporter because the dynamo exporter trips on
     window-attention view-reshapes — same bug pattern as the CLIP MHA
     view-reshape; see `docs/getting-started.md#8`).
  2. Build a TRT engine with fp16 precision.
  3. `TensorRTSamEncoder` is a drop-in replacement for
     `sam.image_encoder(x)` — same input/output shape, same semantics.
     The SAM automatic-mask-generator downstream doesn't care whether
     the encoder is PyTorch or TRT; it only needs `(B, 256, 64, 64)`.

The PyTorch encoder remains the reference — TRT must agree with it to
within a cosine similarity of at least 0.98 on a fixed batch of inputs
(see `tests/unit/test_trt_sam.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

MOBILE_SAM_IMG_SIZE = 1024  # hard-baked into the TinyViT weights
MOBILE_SAM_OUT_C = 256
MOBILE_SAM_OUT_HW = 64


@dataclass
class TRTSamConfig:
    sam_weights_path: Path = Path("~/.cache/openvocab_tsdf/weights/mobile_sam.pt").expanduser()
    sam_model_type: str = "vit_t"
    engine_path: Path = Path("outputs/trt/mobile_sam_vit_t_fp32.engine")
    onnx_path: Path = Path("outputs/trt/mobile_sam_vit_t_fp32.onnx")
    input_size: int = MOBILE_SAM_IMG_SIZE
    batch_size: int = 1
    # Default to fp32: on real images (not the random-noise parity inputs),
    # TRT fp16 perturbs MobileSAM's image embedding enough that downstream
    # mask proposals shift and grounding accuracy drops ~30 pp hit@1 on
    # Replica room0. TRT fp32 preserves parity (mean cos 0.9995 vs PyTorch
    # fp32 on a real frame) and still gives ~10 % encoder speedup. The fp16
    # path remains available as an opt-in when encoder throughput matters
    # more than grounding accuracy (e.g., live demo at 3+ Hz). See
    # `docs/decisions.md` 2026-04-13 "TensorRT MobileSAM" ADR.
    fp16: bool = False
    device: str = "cuda:0"
    workspace_gb: int = 4


def _export_onnx(cfg: TRTSamConfig, image_encoder: torch.nn.Module) -> None:
    """Export MobileSAM's image_encoder to ONNX for TRT consumption."""
    cfg.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = next(image_encoder.parameters()).dtype
    dummy = torch.zeros(
        cfg.batch_size, 3, cfg.input_size, cfg.input_size, device=cfg.device, dtype=dtype
    )
    # Static batch + legacy tracing exporter — the dynamo exporter trips on the
    # window-attention view/reshape the same way it trips on CLIP's MHA
    # view-reshape (see trt_encoder.py). Fixed-batch static-shape tracing
    # is fine for our use (B=1 per-frame).
    torch.onnx.export(
        image_encoder,
        (dummy,),
        str(cfg.onnx_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _build_engine(cfg: TRTSamConfig) -> None:
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
    profile.set_shape("image", shape, shape, shape)
    config.add_optimization_profile(profile)

    cfg.engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")
    cfg.engine_path.write_bytes(serialized)


class TensorRTSamEncoder(torch.nn.Module):
    """Drop-in replacement for `sam.image_encoder(x)` that runs the TinyViT
    forward through a pre-built fp16 TRT engine.

    Inherits `nn.Module` so it can be assigned to `sam.image_encoder`
    without tripping the child-module type check that `nn.Module.__setattr__`
    enforces on previously-registered submodules. The TRT engine and IO
    tensors are stored as plain attributes (not parameters / buffers), so
    `.to(device)` and `.train() / .eval()` are effectively no-ops — the
    engine is already pinned to the device it was built on.
    """

    def __init__(self, cfg: TRTSamConfig) -> None:
        super().__init__()
        import tensorrt as trt

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        if not cfg.engine_path.exists():
            raise FileNotFoundError(
                f"TRT engine missing at {cfg.engine_path}. Build it with "
                f"`build_engine_from_mobile_sam(cfg)`."
            )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(cfg.engine_path.read_bytes())
        self.context = self.engine.create_execution_context()

        in_shape = (cfg.batch_size, 3, cfg.input_size, cfg.input_size)
        out_shape = (cfg.batch_size, MOBILE_SAM_OUT_C, MOBILE_SAM_OUT_HW, MOBILE_SAM_OUT_HW)
        self.context.set_input_shape("image", in_shape)
        self._io_dtype = torch.float16 if cfg.fp16 else torch.float32
        self._input = torch.empty(in_shape, dtype=self._io_dtype, device=self.device)
        self._output = torch.empty(out_shape, dtype=self._io_dtype, device=self.device)
        self.context.set_tensor_address("image", int(self._input.data_ptr()))
        self.context.set_tensor_address("embedding", int(self._output.data_ptr()))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-2] != self.cfg.input_size or x.shape[-1] != self.cfg.input_size:
            raise ValueError(
                f"expected (B, 3, {self.cfg.input_size}, {self.cfg.input_size}), got {tuple(x.shape)}"
            )
        B = int(x.shape[0])
        out_dtype = x.dtype
        out = torch.empty(
            (B, MOBILE_SAM_OUT_C, MOBILE_SAM_OUT_HW, MOBILE_SAM_OUT_HW),
            dtype=out_dtype,
            device=x.device,
        )
        bs = self.cfg.batch_size
        for i in range(0, B, bs):
            n = min(bs, B - i)
            # pad trailing partial batch with the last frame so kernel shapes stay static
            chunk = x[i : i + n]
            if n < bs:
                pad = chunk[-1:].expand(bs - n, -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=0)
            self._input.copy_(chunk.to(self._io_dtype))
            ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 returned False")
            torch.cuda.current_stream().synchronize()
            out[i : i + n] = self._output[:n].to(out_dtype)
        return out

    # Compatibility shim — MobileSAM's SamAutomaticMaskGenerator sometimes
    # consults `.img_size` on the image_encoder to set up preprocessing.
    @property
    def img_size(self) -> int:
        return self.cfg.input_size


def build_engine_from_mobile_sam(cfg: TRTSamConfig) -> TensorRTSamEncoder:
    """One-shot: load MobileSAM, export ONNX, build TRT engine, return encoder.

    Subsequent runs can skip the build by constructing `TensorRTSamEncoder(cfg)`
    directly — the constructor only reads the engine bytes from disk.
    """
    from mobile_sam import sam_model_registry

    sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_weights_path))
    sam = sam.to(cfg.device).eval()
    if cfg.fp16:
        sam = sam.half()
    image_encoder = sam.image_encoder
    _export_onnx(cfg, image_encoder)
    _build_engine(cfg)
    return TensorRTSamEncoder(cfg)
