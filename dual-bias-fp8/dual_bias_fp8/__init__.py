"""dual-bias-fp8: per-block dual format + dual-bias FP8 GEMV (numpy reference)."""

from .core import (
    BLOCK,
    E4M3_MAX,
    E5M2_MAX,
    HAS_TRITON,
    calibrate_bias,
    choose_format,
    pack_weights,
    gemv_reference,
    gemv_triton,
    quantize_block,
    dequantize_block,
    bytes_model,
)

__all__ = [
    "BLOCK",
    "E4M3_MAX",
    "E5M2_MAX",
    "HAS_TRITON",
    "calibrate_bias",
    "choose_format",
    "pack_weights",
    "gemv_reference",
    "gemv_triton",
    "quantize_block",
    "dequantize_block",
    "bytes_model",
]

__version__ = "0.1.0"
