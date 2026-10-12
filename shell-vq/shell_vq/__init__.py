"""shell-vq: norm-shelled grouped vector quantization."""

from .core import (
    build_shell_codebook,
    quantize_shelled,
    rate_distortion_floor,
    residual_refine,
)

__all__ = [
    "build_shell_codebook",
    "quantize_shelled",
    "rate_distortion_floor",
    "residual_refine",
]

__version__ = "0.1.0"
