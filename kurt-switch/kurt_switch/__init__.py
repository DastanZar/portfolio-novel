"""kurt-switch: per-row kurtosis-switched 4-bit codebooks."""

from .core import (
    CODEBOOK_NAMES,
    compute_row_stats,
    get_codebook,
    quantize_block,
    quantize_row,
    select_codebook,
)

__all__ = [
    "CODEBOOK_NAMES",
    "compute_row_stats",
    "get_codebook",
    "quantize_block",
    "quantize_row",
    "select_codebook",
]

__version__ = "0.1.0"
