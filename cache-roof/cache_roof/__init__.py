"""cache-roof: cache-aware roofline harness with confidence intervals."""
from .harness import time_fn, run_case
from .roofline import peak_measure, cache_adjusted_roof, classic_roof, analyze

__all__ = ["time_fn", "run_case", "peak_measure", "cache_adjusted_roof", "classic_roof", "analyze"]
__version__ = "0.1.0"
