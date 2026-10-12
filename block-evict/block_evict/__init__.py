"""block-evict: block-aligned eviction for paged KV caches."""
from .cache import BlockCache
from .scoring import block_score, fragmentation_cost, mean_attention, rank_blocks_for_eviction

__all__ = ["BlockCache", "block_score", "fragmentation_cost", "mean_attention", "rank_blocks_for_eviction"]
__version__ = "0.1.0"
