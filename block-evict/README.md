# block-evict — block-aligned eviction for PagedAttention-style KV caches

Novel successor to a token-level paged-KV + eviction lab. All code is fresh and
original for this portfolio project.

**Novelty (what's new here):** token-granular eviction (H2O / StreamingLLM-style
per-token dropping, sliding windows) leaves partially-filled vLLM blocks pinned —
the block is still allocated even though most of its tokens are gone (the infra
problem noted in the NVIDIA EAI paged-KV blog). This project instead:

1. **Scores whole blocks, not tokens** — `block_evict/scoring.py` ranks each
   physical block by `mean future-attention + fragmentation cost`.
2. **Evicts full blocks** — `BlockCache.block_evict(k)` removes the lowest-score
   prefix blocks (never the active tail) and returns entire physical blocks to
   the free pool. Every eviction frees `k × block_size` tokens of capacity.
3. **Retains COW prefix sharing** — `fork()` shares physical blocks with
   refcounts; shared blocks are never evicted and `append()` copies-on-write, so
   a shared system-prompt prefix survives pressure while private low-attention
   blocks are reclaimed first.

## Quickstart

```bash
pip install -e .
pytest
python demos/demo.py
```

Outputs: `receipts/receipt.json`, `assets/blockmap.png` (physical-block map).

## API

```python
from block_evict.cache import BlockCache
from block_evict.scoring import block_score, rank_blocks_for_eviction

c = BlockCache(num_blocks=16, block_size=8)
c.alloc("s0", tokens, attn=[...])
c.fork("s0", "s1")          # COW prefix sharing
c.append("s1", tok, attn=..) # copy-on-write tail
freed = c.block_evict(2)     # frees 2 whole physical blocks
```

## Attribution / related work

- PagedAttention (Kwon et al.) — non-contiguous block-paged KV cache.
- H2O — heavy-hitter token eviction policy (per-token scoring baseline).
- StreamingLLM — sliding-window / sink-token retention baseline.
- PagedEviction (arXiv 2509.04377) — block-level eviction framing.
- NVIDIA EAI blog — paged-KV infra: partial-block pinning problem.

This repo re-implements the *ideas* above from scratch; no code is copied.

## License

MIT — see `LICENSE`.
