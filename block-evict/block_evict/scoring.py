"""Block-aligned eviction scoring for PagedAttention-style KV caches.

Novelty: score WHOLE blocks (not tokens) by mean future-attention plus a
fragmentation cost, so eviction frees entire physical blocks instead of
punching holes inside them. Original code for this portfolio project.
"""
from __future__ import annotations


def mean_attention(attn: list[float]) -> float:
    if not attn:
        return 0.0
    return sum(float(x) for x in attn) / len(attn)


def fragmentation_cost(used_slots: int, block_size: int) -> float:
    """Fraction of wasted slots in a block, in [0, 1]."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    used_slots = max(0, min(int(used_slots), block_size))
    return 1.0 - used_slots / block_size


def block_score(mean_attn: float, occupancy: float = 1.0, frag_weight: float = 0.1) -> float:
    """Eviction score for one block. LOWER = evict first.

    score = mean_attn - frag_weight * (1 - occupancy)

    Full blocks (occupancy 1) score purely on attention; fragmented
    (partially-filled) blocks get a small discount because evicting them
    still frees a whole physical block while discarding less useful state.
    """
    return float(mean_attn) - float(frag_weight) * (1.0 - float(occupancy))


def rank_blocks_for_eviction(blocks: list[dict], frag_weight: float = 0.1) -> list[dict]:
    """Rank whole-block candidates for eviction (lowest score first).

    Each entry must have at least ``block_id`` and ``mean_attn``; optional
    ``occupancy`` (default 1.0). Entries may carry extra keys (seq_id, ...).
    Returns a new sorted list with ``score`` filled in. Ties broken by
    ``block_id`` for determinism.
    """
    ranked = []
    for b in blocks:
        occ = float(b.get("occupancy", 1.0))
        score = block_score(float(b.get("mean_attn", 0.0)), occ, frag_weight)
        entry = dict(b)
        entry["score"] = score
        entry["occupancy"] = occ
        ranked.append(entry)
    ranked.sort(key=lambda e: (e["score"], str(e.get("block_id"))))
    return ranked
