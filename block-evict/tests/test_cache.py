"""Tests for block-evict (6 tests)."""
import pytest

from block_evict.cache import BlockCache
from block_evict.scoring import block_score, rank_blocks_for_eviction


def test_alloc_read_roundtrip():
    c = BlockCache(num_blocks=8, block_size=4)
    toks = list(range(10))
    assert c.alloc("s0", toks, attn=[0.1] * 10) is True
    assert c.read("s0") == toks
    assert c.seq_len("s0") == 10
    assert c.num_used == 3  # ceil(10/4)


def test_fork_shares_physical_blocks():
    c = BlockCache(num_blocks=8, block_size=4)
    assert c.alloc("p", [1, 2, 3, 4, 5], attn=[0.5] * 5)
    c.fork("p", "c")
    assert c.seq_blocks("p") == c.seq_blocks("c")  # same physical blocks
    for p in c.seq_blocks("p"):
        assert c.refcount(p) == 2
    assert c.read("c") == c.read("p")


def test_cow_append_preserves_parent():
    c = BlockCache(num_blocks=8, block_size=4)
    assert c.alloc("p", [1, 2, 3], attn=[0.5] * 3)
    c.fork("p", "c")
    before_parent = c.read("p")
    assert c.append("c", 99, attn=0.9) is True
    assert c.read("p") == before_parent  # parent untouched
    assert c.read("c") == before_parent + [99]
    # tail block was COW-copied: parent and child tails now differ physically
    assert c.seq_blocks("p")[-1] != c.seq_blocks("c")[-1]


def test_block_evict_frees_whole_blocks():
    c = BlockCache(num_blocks=8, block_size=4)
    assert c.alloc("s0", list(range(8)), attn=[0.1] * 8)   # 2 blocks, low attn
    assert c.alloc("s1", list(range(8)), attn=[0.9] * 8)   # 2 blocks, high attn
    free_before = c.num_free
    assert free_before == 4
    len0, len1 = c.seq_len("s0"), c.seq_len("s1")
    freed = c.block_evict(2, frag_weight=0.0)
    assert len(freed) == 2
    assert c.num_free == free_before + 2  # whole blocks returned to pool
    # block-aligned: each evicted prefix removes exactly block_size tokens
    assert (len0 - c.seq_len("s0")) + (len1 - c.seq_len("s1")) == 2 * c.block_size
    # low-attention seq evicted first
    assert c.seq_len("s0") < len0


def test_scoring_ranks_low_attention_first():
    blocks = [
        {"block_id": "hi", "mean_attn": 0.9, "occupancy": 1.0},
        {"block_id": "lo", "mean_attn": 0.1, "occupancy": 1.0},
        {"block_id": "mid", "mean_attn": 0.5, "occupancy": 1.0},
    ]
    ranked = rank_blocks_for_eviction(blocks)
    assert [b["block_id"] for b in ranked] == ["lo", "mid", "hi"]
    assert block_score(0.1) < block_score(0.9)


def test_oom_then_evict_recovers_and_shared_prefix_retained():
    c = BlockCache(num_blocks=4, block_size=4)
    assert c.alloc("a", list(range(8)), attn=[0.2] * 8)  # 2 blocks
    assert c.alloc("b", list(range(8)), attn=[0.8] * 8)  # 2 blocks -> full
    assert c.num_free == 0
    assert c.alloc("c", [1, 2], attn=[0.5, 0.5]) is False  # OOM
    c.fork("a", "a2")  # a's blocks now shared (refcount 2) -> must be retained
    shared = set(c.seq_blocks("a"))
    freed = c.block_evict(1)
    assert len(freed) == 1
    assert freed[0] not in shared  # shared COW prefix never evicted
    for p in shared:
        assert c.refcount(p) == 2
    assert c.read("a") == c.read("a2")  # prefix sharing intact
    assert c.alloc("c", [1, 2], attn=[0.5, 0.5]) is True  # pressure relieved
    assert c.read("c") == [1, 2]
