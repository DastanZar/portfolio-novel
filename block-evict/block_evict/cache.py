"""BlockCache: PagedAttention-style block manager with COW + block-aligned eviction.

Original implementation (no code copied from vLLM). Key properties:

* Fixed pool of physical blocks, each holding up to ``block_size`` tokens.
* Logical -> physical mapping per sequence; ``fork`` shares physical blocks
  with copy-on-write (refcount > 1); ``append`` copies the tail block on write
  when shared (true COW) so parents are never corrupted.
* ``block_evict`` scores WHOLE blocks by mean future-attention + fragmentation
  cost and evicts full blocks (prefix, all except each seq's active tail), so
  every eviction actually returns physical blocks to the free pool — unlike
  token-granular sliding windows that leave partially-filled blocks pinned.
"""
from __future__ import annotations

from .scoring import fragmentation_cost, rank_blocks_for_eviction


class BlockCache:
    def __init__(self, num_blocks: int, block_size: int = 16):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be > 0")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self._free: list[int] = list(reversed(range(self.num_blocks)))
        self._refcount: list[int] = [0] * self.num_blocks
        self._tokens: list[list[int]] = [[] for _ in range(self.num_blocks)]
        self._attn: list[list[float]] = [[] for _ in range(self.num_blocks)]
        # seq_id -> {"blocks": [phys_id...], "length": total live tokens}
        self._seqs: dict[str, dict] = {}

    # ---- introspection ----
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def seq_ids(self) -> list[str]:
        return sorted(self._seqs.keys())

    def seq_len(self, seq_id: str) -> int:
        return self._seqs[seq_id]["length"]

    def seq_blocks(self, seq_id: str) -> list[int]:
        return list(self._seqs[seq_id]["blocks"])

    def refcount(self, phys_id: int) -> int:
        return self._refcount[phys_id]

    def fragmentation(self) -> float:
        """1 - used_tokens / (used_blocks * block_size); 0 = perfectly packed."""
        used_tokens = sum(len(self._tokens[p]) for p in range(self.num_blocks) if self._refcount[p] > 0)
        denom = self.num_used * self.block_size
        if denom == 0:
            return 0.0
        return 1.0 - used_tokens / denom

    # ---- core ops ----
    def _take_block(self) -> int | None:
        if not self._free:
            return None
        p = self._free.pop()
        self._refcount[p] = 1
        self._tokens[p] = []
        self._attn[p] = []
        return p

    def _release_block(self, phys_id: int) -> None:
        self._refcount[phys_id] = 0
        self._tokens[phys_id] = []
        self._attn[phys_id] = []
        self._free.append(phys_id)

    def alloc(self, seq_id: str, tokens: list[int], attn: list[float] | None = None) -> bool:
        """Allocate a new sequence. Returns False on OOM (no partial alloc)."""
        if seq_id in self._seqs:
            raise KeyError(f"seq {seq_id!r} already exists")
        n = len(tokens)
        need = (n + self.block_size - 1) // self.block_size if n else 1
        if len(self._free) < need:
            return False
        attn = list(attn) if attn is not None else [0.5] * n
        if len(attn) != n:
            raise ValueError("attn length must match tokens")
        blocks: list[int] = []
        for i in range(need):
            p = self._take_block()
            assert p is not None
            chunk_t = tokens[i * self.block_size : (i + 1) * self.block_size]
            chunk_a = attn[i * self.block_size : (i + 1) * self.block_size]
            self._tokens[p] = list(chunk_t)
            self._attn[p] = [float(x) for x in chunk_a]
            blocks.append(p)
        self._seqs[seq_id] = {"blocks": blocks, "length": n if n else 0}
        # zero-length edge: keep the 1 block but length 0 (allows append path)
        return True

    def fork(self, src_id: str, dst_id: str) -> None:
        """Share all physical blocks of src with dst (COW prefix sharing)."""
        if dst_id in self._seqs:
            raise KeyError(f"seq {dst_id!r} already exists")
        src = self._seqs[src_id]  # KeyError if missing — intentional
        for p in src["blocks"]:
            self._refcount[p] += 1
        self._seqs[dst_id] = {"blocks": list(src["blocks"]), "length": src["length"]}

    def append(self, seq_id: str, token: int, attn: float = 0.5) -> bool:
        """Append one token with COW. Returns False on OOM."""
        seq = self._seqs[seq_id]
        blocks = seq["blocks"]
        if not blocks:
            p = self._take_block()
            if p is None:
                return False
            blocks.append(p)
        last = blocks[-1]
        if len(self._tokens[last]) >= self.block_size:
            p = self._take_block()
            if p is None:
                return False
            blocks.append(p)
            last = p
        elif self._refcount[last] > 1:
            # COW: copy shared tail block before mutating
            p = self._take_block()
            if p is None:
                return False
            self._tokens[p] = list(self._tokens[last])
            self._attn[p] = list(self._attn[last])
            self._refcount[last] -= 1
            blocks[-1] = p
            last = p
        self._tokens[last].append(int(token))
        self._attn[last].append(float(attn))
        seq["length"] += 1
        return True

    def read(self, seq_id: str) -> list[int]:
        seq = self._seqs[seq_id]
        out: list[int] = []
        for p in seq["blocks"]:
            out.extend(self._tokens[p])
        return out[: seq["length"]]

    def free_seq(self, seq_id: str) -> None:
        seq = self._seqs.pop(seq_id)
        for p in seq["blocks"]:
            self._refcount[p] -= 1
            if self._refcount[p] == 0:
                self._release_block(p)

    # ---- block-aligned eviction ----
    def _candidates(self) -> list[dict]:
        """Whole-block eviction candidates: every block except each seq's
        active tail (append target). Only exclusively-owned (refcount==1)
        full blocks are eligible so eviction always frees whole blocks."""
        cands: list[dict] = []
        for seq_id, seq in self._seqs.items():
            blocks = seq["blocks"]
            for pos, p in enumerate(blocks):
                is_tail = pos == len(blocks) - 1
                if is_tail:
                    continue  # keep the active working set
                if self._refcount[p] != 1:
                    continue  # shared COW prefix: retained
                toks = self._tokens[p]
                att = self._attn[p]
                mean_a = sum(att) / len(att) if att else 0.0
                occ = len(toks) / self.block_size
                cands.append({
                    "block_id": p,
                    "seq_id": seq_id,
                    "pos": pos,
                    "mean_attn": mean_a,
                    "occupancy": occ,
                    "frag": fragmentation_cost(len(toks), self.block_size),
                })
        return cands

    def block_evict(self, n_blocks: int, frag_weight: float = 0.1) -> list[int]:
        """Evict up to ``n_blocks`` whole blocks with the lowest
        (attention, fragmentation) score. Returns freed physical block ids.

        Eviction removes the block from its owning sequence's prefix
        (block-aligned truncation) and returns it to the free pool.
        Shared (COW) blocks are never evicted — prefix sharing is retained.
        """
        if n_blocks <= 0:
            return []
        ranked = rank_blocks_for_eviction(self._candidates(), frag_weight=frag_weight)
        freed: list[int] = []
        for entry in ranked:
            if len(freed) >= n_blocks:
                break
            p = entry["block_id"]
            seq_id = entry["seq_id"]
            if seq_id not in self._seqs:
                continue
            seq = self._seqs[seq_id]
            if p not in seq["blocks"]:
                continue
            if self._refcount[p] != 1:
                continue
            # remove whole block from prefix; adjust live length by tokens held
            seq["blocks"].remove(p)
            seq["length"] = max(0, seq["length"] - self.block_size)
            # NOTE: tokens list len == block_size for non-tail full blocks
            self._release_block(p)
            freed.append(p)
        return freed
