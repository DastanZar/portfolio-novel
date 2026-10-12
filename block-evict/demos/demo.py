"""Demo: block-aligned eviction under pressure with COW prefix sharing.

Outputs:
  receipts/receipt.json
  assets/blockmap.png  (physical-block map before/after eviction)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from block_evict.cache import BlockCache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def snapshot(cache: BlockCache):
    """Per-physical-block state: -1.0 if free else mean attn."""
    vals = []
    for p in range(cache.num_blocks):
        if cache._refcount[p] == 0:
            vals.append(-1.0)
        else:
            att = cache._attn[p]
            vals.append(sum(att) / len(att) if att else 0.0)
    return vals


def plot_blockmap(before, after, ncols, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = mcolors.ListedColormap(["#d3d3d3", "#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"])
    bounds = [-1.5, -0.5, 0.25, 0.5, 0.75, 1.01]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    for ax, vals, title in ((axes[0], before, "before eviction"), (axes[1], after, "after block_evict(4)")):
        nrows = (len(vals) + ncols - 1) // ncols
        grid = [[-1.0] * ncols for _ in range(nrows)]
        for i, v in enumerate(vals):
            grid[i // ncols][i % ncols] = v
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < len(vals):
                    txt = "free" if vals[idx] < 0 else f"{vals[idx]:.2f}"
                    ax.text(j, i, f"B{idx}\n{txt}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="mean future-attention (gray=free)", shrink=0.8)
    fig.suptitle("Block-aligned eviction frees whole physical blocks (COW prefix retained)")
    fig.subplots_adjust(top=0.85, bottom=0.08, wspace=0.15)
    fig.savefig(path, dpi=130)


def main():
    rng = random.Random(11)
    cache = BlockCache(num_blocks=16, block_size=8)
    # low-attention background seqs (evictable) + high-attention hot seqs (retained)
    cache.alloc("bg0", list(range(24)), attn=[0.10 + rng.random() * 0.1 for _ in range(24)])
    cache.alloc("bg1", list(range(24)), attn=[0.15 + rng.random() * 0.1 for _ in range(24)])
    cache.alloc("hot0", list(range(16)), attn=[0.85 + rng.random() * 0.1 for _ in range(16)])
    cache.fork("hot0", "hot0-chat")  # COW prefix sharing
    cache.append("hot0-chat", 999, attn=0.9)
    free_before = cache.num_free
    frag_before = cache.fragmentation()
    before = snapshot(cache)
    freed = cache.block_evict(4, frag_weight=0.1)
    after = snapshot(cache)
    # pressure relief: new arrival fits now
    ok = cache.alloc("new", list(range(8)), attn=[0.5] * 8)

    os.makedirs(os.path.join(ROOT, "receipts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    receipt = {
        "num_blocks": cache.num_blocks,
        "block_size": cache.block_size,
        "free_before": free_before,
        "free_after": cache.num_free,
        "freed_phys_ids": sorted(freed),
        "frag_before": frag_before,
        "frag_after": cache.fragmentation(),
        "new_alloc_ok_after_evict": ok,
        "cow_pair_read_equal_prefix": cache.read("hot0") == cache.read("hot0-chat")[: cache.seq_len("hot0")],
        "seqs": {s: {"len": cache.seq_len(s), "blocks": cache.seq_blocks(s)} for s in cache.seq_ids()},
    }
    with open(os.path.join(ROOT, "receipts", "receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    plot_blockmap(before, after, ncols=4, path=os.path.join(ROOT, "assets", "blockmap.png"))
    print(json.dumps(receipt, indent=2))
    print("wrote receipts/receipt.json + assets/blockmap.png")


if __name__ == "__main__":
    main()
