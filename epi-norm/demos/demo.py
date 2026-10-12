"""epi-norm demo: HBM traffic across 3 designs + Triton status.

Saves assets/traffic_sankey.png (grouped-bar Sankey-style) and
receipts/receipt.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from epi_norm.core import hbm_accountant
from epi_norm.kernels import kernel_status, triton_rmsnorm_fwd

ASSETS = ROOT / "assets"
RECEIPTS = ROOT / "receipts"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)

    # Representative LLM-ish shape (small T for demo speed)
    B, T, D, H = 2, 512, 1024, 2752
    acc = hbm_accountant(B, T, D, H, dtype_bytes=2)
    status = kernel_status()
    rng = np.random.default_rng(0)
    y, rr, info = triton_rmsnorm_fwd(rng.normal(size=(4, 32)), np.ones(32))
    print("kernel status:", status, "->", info["backend"])

    designs = ["baseline", "fused-store", "fused-recompute"]
    labels = ["Baseline\n3 passes", "Fused\nstore all", "Fused\nrecompute rrms"]
    fwd = np.array([acc[d]["fwd_bytes"] for d in designs]) / 1e6
    stored = np.array([acc[d]["stored_bytes"] for d in designs]) / 1e6
    bwd = np.array([acc[d]["bwd_bytes"] for d in designs]) / 1e6

    x = np.arange(len(designs))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x, fwd, label="forward traffic (MB)")
    b2 = ax.bar(x, stored, bottom=fwd, label="stashed for backward (MB)")
    b3 = ax.bar(x, bwd, bottom=fwd + stored, label="backward reload (MB)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("HBM bytes (MB, analytic model)")
    ax.set_title(f"epi-norm HBM accountant  B={B} T={T} D={D} H={H} (bf16)")
    ax.legend()
    for i, d in enumerate(designs):
        tot = acc[d]["total_bytes"] / 1e6
        ax.text(i, fwd[i] + stored[i] + bwd[i] + 2, f"{tot:.0f} MB", ha="center", fontsize=9)
    # annotate saving
    save = (acc["baseline"]["total_bytes"] - acc["fused-recompute"]["total_bytes"]) / acc[
        "baseline"
    ]["total_bytes"]
    ax.annotate(
        f"recompute saves {save*100:.1f}% vs baseline",
        xy=(2, fwd[2] + stored[2] + bwd[2]),
        xytext=(1.0, max(fwd + stored + bwd) * 0.7),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
        ha="center",
    )
    fig.tight_layout()
    out = ASSETS / "traffic_sankey.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    receipt = {
        "project": "epi-norm",
        "config": acc["config"],
        "designs": {d: acc[d] for d in designs},
        "saving_vs_baseline": float(save),
        "kernel_status": status,
        "kernel_backend_demo": info["backend"],
        "assets": ["assets/traffic_sankey.png"],
        "method": "RMSNorm+residual+SwiGLU epilogue fusion; recompute rrms in backward",
    }
    (RECEIPTS / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("saved receipts/receipt.json")


if __name__ == "__main__":
    main()
