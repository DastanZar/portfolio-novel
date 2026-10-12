"""Demo: bandwidth vs accuracy tradeoff for dual-bias FP8 GEMV."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dual_bias_fp8.core import bytes_model, pack_weights, gemv_reference

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def fp16_sim(x):
    return x.astype(np.float16).astype(np.float64)


def int8_block_sim(W):
    Q = W.copy()
    K = W.shape[1]
    for s in range(0, K, 32):
        blk = W[:, s : s + 32]
        amax = np.max(np.abs(blk), axis=1, keepdims=True) + 1e-12
        q = np.round(blk / amax * 127) / 127 * amax
        Q[:, s : s + 32] = q
    return Q


def uniform_e5m2_sim(W):
    from dual_bias_fp8.core import _decode, _encode

    flat = W.ravel()
    amax = float(np.max(np.abs(flat))) + 1e-12
    s = amax / (57344.0 * 0.9)
    xs = flat / s
    back = _decode(_encode(xs, "e5m2"), "e5m2") * s
    return back.reshape(W.shape)


def main():
    rng = np.random.default_rng(7)
    M, K = 64, 256
    # mixture: tame rows + spiky rows to reward per-block adaptivity
    W = rng.normal(0, 1.5, (M, K))
    W[::8] *= 40.0
    W[3, 10] = 900.0
    x = rng.normal(0, 1, K)
    y32 = W @ x

    cands = {}
    cands["fp32"] = (W @ x, bytes_model(M, K)["fp32"])
    cands["fp16"] = (fp16_sim(W) @ fp16_sim(x), bytes_model(M, K)["fp16"])
    cands["int8_block"] = (int8_block_sim(W) @ x, bytes_model(M, K)["int8_per_block"])
    cands["fp8_uniform"] = (uniform_e5m2_sim(W) @ x, bytes_model(M, K)["fp8_uniform"])
    packed = pack_weights(W)
    cands["dual_bias_fp8"] = (gemv_reference(packed, x), bytes_model(M, K)["dual_bias_fp8"])

    nrm = float(np.linalg.norm(y32) + 1e-12)
    rows = {}
    for name, (y, nbytes) in cands.items():
        rows[name] = {
            "bytes": int(nbytes),
            "rel_err": float(np.linalg.norm(y - y32) / nrm),
        }

    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "receipts"), exist_ok=True)
    names = list(rows)
    xs = [rows[n]["bytes"] / 1024 for n in names]
    ys = [rows[n]["rel_err"] + 1e-9 for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    colors = {"fp32": "black", "fp16": "gray", "int8_block": "tab:orange",
              "fp8_uniform": "tab:blue", "dual_bias_fp8": "tab:red"}
    for n in names:
        ax.scatter([rows[n]["bytes"] / 1024], [rows[n]["rel_err"] + 1e-9],
                   s=90, label=n, color=colors.get(n, "green"), zorder=3)
    # pareto arrow to novelty point
    ax.annotate("dual-bias: per-block fmt+bias",
                xy=(rows["dual_bias_fp8"]["bytes"] / 1024, rows["dual_bias_fp8"]["rel_err"] + 1e-9),
                xytext=(0.35, 0.25), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("Weight bytes (KiB, log)")
    ax.set_ylabel("GEMV relative error vs fp32 (log)")
    ax.set_title("Bandwidth vs accuracy: dual-bias FP8 GEMV")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets/dual_bias.png"), dpi=150)

    with open(os.path.join(ROOT, "receipts/receipt.json"), "w") as f:
        json.dump({"M": M, "K": K, "methods": rows,
                   "artifact": "assets/dual_bias.png"}, f, indent=2)
    print("wrote assets/dual_bias.png + receipts/receipt.json")
    for n in names:
        print(f"{n:14s} bytes={rows[n]['bytes']:6d} rel_err={rows[n]['rel_err']:.3e}")


if __name__ == "__main__":
    main()
