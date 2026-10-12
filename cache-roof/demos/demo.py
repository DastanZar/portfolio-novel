"""Demo: benchmark 4 CPU kernels -> cache-aware ladder + receipt."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cache_roof.harness import run_case
from cache_roof.roofline import analyze, cache_adjusted_roof, peak_measure

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RECEIPTS = ROOT / "receipts"


def make_kernels():
    rng = np.random.default_rng(0)
    va, vb = rng.random(2_000_000, dtype=np.float32), rng.random(2_000_000, dtype=np.float32)
    m = rng.random((512, 512), dtype=np.float32)
    w8 = rng.integers(0, 127, size=(512, 512)).astype(np.float32)
    scale = np.float32(0.02)
    x512 = rng.random(512).astype(np.float32)
    q, k = rng.random((256, 64), dtype=np.float32), rng.random((1024, 64), dtype=np.float32)
    v = rng.random((1024, 64), dtype=np.float32)

    def attn_decode():
        s = q @ k.T
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s, dtype=np.float32)
        return (e / e.sum(axis=1, keepdims=True)) @ v
    return {
        # memory-bound streaming kernel: 1 add flop per 12 bytes (2 reads + 1 write, fp32)
        "vec-add": (lambda: np.add(va, vb, out=np.empty_like(va)),
                    2_000_000.0, 24_000_000.0, 0.0, 0.0),
        # compute-leaning dense kernel
        "matmul-512": (lambda: m @ m, 2 * 512**3, 2 * 512 * 512 * 4, 512 * 512 * 4, 0.0),
        # quantized GEMV with overlapped dequant (async scale multiply hidden).
        # FLOPs include the N^2 dequant scales + 2N^2 GEMV ops.
        "q-gemv-dequant": (lambda: (w8 * scale) @ x512,
                           3 * 512 * 512, 512 * 512, 512 * 4, 0.4),
        # decode-attention tile with high L2 reuse of K cache
        "attn-decode": (attn_decode,
                        4 * 256 * 1024 * 64, 1024 * 64 * 4, 9 * 1024 * 64 * 4, 0.2),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    peaks = peak_measure()
    print(f"measured peaks: {peaks}")

    kernels = make_kernels()
    rows = []
    for name, (fn, flops, b_dram, b_l2, q) in kernels.items():
        case = run_case(name, fn, flops=flops, bytes_dram=b_dram, bytes_l2=b_l2,
                        trials=20, warmup=3)
        roof = cache_adjusted_roof(
            flops=flops, bytes_dram=b_dram, bytes_l2=b_l2,
            peak_gflops=peaks["peak_gflops"], bw_dram_gbps=peaks["bw_dram_gbps"],
            bw_l2_gbps=peaks["bw_l2_gbps"], dequant_overlap=q)
        graded = analyze(case, roof)
        rows.append({"case": case, "roof": roof, "graded": graded})
        print(f"{name:14s} AI={case['oi_dram']:.2f} ach={case['achieved_gflops']:.2f} "
              f"roof={roof['roof_gflops']:.2f} grade={graded['grade_pct']:.1f}% [{graded['verdict']}]")

    # ---- ladder plot ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.logspace(-1, 2, 200)
    ax.loglog(xs, np.minimum(peaks["peak_gflops"], xs * peaks["bw_dram_gbps"]),
              label=f"DRAM roof ({peaks['bw_dram_gbps']:.1f} GB/s)", lw=2)
    ax.loglog(xs, np.minimum(peaks["peak_gflops"], xs * peaks["bw_l2_gbps"]),
              "--", label=f"L2 roof ({peaks['bw_l2_gbps']:.1f} GB/s)")
    ax.axhline(peaks["peak_gflops"], color="k", lw=1.5,
               label=f"compute ({peaks['peak_gflops']:.1f} GFLOP/s)")
    for r in rows:
        c, g = r["case"], r["graded"]
        yerr = np.array([[[c["achieved_gflops"] - c["achieved_gflops_lo"]],
                          [c["achieved_gflops_hi"] - c["achieved_gflops"]]]]).reshape(2, 1)
        ax.errorbar([c["oi_dram"]], [c["achieved_gflops"]], yerr=yerr, fmt="o",
                    capsize=4, label=f"{c['name']} {g['grade_pct']:.0f}% of cache-roof")
    ax.set_xlabel("Operational intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Achieved (GFLOP/s)")
    ax.set_title("cache-roof ladder: % of cache-adjusted peak (95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    out_png = ASSETS / "ladder.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")

    receipt = {
        "project": "cache-roof",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "peaks": peaks,
        "kernels": [
            {"name": r["case"]["name"], "oi_dram": r["case"]["oi_dram"],
             "achieved_gflops": r["case"]["achieved_gflops"],
             "achieved_ci": [r["case"]["achieved_gflops_lo"], r["case"]["achieved_gflops_hi"]],
             "roof_gflops": r["roof"]["roof_gflops"], "classic_gflops": r["roof"]["classic_gflops"],
             "bound": r["roof"]["bound"], "l2_hit_rate": r["roof"]["l2_hit_rate"],
             "grade_pct": r["graded"]["grade_pct"],
             "grade_ci": [r["graded"]["grade_lo"], r["graded"]["grade_hi"]],
             "verdict": r["graded"]["verdict"]}
            for r in rows
        ],
        "assets": ["assets/ladder.png"],
    }
    (RECEIPTS / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("wrote receipts/receipt.json")


if __name__ == "__main__":
    main()
