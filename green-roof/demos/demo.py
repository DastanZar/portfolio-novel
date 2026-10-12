"""Demo: cost vs carbon tradeoff across GPUs + SLO feasibility -> PNG + receipt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from green_roof.gpu import GPU_TABLE
from green_roof.model import get_model
from green_roof.planner import full_plan, replicas_for_slo

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RECEIPTS = ROOT / "receipts"

MODEL = "llama-3-8b"
SEQ = 2048
BATCH = 8
QPS = 10.0
SLO_MS = 250.0


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    model = get_model(MODEL)
    rows = []
    for gname in sorted(GPU_TABLE):
        plan = full_plan(model, GPU_TABLE[gname], seq_len=SEQ, batch=BATCH)
        slo = replicas_for_slo(model, GPU_TABLE[gname], qps_target=QPS,
                               p99_budget_ms=SLO_MS, seq_len=SEQ, batch=BATCH)
        rows.append({**plan, **{f"slo_{k}": v for k, v in slo.items()}})
        print(f"{gname:10s} {plan['tok_per_s']:7.1f} tok/s  ${plan['usd_per_1m']:6.2f}/1M  "
              f"{plan['co2e_g_per_1m']:7.1f} gCO2e/1M  feas={slo['feasible']} x{slo['replicas']}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    feas = [r for r in rows if r["slo_feasible"]]
    miss = [r for r in rows if not r["slo_feasible"]]
    for group, marker, color, label in (
        (feas, "o", "green", f"SLO-feasible (p99<{SLO_MS:.0f}ms, {QPS:.0f} qps)"),
        (miss, "x", "red", "SLO-miss"),
    ):
        if not group:
            continue
        ax.scatter([r["usd_per_1m"] for r in group], [r["co2e_g_per_1m"] for r in group],
                   marker=marker, color=color, s=80, label=label)
        for r in group:
            ax.annotate(f"{r['gpu']} x{r['slo_replicas']}", (r["usd_per_1m"], r["co2e_g_per_1m"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("$/1M tokens (log)")
    ax.set_ylabel("gCO2e / 1M tokens (log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"green-roof: cost vs carbon — {MODEL} (batch={BATCH}, seq={SEQ})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    out = ASSETS / "green_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    greenest = min(feas or rows, key=lambda r: r["co2e_g_per_1m"])
    cheapest = min(feas or rows, key=lambda r: r["usd_per_1m"])
    receipt = {
        "project": "green-roof",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"model": MODEL, "seq_len": SEQ, "batch": BATCH, "qps": QPS, "slo_ms": SLO_MS},
        "rows": [
            {"gpu": r["gpu"], "tok_per_s": r["tok_per_s"], "ms_per_token": r["ms_per_token"],
             "bound": r["bound"], "usd_per_1m": r["usd_per_1m"],
             "joules_per_1m": r["joules_per_1m"], "co2e_g_per_1m": r["co2e_g_per_1m"],
             "replicas": r["slo_replicas"], "feasible": r["slo_feasible"]}
            for r in rows
        ],
        "cheapest_feasible": cheapest["gpu"],
        "greenest_feasible": greenest["gpu"],
        "assets": ["assets/green_tradeoff.png"],
    }
    (RECEIPTS / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("wrote receipts/receipt.json")


if __name__ == "__main__":
    main()
