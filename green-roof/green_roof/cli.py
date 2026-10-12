"""CLI: plan | gpus | models | green."""
from __future__ import annotations

import argparse
import json
import sys

from .gpu import GPU_TABLE
from .model import MODEL_ZOO, get_model
from .gpu import get_gpu
from .planner import full_plan, replicas_for_slo


def _plan(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    gpu = get_gpu(args.gpu)
    plan = full_plan(model, gpu, seq_len=args.seq_len, batch=args.batch)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"{model.name} on {gpu.name}  (batch={args.batch}, seq={args.seq_len})")
        print(f"  {plan['tok_per_s']:.1f} tok/s  {plan['ms_per_token']:.2f} ms/tok  [{plan['bound']}-bound]")
        print(f"  ${plan['usd_per_1m']:.2f}/1M-tok  {plan['joules_per_1m']/1e6:.2f} MJ/1M  {plan['co2e_g_per_1m']:.1f} gCO2e/1M")
    return 0


def _gpus(_args: argparse.Namespace) -> int:
    print(f"{'GPU':10s} {'mem':>6s} {'BW':>7s} {'BF16':>8s} {'TDP':>5s} {'$/hr':>6s}")
    for name in sorted(GPU_TABLE):
        g = GPU_TABLE[name]
        print(f"{g.name:10s} {g.mem_gb:5.0f}GB {g.bw_tbs:5.2f}TB/s {g.tflops_bf16:7.0f}TF {g.tdp_w:4.0f}W ${g.price_per_hr:5.2f}")
    return 0


def _models(_args: argparse.Namespace) -> int:
    for name in sorted(MODEL_ZOO):
        m = MODEL_ZOO[name]
        print(f"{m.name:12s} {m.params_b:5.1f}B  {m.quant_bits:.0f}-bit  kv={m.kv_bytes_per_token:.0f} B/tok")
    return 0


def _green(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    rows = []
    for gname in sorted(GPU_TABLE):
        gpu = GPU_TABLE[gname]
        plan = full_plan(model, gpu, seq_len=args.seq_len, batch=args.batch)
        slo = replicas_for_slo(model, gpu, qps_target=args.qps,
                               p99_budget_ms=args.slo_ms, seq_len=args.seq_len,
                               batch=args.batch)
        rows.append((plan, slo))
    feas = [r for r in rows if r[1]["feasible"]]
    pool = feas if feas else rows
    by_cost = min(pool, key=lambda r: r[0]["usd_per_1m"])
    by_co2 = min(pool, key=lambda r: r[0]["co2e_g_per_1m"])
    if args.json:
        print(json.dumps({"feasible_count": len(feas),
                          "cheapest": by_cost[0], "greenest": by_co2[0]}, indent=2))
    else:
        print(f"model={model.name} qps={args.qps} p99-budget={args.slo_ms}ms  feasible={len(feas)}/{len(rows)}")
        print(f"  cheapest: {by_cost[0]['gpu']} ${by_cost[0]['usd_per_1m']:.2f}/1M ({by_cost[1]['replicas']} replicas)")
        print(f"  greenest: {by_co2[0]['gpu']} {by_co2[0]['co2e_g_per_1m']:.1f} gCO2e/1M ({by_co2[1]['replicas']} replicas)")
        for plan, slo in sorted(rows, key=lambda r: r[0]["usd_per_1m"]):
            mark = "OK " if slo["feasible"] else "MISS"
            print(f"  [{mark}] {plan['gpu']:10s} ${plan['usd_per_1m']:7.2f}/1M  "
                  f"{plan['co2e_g_per_1m']:8.1f} gCO2e/1M  {plan['tok_per_s']:7.1f} tok/s  x{slo['replicas']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="green-roof", description="Cost/carbon/SLO roofline planner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="predict latency/cost/carbon for one model+gpu")
    pp.add_argument("--model", default="llama-3-8b", choices=sorted(MODEL_ZOO))
    pp.add_argument("--gpu", default="A100-80", choices=sorted(GPU_TABLE))
    pp.add_argument("--seq-len", type=int, default=2048)
    pp.add_argument("--batch", type=int, default=8)
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=_plan)

    pg = sub.add_parser("gpus", help="list GPU table")
    pg.set_defaults(func=_gpus)

    pm = sub.add_parser("models", help="list model zoo")
    pm.set_defaults(func=_models)

    pz = sub.add_parser("green", help="compare all GPUs for cost/carbon under an SLO")
    pz.add_argument("--model", default="llama-3-8b", choices=sorted(MODEL_ZOO))
    pz.add_argument("--seq-len", type=int, default=2048)
    pz.add_argument("--batch", type=int, default=8)
    pz.add_argument("--qps", type=float, default=10.0)
    pz.add_argument("--slo-ms", type=float, default=100.0)
    pz.add_argument("--json", action="store_true")
    pz.set_defaults(func=_green)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
