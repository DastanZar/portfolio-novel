# green-roof — cost / carbon / SLO roofline planner

Novel successor to a plain roofline capacity planner. Classic roofline
(Williams et al. 2009) predicts latency; `green-roof` extends it to
deployment decisions:

- **$/1M tokens** from throughput ÷ cloud $/hr,
- **J/1M tokens + gCO2e/1M** from board power × PUE ÷ throughput × grid
  intensity (methodology follows the "LLM Inference Unveiled" survey's
  energy accounting),
- **Autoscale replica solver** — `replicas_for_slo` sizes replicas for a
  p99-latency budget at a target QPS and reports infeasible SLOs with a
  reason instead of silently rounding up,
- **Tenant fairness** — `fair_split` max-min water-filling across tenants,
- **CLI** — `plan | gpus | models | green` commands.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python demos/demo.py   # -> assets/green_tradeoff.png + receipts/receipt.json
green-roof plan --model llama-3-8b --gpu H100
green-roof green --model llama-3-8b --qps 10 --slo-ms 250
```

## Layout

- `green_roof/model.py` — `LLMModel` + `MODEL_ZOO` (FLOP/byte per token)
- `green_roof/gpu.py` — `GPU_TABLE` ($/hr, W, BW, TFLOPs, CO2)
- `green_roof/planner.py` — `predict`, `cost/energy/co2e_per_million`,
  `replicas_for_slo`, `fair_split`
- `green_roof/cli.py` — argparse CLI
- `tests/test_planner.py` — 7 tests (+1 table smoke test)
- `demos/demo.py` — cost-vs-carbon tradeoff plot + receipt

## GPU table sources (rounded public datasheet figures)

- NVIDIA T4 / L4 / A10 / A100-40GB / A100-80GB / H100-SXM / H200 specs
  (memory, bandwidth, BF16 TFLOPs, TDP) — NVIDIA datasheets / tech blogs.
- RTX 4090 / 5090 (memory, bandwidth, TDP) — NVIDIA GeForce spec pages.
- AMD MI300X (192GB HBM3, ~5.3 TB/s, 1307 BF16 TFLOPs, 750W) — AMD datasheet.
- `price_per_hr` = approximate 2025–26 on-demand cloud rates; grid factor
  defaults to 380 gCO2/kWh (US average, override per deployment).

## References

- Williams, Waterman & Patterson, Roofline (CACM 2009).
- "LLM Inference Unveiled" survey — decode energy/latency taxonomy.
- KernelSight-LM — kernel-aware LLM inference analysis (naming inspiration).

All code is original and CPU-only (no GPU required; the planner is analytic).

## Provenance & Distillation

Distilled from (verified to exist ):
- dengls24/LLM-para (NOASSERTION): https://github.com/dengls24/LLM-para
- Giotyp/GPU-Roofline-Python (MIT): https://github.com/Giotyp/GPU-Roofline-Python
via the earlier study repo `DastanZar/roofline-planner` (archived; do not present as original).

- Retained: roofline capacity math.
- Newly built here (fresh code in this folder, no copied files): $/J/CO2e + replicas_for_slo autosolver + fair_split + CLI.
- License duty: upstream licenses above still apply to any of their code you reuse — this folder is MIT, but keep these credits + links when you publish. Papers/threads cited in prior-art section remain credited.
