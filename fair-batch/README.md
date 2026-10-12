# fair-batch — fairness + power-aware continuous-batching simulator

Novel successor to a plain continuous-batching simulator. All code is fresh and
original for this portfolio project.

**Novelty (what's new here):**
1. **Deficit-round-robin (DRR) across users** — fair queueing instead of FIFO, so a
   flooding tenant can't starve light users.
2. **DVFS-style power cap** — the decode clock scales down when the running batch
   exceeds a threshold (`power_model`), with cubic `P ~ f^3` power law, per-iteration
   latency inflation, and J/token accounting.
3. **KV-memory pressure admission control** — a request is admitted only if
   `used_KV + prompt <= kv_capacity` and batch slots are free; reported `kv_peak`
   proves no oversubscription.
4. **Metrics**: Jain fairness index (over per-user goodput), p99 latency, J/token,
   makespan, per-user mean latency.

## Quickstart

```bash
pip install -e .
pytest
python demos/demo.py
```

Outputs: `receipts/receipt.json`, `assets/fairness.png` (TTFT scatter, static vs fair).

## API

```python
from fair_batch.sim import Request, simulate_static, simulate_fair_continuous, power_model, jain_index
```

- `power_model(batch_size, threshold=8, ...)` → `{clock, iter_ms, power_w, energy_j, j_per_token}`
- `jain_index(values)` → float in [0, 1]
- `simulate_static(requests, batch_size=4, ...)` → metrics dict
- `simulate_fair_continuous(requests, max_batch=4, kv_capacity_tokens=512, power_threshold=8, quantum=1, ...)` → metrics dict

## Attribution / related work

- Orca (OSDI '22, Yu et al.) — iteration-level (continuous) batching for LLMs.
- NanoFlow — nano-batching execution for dense GPU use.
- Fairness-Aware Batch Formation (arXiv 2510.14392) — fairness-aware batching formulation.
- KernelSight-LM — LLM inference simulator methodology (prefill/decode split, KV tracking).

This repo re-implements the *ideas* above from scratch; no code is copied.

## License

MIT — see `LICENSE`.
