# cache-roof — cache-aware roofline with confidence intervals

Novel successor to a plain kernel benchmark harness. Classic roofline
(Williams et al. 2009) grades kernels against `min(peak, AI × BW_DRAM)`,
which underrates kernels with strong L2 reuse and quantized kernels whose
dequant traffic overlaps compute. `cache-roof` adds:

1. **L2-hit-rate model** — memory time is split into DRAM-miss + L2-hit
   traffic (`t_mem = B_dram/BW_dram + B_l2/BW_l2`), so the roof rises with
   reuse instead of assuming every byte pays DRAM latency.
2. **Dequant-overlap model** — a fraction `q` of memory time is hidden
   behind compute (the KV-dequant skip / async-dequant idea discussed in
   recent RTX 5090 + vLLM bench threads): `t_mem_eff = t_mem × (1 − q)`.
3. **Ladder with CIs** — `time_fn` (warmup + repeated trials, same pattern
   as Triton's `do_bench`) returns a 95% CI; grades are reported as
   **% of cache-adjusted peak with error bars**, not point estimates.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python demos/demo.py   # -> assets/ladder.png + receipts/receipt.json
```

## Layout

- `cache_roof/harness.py` — `time_fn` (mean/median/std/95% CI), `run_case`
- `cache_roof/roofline.py` — `peak_measure`, `cache_adjusted_roof`, `analyze`
- `tests/test_harness.py` — 5 tests
- `demos/demo.py` — 4 CPU kernels → ladder plot + receipt

## References

- Williams, Waterman & Patterson, "Roofline: An Insightful Visual
  Performance Model…", CACM 2009.
- Triton `do_bench` (warmup + percentile timing pattern).
- KernelBench (Scaling et al.) — systematic GPU-kernel correctness/perf harness.
- Reddit r/LocalLLaMA "5090 + vLLM" bench discussion thread (KV-cache
  quantization / dequant-overlap observations) — idea source for the
  dequant-overlap term; numbers here are CPU-measured, not copied.

All code is original and CPU-only.
