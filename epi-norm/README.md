# epi-norm — RMSNorm + residual + SwiGLU epilogue fusion

Successor to fused RMSNorm-residual kernels: fuse the **whole epilogue**
(RMSNorm + residual-add + SwiGLU gate) into one block and ship a
**recompute-vs-store tradeoff explorer** with a guarded Triton forward.

## Novelty (state the delta)

| Prior art | What it does | epi-norm delta |
|---|---|---|
| RMSNorm (Zhang & Sennrich 2019) | `x / rms(x) * w` | same math, reference returns `(y, rrms)` so backward can recompute |
| Liger-Kernel / fused RMSNorm, Tri-RMSNorm | fuse norm (sometimes +SwiGLU) in Triton | fuse **residual-add + norm + SwiGLU gate** as one documented epilogue equation |
| GEMM-epilogue fusion (briefed as arXiv 2605.19269) | attach elementwise ops to GEMM | epilogue framing + analytic **HBM accountant** over 3 designs |

Three designs in `hbm_accountant`:
- **A baseline** — 3 passes, stash `fused+normed+g+u`.
- **B fused-store** — 1 pass, still stash `normed+g+u`.
- **C fused-recompute (recommended)** — 1 pass, stash only `fused+rrms`,
  recompute `normed/g/u` in backward (`fused_epilogue_backward_recompute`).

`epi_norm/kernels.py` attempts a real Triton RMSNorm forward when
`torch+triton+CUDA` exist, else falls back to torch/numpy reference —
importing it never crashes on CPU-only machines.

## Quickstart

```bash
pip install -e ".[test]"
python -m pytest -q
python demos/demo.py
```

Outputs: `assets/traffic_sankey.png`, `receipts/receipt.json`.

## Layout

- `epi_norm/core.py` — `rmsnorm_ref`, `fused_epilogue_ref`, `hbm_accountant`, `pick_block` + both backward variants.
- `epi_norm/kernels.py` — guarded Triton attempt.
- `tests/test_core.py` — 5 tests (numpy parity; torch parity when installed).
- `demos/demo.py` — traffic bar/Sankey-style chart.

All code is original; citations are ideas-only (no code reuse).
