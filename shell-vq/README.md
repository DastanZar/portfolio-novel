# shell-vq — norm-shelled grouped vector quantization

> **Hero:** flat lattice VQ gives every vector the same scale. Real weight matrices don't —
> outer-shell outliers need their own scale plus one more bit. shell-vq learns all three.

## Purpose

Vector quantization for weight blocks viewed as `(N, dim)` row-vectors. shell-vq:

1. Groups vectors into **inner / mid / outer radius shells** via L2-norm quantiles.
2. Learns a **separate closed-form LS scale per shell** (`s = Σv·c / Σc·c`).
3. Applies **1-bit residual sign refinement** on the outer shell
   (per-vector magnitude + per-coordinate sign, per-vector gated so it never hurts).

## Novelty vs prior art

| Method | Grouping | Scale | Refinement |
|---|---|---|---|
| Flat lattice VQ (D8 / D4 / Z2 brute force) | none | single global | none |
| AQLM / QTIP | learned lattice / trellis codes | joint | trellis |
| Leech-lattice VQ (arXiv 2603.11021) | Leech lattice cells | shared | — |
| BTC-LLM (sub-1-bit) | binary | per-row | 1-bit residual |
| **shell-vq (this repo)** | **norm-quantile shells** | **per-shell LS** | **gated outer-shell sign bit** |

The delta: **radius shells instead of flat cells**. Small-norm and large-norm vectors
stop sharing one compromise scale; the outer shell — where almost all the MSE lives —
gets its own scale plus one refinement bit. All code is original to this repo
(numpy + optional torch-CPU path); references are idea-level only.

## How to run

```bash
pip install -e ".[test]"   # torch listed but optional; numpy path is default
python -m pytest tests -v
python demos/demo.py        # writes receipts/receipt.json + assets/shells.png
```

CPU-only. No GPU required.

## Results (reproducible, seed 7, 500×8, 64-point unit-sphere codebook)

| metric | value |
|---|---|
| flat single-scale MSE | 1.9777 |
| shelled MSE (per-shell scales) | 1.5852 |
| shelled + outer refine MSE | **0.7019** |
| improvement vs flat | **64.5%** |
| outer-shell MSE before → after | 3.9259 → **1.2813** |
| Gaussian RD floor (≈6.67 bit/vec) | 1.0346 (theory reference) |

Shells (norm quantiles): t1=2.69, t2=3.83; counts 167/166/167; scales 1.64 / 2.36 / 5.70.

| shell | n | MSE (shelled) |
|---|---|---|
| inner | 167 | 0.2883 |
| mid | 166 | 0.5352 |
| outer | 167 | 3.9259 → 1.2813 after refine |

![shells](assets/shells.png)

## API

```python
from shell_vq.core import build_shell_codebook, quantize_shelled, residual_refine, rate_distortion_floor
C = build_shell_codebook(dim=8, n_points=64, seed=0)
out = quantize_shelled(V, C)                    # {indices, shells, thresholds, scales, reconstruction, mse, ...}
ref = residual_refine(V, out["reconstruction"], out["shells"] == 2)
floor = rate_distortion_floor(var=3.28, dim=8, bits_per_vec=6.67)
```

## Attribution

- AQLM / QTIP — lattice and trellis VQ for LLM weights (grouped-code inspiration).
- Conway & Sloane lattices; Leech-lattice VQ, arXiv 2603.11021 — spherical-code inspiration.
- BTC-LLM sub-1-bit line — 1-bit residual refinement inspiration.
- All implementation is original; citations are idea-level only.

## License

MIT — holder Abhinav Chauhan. See `LICENSE`.
