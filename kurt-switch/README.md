# kurt-switch — kurtosis-switched 4-bit codebooks

> **Hero:** one dumb uniform 4-bit grid leaves 80% of the error on the table for heavy-tailed rows.
> kurt-switch routes each row to the grid it deserves — and proves it with a closed-form least-squares scale.

## Purpose

4-bit weight quantization for LLM-style matrices. Instead of one static grid per tensor,
kurt-switch computes **per-row statistics (excess kurtosis, outlier fraction, skew, sparsity)**
and switches among three 16-entry codebooks, then fits a **closed-form least-squares scale**
`s = (x·c)/(c·c)` that is exactly optimal for the chosen indices.

## Novelty vs prior art

| Method | Grid | Scale | Selector |
|---|---|---|---|
| Static uniform 4-bit (baseline) | 1 fixed grid | min-max | none |
| GPTQ / AWQ | per-row scaled grid | Hessian/salience-based | salience |
| Grouped lattice VQ (arXiv 2510.20984) | learned lattice groups | grouped | lattice membership |
| **kurt-switch (this repo)** | **3 switched grids** | **closed-form LS per row** | **kurtosis + outlier + skew rule** |

The contribution is the **adaptive selector**, not a new grid per se:

- `uniform16` — plain `linspace(-1, 1, 16)` baseline for Gaussian rows.
- `odd16` — **odd-symmetric square-law companded** grid `sign(u)·|u|²`, zero-excluding,
  fine near zero (≈0.004) with full ±1 range. Wins on sparse / heavy-tailed rows
  where most mass clusters near zero plus a few big outliers.
- `shifted` — asymmetric `linspace(-0.75, 1.25, 16)`, mirrored to `[-1.25, 0.75]`
  for left-skewed rows. Wins on skewed rows.

All code here is written from scratch for this project (numpy + optional torch-CPU path);
no code is copied from any other repo.

## How to run

```bash
pip install -e ".[test]"   # torch is listed but optional; numpy path is default
python -m pytest tests -v
python demos/demo.py        # writes receipts/receipt.json + assets/switch_map.png
```

CPU-only. No GPU required.

## Results (reproducible, seed 0, 200×64)

| metric | uniform baseline | kurt-switch (adaptive) |
|---|---|---|
| overall MSE | 0.0578 | **0.0116** |
| improvement | — | **79.9%** |
| rows where adaptive strictly wins | — | 33.0% |

Per kurtosis bin (adaptive win-rate vs static uniform):

| kurtosis bin | n | win-rate | adapt MSE | uniform MSE |
|---|---|---|---|---|
| kurt<0 | 81 | 0.0% (ties — correctly stays on uniform) | 0.0089 | 0.0089 |
| 0–3 | 69 | 23.2% | 0.0148 | 0.0167 |
| 3–10 | 10 | 100.0% | 0.0369 | 0.0579 |
| >10 | 40 | 100.0% | 0.0053 | 0.2279 |

Codebook mix: uniform16 ×132, odd16 ×48, shifted ×20.

![switch map](assets/switch_map.png)

## API

```python
from kurt_switch.core import compute_row_stats, select_codebook, quantize_row, quantize_block
stats = compute_row_stats(row)   # kurtosis, skew, outlier/zero fractions
name = select_codebook(row)      # 'uniform16' | 'odd16' | 'shifted'
out = quantize_row(row)          # {indices, scale, codebook, levels, reconstruction, mse}
blk = quantize_block(matrix)     # per-row adaptive over a 2-D block
```

## Attribution

- GPTQ (Frantar et al.) and AWQ (Lin et al.) — per-row / salience-aware 4-bit scaling inspiration.
- Grouped lattice vector quantization, arXiv 2510.20984 — grouped-codebook idea.
- Reddit r/LocalLLaMA 4-bit benchmark threads — practical discussion of which rows break static grids.
- All implementation is original to this repo; references are idea-level only.

## License

MIT — holder Abhinav Chauhan. See `LICENSE`.
