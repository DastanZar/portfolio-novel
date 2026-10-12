# dual-bias-fp8

Per-block **dual format + dual exponent-bias** FP8 GEMV — a novel successor to
plain software FP8 GEMV / KV-cache quantization.

## Novelty

Standard FP8 GEMV picks **one** format (E5M2 *or* E4M3) and **one** float scale
per tensor. That fails on mixed-magnitude rows: one outlier forces the whole
tensor into E5M2, throwing away mantissa bits everywhere else.

`dual-bias-fp8` instead splits the K-dimension into blocks of 32 and, per block:

1. **Chooses E5M2 vs E4M3 by a dynamic-range test** —
   `amax > 240` or `amax/rms > 12` → E5M2 (range wins),
   else E4M3 (precision wins).
2. **Calibrates an integer exponent bias** `B ∈ [11, 19]` (nominal 15) so
   `amax / 2^(15−B) ≈ 0.9 · fmt_max`. This is a power-of-two scale: dequant is
   a single exponent add, no float multiply, and costs 2 bytes/block
   (fmt bit + bias) vs 4 bytes/block for fp32 scales.
3. **Fuses dequant + GEMV** in the reference kernel: decode one block,
   partial dot, accumulate — the full fp32 weight is never materialized.
   An optional Triton path (`gemv_triton`, guarded by import + CUDA check)
   dispatches to the same reference on CPU.

Result: near-int8 bandwidth with better worst-row error on spiky data
(see `assets/dual_bias.png`).

## Layout

- `dual_bias_fp8/core.py` — `choose_format`, `calibrate_bias`,
  `quantize_block`, `dequantize_block`, `pack_weights`,
  `gemv_reference`, `gemv_triton`, `bytes_model`
- `tests/test_core.py` — 6 tests incl. block-choice correctness and
  bit-exact E5M2 check vs `torch.float8_e5m2` where available
- `demos/demo.py` — bandwidth-vs-accuracy plot → `assets/dual_bias.png`,
  `receipts/receipt.json`

## Run

```bash
pip install -e .[test]
pytest -q
python demos/demo.py
```

## Attribution

- FP8 formats: Micikevicius et al., *FP8 Formats for Deep Learning* (2022);
  OCP *8-bit Floating Point Specification* (E5M2 bias 15 / E4M3 bias 8).
- Prior art: NVIDIA TensorRT-LLM FP8, PyTorch TorchAO FP8 recipes
  (per-tensor/per-row scaling).
- The per-tensor-scale pain point (skipping KV dequant) is documented in the
  Reddit `r/LocalLLaMA` KV-quant discussion thread.
- This project is original work by Abhinav Chauhan (MIT): the per-block
  format test + integer-bias calibration + fused reference kernel above.
