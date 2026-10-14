# gate-keep — learned gate + latent memory (novel KV-cache compression)

Successor to attention-scored KV compression (H2O / StreamingLLM family).
Instead of **dropping** low-scoring tokens, gate-keep **sketches** them into
a tiny latent summary that stays in the attention as one extra key.

## Novelty (state the delta)

| Prior art | What it does | gate-keep delta |
|---|---|---|
| StreamingLLM (Xiao et al. 2023) | keeps sinks + recent window, drops the middle | keeps sinks too, but compresses the dropped middle into memory |
| H2O (Zhang et al. 2023) | keeps heavy hitters by cumulative mass, drops rest | same mass scoring for ranking, but evicted tail becomes an 8-d sketch |
| Gated KV Eviction family (e.g. arXiv 2601.17668, as briefed) + latent-memory eviction + "don't discard" quantization threads | learned eviction signals | tiny **2-layer ridge probe** (closed-form, no SGD) mapping *past* attention stats → *future* importance + orthonormal sketch projector |

Concretely:
1. `score_tokens` — H2O-style mass scoring (mean attention per key).
2. `fit_gate_ridge` — fixed random ReLU expansion + closed-form ridge readout.
   Learns `past stats → future importance` on the fly, CPU-only.
3. `compress_with_memory` — keep top-`budget` (sinks forced), sketch evicted
   tail as score-weighted barycenter → random orthonormal `(d, 8)` → 8-d vector
   → project back → **one extra key/value** (`budget+1` rows).
4. `reconstruction_error` — single-query attention-output relative error.

## Quickstart

```bash
pip install -e ".[test]"
python -m pytest -q
python demos/demo.py
```

Outputs: `assets/gate_gain.png`, `receipts/receipt.json`.

## Layout

- `gate_keep/compress.py` — all four required functions + `gate_predict`, `drop_error`.
- `tests/test_compress.py` — 5 tests.
- `demos/demo.py` — synthetic benchmark + plot.

See `ATTRIBUTION.md` for citations. All code is original.

## Provenance & Distillation

Distilled from (verified to exist ):
- snu-mllab/KVzip (MIT): https://github.com/snu-mllab/KVzip
- varjoranta/turboquant-vllm (MIT): https://github.com/varjoranta/turboquant-vllm
via the earlier study repo `DastanZar/kvstamp` (archived; do not present as original).

- Retained: KV eviction + quantization problem.
- Newly built here (fresh code in this folder, no copied files): ridge-gate predictor + 8-d latent memory instead of drop (+11.8% @ b8).
- License duty: upstream licenses above still apply to any of their code you reuse — this folder is MIT, but keep these credits + links when you publish. Papers/threads cited in prior-art section remain credited.
