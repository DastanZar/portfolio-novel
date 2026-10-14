# bandit-spec

**Thompson-sampling bandit over speculative-decoding draft lengths + SLO guard**
— a novel successor to fixed-K / fixed-analytic-argmax adaptive drafting.

## Novelty

Classical adaptive drafting (e.g. SpecDec++) estimates one global acceptance
rate α and sets `K* = argmax E[tokens|α,K] / cost(K)` analytically. That is
optimal only if acceptance is identical for every K and stationary.

`bandit-spec` instead treats each `K ∈ {1..7}` as an arm with its **own**
`Beta(a_K, b_K)` posterior over per-token acceptance:

- each step samples `θ_K ~ Beta(a_K, b_K)` and picks
  `argmax E[tokens|θ_K,K] / (1 + c·K)` (Thompson sampling — explores
  efficiently, exploits as posteriors sharpen);
- `observe(K, accepted, proposed, latency)` updates the Beta counts;
- a **p99 latency SLO guard** vetoes/shrinks K when tail latency exceeds
  budget (SLO-aware, unlike pure-throughput drafts).

This learns K-dependent, drifting acceptance where a single-α formula cannot.

## Layout

- `bandit_spec/core.py` — `expected_tokens`, `thompson_pick`,
  `BanditSpeculator` (`pick`/`observe`/`update`), `slo_guard`
- `tests/test_core.py` — 6 tests (formula, cost sensitivity, posterior math,
  SLO veto, learning, loop bounds)
- `demos/demo.py` — race chart + regret curve → `assets/bandit_race.png`,
  `receipts/receipt.json`

## Run

```bash
pip install -e .[test]
pytest -q
python demos/demo.py
```

## Attribution

- Speculative decoding: Leviathan et al., *Fast Inference from Transformers
  via Speculative Decoding* (2023); Chen et al., *Accelerating Large Language
  Model Decoding with Speculative Sampling* (2023).
- Adaptive drafting: SpecDec++ (arXiv 2405.19715); AdaSpec (SLO-aware
  speculative serving); *Not-a-Bandit* no-regret drafting analysis.
- This project is original work by Abhinav Chauhan (MIT): the per-K
  Beta–Thompson rule with cost-normalized reward plus the p99 SLO veto above.

## Provenance & Distillation

Distilled from (verified to exist ):
- lightseekorg/TorchSpec (MIT): https://github.com/lightseekorg/TorchSpec
via the earlier study repo `DastanZar/sdecode` (archived; do not present as original).

- Retained: draft-verify speculative decoding.
- Newly built here (fresh code in this folder, no copied files): Thompson bandit over K + p99 SLO veto (regret 13.7, tracks oracle).
- License duty: upstream licenses above still apply to any of their code you reuse — this folder is MIT, but keep these credits + links when you publish. Papers/threads cited in prior-art section remain credited.
