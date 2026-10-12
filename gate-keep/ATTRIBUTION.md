# Attribution

All implementation code in this repo is original and written for this project.

Prior / inspirational work (ideas only, no code reuse):

- StreamingLLM (Xiao et al., 2023) — attention sinks: first tokens must be
  preserved. We force-keep `sink_keep` leading tokens.
- H2O (Zhang et al., 2023) — Heavy-Hitter Oracle: rank tokens by accumulated
  attention mass. Our `score_tokens` uses the same mass principle.
- Gated KV Eviction line (briefed as arXiv 2601.17668) — learning a small
  gating signal for eviction. Our gate is a 2-layer ridge probe instead.
- Learned eviction with latent memory (general family) — compress rather than
  drop. Our 8-d orthonormal sketch + append-as-extra-key is our own variant.
- KV-quantization community threads (e.g. "turboquant-KV" style discussions
  about not discarding the tail) — motivational intuition only.

No weights, checkpoints, or third-party code are bundled.
Fictional/future-dated arXiv IDs from the brief are cited as inspiration
labels only; the delta above is implemented from scratch.
