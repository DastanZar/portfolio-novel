"""LLM model specs for roofline planning (original estimates)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModel:
    """Minimal decode-planning spec for one LLM."""
    name: str
    params_b: float          # billions of parameters
    quant_bits: float = 16.0  # weight precision
    kv_bytes_per_token: float = 0.0  # KV-cache bytes per token (per sequence)
    hidden: int = 4096

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * (self.quant_bits / 8.0)

    def flops_per_token(self) -> float:
        """~2 FLOP per parameter per decode token (standard estimate)."""
        return 2.0 * self.params_b * 1e9

    def bytes_per_token(self, seq_len: int = 2048) -> float:
        """Weight bytes (streamed once per token) + KV-cache read per token."""
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        return self.weight_bytes + self.kv_bytes_per_token * seq_len


# KV bytes/token rows are back-of-envelope for GQA configs at bf16
# (2 tensors x layers x kv_dim x 2 bytes); documented as estimates.
MODEL_ZOO: dict[str, LLMModel] = {
    "phi-3-mini": LLMModel("phi-3-mini", params_b=3.8, hidden=3072, kv_bytes_per_token=192.0),
    "mistral-7b": LLMModel("mistral-7b", params_b=7.3, hidden=4096, kv_bytes_per_token=320.0),
    "llama-3-8b": LLMModel("llama-3-8b", params_b=8.0, hidden=4096, kv_bytes_per_token=512.0),
    "qwen2.5-32b": LLMModel("qwen2.5-32b", params_b=32.5, hidden=5120, kv_bytes_per_token=640.0),
    "llama-3-70b": LLMModel("llama-3-70b", params_b=70.6, hidden=8192, kv_bytes_per_token=1280.0),
}


def get_model(name: str) -> LLMModel:
    try:
        return MODEL_ZOO[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_ZOO)}") from None
