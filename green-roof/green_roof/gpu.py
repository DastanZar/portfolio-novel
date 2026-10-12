"""GPU table: public-datasheet specs + approximate cloud pricing.

Bandwidth / TDP / memory figures are rounded from vendor datasheets
(NVIDIA Ampere/Ada/Hopper specs, AMD MI300X spec, Intel/CPU-adjacent
entries omitted). ``price_per_hr`` is an approximate on-demand cloud
rate (USD/hr, 2025-2026 era) recorded so cost comparisons are
reproducible — see README for the per-GPU source notes. ``co2_g_per_kwh``
defaults to a US-average grid factor and is overridable per deployment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float          # HBM/GDDR memory bandwidth, TB/s
    tflops_bf16: float     # dense bf16 TFLOP/s (no sparsity)
    tdp_w: float           # board TDP, W
    price_per_hr: float    # USD/hr approximate cloud rate
    co2_g_per_kwh: float = 380.0  # grid carbon intensity


GPU_TABLE: dict[str, GPU] = {
    "T4": GPU("T4", mem_gb=16, bw_tbs=0.30, tflops_bf16=65, tdp_w=70, price_per_hr=0.35),
    "L4": GPU("L4", mem_gb=24, bw_tbs=0.30, tflops_bf16=121, tdp_w=72, price_per_hr=0.60),
    "A10": GPU("A10", mem_gb=24, bw_tbs=0.60, tflops_bf16=125, tdp_w=150, price_per_hr=0.75),
    "RTX4090": GPU("RTX4090", mem_gb=24, bw_tbs=1.01, tflops_bf16=330, tdp_w=450, price_per_hr=0.80),
    "RTX5090": GPU("RTX5090", mem_gb=32, bw_tbs=1.79, tflops_bf16=400, tdp_w=575, price_per_hr=1.10),
    "A100-40": GPU("A100-40", mem_gb=40, bw_tbs=1.56, tflops_bf16=312, tdp_w=400, price_per_hr=1.80),
    "A100-80": GPU("A100-80", mem_gb=80, bw_tbs=2.04, tflops_bf16=312, tdp_w=400, price_per_hr=2.20),
    "H100": GPU("H100", mem_gb=80, bw_tbs=3.35, tflops_bf16=989, tdp_w=700, price_per_hr=3.20),
    "H200": GPU("H200", mem_gb=141, bw_tbs=4.80, tflops_bf16=989, tdp_w=700, price_per_hr=3.80),
    "MI300X": GPU("MI300X", mem_gb=192, bw_tbs=5.30, tflops_bf16=1307, tdp_w=750, price_per_hr=3.00),
}


def get_gpu(name: str) -> GPU:
    try:
        return GPU_TABLE[name]
    except KeyError:
        raise KeyError(f"unknown gpu {name!r}; choose from {sorted(GPU_TABLE)}") from None


def list_gpus() -> list[GPU]:
    return [GPU_TABLE[k] for k in sorted(GPU_TABLE)]
