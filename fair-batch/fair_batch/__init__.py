"""fair-batch: fairness + power-aware batching simulator."""
from .sim import Request, jain_index, power_model, simulate_static, simulate_fair_continuous

__all__ = ["Request", "jain_index", "power_model", "simulate_static", "simulate_fair_continuous"]
__version__ = "0.1.0"
