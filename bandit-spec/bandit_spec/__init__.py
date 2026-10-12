"""bandit-spec: Thompson-sampling draft-length bandit with SLO guard."""

from .core import BanditSpeculator, expected_tokens, slo_guard, thompson_pick

__all__ = ["BanditSpeculator", "expected_tokens", "slo_guard", "thompson_pick"]

__version__ = "0.1.0"
