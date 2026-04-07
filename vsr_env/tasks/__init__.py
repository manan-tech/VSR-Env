"""Task implementations for VSR-Env.

This module provides the three graded tasks:
- DeltaHedgingTask (medium): Neutralize portfolio delta through market shock
- EarningsVolCrushTask (hard): Position for and recover from earnings vol crush
- GammaScalpingTask (expert): Profit from gamma scalping through spot oscillations
"""

from vsr_env.tasks.delta_hedging import DeltaHedgingTask
from vsr_env.tasks.earnings_vol_crush import EarningsVolCrushTask
from vsr_env.tasks.gamma_scalping import GammaScalpingTask
from vsr_env.tasks.vol_regime_detection import VolRegimeDetectionTask

__all__ = [
    "DeltaHedgingTask",
    "EarningsVolCrushTask",
    "GammaScalpingTask",
    "VolRegimeDetectionTask",
]
