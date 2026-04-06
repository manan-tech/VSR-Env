"""Task implementations for VSR-Env.

This module provides the three graded tasks:
- IVReadingTask (easy): Identify mispriced options on IV surface
- DeltaHedgingTask (medium): Neutralize portfolio delta cost-efficiently
- ArbCaptureTask (hard): Execute arbitrage workflow with regime shifts
"""

from vsr_env.tasks.arb_capture import ArbCaptureTask
from vsr_env.tasks.delta_hedging import DeltaHedgingTask
from vsr_env.tasks.iv_reading import IVReadingTask

__all__ = [
    "IVReadingTask",
    "DeltaHedgingTask",
    "ArbCaptureTask",
]