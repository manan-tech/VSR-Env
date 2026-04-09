"""Straddle Trading Task implementation for VSR-Env.

This module implements a hard task where agents must decide whether to
buy or sell a straddle based on volatility analysis.

Requirements: Multi-leg strategy support
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


class StraddleTradingTask:
    """Hard task: Volatility speculation using straddles.

    The agent must analyze IV levels and decide whether to go long vol
    (buy straddle) or short vol (sell straddle). The market then realizes
    volatility that may differ from implied.

    Attributes:
        max_steps: Maximum steps per episode (8)
        difficulty: Task difficulty level ("hard")
    """

    max_steps: int = 8
    difficulty: str = "hard"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the straddle trading task.

        Sets up elevated IV with hints about realized vol forecast.
        The agent should construct an appropriate straddle position.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)
        """
        # Create engine for pricing
        engine = OptionChainEngine()

        # Set elevated IV (1.3-1.5x normal)
        base_var = 0.04  # 20% vol
        vol_mult = rng.uniform(1.3, 1.5)
        state.variance = base_var * vol_mult**2  # Variance scales with vol^2

        # Determine realized vol outcome (stored in expected_outcome)
        # Realized vol can be lower (crush), similar, or higher than implied
        realized_type = rng.choice(["crush", "stable", "spike"], p=[0.4, 0.3, 0.3])

        if realized_type == "crush":
            realized_mult = rng.uniform(0.6, 0.85)  # 60-85% of implied
            state.expected_outcome = "short_vol"  # Sell straddle
        elif realized_type == "stable":
            realized_mult = rng.uniform(0.95, 1.05)
            state.expected_outcome = "neutral"  # Limited position
        else:  # spike
            realized_mult = rng.uniform(1.2, 1.5)  # 120-150% of implied
            state.expected_outcome = "long_vol"  # Buy straddle

        # Store realized vol forecast for grading
        state.straddle_realized_mult = realized_mult

        # No initial positions - agent constructs straddle
        state.positions = []
        state.portfolio_delta = 0.0
        state.portfolio_gamma = 0.0
        state.portfolio_vega = 0.0
        state.portfolio_pnl = 0.0

        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent
        """
        return (
            "Analyze the current elevated implied volatility and construct a straddle position. "
            "You may buy a straddle (long volatility) if you expect realized vol to exceed implied, "
            "or sell a straddle (short volatility) if you expect realized vol to be lower. "
            "Use the strategy_type='straddle' action for atomic execution. "
            "Consider theta decay costs and spot price uncertainty. You have 8 steps total."
        )


class StraddleTradingGrader:
    """Grader for Straddle Trading task.

    Scores based on:
    - Direction correctness (long/short vol decision)
    - Entry timing
    - P&L realized
    - Risk management

    Requirements: Multi-leg strategy evaluation
    """

    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Straddle Trading task.

        Score = direction_correctness × 0.30
              + entry_timing × 0.20
              + pnl_realized × 0.30
              + risk_management × 0.20

        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_pnl

        Returns:
            Score in [0.01, 0.99]
        """
        # Check if agent constructed a straddle
        has_straddle = False
        direction_correctness = 0.0
        entry_step = None
        exit_step = None
        max_pnl = 0.0

        expected_outcome = getattr(state, "expected_outcome", "neutral")

        for i, step in enumerate(episode_history):
            action = step.get("action")
            if action is None:
                continue

            # Check for straddle action
            if hasattr(action, "strategy_type") and action.strategy_type is not None:
                from vsr_env.models import StrategyType

                if action.strategy_type == StrategyType.STRADDLE:
                    has_straddle = True
                    if entry_step is None:
                        entry_step = i

                    # Check direction correctness
                    # Long straddle = buy both legs
                    if action.legs and len(action.legs) >= 2:
                        is_long = action.legs[0].direction == "buy"
                        if is_long and expected_outcome == "long_vol":
                            direction_correctness = 1.0
                        elif not is_long and expected_outcome == "short_vol":
                            direction_correctness = 1.0
                        elif expected_outcome == "neutral":
                            direction_correctness = 0.5

            # Track P&L
            obs = step.get("observation")
            if obs is not None:
                current_pnl = obs.portfolio_pnl if hasattr(obs, "portfolio_pnl") else 0.0
                max_pnl = max(max_pnl, current_pnl)

        # Bonus for using atomic straddle action
        if not has_straddle:
            # Check if agent built straddle manually with single legs
            has_manual_straddle = self._check_manual_straddle(episode_history)
            if has_manual_straddle:
                direction_correctness *= 0.8  # Penalty for non-atomic execution

        # Entry timing score (earlier is better for this task)
        if entry_step is not None:
            entry_timing = max(0.0, 1.0 - entry_step / 5.0)
        else:
            entry_timing = 0.0

        # P&L realized score
        final_pnl = state.portfolio_pnl
        pnl_score = self._sigmoid(final_pnl, scale=0.5)

        # Risk management (based on position sizing and delta management)
        avg_delta = self._compute_avg_delta(episode_history)
        risk_score = max(0.0, 1.0 - abs(avg_delta) / 0.3)

        # Final score: weighted combination
        score = (
            direction_correctness * 0.30
            + entry_timing * 0.20
            + pnl_score * 0.30
            + risk_score * 0.20
        )

        # Clamp to [0.01, 0.99]
        return min(max(score, 0.01), 0.99)

    def _check_manual_straddle(self, episode_history: List[Any]) -> bool:
        """Check if agent built a straddle manually with single-leg actions."""
        call_position = False
        put_position = False

        for step in episode_history:
            action = step.get("action")
            if action is None or not hasattr(action, "option_type"):
                continue

            if hasattr(action, "quantity") and action.quantity > 0:
                if action.option_type == "call":
                    call_position = True
                elif action.option_type == "put":
                    put_position = True

        return call_position and put_position

    def _compute_avg_delta(self, episode_history: List[Any]) -> float:
        """Compute average absolute delta across episode."""
        deltas = []
        for step in episode_history:
            obs = step.get("observation")
            if obs is not None and hasattr(obs, "portfolio_greeks"):
                delta = abs(obs.portfolio_greeks.get("delta", 0.0))
                deltas.append(delta)

        return sum(deltas) / len(deltas) if deltas else 0.0

    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        """Sigmoid function centered at 0."""
        import math

        return 1.0 / (1.0 + math.exp(-x / scale))