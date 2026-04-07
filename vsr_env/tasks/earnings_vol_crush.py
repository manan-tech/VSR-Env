"""Earnings Vol Crush Task implementation for VSR-Env.

This module implements the hard task where agents position for and recover
from an earnings volatility crush event.

Requirements: 2.3, 5.1, 5.2
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.models import VSRState


class EarningsVolCrushTask:
    """Hard task: Position for and recover from an earnings vol crush event.

    The surface starts with elevated IV (pre-earnings state). At a random
    step between 3-6, an "earnings event" fires: IV drops 30-50% instantly.
    The agent must position correctly before the event and re-hedge after.

    Attributes:
        max_steps: Maximum steps per episode (8)
        difficulty: Task difficulty level ("hard")
    """

    max_steps: int = 8
    difficulty: str = "hard"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the earnings vol crush task with elevated IV.

        Sets elevated IV (base_vol × 1.3-1.5) and schedules a vol crush
        event at a random step between 3-6.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)

        Requirements: 2.3, 5.1, 5.2
        """
        # Elevate initial variance (pre-earnings state)
        # Multiply base variance by 1.3-1.5 to get elevated IV
        elevation_factor = rng.uniform(1.3, 1.5)
        state.variance *= elevation_factor
        state.variance = np.clip(state.variance, 0.01, 0.16)

        # Set vol crush step (3-6)
        # Requirements: 5.2
        state.vol_crush_step = int(rng.randint(3, 7))

        # Initialize empty portfolio
        state.positions = []
        state.portfolio_delta = 0.0
        state.portfolio_gamma = 0.0
        state.portfolio_vega = 0.0
        state.portfolio_pnl = 0.0

        # Track pre-crush vega for grading
        state.pre_crush_vega = 0.0

        # No mispricings for this task
        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "You are positioning for an earnings volatility event. "
            "The market has elevated implied volatility (pre-earnings state). "
            "An earnings announcement will occur during the episode, causing "
            "a significant volatility crush (30-50% IV drop). Position your "
            "portfolio to profit from the vol crush (short vega before the event) "
            "and re-hedge delta after the event. You have 8 steps total."
        )


class EarningsVolCrushGrader:
    """Grader for Earnings Vol Crush task.

    Scores based on pre-crush positioning, post-crush re-hedging, and P&L outcome.

    Requirements: 5.4, 5.5, 5.6
    """

    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Earnings Vol Crush task.

        Score = pre_crush_positioning × 0.40 + post_crush_rehedge × 0.35 + pnl_outcome × 0.25

        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_vega, vol_crush_step, and portfolio_pnl

        Returns:
            Score in [0.0, 1.0]

        Requirements: 5.4, 5.5, 5.6
        """
        # Get vol crush step
        vol_crush_step = getattr(state, "vol_crush_step", 4)

        # Compute pre-crush positioning (was vega negative before crush?)
        # Requirements: 5.4
        pre_crush_vega = 0.0
        for i, step in enumerate(episode_history):
            # episode_history[i] = step (i+1). We want the step before crush.
            if i + 1 == vol_crush_step - 1:  # Step immediately before crush
                obs = step.get("observation")
                if obs is not None:
                    greeks = (
                        obs.portfolio_greeks if hasattr(obs, "portfolio_greeks") else {}
                    )
                    pre_crush_vega = greeks.get("vega", 0.0)
                break

        # Pre-crush positioning score: gradient based on how short vega is.
        # Actively short vega (< -0.01) gets credit; zero/positive does not.
        if pre_crush_vega < -0.01:
            pre_crush_positioning = min(1.0, abs(pre_crush_vega) / 0.1)
        else:
            pre_crush_positioning = 0.0

        # Compute post-crush re-hedging (delta neutrality after crush)
        # Requirements: 5.5
        post_crush_deltas = []
        for i, step in enumerate(episode_history):
            if i + 1 > vol_crush_step:  # Steps after crush
                obs = step.get("observation")
                if obs is not None:
                    greeks = (
                        obs.portfolio_greeks if hasattr(obs, "portfolio_greeks") else {}
                    )
                    post_crush_deltas.append(abs(greeks.get("delta", 0.0)))

        # Post-crush re-hedge score: average delta neutrality after crush
        if post_crush_deltas:
            avg_post_crush_delta = sum(post_crush_deltas) / len(post_crush_deltas)
            post_crush_rehedge = max(0.0, 1.0 - avg_post_crush_delta / 0.5)
        else:
            post_crush_rehedge = 0.0

        # P&L outcome (sigmoid-normalized)
        # Requirements: 5.6
        final_pnl = state.portfolio_pnl
        pnl_outcome = self._sigmoid(final_pnl, scale=0.3)

        # Final score: weighted combination
        score = (
            pre_crush_positioning * 0.40
            + post_crush_rehedge * 0.35
            + pnl_outcome * 0.25
        )

        # Clamp to [0.0, 1.0]
        return min(max(score, 0.0), 1.0)

    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        """Sigmoid function centered at 0."""
        import math

        return 1.0 / (1.0 + math.exp(-x / scale))
