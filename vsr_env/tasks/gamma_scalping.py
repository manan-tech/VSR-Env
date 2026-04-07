"""Gamma Scalping Task implementation for VSR-Env.

This module implements the expert task where agents profit from gamma scalping
by re-hedging a high-gamma position through spot oscillations.

Requirements: 2.4, 6.1
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


class GammaScalpingTask:
    """Expert task: Profit from gamma scalping through spot oscillations.

    The agent starts with a long ATM straddle (high gamma position).
    The spot price oscillates significantly (±2-3% per step).
    The agent must re-hedge delta frequently to "scalp" the gamma.

    Attributes:
        max_steps: Maximum steps per episode (10)
        difficulty: Task difficulty level ("expert")
    """

    max_steps: int = 10
    difficulty: str = "expert"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the gamma scalping task with a long ATM straddle.

        Creates a long ATM straddle position (buy call + buy put at same strike).
        This gives maximum gamma exposure.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)

        Requirements: 2.4, 6.1
        """
        # Create engine for pricing and Greeks
        engine = OptionChainEngine()

        # ATM strike (index 4 = 100)
        strike_idx = 4

        # Short maturity (index 0 = 30-day) for maximum gamma
        maturity_idx = 0

        # Get strike and maturity
        K = engine.STRIKES[strike_idx]
        T = engine.MATURITIES[maturity_idx]

        # Current market conditions
        S = state.spot_price
        sigma = np.sqrt(state.variance)

        # Compute option prices and Greeks
        call_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call"
        )[0]
        put_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="put"
        )[0]

        call_delta = engine.delta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call"
        )[0]
        put_delta = engine.delta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="put"
        )[0]

        call_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        put_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

        call_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        put_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

        call_theta = engine.theta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call"
        )[0]
        put_theta = engine.theta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="put"
        )[0]

        # Quantity for straddle (1 contract each)
        quantity = 1.0

        # Create call position (long)
        call_position = {
            "strike_idx": int(strike_idx),
            "maturity_idx": int(maturity_idx),
            "direction": "buy",
            "option_type": "call",
            "quantity": float(quantity),
            "entry_price": float(call_price),
            "entry_iv": float(sigma),
            "entry_spot": float(S),
            "current_price": float(call_price),
            "pnl": 0.0,
            "delta": float(call_delta * quantity),
            "gamma": float(call_gamma * quantity),
            "vega": float(call_vega * quantity),
            "theta": float(call_theta * quantity),
        }

        # Create put position (long)
        put_position = {
            "strike_idx": int(strike_idx),
            "maturity_idx": int(maturity_idx),
            "direction": "buy",
            "option_type": "put",
            "quantity": float(quantity),
            "entry_price": float(put_price),
            "entry_iv": float(sigma),
            "entry_spot": float(S),
            "current_price": float(put_price),
            "pnl": 0.0,
            "delta": float(put_delta * quantity),  # Put delta is negative
            "gamma": float(put_gamma * quantity),
            "vega": float(put_vega * quantity),
            "theta": float(put_theta * quantity),
        }

        # Update state with straddle position
        state.positions = [call_position, put_position]

        # Portfolio Greeks (straddle: call + put)
        # Delta: call_delta + put_delta ≈ 0 for ATM straddle
        # Gamma: call_gamma + put_gamma = 2 * call_gamma (maximum gamma)
        # Vega: call_vega + put_vega = 2 * call_vega
        state.portfolio_delta = float((call_delta + put_delta) * quantity)
        state.portfolio_gamma = float((call_gamma + put_gamma) * quantity)
        state.portfolio_vega = float((call_vega + put_vega) * quantity)
        state.portfolio_pnl = 0.0

        # Store initial theta for grading (total theta decay cost)
        state.initial_theta = float((call_theta + put_theta) * quantity)

        # No mispricings for this task
        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "You are exploiting gamma convexity through a long ATM straddle position. "
            "The spot price will oscillate significantly (±2-3% per step). "
            "Your objective is to scalp gamma by re-hedging delta after large moves "
            "and holding through small moves. Profit from realized volatility exceeding "
            "implied volatility, while managing theta decay costs. You have 10 steps total."
        )


class GammaScalpingGrader:
    """Grader for Gamma Scalping task.

    Scores based on re-hedge quality, P&L above theta, and timing score.

    Requirements: 6.4, 6.5, 6.6
    """

    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Gamma Scalping task.

        Score = rehedge_quality × 0.40 + pnl_above_theta × 0.35 + timing_score × 0.25

        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_delta, portfolio_pnl, and initial_theta

        Returns:
            Score in [0.0, 1.0]

        Requirements: 6.4, 6.5, 6.6
        """
        # Compute re-hedge quality (average delta neutrality across steps)
        # Requirements: 6.4
        deltas = []
        for step in episode_history:
            obs = step.get("observation")
            if obs is not None:
                greeks = (
                    obs.portfolio_greeks if hasattr(obs, "portfolio_greeks") else {}
                )
                deltas.append(abs(greeks.get("delta", 0.0)))

        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            rehedge_quality = max(0.0, 1.0 - avg_delta / 0.5)
        else:
            rehedge_quality = 0.0

        # Compute P&L above theta decay
        # Requirements: 6.5
        final_pnl = state.portfolio_pnl
        initial_theta = getattr(state, "initial_theta", -0.05)
        total_theta_cost = abs(initial_theta) * len(episode_history)  # Approximate

        # P&L above theta: positive P&L after accounting for theta decay
        pnl_above_theta_raw = final_pnl + total_theta_cost  # Add back theta cost
        pnl_above_theta = self._sigmoid(pnl_above_theta_raw, scale=0.3)

        # Compute timing score (correlation between spot moves and hedge quantities)
        # Requirements: 6.6
        spot_moves = []
        hedge_quantities = []

        prev_spot = None
        for step in episode_history:
            obs = step.get("observation")
            action = step.get("action")

            if obs is not None and prev_spot is not None:
                spot_move = abs(obs.spot_price - prev_spot)
                spot_moves.append(spot_move)

                if action is not None:
                    action_direction = (
                        action.direction.value
                        if hasattr(action.direction, "value")
                        else action.direction
                    )
                    if action_direction != "hold":
                        hedge_quantities.append(abs(action.quantity))
                    else:
                        hedge_quantities.append(0.0)

            if obs is not None:
                prev_spot = obs.spot_price

        # Timing score: correlation between spot moves and hedge quantities
        if len(spot_moves) > 2 and len(hedge_quantities) > 2:
            # Ensure same length
            min_len = min(len(spot_moves), len(hedge_quantities))
            spot_moves = spot_moves[:min_len]
            hedge_quantities = hedge_quantities[:min_len]

            # Compute correlation
            if np.std(spot_moves) > 1e-6 and np.std(hedge_quantities) > 1e-6:
                correlation = np.corrcoef(spot_moves, hedge_quantities)[0, 1]
                timing_score = max(
                    0.0, (correlation + 1.0) / 2.0
                )  # Normalize to [0, 1]
            else:
                timing_score = 0.5
        else:
            timing_score = 0.5

        # Final score: weighted combination
        score = rehedge_quality * 0.40 + pnl_above_theta * 0.35 + timing_score * 0.25

        # Clamp to [0.0, 1.0]
        return min(max(score, 0.0), 1.0)

    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        """Sigmoid function centered at 0."""
        import math

        return 1.0 / (1.0 + math.exp(-x / scale))
