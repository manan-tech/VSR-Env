"""Delta Hedging Task implementation for VSR-Env.

This module implements the medium task where agents neutralize portfolio delta
through a market shock, demonstrating event-driven risk management.

Requirements: 2.2, 4.1, 4.2
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


class DeltaHedgingTask:
    """Medium task: Neutralize portfolio delta through a market shock.

    The agent starts with a portfolio that has non-zero delta (0.2-0.8).
    At a random step (2 or 3), a market shock occurs changing spot price
    and volatility. The agent must maintain neutrality through the disruption.

    Attributes:
        max_steps: Maximum steps per episode (5)
        difficulty: Task difficulty level ("medium")
    """

    max_steps: int = 5
    difficulty: str = "medium"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the delta hedging task with a non-zero delta portfolio.

        Creates an initial position with delta between 0.2 and 0.8.
        Sets a random regime shift step (2 or 3) for the market shock.
        Stores the initial delta in state for grading.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)

        Requirements: 2.2, 4.1, 4.2
        """
        # Create engine for pricing and Greeks
        engine = OptionChainEngine()

        # Select a near-ATM strike (indices 2-5 are 95, 97.5, 100, 102.5)
        strike_idx = rng.randint(2, 6)

        # Use 3-month maturity (index 1)
        maturity_idx = 1

        # Random direction
        direction = rng.choice(["buy", "sell"])

        # Get strike and maturity
        K = engine.STRIKES[strike_idx]
        T = engine.MATURITIES[maturity_idx]

        # Current market conditions
        S = state.spot_price
        sigma = np.sqrt(state.variance)

        # Compute per-contract delta
        pos_delta = engine.delta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call"
        )[0]
        pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

        # Target portfolio delta between 0.2 and 0.8 per Requirement 4.1
        target_delta = rng.uniform(0.2, 0.8)

        # Calculate quantity needed to achieve target delta
        quantity = target_delta / abs(pos_delta)

        # Compute entry price
        entry_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call"
        )[0]

        # Adjust sign based on direction
        quantity_signed = quantity if direction == "buy" else -quantity

        # Compute actual portfolio delta
        portfolio_delta = pos_delta * quantity_signed

        # Create initial position
        position = {
            "strike_idx": int(strike_idx),
            "maturity_idx": int(maturity_idx),
            "direction": direction,
            "quantity": float(quantity),
            "entry_price": float(entry_price),
            "entry_iv": float(sigma),
            "entry_spot": float(S),
            "current_price": float(entry_price),
            "pnl": 0.0,
            "delta": float(portfolio_delta),
            "gamma": float(pos_gamma * quantity_signed),
            "vega": float(pos_vega * quantity_signed),
        }

        # Update state
        state.positions = [position]
        state.portfolio_delta = float(portfolio_delta)
        state.portfolio_gamma = float(pos_gamma * quantity_signed)
        state.portfolio_vega = float(pos_vega * quantity_signed)

        # Store initial delta for grading (absolute value)
        state.initial_delta = abs(portfolio_delta)

        # Set regime shift step (2 or 3) for market shock
        # Requirements: 4.2
        state.regime_shift_step = int(rng.randint(2, 4))

        # No mispricings for delta hedging task
        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "You are managing an options portfolio through market disruption. "
            "Your objective is to maintain delta neutrality (within ±0.05) through "
            "a market shock event. A random shock will occur during the episode, "
            "changing spot price and volatility. You must hedge before and after "
            "the shock while minimizing transaction costs. You have 5 steps total."
        )


class DeltaHedgingGrader:
    """Grader for Delta Hedging task.

    Scores based on pre-shock and post-shock delta neutralization quality
    and cost efficiency.

    Requirements: 4.5, 4.6
    """

    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Delta Hedging task.

        Score = pre_shock_neutrality × 0.30 + post_shock_neutrality × 0.40 + cost_efficiency × 0.30

        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_delta, initial_delta, and regime_shift_step

        Returns:
            Score in [0.0, 1.0]

        Requirements: 4.5, 4.6
        """
        # Get initial delta (stored during task initialization)
        initial_delta = getattr(state, "initial_delta", 0.5)
        if initial_delta < 1e-6:
            initial_delta = 0.5

        # Get regime shift step
        regime_shift_step = getattr(state, "regime_shift_step", 3)

        # Compute pre-shock neutrality (delta before the shock)
        pre_shock_delta = initial_delta  # Default to initial if no steps before shock
        for i, step in enumerate(episode_history):
            if i + 1 < regime_shift_step:  # Steps are 1-indexed
                obs = step.get("observation")
                if obs is not None:
                    greeks = (
                        obs.portfolio_greeks if hasattr(obs, "portfolio_greeks") else {}
                    )
                    pre_shock_delta = abs(greeks.get("delta", pre_shock_delta))

        # Compute post-shock neutrality (final delta)
        final_delta = abs(state.portfolio_delta)

        # Pre-shock neutrality score (0.0 - 1.0)
        if initial_delta < 1e-6:
            pre_shock_neutrality = 1.0 if pre_shock_delta < 0.05 else 0.0
        else:
            pre_shock_neutrality = max(0.0, 1.0 - pre_shock_delta / initial_delta)

        # Post-shock neutrality score (0.0 - 1.0)
        if initial_delta < 1e-6:
            post_shock_neutrality = 1.0 if final_delta < 0.05 else 0.0
        else:
            post_shock_neutrality = max(0.0, 1.0 - final_delta / initial_delta)

        # Cost efficiency (0.0 - 1.0)
        total_cost = 0.0
        for step in episode_history:
            action = step.get("action")
            if action is None:
                continue

            action_direction = (
                action.direction.value
                if hasattr(action.direction, "value")
                else action.direction
            )

            # Skip hold actions (no cost)
            if action_direction == "hold":
                continue

            # Simplified cost model: 0.01 per contract traded
            total_cost += abs(action.quantity) * 0.01

        # Max reasonable cost is proportional to initial delta
        max_cost = max(initial_delta * 2.0, 0.1)
        cost_efficiency = max(0.0, 1.0 - total_cost / max_cost)

        # Final score: weighted combination
        # Requirements: 4.5
        score = (
            pre_shock_neutrality * 0.30
            + post_shock_neutrality * 0.40
            + cost_efficiency * 0.30
        )

        # Clamp to [0.0, 1.0]
        return min(max(score, 0.0), 1.0)
