"""Delta Hedging Task implementation for VSR-Env.

This module implements the medium task where agents neutralize portfolio delta
cost-efficiently.

Requirements: 2.2, 4.1
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


class DeltaHedgingTask:
    """Medium task: Neutralize portfolio delta within ±0.05 cost-efficiently.

    The agent starts with a portfolio that has non-zero delta (0.2-0.8)
    and must execute trades to bring delta close to zero while minimizing
    transaction costs.

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
        Stores the initial delta in state for grading.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)

        Requirements: 2.2, 4.1
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
        r = engine.r

        # Compute per-contract delta
        pos_delta = engine.delta(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]
        pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

        # Target portfolio delta between 0.2 and 0.8 per Requirement 4.1
        target_delta = rng.uniform(0.2, 0.8)

        # Calculate quantity needed to achieve target delta
        # portfolio_delta = pos_delta * quantity_signed
        # For buy: quantity = target_delta / pos_delta
        # For sell: quantity = target_delta / pos_delta (portfolio_delta will be negative)
        quantity = target_delta / abs(pos_delta)

        # Compute entry price
        entry_price = engine.bs_price(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]

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
        # Using a private attribute that will be accessed by the grader
        state.initial_delta = abs(portfolio_delta)

        # No mispricings for delta hedging task
        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "You are managing an options portfolio with non-zero delta exposure. "
            "Your objective is to neutralize the portfolio delta to within ±0.05 "
            "while minimizing transaction costs. Execute trades (buy/sell options) "
            "to offset the existing delta. Consider the cost-efficiency of your hedging "
            "strategy. You have 5 steps to achieve delta neutrality."
        )
class DeltaHedgingGrader:
    """Grader for Delta Hedging task.
    
    Scores based on delta neutralization quality and cost efficiency.
    
    Requirements: 4.5, 4.6
    """
    
    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Delta Hedging task.
        
        Score = neutralization_quality × 0.7 + cost_efficiency × 0.3
        
        neutralization_quality = max(0, 1.0 - |final_delta| / |initial_delta|)
        cost_efficiency = max(0, 1.0 - total_cost / max_cost)
        
        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_delta and initial_delta
        
        Returns:
            Score in [0.0, 1.0]
        
        Requirements: 4.5, 4.6
        """
        # Get initial delta (stored during task initialization)
        initial_delta = state.initial_delta if state.initial_delta > 1e-6 else 0.5
        
        final_delta = abs(state.portfolio_delta)
        
        # Neutralization quality (0.0 - 1.0)
        # Requirements: 4.6
        if initial_delta < 1e-6:
            # Already neutral from start
            neutralization_quality = 1.0 if final_delta < 0.05 else 0.0
        else:
            neutralization_quality = max(0.0, 1.0 - final_delta / initial_delta)
        
        # Cost efficiency (0.0 - 1.0)
        # Compute total cost from trades executed
        total_cost = 0.0
        for step in episode_history:
            action = step.get("action")
            if action is None:
                continue
            
            action_direction = action.direction.value if hasattr(action.direction, 'value') else action.direction
            
            # Skip hold actions (no cost)
            if action_direction == "hold":
                continue
            
            # Simplified cost model: 0.01 per contract traded
            total_cost += abs(action.quantity) * 0.01
        
        # Max reasonable cost is proportional to initial delta
        # A generous upper bound: 2x the initial delta in cost units
        max_cost = max(initial_delta * 2.0, 0.1)  # At least 0.1 to avoid division issues
        cost_efficiency = max(0.0, 1.0 - total_cost / max_cost)
        
        # Final score: weighted combination
        # Requirements: 4.5
        score = neutralization_quality * 0.7 + cost_efficiency * 0.3
        
        # Clamp to [0.0, 1.0]
        return min(max(score, 0.0), 1.0)