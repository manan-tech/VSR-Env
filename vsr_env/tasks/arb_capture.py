"""Arbitrage Capture Task implementation for VSR-Env.

This module implements the hard task where agents execute a full arbitrage
workflow with regime shifts.

Requirements: 2.3, 5.1, 5.2
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine, inject_mispricings
from vsr_env.models import VSRState


class ArbCaptureTask:
    """Hard task: Execute full arbitrage workflow with regime shifts.

    The agent must identify an exploitable mispricing, execute trades to
    capture the arbitrage, and manage the position through a regime shift
    that occurs at step 4 or 5.

    Attributes:
        max_steps: Maximum steps per episode (8)
        difficulty: Task difficulty level ("hard")
    """

    max_steps: int = 8
    difficulty: str = "hard"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the arbitrage capture task with an exploitable mispricing.

        Generates an IV surface with 1 significant mispricing and sets up
        regime shift parameters.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            List of mispriced cells: [((strike_idx, mat_idx), direction, magnitude)]

        Requirements: 2.3, 5.1, 5.2
        """
        # Generate 1 mispriced cell for exploitation
        mispriced_cells = inject_mispricings(rng, num_mispricings=1)

        # Extract strike indices and directions for grading
        true_mispriced_strikes = [si for (si, mi), _, _ in mispriced_cells]
        true_mispriced_directions = {
            si: direction for (si, mi), direction, _ in mispriced_cells
        }

        # Store in state for grading
        state.true_mispriced_strikes = true_mispriced_strikes
        state.true_mispriced_directions = true_mispriced_directions

        # Set regime shift step (4 or 5) per Requirement 5.2
        regime_shift_step = int(rng.randint(4, 6))  # 4 or 5

        # Store regime shift parameters using private attributes
        state.regime_shift_step = regime_shift_step
        state.regime = "normal"

        # Return mispriced_cells for IV surface generation
        return mispriced_cells

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "You are an options arbitrage trader. Identify and exploit mispricings "
            "on the implied volatility surface. Execute trades to capture arbitrage "
            "profits while managing your portfolio risk. Be prepared for potential "
            "market regime shifts that may occur mid-episode. Monitor your P&L, "
            "Greeks, and market conditions throughout. You have 8 steps to complete "
            "the arbitrage workflow."
        )
import math

from vsr_env.reward.reward_computer import score_reasoning_quality


class ArbCaptureGrader:
    """Grader for Arbitrage Capture task.
    
    Scores based on P&L, delta neutrality, and reasoning quality.
    
    Requirements: 5.4, 5.5, 5.6, 5.7
    """
    
    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        """Sigmoid function centered at 0.
        
        Args:
            x: Input value
            scale: Scale parameter (default 0.3)
        
        Returns:
            Sigmoid output in range (0.0, 1.0)
        """
        return 1.0 / (1.0 + math.exp(-x / scale))
    
    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Arbitrage Capture task.
        
        Score = pnl_score × 0.4 + neutrality_score × 0.3 + reasoning_score × 0.3
        
        pnl_score = sigmoid(final_pnl, scale=0.3)
        neutrality_score = max(0, 1.0 - avg_delta / 0.5)
        reasoning_score = average reasoning quality across steps
        
        Args:
            episode_history: List of step records with 'action' and 'observation'
            state: Final VSRState with portfolio_pnl and portfolio_delta
        
        Returns:
            Score in [0.0, 1.0]
        
        Requirements: 5.4, 5.5, 5.6, 5.7
        """
        if not episode_history:
            return 0.0
        
        # P&L component (0.0 - 1.0)
        # Requirements: 5.5
        final_pnl = state.portfolio_pnl
        pnl_score = self._sigmoid(final_pnl, scale=0.3)
        
        # Neutrality component (0.0 - 1.0)
        # Requirements: 5.6
        # Compute average delta across all steps
        total_delta = 0.0
        step_count = 0
        for step in episode_history:
            obs = step.get("observation")
            if obs is not None:
                greeks = obs.portfolio_greeks if hasattr(obs, 'portfolio_greeks') else {}
                delta = greeks.get("delta", 0.0) if isinstance(greeks, dict) else 0.0
                total_delta += abs(delta)
                step_count += 1
        
        avg_delta = total_delta / step_count if step_count > 0 else 0.0
        neutrality_score = max(0.0, 1.0 - avg_delta / 0.5)
        
        # Reasoning component (0.0 - 1.0)
        # Requirements: 5.7
        reasoning_scores = []
        for step in episode_history:
            action = step.get("action")
            obs = step.get("observation")
            if action is not None and obs is not None:
                reasoning = action.reasoning if hasattr(action, 'reasoning') else ""
                reasoning_score = score_reasoning_quality(reasoning, obs, state)
                reasoning_scores.append(reasoning_score)
        
        avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        
        # Final score: weighted combination
        # Requirements: 5.4
        score = pnl_score * 0.4 + neutrality_score * 0.3 + avg_reasoning_score * 0.3
        
        # Clamp to [0.0, 1.0]
        return min(max(score, 0.0), 1.0)