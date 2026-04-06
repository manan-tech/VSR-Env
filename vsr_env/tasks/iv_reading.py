"""IV Reading Task implementation for VSR-Env.

This module implements the easy task where agents identify mispriced options
on a volatility surface.

Requirements: 2.1, 3.1
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine, inject_mispricings
from vsr_env.models import VSRState


class IVReadingTask:
    """Easy task: Identify 2 mispriced options on the IV surface.

    The agent must analyze the implied volatility surface and identify
    which options are mispriced (overpriced or underpriced).

    Attributes:
        max_steps: Maximum steps per episode (3)
        difficulty: Task difficulty level ("easy")
    """

    max_steps: int = 3
    difficulty: str = "easy"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the IV reading task with 2 mispriced options.

        Generates an IV surface with exactly 2 deliberately mispriced cells.
        Stores the ground truth in state for grading.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            List of mispriced cells: [((strike_idx, mat_idx), direction, magnitude)]

        Requirements: 2.1, 3.1
        """
        # Generate 2 mispriced cells using the inject_mispricings function
        mispriced_cells = inject_mispricings(rng, num_mispricings=2)

        # Extract strike indices and directions for grading
        true_mispriced_strikes = [si for (si, mi), _, _ in mispriced_cells]
        true_mispriced_directions = {
            si: direction for (si, mi), direction, _ in mispriced_cells
        }

        # Store in state for grading
        state.true_mispriced_strikes = true_mispriced_strikes
        state.true_mispriced_directions = true_mispriced_directions

        # Return mispriced_cells for IV surface generation
        return mispriced_cells

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent

        Requirements: 2.5
        """
        return (
            "Analyze the implied volatility surface and identify the 2 mispriced options. "
            "For each mispriced option, indicate whether it is overpriced or underpriced. "
            "Select the strike index and maturity index of each mispriced cell, "
            "and specify the direction (overpriced or underpriced). "
            "You have 3 steps to identify both mispricings."
        )
class IVReadingGrader:
    """Grader for IV Reading task.
    
    Scores based on correct identification of mispriced options.
    A correct identification requires both the correct strike AND direction.
    
    Requirements: 3.5, 3.6, 6.4
    """
    
    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for IV Reading task.
        
        Score = correct_identifications / 2.0, clamped to [0.0, 1.0]
        
        A correct identification is when:
        - Agent selects a mispriced strike (correct strike)
        - Agent's direction matches the true mispricing direction
        
        Direction mapping:
        - 'sell' action → agent thinks option is overpriced
        - 'buy' action → agent thinks option is underpriced
        
        Args:
            episode_history: List of step records with 'action' and 'observation'
            state: Final VSRState with true_mispriced_strikes and true_mispriced_directions
        
        Returns:
            Score in [0.0, 1.0]
        
        Requirements: 3.5, 3.6, 6.4
        """
        correct_identifications = 0
        identified_strikes = set()
        
        for step in episode_history:
            action = step.get("action")
            if action is None:
                continue
            
            strike_idx = action.selected_strike
            
            # Skip if already identified this strike
            if strike_idx in identified_strikes:
                continue
            
            # Check if this strike is mispriced
            if strike_idx in state.true_mispriced_strikes:
                expected_direction = state.true_mispriced_directions.get(strike_idx)
                
                # Map action direction to mispricing assessment
                action_direction = action.direction.value if hasattr(action.direction, 'value') else action.direction
                if action_direction == "sell":
                    agent_assessment = "over"
                elif action_direction == "buy":
                    agent_assessment = "under"
                else:
                    # 'hold' doesn't count as an identification
                    continue
                
                # Check if direction matches
                if agent_assessment == expected_direction:
                    correct_identifications += 1
                    identified_strikes.add(strike_idx)
        
        # Score is correct identifications divided by 2 (max 2 mispricings)
        score = correct_identifications / 2.0
        
        # Clamp to [0.0, 1.0] per Requirement 6.4
        return min(max(score, 0.0), 1.0)