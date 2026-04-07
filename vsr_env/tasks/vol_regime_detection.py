"""Volatility Regime Detection Task for VSR-Env.

Easy difficulty tier: Purely analytical task where the agent must
identify the current market volatility regime based on the IV surface.
"""

from typing import Dict, Any
import numpy as np

from vsr_env.models import VSRState
from vsr_env.reward.rubrics import ExactMatchRubric


class VolRegimeDetectionTask:
    """Easy tier task: Identify market volatility regime from IV surface."""

    def __init__(self):
        self.base_variance = 0.04
        self.regimes = {"low": 0.01, "normal": 0.04, "high": 0.09}
        self.selected_regime = "normal"

    def initialize(self, state: VSRState, seed: int) -> Dict[str, Any]:
        """Set up the volatility regime challenge."""
        np.random.seed(seed)

        # Randomly select a regime
        regimes_list = list(self.regimes.keys())
        self.selected_regime = np.random.choice(regimes_list)
        state.variance = self.regimes[self.selected_regime]
        state.regime = self.selected_regime
        state.expected_outcome = self.selected_regime

        return {
            "task_name": "vol_regime_detection",
            "task_description": (
                "Analyze the provided IV surface and determine the current volatility "
                "regime ('low', 'normal', or 'high'). Output an action holding 0 quantity "
                "and state the exact regime in your reasoning."
            ),
            "variance": state.variance,
            "regime": state.regime,
        }

    def get_description(self) -> str:
        return (
            "Analyze the provided IV surface and determine the current volatility "
            "regime ('low', 'normal', or 'high'). Output an action holding 0 quantity "
            "and state the exact regime in your reasoning."
        )


class VolRegimeDetectionGrader:
    """Grader for VolRegimeDetectionTask.

    Scores pure identification capability of the agent against the ground truth regime.
    """

    def __init__(self):
        self.exact_match = ExactMatchRubric(case_sensitive=False)

    def score(self, episode_history: Dict[str, Any], state: VSRState) -> float:
        """Compute the final grade for the episode.

        Args:
            episode_history: Full trace of actions, observations, and rewards
            state: Final environment state (contains ground truth)

        Returns:
            Float between 0.0 and 1.0 indicating detection accuracy
        """
        steps = episode_history.get("steps", [])
        if not steps:
            return 0.0

        # Extract the reasoning from the first action (since it's a 1-step task usually)
        first_action = steps[0].get("action", {})
        reasoning = first_action.get("reasoning", "")

        expected = state.expected_outcome or state.regime

        # Check if the correct regime string is present in the reasoning payload
        score = self.exact_match.evaluate(reasoning, target=expected)

        return float(score)
