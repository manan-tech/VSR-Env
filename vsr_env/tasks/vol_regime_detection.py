"""Volatility Regime Detection Task for VSR-Env.

Easy difficulty tier: Purely analytical task where the agent must
identify the current market volatility regime based on the IV surface.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from vsr_env.models import VSRState
from vsr_env.reward.rubrics import ExactMatchRubric


class VolRegimeDetectionTask:
    """Easy tier task: Identify market volatility regime from IV surface."""

    def __init__(self):
        self.base_variance = 0.04
        self.regimes = {"low": 0.01, "normal": 0.04, "high": 0.09}
        self.selected_regime = "normal"

    def initialize(self, state: VSRState, rng: Any = None) -> List[Tuple[Tuple[int, int], str, float]]:
        """Set up the volatility regime challenge."""
        if rng is None:
            rng = np.random.RandomState()

        # Randomly select a regime
        regimes_list = list(self.regimes.keys())
        self.selected_regime = rng.choice(regimes_list)
        state.variance = self.regimes[self.selected_regime]
        state.regime = self.selected_regime
        state.expected_outcome = self.selected_regime

        # Create ambiguity cases to make inference harder (mixed-signal surfaces)
        mispriced_cells = []
        
        # 60% chance of creating ambiguous surface to force reasoning
        if rng.uniform() < 0.6:
            ambiguity_type = rng.choice(["short_term_spike", "long_term_crush", "inversion", "smile_distortion"])
            
            if ambiguity_type == "short_term_spike":
                # Front-month heavily overpriced
                for strike_idx in range(8):
                    mispriced_cells.append(((strike_idx, 0), "over", rng.uniform(0.08, 0.15)))
            elif ambiguity_type == "long_term_crush":
                # Long-term heavily underpriced
                for strike_idx in range(8):
                    mispriced_cells.append(((strike_idx, 2), "under", rng.uniform(0.08, 0.15)))
            elif ambiguity_type == "inversion":
                # Term structure inverted
                for strike_idx in range(8):
                    mispriced_cells.append(((strike_idx, 0), "over", rng.uniform(0.05, 0.10)))
                    mispriced_cells.append(((strike_idx, 2), "under", rng.uniform(0.05, 0.10)))
            elif ambiguity_type == "smile_distortion":
                # OTM options spiked
                for mat_idx in range(3):
                    mispriced_cells.append(((0, mat_idx), "over", rng.uniform(0.08, 0.15)))
                    mispriced_cells.append(((1, mat_idx), "over", rng.uniform(0.06, 0.12)))
                    mispriced_cells.append(((6, mat_idx), "over", rng.uniform(0.06, 0.12)))
                    mispriced_cells.append(((7, mat_idx), "over", rng.uniform(0.08, 0.15)))

        return mispriced_cells

    def get_description(self) -> str:
        return (
            "Analyze the provided IV surface and determine the underlying fundamental volatility "
            "regime ('low', 'normal', or 'high'). Beware that the surface may feature transient "
            "mixed signals (e.g., short-term spikes, inverted term structures). You have 3 steps. "
            "Use 'hold' actions to observe surface evolution, or state your best guess in your "
            "reasoning at each step. Your final step's reasoning will be graded."
        )


class VolRegimeDetectionGrader:
    """Grader for VolRegimeDetectionTask.

    Scores pure identification capability of the agent against the ground truth regime.
    """

    def __init__(self):
        self.exact_match = ExactMatchRubric()

    def score(self, episode_history: List[Dict[str, Any]], state: VSRState) -> float:
        """Compute the final grade for the episode.

        Args:
            episode_history: Full trace of actions, observations, and rewards
            state: Final environment state (contains ground truth)

        Returns:
            Float between 0.01 and 0.99 indicating detection accuracy
        """
        steps = episode_history
        if not steps:
            return 0.01

        # Extract the reasoning from the final action
        last_action = steps[-1].get("action")
        if last_action is None:
            return 0.01
            
        reasoning = getattr(last_action, "reasoning", "")
        if not reasoning and isinstance(last_action, dict):
            reasoning = last_action.get("reasoning", "")

        expected = state.expected_outcome or state.regime

        # Check if the correct regime string is present in the reasoning payload
        if expected and isinstance(reasoning, str):
            if expected.lower() in reasoning.lower():
                return 0.99

        return 0.01
