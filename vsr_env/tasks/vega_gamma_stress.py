"""Vega Gamma Stress Task for VSR-Env.

Super-Boss difficulty tier: Agent starts with a deeply negative options portfolio
(short vega, short gamma) and must quickly neutralize the position before the
environment undergoes a dual market shock (-20% spot drop alongside a 400% IV spike).
Grading focuses on standard deviation boundaries of risk neutralization.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from vsr_env.models import VSRState
from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.reward.rubrics import ReasoningQualityRubric


class VegaGammaStressTask:
    """Super-Boss tier task: Neutralize negative convexity before a dual shock."""

    def __init__(self):
        self.max_steps = 10
        self.difficulty = "super-boss"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the task with massive short straddles.

        Args:
            state: VSRState to initialize
            rng: Reproducibility logic

        Returns:
            Empty list (no mispricings)
        """
        engine = OptionChainEngine()

        # Start the agent out with deeply short ATMs (short vega and short gamma)
        strike_idx = 4  # ATM ~100
        maturity_idx = 1  # 90d

        K = engine.STRIKES[strike_idx]
        T = engine.MATURITIES[maturity_idx]
        S = state.spot_price
        sigma = np.sqrt(state.variance)

        # Assign agent a massive 5.0 lot short Call/Put Straddle position
        pos_entry_call = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), "call"
        )[0]
        pos_entry_put = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), "put"
        )[0]

        state.positions = [
            {
                "strike_idx": strike_idx,
                "maturity_idx": maturity_idx,
                "direction": "sell",
                "quantity": 5.0,
                "entry_price": pos_entry_call,
                "option_type": "call",
            },
            {
                "strike_idx": strike_idx,
                "maturity_idx": maturity_idx,
                "direction": "sell",
                "quantity": 5.0,
                "entry_price": pos_entry_put,
                "option_type": "put",
            },
        ]

        # Shock occurs abruptly between steps 4 and 8
        state.dual_shock_step = int(rng.randint(4, 9))

        return []

    def get_description(self) -> str:
        return (
            "SUPER-BOSS TIER: You start deeply short vega and gamma. A catastrophic dual market "
            "shock (-15% to -20% spot drop alongside a 300% to 500% IV spike) will occur mid-episode. "
            "You must neutralize your vega and gamma exposure (targeting ~0 bounds) before the "
            "shock hits, or your PnL will be decimated. Ensure your reasoning mentions vega/gamma "
            "hedging and crash preparation."
        )


class VegaGammaStressGrader:
    """Grader for the dual shock VegaGammaStress task.

    Scores based on standard deviation boundaries around a targeted neutral state
    prior to the shock, plus reward for surviving PnL outcome.
    """

    def __init__(self):
        self.reasoning_rubric = ReasoningQualityRubric(
            keywords=["vega", "gamma", "shock", "hedge", "convexity", "crash"]
        )

    def score(self, episode_history: List[Dict[str, Any]], state: VSRState) -> float:
        """Compute the final grade using strict standard deviation bounds."""
        steps = episode_history
        if not steps:
            return 0.0

        shock_step = getattr(state, "dual_shock_step", 6)

        pre_shock_vegas = []
        pre_shock_gammas = []

        # Analyze trajectory leading up to the shock
        for i, step in enumerate(steps):
            if i + 1 < shock_step:
                obs = step.get("observation")
                if obs:
                    greeks = getattr(obs, "portfolio_greeks", {})
                    if hasattr(greeks, "get"):
                        pre_shock_vegas.append(greeks.get("vega", -5.0))
                        pre_shock_gammas.append(greeks.get("gamma", -5.0))
                    elif isinstance(greeks, dict):
                        pre_shock_vegas.append(greeks.get("vega", -5.0))
                        pre_shock_gammas.append(greeks.get("gamma", -5.0))
                    else:
                        pre_shock_vegas.append(getattr(greeks, "vega", -5.0))
                        pre_shock_gammas.append(getattr(greeks, "gamma", -5.0))

        # 1. Vega/Gamma Neutrality via SD Bounds (50% of Grade)
        # Target: Neutralize near 0. If SD bounds are extremely narrow and close to 0, score high.
        # However, due to discrete sizing, an exact 0 is tough. Let's strictly grade average position.
        vg_score = 0.0
        if pre_shock_vegas and pre_shock_gammas:
            avg_vega = np.mean(pre_shock_vegas)
            avg_gamma = np.mean(pre_shock_gammas)
            # Bound function: if average is inside [-0.02, 0.02], max score. Decays drastically via gaussian.
            vega_bounds_score = np.exp(-0.5 * (avg_vega / 0.05) ** 2)
            gamma_bounds_score = np.exp(-0.5 * (avg_gamma / 0.02) ** 2)
            vg_score = (vega_bounds_score * 0.5) + (gamma_bounds_score * 0.5)

        # 2. Survival PnL constraints (30% of Grade)
        pnl = state.portfolio_pnl
        # Surviving entails a PNL drop better than -50 (the unfiltered short straddle will lose 100+)
        pnl_score = 0.0
        if pnl > 0.0:
            pnl_score = 1.0
        elif pnl > -5.0:
            pnl_score = 0.8
        elif pnl > -50.0:
            pnl_score = 0.4

        # 3. Reasoning Analysis (20% of Grade)
        reasoning_score = 0.0
        if steps:
            reasonings = []
            for s in steps:
                act = s.get("action")
                if act:
                    r = getattr(act, "reasoning", "")
                    if not r and isinstance(act, dict):
                        r = act.get("reasoning", "")
                    if r:
                        reasonings.append(r)
            all_reasoning = " ".join(reasonings)
            reasoning_score = float(self.reasoning_rubric.score(all_reasoning, None))

        final_score = (vg_score * 0.5) + (pnl_score * 0.3) + (reasoning_score * 0.2)
        return float(np.clip(final_score, 0.0, 1.0))
