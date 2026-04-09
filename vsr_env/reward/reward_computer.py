"""Reward computation for VSR-Env.

Provides meaningful per-step signals that enable agents to learn from trajectory feedback.
All rewards are normalized to contribute to a total in the approximate range [0.01, 0.99].
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vsr_env.models import VSRAction, VSRObservation, VSRState, VSRReward

# Domain keywords for reasoning quality scoring
DOMAIN_KEYWORDS = [
    "delta",
    "hedge",
    "neutral",
    "skew",
    "smile",
    "regime",
    "overpriced",
    "underpriced",
    "moneyness",
    "vega",
    "gamma",
    "theta",
    "volatility",
    "arbitrage",
    "mispricing",
    # Multi-leg strategy keywords
    "straddle",
    "strangle",
    "spread",
    "calendar",
    "ironcondor",
    "iron_condor",
    "max_profit",
    "breakeven",
    "credit",
    "debit",
    "bull",
    "bear",
]


def sigmoid(x: float, scale: float = 0.3) -> float:
    """Sigmoid function centered at 0.

    Formula: 1.0 / (1.0 + exp(-x / scale))

    Scale calibration:
    - scale=0.3 means P&L of +0.3 → 0.73, P&L of +0.1 → 0.59
    - Typical step P&L range: 0.01 to 0.5

    Args:
        x: Input value
        scale: Scale parameter (default 0.3)

    Returns:
        Sigmoid output in range (0.01, 0.99)

    Requirements: 10.4
    """
    return 1.0 / (1.0 + math.exp(-x / scale))


def score_reasoning_quality(
    reasoning: str, observation: "VSRObservation", state: "VSRState"
) -> float:
    """Score reasoning quality using keyword presence and numeric consistency.

    Components:
    - Keyword presence (max 0.4): count domain keywords, score = min(hits / 4.0, 1.0) * 0.4
    - Numeric consistency (max 0.6): check for spot price, IV values, delta citations
    - Length penalty: multiply by 0.3 if len(reasoning) <= 20

    Args:
        reasoning: Agent's reasoning text
        observation: Current observation with IV surface
        state: Current state with spot price and portfolio delta

    Returns:
        Score in [0.01, 0.99]

    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8
    """
    score = 0.0
    text = reasoning.lower()

    # Component 1: Keyword presence (max 0.4)
    # Requirements: 11.1, 11.2
    keyword_hits = sum(1 for kw in DOMAIN_KEYWORDS if kw in text)
    keyword_score = min(keyword_hits / 4.0, 1.0) * 0.4
    score += keyword_score

    # Component 2: Numeric consistency (max 0.6)
    numeric_score = 0.0

    # Check spot price citation (0.25 points)
    # Requirements: 11.3
    spot_str = f"{state.spot_price:.1f}"
    spot_int = f"{int(round(state.spot_price))}"
    if spot_str in reasoning or spot_int in reasoning:
        numeric_score += 0.25

    # Check IV value citations (0.15 for one, +0.10 for two)
    # Requirements: 11.4, 11.5
    iv_values_cited = 0
    for row in observation.iv_surface:
        for iv_val in row:
            # Check for decimal format (e.g., "0.22") or percentage (e.g., "22%")
            if f"{iv_val:.2f}" in reasoning or f"{iv_val*100:.0f}%" in reasoning:
                iv_values_cited += 1

    if iv_values_cited >= 1:
        numeric_score += 0.15
    if iv_values_cited >= 2:
        numeric_score += 0.10  # Bonus for citing multiple IV values

    # Check portfolio delta citation (0.1 points)
    # Requirements: 11.6
    delta_val = state.portfolio_delta
    if f"{delta_val:.2f}" in reasoning or f"{delta_val:.1f}" in reasoning:
        numeric_score += 0.1

    score += numeric_score

    # Apply length penalty for trivial reasoning
    # Requirements: 11.7
    if len(reasoning) <= 20:
        score *= 0.3

    # Clamp to valid range
    # Requirements: 11.8
    return min(max(score, 0.01), 0.99)


class RewardComputer:
    """Computes per-step rewards for each task.

    Provides meaningful per-step signals that enable agents to learn
    from trajectory feedback. All rewards are normalized to contribute
    to a total in the approximate range [0.01, 0.99].

    Requirements: 10.1
    """

    def compute_iv_reading_reward(
        self, action: "VSRAction", state: "VSRState", observation: "VSRObservation"
    ) -> "VSRReward":
        """Compute reward for IV reading task.

        Reward = identification_component + reasoning_component

        identification_component:
          - 0.5 if correct strike and correct direction
          - 0.1 if correct strike but wrong direction
          - 0.0 otherwise

        Direction mapping:
          - 'sell' action → agent thinks option is overpriced
          - 'buy' action → agent thinks option is underpriced

        Args:
            action: Agent's action
            state: Current state with true mispriced strikes
            observation: Current observation

        Returns:
            VSRReward with total and component breakdown

        Requirements: 10.2
        """
        from vsr_env.models import VSRReward

        identification = 0.0

        # Check if the selected strike is mispriced
        if action.selected_strike in state.true_mispriced_strikes:
            expected_mispricing = state.true_mispriced_directions.get(
                action.selected_strike
            )

            # Map action direction to mispricing assessment:
            # - 'sell' means agent thinks it's overpriced
            # - 'buy' means agent thinks it's underpriced
            action_direction = action.direction.value
            if action_direction == "sell":
                agent_assessment = "over"
            elif action_direction == "buy":
                agent_assessment = "under"
            else:
                # 'hold' doesn't indicate a mispricing assessment
                agent_assessment = None

            if agent_assessment == expected_mispricing:
                identification = 0.5  # Correct strike and direction
            else:
                identification = 0.1  # Correct strike, wrong direction

        # Compute reasoning component
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_component = reasoning_score * 0.2

        # Total is clamped to [0.01, 0.99]
        total = min(max(identification + reasoning_component, 0.01), 0.99)

        return VSRReward(
            total=total,
            identification_component=identification,
            reasoning_component=reasoning_component,
        )

    def compute_delta_hedging_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_delta: float,
        trade_cost: float,
    ) -> "VSRReward":
        """Compute reward for delta hedging task.

        Reward = delta_improvement + cost_efficiency + neutrality_bonus

        delta_improvement = max(0, (old_delta - new_delta) / old_delta) * 0.6
        cost_efficiency = max(0, 0.4 - trade_cost * 0.1)
        neutrality_bonus = 0.1 if |delta| < 0.05 else 0.0

        Args:
            action: Agent's action
            state: Current state with portfolio delta
            observation: Current observation
            prev_delta: Portfolio delta before the action
            trade_cost: Cost of the trade (simplified model)

        Returns:
            VSRReward with total and component breakdown

        Requirements: 10.3
        """
        from vsr_env.models import VSRReward

        new_delta = abs(state.portfolio_delta)
        old_delta = abs(prev_delta)

        # Delta improvement component (0.0 - 0.5)
        if old_delta > 1e-6:
            improvement = max(0.0, (old_delta - new_delta) / old_delta)
        else:
            # Already neutral
            improvement = 1.0 if new_delta < 0.05 else 0.0
        delta_reward = improvement * 0.5

        # Cost efficiency component (0.0 - 0.3)
        cost_reward = max(0.0, 0.3 - trade_cost * 0.1)

        # Neutrality bonus (0.0 or 0.1)
        neutrality_bonus = 0.1 if new_delta < 0.05 else 0.0

        # Reasoning coherence (0.0 - 0.2)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.2

        # Total is clamped to [0.01, 0.99]
        total = min(
            max(delta_reward + cost_reward + neutrality_bonus + reasoning_reward, 0.01), 0.99
        )

        return VSRReward(
            total=total,
            greek_component=delta_reward + neutrality_bonus,
            pnl_component=cost_reward,
            reasoning_component=reasoning_reward,
        )

    def compute_arb_capture_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward for arbitrage capture task.

        Reward = pnl_component + greek_component + reasoning_component

        pnl_component = sigmoid(pnl_change, scale=0.3) * 0.4
        greek_component = (1.0 - min(|delta| / 0.5, 1.0)) * 0.3
        reasoning_component = score_reasoning_quality * 0.3

        Args:
            action: Agent's action
            state: Current state with portfolio delta and P&L
            observation: Current observation
            prev_pnl: Portfolio P&L before the action

        Returns:
            VSRReward with total and component breakdown

        Requirements: 10.4
        """
        from vsr_env.models import VSRReward

        # P&L improvement (0.0 - 0.4)
        pnl_change = state.portfolio_pnl - prev_pnl
        pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.4

        # Greek neutrality (0.0 - 0.3)
        delta_abs = abs(state.portfolio_delta)
        delta_penalty = min(delta_abs / 0.5, 1.0)
        greek_reward = (1.0 - delta_penalty) * 0.3

        # Reasoning coherence (0.0 - 0.3)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.3

        # Total is clamped to [0.01, 0.99]
        total = min(max(pnl_reward + greek_reward + reasoning_reward, 0.01), 0.99)

        return VSRReward(
            total=total,
            pnl_component=pnl_reward,
            greek_component=greek_reward,
            reasoning_component=reasoning_reward,
        )

    def compute_earnings_crush_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward for earnings vol crush task.

        Reward = pnl_component + greek_component + reasoning_component

        Args:
            action: Agent's action
            state: Current state with portfolio delta, vega, and P&L
            observation: Current observation
            prev_pnl: Portfolio P&L before the action

        Returns:
            VSRReward with total and component breakdown

        Requirements: 5.4, 5.5, 5.6
        """
        from vsr_env.models import VSRReward

        # P&L improvement (0.0 - 0.4)
        pnl_change = state.portfolio_pnl - prev_pnl
        pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.4

        # Greek neutrality (0.0 - 0.3)
        delta_abs = abs(state.portfolio_delta)
        delta_penalty = min(delta_abs / 0.5, 1.0)
        greek_reward = (1.0 - delta_penalty) * 0.3

        # Reasoning coherence (0.0 - 0.3)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.3

        # Total is clamped to [0.0, 1.0]
        total = min(pnl_reward + greek_reward + reasoning_reward, 1.0)

        return VSRReward(
            total=total,
            pnl_component=pnl_reward,
            greek_component=greek_reward,
            reasoning_component=reasoning_reward,
        )

    def compute_gamma_scalping_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_delta: float,
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward for gamma scalping task.

        Reward = delta_neutrality + pnl_component + reasoning_component

        Args:
            action: Agent's action
            state: Current state with portfolio delta and P&L
            observation: Current observation
            prev_delta: Portfolio delta before the action
            prev_pnl: Portfolio P&L before the action

        Returns:
            VSRReward with total and component breakdown

        Requirements: 6.4, 6.5, 6.6
        """
        from vsr_env.models import VSRReward

        # Delta neutrality quality (0.0 - 0.5)
        new_delta = abs(state.portfolio_delta)
        delta_neutrality = max(0.0, 1.0 - new_delta / 0.5) * 0.5

        # P&L improvement (0.0 - 0.3)
        pnl_change = state.portfolio_pnl - prev_pnl
        pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.3

        # Reasoning coherence (0.0 - 0.2)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.2

        # Total is clamped to [0.01, 0.99]
        total = min(max(delta_neutrality + pnl_reward + reasoning_reward, 0.01), 0.99)

        return VSRReward(
            total=total,
            greek_component=delta_neutrality,
            pnl_component=pnl_reward,
            reasoning_component=reasoning_reward,
        )

    def compute_vol_regime_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation"
    ) -> "VSRReward":
        """Compute reward for vol regime detection task.

        Reward = identification_component + reasoning_component

        identification_component:
          - 0.8 if correct regime identified in reasoning
          - 0.0 otherwise

        reasoning_component:
          - Up to 0.2 for quality of reasoning

        Args:
            action: Agent's action with reasoning containing regime prediction
            state: Current state with expected_regime
            observation: Current observation

        Returns:
            VSRReward with total and component breakdown
        """
        from vsr_env.models import VSRReward

        expected = getattr(state, "expected_outcome", "normal")

        # Extract regime from reasoning
        text = action.reasoning.lower()
        predicted = "normal"  # Default

        if "low" in text and any(w in text for w in ["vol", "iv", "implied", "regime", "variance"]):
            predicted = "low"
        elif "high" in text and any(w in text for w in ["vol", "iv", "implied", "regime", "variance"]):
            predicted = "high"
        elif "normal" in text and any(w in text for w in ["vol", "iv", "implied", "regime", "variance"]):
            predicted = "normal"

        # Identification component (0.8 for correct)
        identification = 0.8 if predicted == expected else 0.0

        # Reasoning component (up to 0.2)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_component = reasoning_score * 0.2

        total = min(max(identification + reasoning_component, 0.01), 0.99)

        return VSRReward(
            total=total,
            identification_component=identification,
            reasoning_component=reasoning_component
        )

    def compute_vega_gamma_stress_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward for vega-gamma stress task.

        Reward = vega_gamma_neutrality + pnl_survival + reasoning_component

        Args:
            action: Agent's action
            state: Current state with portfolio vega, gamma, and P&L
            observation: Current observation
            prev_pnl: Portfolio P&L before the action

        Returns:
            VSRReward with total and component breakdown
        """
        from vsr_env.models import VSRReward

        # Vega/Gamma neutrality (0.0 - 0.5)
        # Use Gaussian scoring: exp(-0.5 * (value / threshold)^2)
        vega_abs = abs(state.portfolio_vega) if hasattr(state, "portfolio_vega") else 0.0
        gamma_abs = abs(state.portfolio_gamma) if hasattr(state, "portfolio_gamma") else 0.0

        vega_score = math.exp(-0.5 * (vega_abs / 0.05) ** 2)
        gamma_score = math.exp(-0.5 * (gamma_abs / 0.02) ** 2)
        vg_neutrality = (vega_score * 0.5 + gamma_score * 0.5) * 0.5

        # P&L survival (0.0 - 0.3)
        pnl_change = state.portfolio_pnl - prev_pnl
        pnl_reward = sigmoid(pnl_change, scale=0.5) * 0.3

        # Reasoning coherence (0.0 - 0.2)
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.2

        # Total is clamped to [0.01, 0.99]
        total = min(max(vg_neutrality + pnl_reward + reasoning_reward, 0.01), 0.99)

        return VSRReward(
            total=total,
            greek_component=vg_neutrality,
            pnl_component=pnl_reward,
            reasoning_component=reasoning_reward,
        )

    # ========================================================================
    # Multi-Leg Strategy Rewards
    # ========================================================================

    def compute_strategy_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        strategy_type: str,
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward for multi-leg strategy actions.

        Provides unified reward computation for straddle, strangle, spread
        and other multi-leg strategies.

        Args:
            action: Agent's action (should be multi-leg)
            state: Current state with portfolio Greeks and P&L
            observation: Current observation
            strategy_type: Type of strategy ("straddle", "strangle", etc.)
            prev_pnl: Portfolio P&L before the action

        Returns:
            VSRReward with total and component breakdown
        """
        from vsr_env.models import VSRReward

        # Base components
        pnl_change = state.portfolio_pnl - prev_pnl
        pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.3

        # Strategy-specific evaluation
        if strategy_type == "straddle":
            strategy_reward = self._evaluate_straddle(action, state, observation)
        elif strategy_type == "strangle":
            strategy_reward = self._evaluate_strangle(action, state, observation)
        elif strategy_type in ("vertical_spread", "spread"):
            strategy_reward = self._evaluate_spread(action, state, observation)
        elif strategy_type == "calendar_spread":
            strategy_reward = self._evaluate_calendar(action, state, observation)
        else:
            strategy_reward = 0.3  # Default moderate score

        # Reasoning component
        reasoning_score = score_reasoning_quality(action.reasoning, observation, state)
        reasoning_reward = reasoning_score * 0.2

        # Efficiency bonus for using atomic multi-leg action
        efficiency_bonus = 0.0
        if hasattr(action, "strategy_type") and action.strategy_type is not None:
            efficiency_bonus = 0.1

        total = min(max(pnl_reward + strategy_reward + reasoning_reward + efficiency_bonus, 0.01), 0.99)

        return VSRReward(
            total=total,
            pnl_component=pnl_reward,
            greek_component=strategy_reward,
            reasoning_component=reasoning_reward,
        )

    def _evaluate_straddle(
        self, action: "VSRAction", state: "VSRState", observation: "VSRObservation"
    ) -> float:
        """Evaluate straddle strategy quality.

        Scores based on:
        - Delta neutrality (straddle should be delta-neutral)
        - Correct vol direction (long vs short)
        - Position sizing
        """
        delta_abs = abs(state.portfolio_delta)
        delta_score = max(0.0, 1.0 - delta_abs / 0.3) * 0.3

        # Vega exposure indicates vol direction
        vega = state.portfolio_vega
        if abs(vega) > 0.01:
            # Has vol exposure
            vol_exposure_score = 0.2 * min(abs(vega) / 0.1, 1.0)
        else:
            vol_exposure_score = 0.1

        return delta_score + vol_exposure_score

    def _evaluate_strangle(
        self, action: "VSRAction", state: "VSRState", observation: "VSRObservation"
    ) -> float:
        """Evaluate strangle strategy quality.

        Similar to straddle but with different strike selection considerations.
        """
        delta_abs = abs(state.portfolio_delta)
        delta_score = max(0.0, 1.0 - delta_abs / 0.3) * 0.25

        vega = state.portfolio_vega
        vol_exposure_score = 0.2 * min(abs(vega) / 0.08, 1.0)

        return delta_score + vol_exposure_score

    def _evaluate_spread(
        self, action: "VSRAction", state: "VSRState", observation: "VSRObservation"
    ) -> float:
        """Evaluate vertical spread strategy quality.

        Scores based on:
        - Directional exposure (spread should have delta)
        - Risk management (defined risk characteristic)
        """
        delta = state.portfolio_delta

        # For spreads, we expect directional exposure
        if abs(delta) > 0.05:
            direction_score = 0.2 * min(abs(delta) / 0.2, 1.0)
        else:
            direction_score = 0.0

        # Gamma should be limited for spreads
        gamma_abs = abs(state.portfolio_gamma)
        gamma_score = 0.1 * max(0.0, 1.0 - gamma_abs / 0.05)

        return direction_score + gamma_score

    def _evaluate_calendar(
        self, action: "VSRAction", state: "VSRState", observation: "VSRObservation"
    ) -> float:
        """Evaluate calendar spread strategy quality.

        Scores based on term structure positioning.
        """
        # Calendar spreads benefit from term structure stability
        delta_abs = abs(state.portfolio_delta)
        delta_score = max(0.0, 1.0 - delta_abs / 0.3) * 0.2

        # Vega should be positive for long calendars
        vega = state.portfolio_vega
        vega_score = 0.2 * min(abs(vega) / 0.05, 1.0)

        return delta_score + vega_score

    def compute_straddle_trading_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward specifically for straddle trading task.

        Args:
            action: Agent's action
            state: Current state
            observation: Current observation
            prev_pnl: Previous P&L

        Returns:
            VSRReward instance
        """
        return self.compute_strategy_reward(action, state, observation, "straddle", prev_pnl)

    def compute_vertical_spread_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float,
    ) -> "VSRReward":
        """Compute reward specifically for vertical spread task.

        Args:
            action: Agent's action
            state: Current state
            observation: Current observation
            prev_pnl: Previous P&L

        Returns:
            VSRReward instance
        """
        return self.compute_strategy_reward(action, state, observation, "vertical_spread", prev_pnl)
