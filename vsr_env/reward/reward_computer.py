"""Reward computation for VSR-Env.

Provides meaningful per-step signals that enable agents to learn from trajectory feedback.
All rewards are normalized to contribute to a total in the approximate range [0.0, 1.0].
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vsr_env.models import VSRAction, VSRObservation, VSRState, VSRReward

# Domain keywords for reasoning quality scoring
DOMAIN_KEYWORDS = [
    "delta", "hedge", "neutral", "skew", "smile", "regime",
    "overpriced", "underpriced", "moneyness", "vega", "gamma",
    "theta", "volatility", "arbitrage", "mispricing"
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
        Sigmoid output in range (0.0, 1.0)
    
    Requirements: 10.4
    """
    return 1.0 / (1.0 + math.exp(-x / scale))


def score_reasoning_quality(
    reasoning: str,
    observation: "VSRObservation",
    state: "VSRState"
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
        Score in [0.0, 1.0]
    
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
    return min(max(score, 0.0), 1.0)


class RewardComputer:
    """Computes per-step rewards for each task.
    
    Provides meaningful per-step signals that enable agents to learn
    from trajectory feedback. All rewards are normalized to contribute
    to a total in the approximate range [0.0, 1.0].
    
    Requirements: 10.1
    """
    
    def compute_iv_reading_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation"
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
            expected_mispricing = state.true_mispriced_directions.get(action.selected_strike)
            
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
        
        # Total is clamped to [0.0, 1.0]
        total = min(identification + reasoning_component, 1.0)
        
        return VSRReward(
            total=total,
            identification_component=identification,
            reasoning_component=reasoning_component
        )
    
    def compute_delta_hedging_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_delta: float,
        trade_cost: float
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
        
        # Delta improvement component (0.0 - 0.6)
        if old_delta > 1e-6:
            improvement = max(0.0, (old_delta - new_delta) / old_delta)
        else:
            # Already neutral
            improvement = 1.0 if new_delta < 0.05 else 0.0
        delta_reward = improvement * 0.6
        
        # Cost efficiency component (0.0 - 0.4)
        cost_reward = max(0.0, 0.4 - trade_cost * 0.1)
        
        # Neutrality bonus (0.0 or 0.1)
        neutrality_bonus = 0.1 if new_delta < 0.05 else 0.0
        
        # Total is clamped to [0.0, 1.0]
        total = min(delta_reward + cost_reward + neutrality_bonus, 1.0)
        
        return VSRReward(
            total=total,
            greek_component=delta_reward + neutrality_bonus,
            pnl_component=cost_reward
        )
    
    def compute_arb_capture_reward(
        self,
        action: "VSRAction",
        state: "VSRState",
        observation: "VSRObservation",
        prev_pnl: float
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
        
        # Total is clamped to [0.0, 1.0]
        total = min(pnl_reward + greek_reward + reasoning_reward, 1.0)
        
        return VSRReward(
            total=total,
            pnl_component=pnl_reward,
            greek_component=greek_reward,
            reasoning_component=reasoning_reward
        )