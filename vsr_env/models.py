"""Pydantic models for VSR-Env OpenEnv compliance.

All models inherit from pydantic.BaseModel for OpenEnv compliance.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TradeDirection(str, Enum):
    """Trade direction enum for action specification.

    Valid values:
        BUY: Purchase contracts (add positive position)
        SELL: Sell contracts (add negative position)
        HOLD: No action (skip step)
    """

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class StrategyType(str, Enum):
    """Strategy type enum for multi-leg actions.

    Valid values:
        SINGLE: Single-leg trade (backward compatible)
        STRADDLE: ATM straddle (2 legs, same strike/expiry, call+put)
        STRANGLE: OTM strangle (2 legs, different strikes, same expiry)
        VERTICAL_SPREAD: Vertical spread (2 legs, same expiry, same type)
        CALENDAR_SPREAD: Calendar spread (2 legs, same strike, diff expiry)
        IRON_CONDOR: Iron condor (4 legs, put spread + call spread)
    """

    SINGLE = "single"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    VERTICAL_SPREAD = "vertical_spread"
    CALENDAR_SPREAD = "calendar_spread"
    IRON_CONDOR = "iron_condor"


class StrategyLeg(BaseModel):
    """Individual leg of a multi-leg options strategy.

    Attributes:
        strike_idx: Strike index (0-7) into STRIKES array
        maturity_idx: Maturity index (0-2) into MATURITIES array
        option_type: "call" or "put"
        direction: "buy" or "sell"
        quantity: Number of contracts (0.0-10.0)
    """

    strike_idx: int = Field(
        ...,
        ge=0,
        le=7,
        description="Strike index (0-7) into STRIKES array",
    )
    maturity_idx: int = Field(
        ...,
        ge=0,
        le=2,
        description="Maturity index (0-2) into MATURITIES array",
    )
    option_type: Literal["call", "put"] = Field(
        ...,
        description="Option type: call or put",
    )
    direction: Literal["buy", "sell"] = Field(
        ...,
        description="Direction: buy or sell",
    )
    quantity: float = Field(
        ...,
        gt=0.0,
        le=10.0,
        description="Number of contracts",
    )


class VSRAction(BaseModel):
    """Action model for agent decisions.

    Supports both single-leg trades (backward compatible) and multi-leg strategies.
    For single-leg: use selected_strike, selected_maturity, direction, quantity, option_type.
    For multi-leg: use strategy_type and legs.

    Attributes:
        selected_strike: Strike index (0-7) for single-leg trades
        selected_maturity: Maturity index (0-2) for single-leg trades
        direction: Trade direction (buy, sell, or hold) for single-leg
        option_type: "call" or "put" for single-leg (defaults to "call")
        quantity: Trade size (0.0-10.0) for single-leg
        strategy_type: Strategy type for multi-leg actions
        legs: List of StrategyLeg for multi-leg actions
        reasoning: Agent's analysis and trade thesis
    """

    # Single-leg fields (backward compatible)
    selected_strike: int = Field(
        0,
        ge=0,
        le=7,
        description="Strike index (0-7) into STRIKES array [85, 90, 95, 97.5, 100, 102.5, 105, 110]",
    )
    selected_maturity: int = Field(
        0,
        ge=0,
        le=2,
        description="Maturity index (0-2) into MATURITIES array [30/365, 90/365, 180/365]",
    )
    direction: TradeDirection = Field(
        TradeDirection.HOLD,
        description="Trade direction: buy, sell, or hold",
    )
    option_type: Literal["call", "put"] = Field(
        "call",
        description="Option type: call or put (single-leg only)",
    )
    quantity: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Trade size in contracts (0.0-10.0)",
    )

    # Multi-leg fields
    strategy_type: Optional[StrategyType] = Field(
        None,
        description="Strategy type for multi-leg actions",
    )
    legs: Optional[List[StrategyLeg]] = Field(
        None,
        description="List of legs for multi-leg strategies",
    )

    # Reasoning (always required)
    reasoning: str = Field(
        "",
        description="Agent's analysis and trade thesis",
    )

    @field_validator("legs")
    @classmethod
    def validate_legs(cls, v: Optional[List[StrategyLeg]], info) -> Optional[List[StrategyLeg]]:
        """Validate that leg count matches strategy type."""
        if v is None:
            return v

        strategy_type = info.data.get("strategy_type")
        if strategy_type is None:
            raise ValueError("strategy_type is required when legs are provided")

        leg_count = len(v)
        expected_counts = {
            StrategyType.SINGLE: 1,
            StrategyType.STRADDLE: 2,
            StrategyType.STRANGLE: 2,
            StrategyType.VERTICAL_SPREAD: 2,
            StrategyType.CALENDAR_SPREAD: 2,
            StrategyType.IRON_CONDOR: 4,
        }

        expected = expected_counts.get(strategy_type)
        if expected is not None and leg_count != expected:
            raise ValueError(
                f"Strategy type {strategy_type.value} requires {expected} legs, got {leg_count}"
            )

        return v


class StrategyInfo(BaseModel):
    """Information about an active multi-leg strategy.

    Exposed in observations to help agents track strategy-level performance.

    Attributes:
        strategy_id: Unique identifier for the strategy
        strategy_type: Type of strategy (straddle, strangle, etc.)
        net_greeks: Aggregate Greeks for the strategy
        unrealized_pnl: Current unrealized P&L
        legs_summary: Human-readable summary of legs
    """

    strategy_id: str = Field(
        ...,
        description="Unique identifier for the strategy",
    )
    strategy_type: str = Field(
        ...,
        description="Type of strategy (straddle, strangle, etc.)",
    )
    net_greeks: Dict[str, float] = Field(
        ...,
        description="Aggregate Greeks for the strategy",
    )
    unrealized_pnl: float = Field(
        0.0,
        description="Current unrealized P&L",
    )
    legs_summary: str = Field(
        "",
        description="Human-readable summary of legs",
    )


class VSRObservation(BaseModel):
    """Observation model for environment state visible to agent.

    Provides comprehensive market state including IV surface,
    portfolio Greeks, P&L, and task information.

    Attributes:
        iv_surface: 8×3 implied volatility matrix
        spot_price: Current underlying price
        portfolio_greeks: Dict with delta, gamma, vega, theta values
        portfolio_pnl: Cumulative profit/loss
        portfolio_positions: List of current open positions
        market_sentiment: Sentiment indicator in [-1.0, 1.0]
        step_number: Current step in episode
        steps_remaining: Steps until episode end
        task_name: Current task identifier
        task_description: Task objective description
        last_action_error: Validation error from last action (if any)
    """

    iv_surface: List[List[float]] = Field(
        ...,
        description="8×3 implied volatility matrix (strikes × maturities)",
    )
    spot_price: float = Field(
        ...,
        description="Current underlying price",
    )
    portfolio_greeks: Dict[str, float] = Field(
        ...,
        description="Portfolio Greeks: delta, gamma, vega, theta",
    )
    portfolio_pnl: float = Field(
        0.0,
        description="Cumulative profit/loss",
    )
    portfolio_positions: List[Dict] = Field(
        default_factory=list,
        description="List of current open positions",
    )
    market_sentiment: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Market sentiment indicator in [-1.0, 1.0]",
    )
    step_number: int = Field(
        0,
        description="Current step in episode",
    )
    steps_remaining: int = Field(
        10,
        description="Steps remaining until episode end",
    )
    task_name: str = Field(
        "",
        description="Current task identifier",
    )
    task_description: str = Field(
        "",
        description="Task objective description for agent",
    )
    earnings_proximity: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Proximity float decreasing from 1.0 to 0.0 leading up to the vol crush",
    )
    last_action_error: Optional[str] = Field(
        None,
        description="Validation error from last action (if any)",
    )
    expected_outcome: Optional[str] = Field(
        None,
        description="Ground-truth expected outcome for grader",
    )
    active_strategies: List[StrategyInfo] = Field(
        default_factory=list,
        description="Active multi-leg strategies with their Greeks and P&L",
    )


class VSRState(BaseModel):
    """Internal state model including hidden information.

    Tracks all environment state including ground truth for grading,
    market simulation parameters, and portfolio details.

    Attributes:
        episode_id: Unique episode identifier
        step_count: Steps taken in current episode
        task_name: Active task name
        true_mispriced_strikes: Hidden mispriced strike indices for grading
        true_mispriced_directions: Hidden mispricing directions for grading
        regime: Current market regime
        spot_price: Current underlying price
        variance: Current variance for simulation
        portfolio_delta: Portfolio delta exposure
        portfolio_gamma: Portfolio gamma exposure
        portfolio_vega: Portfolio vega exposure
        portfolio_pnl: Portfolio profit/loss
        positions: List of position details
    """

    episode_id: str = Field(
        default="",
        description="Unique episode identifier",
    )
    step_count: int = Field(
        0,
        description="Steps taken in current episode",
    )
    task_name: str = Field(
        "",
        description="Active task name",
    )
    true_mispriced_strikes: List[int] = Field(
        default_factory=list,
        description="Hidden mispriced strike indices for grading",
    )
    true_mispriced_directions: Dict[int, str] = Field(
        default_factory=dict,
        description="Hidden mispricing directions (strike_idx -> 'over'/'under')",
    )
    regime: str = Field(
        "normal",
        description="Current market regime",
    )
    spot_price: float = Field(
        100.0,
        description="Current underlying price",
    )
    variance: float = Field(
        0.04,
        description="Current variance for simulation",
    )
    portfolio_delta: float = Field(
        0.0,
        description="Portfolio delta exposure",
    )
    portfolio_gamma: float = Field(
        0.0,
        description="Portfolio gamma exposure",
    )
    portfolio_vega: float = Field(
        0.0,
        description="Portfolio vega exposure",
    )
    portfolio_pnl: float = Field(
        0.0,
        description="Portfolio profit/loss",
    )
    positions: List[Dict] = Field(
        default_factory=list,
        description="List of position details",
    )
    # Multi-leg strategy tracking
    strategies: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Map of strategy_id -> strategy metadata (type, entry_prices)",
    )
    # Task-specific fields (set during task initialization)
    initial_delta: float = Field(
        0.0,
        description="Initial portfolio delta at episode start (for delta_hedging grading)",
    )
    regime_shift_step: int = Field(
        5,
        description="Step at which regime shift occurs (for delta_hedging task)",
    )
    dual_shock_step: int = Field(
        5,
        description="Step at which the dual shock occurs (for vega_gamma_stress task)",
    )
    vol_crush_step: int = Field(
        5,
        description="Step at which vol crush occurs (for earnings_vol_crush task)",
    )
    pre_crush_vega: float = Field(
        0.0,
        description="Portfolio vega before the vol crush event (for grading)",
    )
    initial_theta: float = Field(
        0.0,
        description="Initial portfolio theta at episode start (for gamma_scalping grading)",
    )
    mispriced_cells: List = Field(
        default_factory=list,
        description="List of mispriced cell specs for grading",
    )
    earnings_proximity: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Proximity float decreasing from 1.0 to 0.0 leading up to the vol crush",
    )
    expected_outcome: Optional[str] = Field(
        None,
        description="Ground-truth expected outcome for grader",
    )


class VSRReward(BaseModel):
    """Reward model with component breakdown.

    Provides structured reward decomposition for analysis of
    which components drive agent behavior.

    Attributes:
        total: Aggregate reward for step
        pnl_component: Profit/loss contribution
        greek_component: Greek neutrality contribution
        identification_component: Mispricing identification contribution
        reasoning_component: Reasoning quality contribution
    """

    total: float = Field(
        ...,
        description="Aggregate reward for step",
    )
    pnl_component: float = Field(
        0.0,
        description="Profit/loss contribution",
    )
    greek_component: float = Field(
        0.0,
        description="Greek neutrality contribution",
    )
    identification_component: float = Field(
        0.0,
        description="Mispricing identification contribution",
    )
    reasoning_component: float = Field(
        0.0,
        description="Reasoning quality contribution",
    )
