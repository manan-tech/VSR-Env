"""Pydantic models for VSR-Env OpenEnv compliance.

All models inherit from pydantic.BaseModel for OpenEnv compliance.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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


class VSRAction(BaseModel):
    """Action model for agent decisions.

    Defines the action space for trading on the volatility surface.
    Agents select a strike, maturity, direction, and quantity.

    Attributes:
        selected_strike: Strike index (0-7) into STRIKES array
        selected_maturity: Maturity index (0-2) into MATURITIES array
        direction: Trade direction (buy, sell, or hold)
        quantity: Trade size in contracts (0.0-10.0)
        reasoning: Agent's analysis and trade thesis
    """

    selected_strike: int = Field(
        ...,
        ge=0,
        le=7,
        description="Strike index (0-7) into STRIKES array [85, 90, 95, 97.5, 100, 102.5, 105, 110]",
    )
    selected_maturity: int = Field(
        ...,
        ge=0,
        le=2,
        description="Maturity index (0-2) into MATURITIES array [30/365, 90/365, 180/365]",
    )
    direction: TradeDirection = Field(
        ...,
        description="Trade direction: buy, sell, or hold",
    )
    quantity: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Trade size in contracts (0.0-10.0)",
    )
    reasoning: str = Field(
        "",
        description="Agent's analysis and trade thesis",
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
    # Task-specific fields (set during task initialization)
    initial_delta: float = Field(
        0.0,
        description="Initial portfolio delta at episode start (for delta_hedging grading)",
    )
    regime_shift_step: int = Field(
        5,
        description="Step at which regime shift occurs (for delta_hedging task)",
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
