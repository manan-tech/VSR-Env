"""Portfolio manager for VSR-Env.

Tracks open positions, computes aggregate Greeks, and calculates
mark-to-market P&L based on current market prices.

Enhanced with multi-leg strategy support for straddles, strangles, spreads.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState

if TYPE_CHECKING:
    from vsr_env.strategies.base import OptionStrategy


def add_position(
    state: VSRState,
    strike_idx: int,
    maturity_idx: int,
    direction: str,
    quantity: float,
    engine: OptionChainEngine,
    option_type: str = "call",
) -> None:
    """Add a new position to the portfolio.

    Computes entry price, IV, and Greeks using the OptionChainEngine.
    Adjusts Greek signs based on direction (buy = positive, sell = negative).

    Args:
        state: Current VSRState (modified in place)
        strike_idx: Strike index (0-7) into STRIKES array
        maturity_idx: Maturity index (0-2) into MATURITIES array
        direction: Trade direction ("buy" or "sell")
        quantity: Number of contracts
        engine: OptionChainEngine for pricing and Greeks

    Requirements: 22.1, 22.2, 22.7
    """
    # Get strike and maturity from engine constants
    K = engine.STRIKES[strike_idx]
    T = engine.MATURITIES[maturity_idx]

    # Current market conditions
    S = state.spot_price
    sigma = np.sqrt(state.variance)

    # Compute entry price and Greeks using the specified option type
    entry_price = engine.bs_price(
        S, np.array([K]), np.array([T]), np.array([sigma]), option_type=option_type
    )[0]
    pos_delta = engine.delta(
        S, np.array([K]), np.array([T]), np.array([sigma]), option_type=option_type
    )[0]
    pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
    pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

    # Adjust sign based on direction (buy = positive, sell = negative)
    quantity_signed = quantity if direction == "buy" else -quantity

    # Create position dictionary
    position = {
        "strike_idx": strike_idx,
        "maturity_idx": maturity_idx,
        "direction": direction,
        "option_type": option_type,
        "quantity": quantity,
        "entry_price": entry_price,
        "entry_iv": sigma,
        "entry_spot": S,
        "current_price": entry_price,
        "pnl": 0.0,
        "delta": pos_delta * quantity_signed,
        "gamma": pos_gamma * quantity_signed,
        "vega": pos_vega * quantity_signed,
    }

    # Append to state positions
    state.positions.append(position)


def compute_portfolio_greeks(
    state: VSRState,
    engine: OptionChainEngine,
) -> dict[str, float]:
    """Compute aggregate portfolio Greeks at current market conditions.

    Recomputes Greeks for all positions and sums them to get portfolio totals.

    Args:
        state: Current VSRState with positions
        engine: OptionChainEngine for Greeks computation

    Returns:
        Dict with delta, gamma, vega, theta values

    Requirements: 22.4
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0

    # Current market conditions
    S = state.spot_price
    sigma = np.sqrt(state.variance)

    for pos in state.positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]
        opt_type = pos.get("option_type", "call")

        # Recompute Greeks at current market conditions using stored option type
        pos_delta = engine.delta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]
        pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_theta = engine.theta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]

        # Apply position quantity and direction
        quantity_signed = (
            pos["quantity"] if pos["direction"] == "buy" else -pos["quantity"]
        )

        total_delta += pos_delta * quantity_signed
        total_gamma += pos_gamma * quantity_signed
        total_vega += pos_vega * quantity_signed
        total_theta += pos_theta * quantity_signed

    return {
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
        "theta": total_theta,
    }


def compute_portfolio_pnl(
    state: VSRState,
    engine: OptionChainEngine,
) -> float:
    """Compute mark-to-market P&L for all positions.

    Recomputes current price for each position and calculates P&L:
    - Buy: P&L = (current_price - entry_price) * quantity
    - Sell: P&L = (entry_price - current_price) * quantity

    Args:
        state: Current VSRState with positions (modified in place)
        engine: OptionChainEngine for pricing

    Returns:
        Total portfolio P&L

    Requirements: 22.5, 22.6
    """
    total_pnl = 0.0

    # Current market conditions
    S = state.spot_price
    sigma = np.sqrt(state.variance)

    for pos in state.positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]
        opt_type = pos.get("option_type", "call")

        # Recompute current market price using stored option type
        current_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]

        # P&L calculation based on direction
        if pos["direction"] == "buy":
            pnl = (current_price - pos["entry_price"]) * pos["quantity"]
        else:  # sell
            pnl = (pos["entry_price"] - current_price) * pos["quantity"]

        # Update position with current values
        pos["current_price"] = current_price
        pos["pnl"] = pnl

        total_pnl += pnl

    return total_pnl


def update_positions_on_market_move(
    state: VSRState,
    engine: OptionChainEngine,
) -> None:
    """Update all position Greeks and P&L after market moves.

    Called after each step's market simulation to recompute
    portfolio state at new market conditions.

    Args:
        state: Current VSRState (modified in place)
        engine: OptionChainEngine for computations

    Requirements: 22.4, 22.5
    """
    # Recompute and update portfolio Greeks
    greeks = compute_portfolio_greeks(state, engine)
    state.portfolio_delta = greeks["delta"]
    state.portfolio_gamma = greeks["gamma"]
    state.portfolio_vega = greeks["vega"]

    # Recompute and update portfolio P&L
    pnl = compute_portfolio_pnl(state, engine)
    state.portfolio_pnl = pnl


# ============================================================================
# Multi-Leg Strategy Support
# ============================================================================


def add_strategy(
    state: VSRState,
    strategy: "OptionStrategy",
    engine: OptionChainEngine,
    entry_prices: Optional[List[float]] = None,
) -> None:
    """Add a multi-leg strategy to the portfolio atomically.

    All legs of the strategy are added with the same strategy_id,
    enabling strategy-level P&L tracking and closing.

    Args:
        state: Current VSRState (modified in place)
        strategy: OptionStrategy instance (Straddle, Strangle, etc.)
        engine: OptionChainEngine for pricing and Greeks
        entry_prices: Optional list of entry prices for each leg.
                     If not provided, computes from current market.

    Note:
        Positions are added in the order of strategy.legs
    """
    strategy_id = strategy.strategy_id

    for i, leg in enumerate(strategy.legs):
        # Get entry price
        if entry_prices and i < len(entry_prices):
            entry_price = entry_prices[i]
        else:
            # Compute from current market
            K = engine.STRIKES[leg["strike_idx"]]
            T = engine.MATURITIES[leg["maturity_idx"]]
            S = state.spot_price
            sigma = np.sqrt(state.variance)
            entry_price = engine.bs_price(
                S, np.array([K]), np.array([T]), np.array([sigma]),
                option_type=leg["option_type"]
            )[0]

        # Add position with strategy_id
        add_position_with_strategy(
            state=state,
            strike_idx=leg["strike_idx"],
            maturity_idx=leg["maturity_idx"],
            direction=leg["direction"],
            quantity=leg["quantity"],
            engine=engine,
            option_type=leg["option_type"],
            strategy_id=strategy_id,
            entry_price=entry_price,
        )


def add_position_with_strategy(
    state: VSRState,
    strike_idx: int,
    maturity_idx: int,
    direction: str,
    quantity: float,
    engine: OptionChainEngine,
    option_type: str = "call",
    strategy_id: Optional[str] = None,
    entry_price: Optional[float] = None,
) -> None:
    """Add a position with optional strategy grouping.

    Enhanced version of add_position() that supports strategy tracking.

    Args:
        state: Current VSRState (modified in place)
        strike_idx: Strike index (0-7) into STRIKES array
        maturity_idx: Maturity index (0-2) into MATURITIES array
        direction: Trade direction ("buy" or "sell")
        quantity: Number of contracts
        engine: OptionChainEngine for pricing and Greeks
        option_type: "call" or "put"
        strategy_id: Optional strategy ID for grouping legs
        entry_price: Optional pre-computed entry price
    """
    # Get strike and maturity from engine constants
    K = engine.STRIKES[strike_idx]
    T = engine.MATURITIES[maturity_idx]

    # Current market conditions
    S = state.spot_price
    sigma = np.sqrt(state.variance)

    # Compute entry price if not provided
    if entry_price is None:
        entry_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=option_type
        )[0]

    # Compute Greeks
    pos_delta = engine.delta(
        S, np.array([K]), np.array([T]), np.array([sigma]), option_type=option_type
    )[0]
    pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
    pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

    # Adjust sign based on direction
    quantity_signed = quantity if direction == "buy" else -quantity

    # Create position dictionary
    position = {
        "strike_idx": strike_idx,
        "maturity_idx": maturity_idx,
        "direction": direction,
        "option_type": option_type,
        "quantity": quantity,
        "entry_price": entry_price,
        "entry_iv": sigma,
        "entry_spot": S,
        "current_price": entry_price,
        "pnl": 0.0,
        "delta": pos_delta * quantity_signed,
        "gamma": pos_gamma * quantity_signed,
        "vega": pos_vega * quantity_signed,
    }

    # Add strategy_id if provided
    if strategy_id:
        position["strategy_id"] = strategy_id

    # Append to state positions
    state.positions.append(position)


def get_positions_by_strategy(
    state: VSRState,
    strategy_id: str,
) -> List[Dict[str, Any]]:
    """Get all positions belonging to a specific strategy.

    Args:
        state: Current VSRState
        strategy_id: Strategy identifier

    Returns:
        List of position dictionaries for the strategy
    """
    return [
        pos for pos in state.positions
        if pos.get("strategy_id") == strategy_id
    ]


def compute_strategy_pnl(
    state: VSRState,
    strategy_id: str,
    engine: OptionChainEngine,
) -> float:
    """Compute P&L for a specific strategy.

    Args:
        state: Current VSRState
        strategy_id: Strategy identifier
        engine: OptionChainEngine for current pricing

    Returns:
        Total unrealized P&L for the strategy
    """
    positions = get_positions_by_strategy(state, strategy_id)

    if not positions:
        return 0.0

    total_pnl = 0.0
    S = state.spot_price
    sigma = np.sqrt(state.variance)

    for pos in positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]
        opt_type = pos.get("option_type", "call")

        current_price = engine.bs_price(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]

        if pos["direction"] == "buy":
            pnl = (current_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["quantity"]

        total_pnl += pnl

    return total_pnl


def compute_strategy_greeks(
    state: VSRState,
    strategy_id: str,
    engine: OptionChainEngine,
) -> Dict[str, float]:
    """Compute aggregate Greeks for a specific strategy.

    Args:
        state: Current VSRState
        strategy_id: Strategy identifier
        engine: OptionChainEngine for Greeks computation

    Returns:
        Dict with delta, gamma, vega, theta for the strategy
    """
    positions = get_positions_by_strategy(state, strategy_id)

    if not positions:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0

    S = state.spot_price
    sigma = np.sqrt(state.variance)

    for pos in positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]
        opt_type = pos.get("option_type", "call")

        pos_delta = engine.delta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]
        pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_theta = engine.theta(
            S, np.array([K]), np.array([T]), np.array([sigma]), option_type=opt_type
        )[0]

        quantity_signed = pos["quantity"] if pos["direction"] == "buy" else -pos["quantity"]

        total_delta += pos_delta * quantity_signed
        total_gamma += pos_gamma * quantity_signed
        total_vega += pos_vega * quantity_signed
        total_theta += pos_theta * quantity_signed

    return {
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
        "theta": total_theta,
    }


def close_strategy(
    state: VSRState,
    strategy_id: str,
    engine: OptionChainEngine,
) -> float:
    """Close all positions in a strategy atomically.

    Creates opposite trades for all legs and realizes P&L.

    Args:
        state: Current VSRState (modified in place)
        strategy_id: Strategy identifier
        engine: OptionChainEngine for pricing

    Returns:
        Realized P&L from closing the strategy
    """
    positions = get_positions_by_strategy(state, strategy_id)

    if not positions:
        return 0.0

    # Compute final P&L
    realized_pnl = compute_strategy_pnl(state, strategy_id, engine)

    # Remove all positions belonging to this strategy
    state.positions = [
        pos for pos in state.positions
        if pos.get("strategy_id") != strategy_id
    ]

    # Update portfolio P&L
    state.portfolio_pnl += realized_pnl

    return realized_pnl


def get_active_strategies(state: VSRState) -> List[str]:
    """Get list of active strategy IDs in the portfolio.

    Args:
        state: Current VSRState

    Returns:
        List of unique strategy_id strings
    """
    strategy_ids = set()
    for pos in state.positions:
        if "strategy_id" in pos:
            strategy_ids.add(pos["strategy_id"])
    return list(strategy_ids)
