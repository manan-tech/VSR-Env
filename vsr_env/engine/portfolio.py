"""Portfolio manager for VSR-Env.

Tracks open positions, computes aggregate Greeks, and calculates
mark-to-market P&L based on current market prices.
"""

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


def add_position(
    state: VSRState,
    strike_idx: int,
    maturity_idx: int,
    direction: str,
    quantity: float,
    engine: OptionChainEngine,
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
    r = engine.r

    # Compute entry price and Greeks (using call options)
    entry_price = engine.bs_price(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]
    pos_delta = engine.delta(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]
    pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
    pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]

    # Adjust sign based on direction (buy = positive, sell = negative)
    quantity_signed = quantity if direction == "buy" else -quantity

    # Create position dictionary
    position = {
        "strike_idx": strike_idx,
        "maturity_idx": maturity_idx,
        "direction": direction,
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
    r = engine.r

    for pos in state.positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]

        # Recompute Greeks at current market conditions
        pos_delta = engine.delta(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]
        pos_gamma = engine.gamma(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_vega = engine.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0]
        pos_theta = engine.theta(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]

        # Apply position quantity and direction
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
    r = engine.r

    for pos in state.positions:
        K = engine.STRIKES[pos["strike_idx"]]
        T = engine.MATURITIES[pos["maturity_idx"]]

        # Recompute current market price
        current_price = engine.bs_price(S, np.array([K]), np.array([T]), np.array([sigma]), option_type="call")[0]

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