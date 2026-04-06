"""Market simulator for VSR-Env.

Implements realistic price dynamics using Geometric Brownian Motion
and regime shifts for the arbitrage capture task.
"""

import numpy as np

from vsr_env.models import VSRState


def advance_market(state: VSRState, rng: np.random.RandomState, dt: float = 1/252) -> None:
    """Advance market by one time step using Geometric Brownian Motion.
    
    Updates spot price using GBM: dS = μ*S*dt + σ*S*dW
    Also updates variance using mean-reverting Ornstein-Uhlenbeck process.
    
    Args:
        state: Current VSRState (modified in place)
        rng: Seeded numpy RandomState for reproducibility
        dt: Time step (default 1/252 = one trading day)
    
    Requirements: 21.1, 21.2, 21.6
    """
    # Geometric Brownian Motion for spot price
    mu = 0.0  # Risk-neutral drift
    sigma = np.sqrt(state.variance)
    
    # Generate random shock
    dW = rng.normal(0, np.sqrt(dt))
    
    # Price change: dS = μ*S*dt + σ*S*dW
    dS = mu * state.spot_price * dt + sigma * state.spot_price * dW
    state.spot_price += dS
    
    # Clamp spot price to realistic range
    state.spot_price = np.clip(state.spot_price, 50.0, 150.0)
    
    # Mean-reverting variance dynamics (Ornstein-Uhlenbeck)
    # dV = θ*(var_mean - variance)*dt + var_vol*dW
    theta = 0.1  # Mean reversion speed
    var_mean = 0.04  # Long-term variance (20% vol)
    var_vol = 0.01  # Variance of variance
    
    dW_var = rng.normal(0, np.sqrt(dt))
    dV = theta * (var_mean - state.variance) * dt + var_vol * dW_var
    state.variance += dV
    
    # Clamp variance to realistic range (10% to 40% vol)
    state.variance = np.clip(state.variance, 0.01, 0.16)


def trigger_regime_shift(state: VSRState, rng: np.random.RandomState) -> None:
    """Trigger regime shift by modifying volatility parameters.
    
    Simulates market stress events where volatility spikes or crashes.
    Randomly chooses between vol_spike and vol_crash scenarios.
    
    Args:
        state: Current VSRState (modified in place)
        rng: Seeded numpy RandomState for reproducibility
    
    Requirements: 21.4
    """
    shift_type = rng.choice(["vol_spike", "vol_crash"])
    
    if shift_type == "vol_spike":
        # Volatility increases 20-40%
        multiplier = rng.uniform(1.2, 1.4)
        state.variance *= multiplier
        state.regime = "high_vol"
    else:
        # Volatility decreases 20-30%
        multiplier = rng.uniform(0.7, 0.8)
        state.variance *= multiplier
        state.regime = "low_vol"
    
    # Ensure variance stays in valid range
    state.variance = np.clip(state.variance, 0.01, 0.16)