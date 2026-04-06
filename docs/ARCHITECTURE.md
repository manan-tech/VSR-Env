# VSR-Env System Architecture Guide

This document provides a deep dive into the internal architecture of **VSR-Env**, detailing the design of its simulation engine, reward system, and data models.

## 1. Engine Layer (`vsr_env/engine/`)

The simulation engine is designed for high performance using vectorized operations.

### A. Market Dynamics (`market_sim.py`)
Responsible for advancing the state of the market.
- **Spot Price**: Geometric Brownian Motion (GBM).
- **Variance**: Mean-reverting Ornstein-Uhlenbeck (OU) process.
- **Regime Shifts**: Simulation of "high_vol" and "low_vol" market conditions.

### B. Option Chain Engine (`option_chain.py`)
The quantitative core of the environment.
- **Vectorized Black-Scholes**: Computes prices and Greeks for an $8 \times 3$ grid (8 strikes, 3 maturities).
- **IV Solving**: Hybrid numerical methods (Newton-Raphson with Brent's fallback).
- **Surface Generation**: realistic smile/skew generation with Gaussian noise.

### C. Portfolio Management (`portfolio.py`)
Tracks the agent's inventory and risk exposures.
- **Greek Aggregation**: Sums Delta, Gamma, Vega, and Theta across all open positions.
- **Mark-to-Market (MtM)**: Re-evaluates P&L after every market move.

---

## 2. Reward Layer (`vsr_env/reward/`)

The reward system is modular, providing structured feedback across three core tasks with increasing complexity.

### A. Reward Computer (`reward_computer.py`)
The [RewardComputer](file:///Users/mananbansal/Desktop/meta/vsr_env/reward/reward_computer.py) calculates decomposed rewards:
- **Greek Reward**: Incentivizes delta-neutrality (|delta| < 0.05). In Expert tasks, this uses a linear sensitivity lookup `max(0, 1.0 - delta / 0.5)`.
- **Vega Reward**: Rewards correct positioning (short vega) before expected volatility crush events in Hard tasks.
- **P&L Component**: Sigmoid-normalized profit/loss signal, adjusted for time decay (theta) costs in expert tasks.
- **Reasoning Component**: Evaluation of the agent's logic via keyword hits and numeric consistency scoring.

---

## 3. Task Logic (`vsr_env/tasks/`)

Each task defines its own initialization and grading logic, which is aggregated into the final episode score.

### A. Delta Hedging (Medium) - `delta_hedging.py`
- **Mechanism**: Random "market shocks" (spot price jumps) occur at Step 3.
- **Challenge**: The agent must re-balance the portfolio quickly to remain neutral.
- **Grading**: Weighted combination of neutralization quality and transaction cost efficiency.

### B. Earnings Vol Crush (Hard) - `earnings_vol_crush.py`
- **Mechanism**: Implied volatility is artificially elevated at reset. At a hidden step (3-6), a "crush" event drops IV by 30-50%.
- **Challenge**: Anticipate the crush by positioning short vega while managing the resulting delta.
- **Grading**: Focuses on pre-crush vega exposure and post-crush re-hedging.

### C. Gamma Scalping (Expert) - `gamma_scalping.py`
- **Mechanism**: The agent starts with a long ATM straddle (maximum gamma). Spot price oscillates ±2-3% every step.
- **Challenge**: Lock in "gamma profits" by selling into strength and buying into weakness, balancing hedging costs against theta decay.
- **Grading**: Detailed correlation analysis between spot moves and hedge timing.

---

## 4. Data Models (`vsr_env/models.py`)

Standardized Pydantic models ensure compliance with the OpenEnv specification and provide a clear interface for the agent.

- **`VSRAction`**: Inputs from the agent, including the critical `reasoning` field.
- **`VSRObservation`**: Market snapshot including `expected_outcome` for steering.
- **`VSRState`**: Internal ground truth, including `initial_delta`, `initial_theta`, and task-event schedules (`vol_crush_step`, `regime_shift_step`).
- **`VSRReward`**: Decomposed reward signal for granular analysis.

---

## 5. Server Layer (`vsr_env/server/`)

A FastAPI wrapper that exposes the environment through a RESTful API.
- **Endpoints**: `/obs`, `/step`, `/reset`, `/health`, `/state`.
- **Validation**: Strict schema validation for all incoming actions.

---

## 6. Mathematical Summary

### Stochastic Differential Equations (SDEs)
1. **Underlying Asset**: $dS_t = \mu S_t dt + \sigma_t S_t dW_t$
2. **Variance**: $dV_t = \theta (\bar{V} - V_t) dt + \xi dW_{V,t}$

### Model Triggers
- **Volatility Crush**: A discrete jump in $\sigma_t$ (simulating earnings).
- **Hedge-Neutrality**: The agent's goal is to minimize $|\Delta_p| = \left| \sum_{i=1}^n q_i \frac{\partial O_i}{\partial S} \right|$.

### Black-Scholes Formula
- **Call Price**: $C(S, K, T, \sigma, r) = S N(d_1) - K e^{-rT} N(d_2)$
- **$d_1$**: $\frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$
- **$d_2$**: $d_1 - \sigma\sqrt{T}$

---

> [!IMPORTANT]
> The environment constants (Strikes: 85-110, Maturities: 30-180 days) are defined in `option_chain.py` and should be referenced by their indices.
