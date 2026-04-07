# VSR-Env: Volatility Surface Reasoning Environment

VSR-Env is an OpenEnv-compliant reinforcement learning environment that simulates options portfolio management on implied volatility surfaces. It targets the Meta PyTorch OpenEnv Hackathon and provides a realistic simulation of quantitative trading workflows used in the $600T+ notional derivatives market.

## Technical Documentation

For a deep dive into the environment's internals, please refer to the following documents:
- [**Architecture Guide**](docs/ARCHITECTURE.md): Detailed explanation of the engine, reward systems, and server layers.
- [**Walkthrough**](walkthrough.md): A technical summary covering simulation maths, reward logic, and Docker build.

## The VSR-Env Advantage: Moving Beyond P&L

VSR-Env bridges the gap between pure quantitative finance and high-level LLM reasoning. Unlike standard trading environments that focus solely on P&L, VSR-Env requires agents to **synthesize complex, high-dimensional volatility data into coherent trade theses**. 

## Difficulty Matrix

Our environment features a strict, 5-tier adaptive curriculum designed to thoroughly benchmark an agent's quantitative reasoning capabilities:

| Tier | Task | Skill Tested | Max Steps | Expected LLM Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | `vol_regime_detection` | Analytical mapping of Implied Volatility parameters | 1 | ~0.90 |
| **Medium** | `delta_hedging` | Neutralizing directional Greek exposures through shocks | 5 | ~0.65 |
| **Hard** | `earnings_vol_crush` | Predicting and trading short Vega into scheduled crush events | 8 | ~0.40 |
| **Expert** | `gamma_scalping` | Dynamic path-dependent spot re-hedging against Theta bleed | 10 | ~0.15 |
| **Super-Boss** | `vega_gamma_stress` | Surviving massive dual-shock scenarios using strict SD bounds | 10 | ~0.05 |

## Mathematical Foundations

VSR-Env is built on robust quantitative finance principles:

- **Spot Price (GBM)**: $dS_t = \mu S_t dt + \sigma_t S_t dW_t$
- **Variance (OU)**: $dV_t = \theta (\bar{V} - V_t) dt + \xi dW_{V,t}$
- **Option Pricing**: Black-Scholes-Merton model for call/put prices and Greeks.
- **IV Surface**: Modeled with log-moneyness skew and term structure.

## Overview

VSR-Env enables LLM agents to act as junior options traders, performing five core tasks with increasing difficulty in a progressive curriculum:

1. **Vol Regime Detection (Easy)**: Identify the pure variance regime analytically from the IV surface.
2. **Delta Hedging (Medium)**: Neutralize portfolio directionality by managing Delta exposures through market shocks.
3. **Earnings Vol Crush (Hard)**: Position for and recover from massive volatility drops (30-50% IV crush) via Vega management.
4. **Gamma Scalping (Expert)**: Profit from path-dependent spot oscillations by dynamically re-hedging a high-gamma position against time decay.
5. **Vega-Gamma Stress (Super-Boss)**: Pre-emptively stabilize deeply negative convexity (Vega/Gamma) before a random, catastrophic dual-market shock crashes the portfolio.

### Real-World Utility

This environment simulates genuine quantitative trading workflows used at major options desks:

- **Volatility Surface Analysis**: Traders analyze implied volatility surfaces to identify mispricings and construct delta-neutral trades.
- **Risk Management**: Portfolio managers hedge delta exposure to remain direction-neutral, especially through market events.
- **Advanced Strategies**: Agents must manage complex exposures like **Vega** (volatility sensitivity) and **Gamma** (rate of change of delta), matching the requirements of professional trading systems.

The environment uses Black-Scholes pricing, Greeks calculation, and realistic IV surface generation with skew and term structure.

## Action Space

The agent interacts with the environment through `VSRAction`, which specifies trades on the option chain:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `selected_strike` | int | 0-7 | Strike index into STRIKES array: [85, 90, 95, 97.5, 100, 102.5, 105, 110] |
| `selected_maturity` | int | 0-2 | Maturity index into MATURITIES array: [30, 90, 180] days |
| `direction` | enum | buy, sell, hold | Trade direction |
| `quantity` | float | 0.0-10.0 | Trade size in contracts |
| `reasoning` | string | any | Agent's analysis and trade thesis |

### Action Validation

The environment validates actions before execution:
- Strike index must be in range [0, 7]
- Maturity index must be in range [0, 2]
- Quantity must be non-negative and ≤ 10.0
- Direction must be one of: buy, sell, hold
- Hold actions should have quantity = 0

Invalid actions return an error message in `last_action_error` without modifying portfolio state.

## Observation Space

The agent receives comprehensive market state through `VSRObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `iv_surface` | List[List[float]] | 8×3 implied volatility matrix (strikes × maturities) |
| `spot_price` | float | Current underlying price |
| `portfolio_greeks` | Dict[str, float] | Portfolio Greeks: delta, gamma, vega, theta |
| `portfolio_pnl` | float | Cumulative profit/loss |
| `portfolio_positions` | List[Dict] | List of current open positions |
| `market_sentiment` | float | Market sentiment indicator in [-1.0, 1.0] |
| `step_number` | int | Current step in episode |
| `steps_remaining` | int | Steps until episode end |
| `task_name` | str | Current task identifier |
| `task_description` | str | Task objective description |
| `last_action_error` | Optional[str] | Validation error from last action (if any) |
| `expected_outcome` | Optional[str] | Ground-truth expected outcome for the task |

## Grading Transparency

A full breakdown of how tasks are graded and penalized mathematically is available in [docs/GRADING.md](docs/GRADING.md).

## Installation & Deployment (OpenEnv CLI)

### 1. Dependencies
- Python 3.11+
- Docker (optional but recommended for evaluating)

### 2. Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd vsr-env

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. OpenEnv Integration
To test and push this workload into the OpenEnv evaluating cloud:
```bash
# Initialize OpenEnv
openenv init

# Test the manifest locally
openenv test

# Push the environment workload to registry
openenv push --repo-id <username>/vsr-env
```

## Evaluation & Inference

The `inference.py` script evaluates an LLM agent sequentially across all 5 tasks, enforcing the adaptive curriculum. If an agent fails to score at least 0.3 on an easier task, the curriculum breaks early.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-huggingface-token"

# Run progressive inference
python inference.py
```

### Inference Terminal Trace Example

The server natively logs the strictly formatted OpenEnv terminal traces:

```
[START] task=vol_regime_detection env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=hold(0,0,0.0) reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=delta_hedging env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=sell(4,1,2.0) reward=0.65 done=false error=null
...
[STEP] step=5 action=hold(0,0,0.0) reward=0.10 done=true error=null
[END] success=true steps=5 score=0.85 rewards=0.65,...

[START] task=earnings_vol_crush env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=sell(4,2,5.0) reward=0.55 done=false error=null
...
```

## Global Telemetry

The environment exposes a global `/telemetry` endpoint that tracks episodic trajectories, Greeks over time, portfolio PnL, and step rewards natively to power diagnostic plotting.

## License

MIT License

## Acknowledgments

Built for the Meta PyTorch OpenEnv Hackathon.
