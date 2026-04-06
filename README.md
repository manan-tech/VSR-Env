# VSR-Env: Volatility Surface Reasoning Environment

VSR-Env is an OpenEnv-compliant reinforcement learning environment that simulates options portfolio management on implied volatility surfaces. It targets the Meta PyTorch OpenEnv Hackathon and provides a realistic simulation of quantitative trading workflows used in the $600T+ notional derivatives market.

## Technical Documentation

For a deep dive into the environment's internals, please refer to the following documents:
- [**Architecture Guide**](docs/ARCHITECTURE.md): Detailed explanation of the engine, reward systems, and server layers.
- [**Walkthrough**](walkthrough.md): A technical summary covering simulation maths, reward logic, and Docker build.

## The VSR-Env Advantage: Moving Beyond P&L

VSR-Env bridges the gap between pure quantitative finance and high-level LLM reasoning. Unlike standard trading environments that focus solely on P&L, VSR-Env requires agents to **synthesize complex, high-dimensional volatility data into coherent trade theses**. 

It challenges agents to move beyond simple "buy low, sell high" logic into the realm of multi-asset Greeks management, regime-shift adaptation, and qualitative reasoning—critical skills for next-generation AI financial assistants that are rarely captured in existing benchmarks.

## Mathematical Foundations

VSR-Env is built on robust quantitative finance principles:

- **Spot Price (GBM)**: $dS_t = \mu S_t dt + \sigma_t S_t dW_t$
- **Variance (OU)**: $dV_t = \theta (\bar{V} - V_t) dt + \xi dW_{V,t}$
- **Option Pricing**: Black-Scholes-Merton model for call/put prices and Greeks.
- **IV Surface**: Modeled with log-moneyness skew and term structure.

## Overview

VSR-Env enables LLM agents to act as junior options traders, performing three core tasks with increasing difficulty:

1. **Delta Hedging (Medium)**: Neutralize portfolio risk by managing Greek exposures through market shocks.
2. **Earnings Vol Crush (Hard)**: Position for and recover from massive volatility drops (30-50% IV crush).
3. **Gamma Scalping (Expert)**: Profit from path-dependent spot oscillations by dynamically re-hedging a high-gamma position.

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

## Tasks

### Task 1: Delta Hedging (Medium)

**Objective**: Neutralize portfolio delta to within ±0.05 while minimizing transaction costs during automated market shocks.

- **Max Steps**: 5
- **Difficulty**: Medium
- **Grading (Episode)**: Score = neutralization_quality × 0.7 + cost_efficiency × 0.3
- **Per-Step Reward**: delta_improvement × 0.5 + cost_efficiency × 0.3 + neutrality_bonus (0.1 if |delta| < 0.05) + reasoning_coherence × 0.2

### Task 2: Earnings Vol Crush (Hard)

**Objective**: Position the portfolio short vega before an earnings event (vol crush) and re-hedge delta after the event.

- **Max Steps**: 8
- **Difficulty**: Hard
- **Grading (Episode)**: Score = pre_crush_positioning × 0.40 + post_crush_rehedge × 0.35 + pnl_outcome × 0.25
- **Per-Step Reward**: pnl_change × 0.4 + greek_neutrality × 0.3 + reasoning_quality × 0.3

### Task 3: Gamma Scalping (Expert)

**Objective**: Profit from spot price oscillations by dynamically re-hedging a long ATM straddle position (high gamma).

- **Max Steps**: 10
- **Difficulty**: Expert
- **Grading (Episode)**: Score = rehedge_quality × 0.40 + pnl_above_theta × 0.35 + timing_score × 0.25
- **Per-Step Reward**: delta_neutrality × 0.5 + pnl_change × 0.3 + reasoning_quality × 0.2

## Installation

### Prerequisites

- Python 3.11+
- pip

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd vsr-env

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Docker Build

```bash
# Build the Docker image
docker build -t vsr-env:latest .

# Run the container
docker run -p 8000:8000 vsr-env:latest
```

The server will start on `http://localhost:8000`.

## Usage

### API Endpoints

The environment exposes a FastAPI server with the following endpoints:

#### Reset

```bash
curl -X POST http://localhost:8000/reset?task_name=delta_hedging&seed=42
```

Response:
```json
{
  "observation": {
    "iv_surface": [[0.18, 0.19, 0.20], ...],
    "spot_price": 100.0,
    "portfolio_greeks": {"delta": 0.5, "gamma": 0.01, "vega": 0.05, "theta": -0.02},
    "portfolio_pnl": 0.0,
    "portfolio_positions": [],
    "market_sentiment": 0.0,
    "step_number": 0,
    "steps_remaining": 5,
    "task_name": "delta_hedging",
    "task_description": "Neutralize portfolio delta to within ±0.05",
    "last_action_error": null
  }
}
```

#### Step

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "selected_strike": 2,
    "selected_maturity": 0,
    "direction": "sell",
    "quantity": 1.0,
    "reasoning": "Reducing long delta exposure by selling calls"
  }'
```

Response:
```json
{
  "observation": {...},
  "reward": {"total": 0.65, "pnl_component": 0.0, "greek_component": 0.65, "reasoning_component": 0.0},
  "done": false,
  "info": {}
}
```

#### State

```bash
curl http://localhost:8000/state
```

Response:
```json
{
  "state": {
    "episode_id": "...",
    "step_count": 1,
    "task_name": "delta_hedging",
    "regime": "normal",
    "spot_price": 100.5,
    ...
  }
}
```

### Running Inference

The `inference.py` script demonstrates environment usage with an LLM agent:

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-huggingface-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run inference
python inference.py
```

The script runs all three tasks sequentially and outputs progress in the required format:

```
[START] task=delta_hedging env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=sell(2,0,1.0) reward=0.65 done=false error=null
[STEP] step=2 action=sell(3,0,0.5) reward=0.20 done=false error=null
[STEP] step=3 action=hold(0,0,0.0) reward=0.10 done=true error=null
[END] success=true steps=3 score=0.85 rewards=0.65,0.20,0.10
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | API endpoint for the LLM |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for inference |
| `HF_TOKEN` | Yes | - | Hugging Face API key (also accepts `API_KEY`) |
| `IMAGE_NAME` | No | `vsr-env:latest` | Docker image name for deployment |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Architecture

```
vsr-env/
├── vsr_env/
│   ├── __init__.py
│   ├── models.py           # Pydantic models (VSRAction, VSRObservation, etc.)
│   ├── engine/
│   │   ├── option_chain.py # Black-Scholes pricing, Greeks, IV solver
│   │   ├── market_sim.py   # GBM simulation, regime shifts
│   │   └── portfolio.py    # Position tracking, P&L computation
│   ├── tasks/
│   │   ├── delta_hedging.py # Delta Hedging task and grader
│   │   ├── earnings_vol_crush.py # Vol Crush task and grader
│   │   └── gamma_scalping.py # Gamma Scalping task and grader
│   ├── reward/
│   │   └── reward_computer.py # Per-step reward computation
│   └── server/
│       ├── app.py          # FastAPI application
│       └── vsr_environment.py # Core environment implementation
├── docs/
│   └── ARCHITECTURE.md     # System architecture deep-dive
├── inference.py            # Baseline inference script
├── openenv.yaml            # OpenEnv manifest
├── walkthrough.md          # Technical walkthrough
└── README.md               # This file
```

## Performance

The environment is designed to run on CPU-only infrastructure (vcpu=2, 8GB RAM):

- Single step execution: < 2 seconds
- Full episode (all 3 tasks): < 5 minutes
- All computations use NumPy/SciPy (no GPU dependencies)

## License

MIT License

## Acknowledgments

Built for the Meta PyTorch OpenEnv Hackathon × SST.