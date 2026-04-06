# VSR-Env: Volatility Surface Reasoning Environment

VSR-Env is an OpenEnv-compliant reinforcement learning environment that simulates options portfolio management on implied volatility surfaces. It targets the Meta PyTorch OpenEnv Hackathon and provides a realistic simulation of quantitative trading workflows used in the $600T+ notional derivatives market.

## Overview

VSR-Env enables LLM agents to act as junior options traders, performing three core tasks with increasing difficulty:

1. **IV Reading (Easy)**: Identify mispriced options by analyzing volatility surface anomalies
2. **Delta Hedging (Medium)**: Neutralize portfolio risk by managing Greek exposures cost-efficiently
3. **Arbitrage Capture (Hard)**: Execute full arbitrage workflows including detection, trading, hedging, and profit-taking through regime shifts

### Real-World Utility

This environment simulates genuine quantitative trading workflows used at major options desks:

- **Volatility Surface Analysis**: Traders analyze implied volatility surfaces to identify mispricings and construct delta-neutral trades
- **Risk Management**: Portfolio managers hedge delta exposure to remain direction-neutral
- **Arbitrage Detection**: Quantitative researchers identify and exploit pricing inefficiencies across strikes and maturities

The environment uses Black-Scholes pricing, Greeks calculation, and realistic IV surface generation with skew and term structure - the same mathematical foundations used in professional trading systems.

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

## Tasks

### Task 1: IV Reading (Easy)

**Objective**: Identify 2 deliberately mispriced options on the implied volatility surface.

- **Max Steps**: 3
- **Difficulty**: Easy
- **Grading**: Score = correct_identifications / 2.0, clamped to [0.0, 1.0]
  - 0.5 points per correct identification (correct strike + correct direction)
  - 0.1 points for correct strike but wrong direction
- **Per-Step Reward**: 0.5 for correct identification, 0.1 for partial identification

**Expected Scores**:
- Baseline (random): ~0.25
- Frontier (optimal): 1.0

### Task 2: Delta Hedging (Medium)

**Objective**: Neutralize portfolio delta to within ±0.05 while minimizing transaction costs.

- **Max Steps**: 5
- **Difficulty**: Medium
- **Grading**: Score = neutralization_quality × 0.7 + cost_efficiency × 0.3
  - neutralization_quality = max(0, 1.0 - |final_delta| / |initial_delta|)
  - cost_efficiency = max(0, 1.0 - total_cost / max_cost)
- **Per-Step Reward**: delta_improvement × 0.6 + cost_efficiency × 0.4 + neutrality_bonus (0.1 if |delta| < 0.05)

**Expected Scores**:
- Baseline (random): ~0.30
- Frontier (optimal): ~0.85

### Task 3: Arbitrage Capture (Hard)

**Objective**: Execute a full arbitrage workflow including detection, trading, and hedging through potential regime shifts.

- **Max Steps**: 8
- **Difficulty**: Hard
- **Grading**: Score = pnl_score × 0.4 + neutrality_score × 0.3 + reasoning_score × 0.3
  - pnl_score: Sigmoid-normalized P&L centered at 0
  - neutrality_score: max(0, 1.0 - average_delta / 0.5)
  - reasoning_score: Keyword presence + numeric consistency
- **Per-Step Reward**: pnl_change × 0.4 + greek_quality × 0.3 + reasoning_quality × 0.3
- **Regime Shifts**: Occur at step 4 or 5, modifying volatility parameters by 20-40%

**Expected Scores**:
- Baseline (random): ~0.35
- Frontier (optimal): ~0.80

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
curl -X POST http://localhost:8000/reset?task_name=iv_reading&seed=42
```

Response:
```json
{
  "observation": {
    "iv_surface": [[0.18, 0.19, 0.20], ...],
    "spot_price": 100.0,
    "portfolio_greeks": {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
    "portfolio_pnl": 0.0,
    "portfolio_positions": [],
    "market_sentiment": 0.0,
    "step_number": 0,
    "steps_remaining": 3,
    "task_name": "iv_reading",
    "task_description": "Identify 2 mispriced options on the IV surface",
    "last_action_error": null
  }
}
```

#### Step

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "selected_strike": 4,
    "selected_maturity": 1,
    "direction": "buy",
    "quantity": 1.0,
    "reasoning": "IV appears low for ATM option"
  }'
```

Response:
```json
{
  "observation": {...},
  "reward": {"total": 0.5, "pnl_component": 0.0, "greek_component": 0.0, "identification_component": 0.5, "reasoning_component": 0.0},
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
    "task_name": "iv_reading",
    "true_mispriced_strikes": [2, 5],
    "true_mispriced_directions": {"2": "over", "5": "under"},
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
[START] task=iv_reading env=vsr_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=buy(4,1,1.0) reward=0.50 done=false error=null
[STEP] step=2 action=sell(2,0,0.5) reward=0.10 done=false error=null
[STEP] step=3 action=hold(0,0,0.0) reward=0.00 done=true error=null
[END] success=true steps=3 score=0.50 rewards=0.50,0.10,0.00
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
│   │   ├── iv_reading.py   # IV Reading task and grader
│   │   ├── delta_hedging.py # Delta Hedging task and grader
│   │   └── arb_capture.py  # Arbitrage Capture task and grader
│   ├── reward/
│   │   └── reward_computer.py # Per-step reward computation
│   └── server/
│       ├── app.py          # FastAPI application
│       └── vsr_environment.py # Core environment implementation
├── inference.py            # Baseline inference script
├── openenv.yaml            # OpenEnv manifest
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
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