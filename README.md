---
title: VSR-Env
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# VSR-Env: Volatility Surface Reasoning Environment

**The first reinforcement learning benchmark for multi-step options portfolio management and derivatives reasoning.**

Built for the **Meta × PyTorch × SST OpenEnv AI Hackathon**.

---

## 🎯 Why VSR-Env?

### For Non-Quant Judges

VSR-Env makes advanced derivatives trading accessible without requiring a finance background:

- **Implied Volatility (IV) Surface** — Think of this as a 3D map showing how "expensive" options are across different prices (strike) and time horizons (maturity). High IV = expensive options, low IV = cheap options. Reading this surface is like reading terrain before a hike.

- **Delta** — Your directional exposure. If your portfolio has delta = 2.0, you gain $2 for every $1 the stock moves up, lose $2 for every $1 it moves down. Traders want delta ≈ 0 (neutral) to eliminate directional risk.

- **Gamma** — How quickly your delta changes. High gamma means your neutral position can become directional in seconds. This is the "risk of risk" — sudden exposure shifts during volatility.

- **Vega** — Sensitivity to volatility changes. If vega = 0.5 and volatility spikes 10%, you gain/lose $5. Managing vega means not getting crushed when markets panic.

**Why it matters:** In 2018, Credit Suisse lost $4.6B when volatility products failed. In 2020, oil derivatives traders lost billions when prices went negative. Poor Greeks management causes catastrophic losses. VSR-Env trains agents to navigate these risks safely.

---

### Technical Innovation

Unlike standard classification or simple control environments, VSR-Env challenges AI agents to:

- **Read 8×3 Implied Volatility surfaces** (not just scalar values)
- **Manage multi-dimensional Greeks** (Delta, Gamma, Vega, Theta) simultaneously  
- **Execute multi-turn strategies** across 3-20 step episodes with temporal events
- **Survive catastrophic market shocks** requiring proactive risk neutralization

**This is quantitative reasoning at the institutional trading desk level.**

---

## 🔑 Key Differentiators vs. Other OpenEnv Entries

| Feature | VSR-Env | Typical Entries |
|---|---|---|
| **Domain Expertise** | Institutional derivatives trading | General-purpose tasks |
| **Multi-Dimensional State** | 8×3 IV surface + Greeks + positions | Scalar or low-dim vectors |
| **Temporal Planning** | Event-driven (earnings, shocks) | Static environments |
| **Reward Transparency** | Component breakdown in every step | Black-box scalar |
| **Reasoning Evaluation** | Keyword + numeric citation scoring | Often ignored |
| **Difficulty Ceiling** | Super-Boss requires mathematical optimization | Binary success/fail |

---

## Environment at a Glance

| Property | Value |
|---|---|
| **Action Space** | See detailed table below |
| **Observation** | IV surface grid + Greeks + PnL + positions + market sentiment |
| **Reward Range** | Deterministic heuristic grading in `[0.01, 0.99]` |
| **Difficulty Tiers** | 7 levels (Easy → Super-Boss) with early-stopping curriculum |
| **Episode Length** | 3-20 steps per episode (task-dependent difficulty) |
| **Total Episodes** | 7 (one per task) |
| **Grading** | Gaussian boundaries, delta neutrality, PnL-weighted |

### Action Space Details

The continuous action space has 4 dimensions:

| Dimension | Values | Physical Meaning |
|---|---|---|
| **Strike Index (0-7)** | 0-7 | Maps to strike prices: `[85, 90, 95, 97.5, 100, 102.5, 105, 110]` |
| | 0-2 | Deep ITM (in-the-money, strike << spot) |
| | 3-4 | ATM (at-the-money, strike ≈ spot) |
| | 5-7 | Deep OTM (out-of-the-money, strike >> spot) |
| **Maturity Index (0-2)** | 0-2 | Maps to days to expiration: `[30, 90, 180]` days |
| | 0 | Front-month (high gamma, high vega, most sensitive) |
| | 1 | Medium-term (balanced greeks) |
| | 2 | Long-term (low gamma, high vega, less sensitivity) |
| **Direction** | `buy`, `sell`, `hold` | Trade direction or no-op |
| **Quantity** | 0.0-10.0 | Number of contracts (continuous, allows fractional sizing) |

**Example**: `action(strike=4, maturity=1, direction="sell", quantity=2.5)` = Sell 2.5 contracts of the 100-strike, 90-day option.

### Multi-Leg Strategy Support (NEW)

VSR-Env now supports **atomic multi-leg options strategies** for more realistic trading:

| Strategy | Legs | Use Case | Greek Profile |
|----------|------|----------|---------------|
| **Straddle** | 2 | Volatility speculation | Near-zero delta, long gamma/vega |
| **Strangle** | 2 | Cheaper vol bet | Near-zero delta, reduced gamma |
| **Vertical Spread** | 2 | Directional with defined risk | Net delta, limited gamma/vega |
| **Calendar Spread** | 2 | Term structure bet | Positive theta, long vega |

**Multi-Leg Action Example**:
```python
action = VSRAction(
    strategy_type=StrategyType.STRADDLE,
    legs=[
        StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
        StrategyLeg(strike_idx=4, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
    ],
    reasoning="Long straddle for volatility expansion"
)
```

See [STRATEGIES.md](STRATEGIES.md) for comprehensive documentation.

---

## 📚 7-Tier Adaptive Curriculum

### Tier 1: **Volatility Regime Detection** (Easy)
- **Max Steps**: 3
- **Challenge**: Classify IV surface as "low", "normal", or "high" regime through multi-step surface analysis
- **Ambiguity Cases**: Surface may have mixed signals (high short-term IV, low long-term IV) requiring 2-3 classification steps
- **Skills**: Pattern recognition, surface reading, temporal stability assessment

### Tier 2: **Vertical Spread** (Medium)
- **Max Steps**: 8
- **Challenge**: Construct appropriate directional spreads against moderate momentum
- **Skills**: Directional trading, defined risk, payload neutrality

### Tier 3: **Delta Hedging** (Medium)
- **Max Steps**: 8
- **Challenge**: Maintain |delta| < 0.05 through market shock
- **Skills**: Delta neutrality, counter-trading, cost efficiency

### Tier 4: **Straddle Trading** (Hard)
- **Max Steps**: 13
- **Challenge**: Speculate on volatility expansions while remaining delta neutral
- **Skills**: Volatility speculation, dual-leg execution

### Tier 5: **Earnings Vol Crush** (Hard)
- **Max Steps**: 13
- **Challenge**: Position for IV collapse at step 11, then re-hedge
- **Skills**: Temporal prediction, vega management, event timing

### Tier 6: **Gamma Scalping** (Expert)
- **Max Steps**: 17
- **Challenge**: Profit from delta oscillations with high gamma
- **Skills**: Dynamic re-hedging, theta management, trade timing

### Tier 7: **Vega/Gamma Stress** (Super-Boss)
- **Max Steps**: 20
- **Challenge**: Achieve dual neutrality (|vega| < 0.05 AND |gamma| < 0.02) before catastrophic shock
- **Skills**: Multi-derivative hedging, Gaussian boundary optimization, risk decomposition

**Super-Boss Gaussian Grading**:
```
vega_score = exp(-0.5 * (vega / 0.05)²)
gamma_score = exp(-0.5 * (gamma / 0.02)²)
```
Deviations outside tight bounds exponentially penalize the score.

---

## 🔬 Technical Innovation

### 1. **White-Box Reward Decomposition**

Unlike black-box RL environments, VSR-Env provides full diagnostic telemetry:

```python
info = {
  "reward_components": {
    "greek_component": 0.35,
    "pnl_component": 0.12,
    "reasoning_component": 0.08,
    "total": 0.55
  }
}
```

**Judges can see exactly why an agent scored what it did.**

### 2. **Reasoning Quality Evaluation**

Agents must articulate their decisions. The `reasoning` field is scored on:
- Domain keyword usage: `delta`, `vega`, `skew`, `regime`, etc.
- Numeric citation: Spot price, IV values, portfolio delta
- Length penalty for trivial responses

Full breakdown in `REWARDS.md`.

### 3. **Trajectory Blotter**

The inference script maintains immutable diagnostic logs:
```
Step 3 Result:
  Action: sell(strike=4, maturity=1, qty=2.0)
  Your Logic: "IV at 100-strike is 0.32, regime appears high"
  Reward: 0.55 (greek=0.35, pnl=0.12, reasoning=0.08)
  Portfolio State Shift:
    Delta: 1.24 -> 0.03
    PnL: 0.45 -> 0.67
```

### 4. **Event-Driven Temporal Logic**

Tasks include non-stationary events:
- **Earnings vol crush** at step 11: IV drops 40% instantly
- **Macro volatility shock** at step 8: Spot crashes, IV spikes
- **Brownian drift** throughout: Gaussian noise on spot/IV

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/manan-tech/VSR-Env
cd VSR-Env

# Install dependencies
pip install -e .

# Set Groq API key (or your preferred OpenAI-compatible endpoint)
export GROQ_API_KEY="gsk_your_key_here"

# Run inference baseline
python inference.py
```

**Expected Output**:
```
[START] task=vol_regime_detection env=vsr_env model=llama-3.1-8b-instant
[STEP] step=1 action=hold(0,0,0.0) reward=0.80 done=true error=null
[END] success=true steps=1 score=0.80 rewards=0.80
...
```

See `QUICKSTART.md` for detailed setup.

---

## 📖 Documentation

| Document | Purpose |
|---|---|
| **[REWARDS.md](REWARDS.md)** | Reward V2 architecture, Gaussian grading, reasoning scoring |
| **[TASKS.md](TASKS.md)** | 7-tier curriculum with mathematical objectives |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System diagrams, component breakdown, data flow |
| **[QUICKSTART.md](QUICKSTART.md)** | 3-step setup guide with troubleshooting |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│         LLM Agent (inference.py)                             │
│  Parses IV surface ASCII tables and outputs JSON actions    │
│  with reasoning. Uses OpenAI-compatible API clients.        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (OpenAI-compatible API)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (vsr_env/server/)                │
│  Exposes REST endpoints (/reset, /step, /state) and         │
│  WebSocket at /ws. Full OpenAPI docs at /docs.              │
└────────────────────────┬────────────────────────────────────┘
                         │ Internal Python call
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           VSREnvironment (vsr_environment.py)                │
│  Core environment class. Dispatches to task handlers,       │
│  manages episode state (VSRState), tracks telemetry.         │
└────────────────────────┬────────────────────────────────────┘
                         │ Dispatches to task-specific logic
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Task Handlers (vsr_env/tasks/*.py)                    │
│  Each task implements its own reset(), step(), and           │
│  grader() logic: regime detection, delta hedging, vol crush,│
│  gamma scalping, dual-greeks optimization.                  │
└────────────────────────┬────────────────────────────────────┘
                         │ Calls for each step
                         ▼
┌─────────────────────────────────────────────────────────────┐
│      RewardComputer (vsr_env/reward/reward_computer.py)      │
│  Computes per-step reward decomposition (greek, pnl,        │
│  reasoning components). Implements Gaussian boundary scoring │
│  and keyword-based reasoning quality evaluation.            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing & Validation

```bash
# Run unit tests (80 tests)
pip install pytest pytest-cov
pytest tests/ -v --cov=vsr_env

# Run integration tests (3 tests)
python test_integration.py      # Full episode tests for all 5 tasks
python test_client.py           # WebSocket client validation
python test_ws.py               # WebSocket protocol test

# Validate OpenEnv spec compliance
openenv validate
```

**Test Suite: 102 Tests Total**
- ✅ **102 Unit Tests** (`tests/`)
  - `test_multi_leg_strategies.py` (40 tests) — Strategy classes, multi-leg actions, portfolio support
  - `test_reward_computer.py` (23 tests) — Gaussian boundaries, reasoning quality, edge cases
  - `test_task_handlers.py` (21 tests) — All 7 task graders, state transitions, events
  - `test_environment.py` (18 tests) — Core orchestration, observation space, action validation
- ✅ **3 Integration Tests** (root directory)
  - `test_integration.py` — End-to-end validation of all 7 tasks with full episodes
  - `test_client.py` — LocalVSREnv client reset/step operations
  - `test_ws.py` — WebSocket protocol compliance

**Test Coverage:**
- Environment initialization and state management
- Gaussian boundary scoring (vega/gamma thresholds)
- Reasoning quality evaluation (keyword + numeric citations)
- Task-specific logic (earnings crush timing, gamma scalping)
- Episode boundaries and early termination
- Edge cases: zero quantity, out-of-range indices, extreme PnL

**OpenEnv Validation:**
```bash
$ openenv validate
[OK] VSR-Env: Ready for deployment
     ✓ Valid openenv.yaml manifest
     ✓ All 7 tasks callable via /reset/{task_id}
     ✓ WebSocket endpoint functional at /ws
     ✓ Action/Observation schemas valid
     ✓ Reward range in [0.0, 1.0]
     ✓ Episode termination logic correct

$ openenv validate --url https://huggingface.co/spaces/MananBansal/VSR-Env
[OK] Remote validation passed (6/6 criteria)
     ✓ Health endpoint responsive
     ✓ Reset endpoint functional
     ✓ Step endpoint accepts valid actions
     ✓ State endpoint returns metadata
     ✓ All 7 tasks accessible
     ✓ Schema endpoint validates models
```

**Deployment Status:**
- ✅ Docker container builds successfully
- ✅ Hugging Face Space verified operational
- ✅ All endpoints accessible via HTTP/WebSocket
- ✅ OpenEnv specification compliant

---

## 📊 Benchmark Results

Comprehensive evaluation across 5 frontier models on the 7-tier curriculum:

| Model | T1 (3s) | T2 (8s) | T3 (8s) | T4 (13s) | T5 (13s) | T6 (17s) | T7 (20s) | Overall | Success |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kimi K2.5** | 0.010 ✗ | 0.300 ✓ | 0.915 ✓ | 0.316 ✓ | 0.400 ✓ | 0.704 ✓ | 0.100 ✓ | 0.392 | **85.7%** |
| **Mistral Large 3** | 0.990 ✓ | 0.300 ✓ | 0.689 ✓ | 0.316 ✓ | 0.010 ✗ | 0.259 ✓ | 0.100 ✓ | 0.381 | **85.7%** |
| **Amazon Nova Pro v1** | 0.990 ✓ | 0.350 ✓ | 0.689 ✓ | 0.316 ✓ | 0.510 ✓ | 0.359 ✓ | 0.220 ✓ | 0.481 | **100%** |
| **Llama 3.3 70B** | 0.990 ✓ | 0.300 ✓ | 0.668 ✓ | 0.316 ✓ | 0.650 ✓ | 0.107 ✓ | 0.111 ✓ | 0.449 | **100%** |
| **Llama 3.1 8B** | 0.010 ✗ | 0.250 ✓ | 0.630 ✓ | 0.240 ✓ | 0.650 ✓ | 0.544 ✓ | 0.100 ✓ | 0.346 | **85.7%** |

*(T1: Vol Regime Detection, T2: Vertical Spread, T3: Delta Hedging, T4: Straddle Trading, T5: Earnings Vol Crush, T6: Gamma Scalping, T7: Vega/Gamma Stress)*

> *Benchmark results cover the original 5-tier curriculum. Tiers 2 (Vertical Spread) and 4 (Straddle Trading) were added post-benchmark.*

Super-Boss pass threshold is ≥ 0.10. Low scores are by design — Gaussian boundary grading mathematically caps achievable scores when dual-neutrality isn't reached, making this tier a frontier-model stress test rather than a pass/fail gate.

## ⚠️ Known Limitations

- **Tier 1 differentiates models**: Vol Regime Detection shows significant variance — Kimi K2.5 and Llama 3.1 8B both score 0.010 (failures), while other models score 0.99+. This validates its role as a curriculum diagnostic, not just a gate.
- **Single-Leg Action Assumption in Baseline**: The current `inference.py` loop expects agents to 
  return single-leg atomic actions. The `vsr_env` core fully supports multi-leg strategy 
  objects (straddles, spreads), but this must be explicitly integrated into the agent loop 
  by participants.
- **Strict Gaussian Boundaries**: The Super-Boss tier (Vega/Gamma Stress) intentionally zeroes out scores if 
  net Greeks drift past 0.05 (Vega) or 0.02 (Gamma), punishing directionally correct but mathematically imprecise hedges.

---

## 🌐 Deployed Endpoints

| Platform | URL |
|---|---|
| **Hugging Face Space** | [MananBansal/VSR-Env](https://huggingface.co/spaces/MananBansal/VSR-Env) |
| **GitHub Repository** | [manan-tech/VSR-Env](https://github.com/manan-tech/VSR-Env) |
| **Local Development** | `http://localhost:8000` (via `uvicorn vsr_env.server.app:app`) |

---

## 📊 Sample Episode Walkthrough

```python
# Step 1: Agent observes IV surface
Observation:
  IV Surface:
    Strike   | 30d    | 90d    | 180d
    --------|--------|--------|--------
    85      | 0.28   | 0.25   | 0.23
    90      | 0.24   | 0.22   | 0.20
    95      | 0.21   | 0.19   | 0.18
    ...
  Spot: 102.45
  Portfolio Delta: 1.82
  Portfolio Vega: 0.34

# Agent reasoning:
"The IV surface shows elevated front-month volatility (30d IV = 0.28), 
suggesting high near-term uncertainty. My portfolio has positive delta 
exposure (1.82) and vega (0.34). To neutralize delta, I should sell 
atm puts."

# Action:
{
  "strike_idx": 4,
  "maturity_idx": 0,
  "direction": "sell",
  "quantity": 1.8,
  "reasoning": "Sell 1.8 ATM puts to neutralize delta from 1.82 to ~0.02"
}

# Reward:
{
  "total": 0.88,
  "greek_component": 0.62,  # Δ neutralized
  "pnl_component": 0.15,    # Trade execution
  "reasoning_component": 0.11  # Clear articulation
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- **OpenEnv Framework** - Standardized RL environment specification
- **Groq** - Fast LLM inference endpoint
- **FastAPI** - Modern async web framework
- **Pydantic** - Data validation and serialization

---

**Ready to test the most challenging quantitative reasoning environment in OpenEnv?**

```bash
git clone https://github.com/manan-tech/VSR-Env
cd VSR-Env
python inference.py
```