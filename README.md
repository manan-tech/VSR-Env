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
| **Difficulty Tiers** | 5 levels (Easy → Super-Boss) with early-stopping curriculum |
| **Episode Length** | 3-20 steps per episode (task-dependent difficulty) |
| **Total Episodes** | 5 (one per task) |
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

---

## 📚 5-Tier Adaptive Curriculum

### Tier 1: **Volatility Regime Detection** (Easy)
- **Max Steps**: 3
- **Challenge**: Classify IV surface as "low", "normal", or "high" regime through multi-step surface analysis
- **Ambiguity Cases**: Surface may have mixed signals (high short-term IV, low long-term IV) requiring 2-3 classification steps
- **Skills**: Pattern recognition, surface reading, temporal stability assessment

### Tier 2: **Delta Hedging** (Medium)
- **Max Steps**: 8
- **Challenge**: Maintain |delta| < 0.05 through market shock
- **Skills**: Delta neutrality, counter-trading, cost efficiency

### Tier 3: **Earnings Vol Crush** (Hard)
- **Max Steps**: 13
- **Challenge**: Position for IV collapse at step 6, then re-hedge
- **Skills**: Temporal prediction, vega management, event timing

### Tier 4: **Gamma Scalping** (Expert)
- **Max Steps**: 17
- **Challenge**: Profit from delta oscillations with high gamma
- **Skills**: Dynamic re-hedging, theta management, trade timing

### Tier 5: **Vega/Gamma Stress** (Super-Boss)
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
- **Earnings vol crush** at step 6: IV drops 40% instantly
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
| **[TASKS.md](TASKS.md)** | 5-tier curriculum with mathematical objectives |
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
# Run integration tests (3 test files)
pip install pytest
python test_integration.py      # Full episode tests for all 5 tasks
python test_client.py           # WebSocket client validation
python test_ws.py               # WebSocket protocol test

# Validate OpenEnv spec compliance
openenv validate
```

**Test Suite: 3 Integration Tests**
- ✅ `test_integration.py` — End-to-end validation of all 5 tasks with full episodes
- ✅ `test_client.py` — LocalVSREnv client reset/step operations
- ✅ `test_ws.py` — WebSocket protocol compliance

**OpenEnv Validation:**
```bash
$ openenv validate
[OK] VSR-Env: Ready for deployment
     ✓ Valid openenv.yaml manifest
     ✓ All 5 tasks callable
     ✓ WebSocket endpoint functional
     ✓ Action/Observation schemas valid

$ openenv validate --url https://huggingface.co/spaces/MananBansal/VSR-Env
[OK] Remote validation passed (6/6 criteria)
```

**Deployment Status:**
- ✅ Docker container builds successfully
- ✅ Hugging Face Space verified operational
- ✅ All endpoints accessible via HTTP/WebSocket

---

## 📊 Benchmark Results

Comprehensive evaluation across 5 frontier models:

| Model | Task 1<br/>Easy (3 steps) | Task 2<br/>Medium (8 steps) | Task 3<br/>Hard (13 steps) | Task 4<br/>Expert (17 steps) | Task 5<br/>Super-Boss (20 steps) | Overall<br/>Success |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kimi K2.5** | 0.990 ✓ | 0.915 ✓ | 0.400 ✓ | 0.683 ✓ | 0.221 ✓ | **100%** (5/5) |
| **Llama 3.3 70B Versatile** | 0.990 ✓ | 0.458 ✓ | 0.650 ✓ | 0.491 ✓ | 0.200 ✓ | **100%** (5/5) |
| **Mistral Large 3** | 0.990 ✓ | 0.253 ✓ | 0.400 ✓ | 0.623 ✓ | 0.207 ✓ | **100%** (5/5) |
| **Amazon Nova Pro v1** | 0.990 ✓ | 0.671 ✓ | 0.650 ✓ | 0.042 ✗ | 0.151 ✓ | **80%** (4/5) |
| **Llama 3.1 8B Instant** | 0.010 ✗ | 0.575 ✓ | 0.650 ✓ | 0.544 ✓ | 0.100 ✓ | **80%** (4/5) |

**Key Findings:**
- **3 models achieve 100% success**: Kimi K2.5, Llama 3.3 70B, Mistral Large 3
- **Kimi K2.5 leads average score** at 0.642 with strongest delta hedging (0.915)
- **Llama 3.3 70B balances** all tasks well (0.558 avg), strong on earnings (0.650)
- **Llama 3.1 8B fails Easy tier** (0.010 on vol detection) — only model to fail Task 1
- **Nova Pro struggles with Expert tier** (gamma scalping: 0.042) despite strong delta hedging
- **Super-Boss tier challenges all models** — scores range 0.100–0.221
- **Expert tier shows 16× variance** between best (Kimi: 0.683) and worst (Nova: 0.042)

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