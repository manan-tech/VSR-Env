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

Unlike standard classification or simple control environments, VSR-Env challenges AI agents to:

- **Read 8×3 Implied Volatility surfaces** (not just scalar values)
- **Manage multi-dimensional Greeks** (Delta, Gamma, Vega, Theta) simultaneously  
- **Execute multi-turn strategies** across 8-15 step episodes with temporal events
- **Survive catastrophic market shocks** requiring proactive risk neutralization

**This is quantitative reasoning at the institutional trading desk level.**

---

## Environment at a Glance

| Property | Value |
|---|---|
| **Action Space** | Continuous: strike (0-7), maturity (0-2), direction (buy/sell/hold), quantity (0-10) |
| **Observation** | IV surface grid + Greeks + PnL + positions + market sentiment |
| **Reward Range** | Deterministic heuristic grading in `[0.01, 0.99]` |
| **Difficulty Tiers** | 5 levels (Easy → Super-Boss) with early-stopping curriculum |
| **Episode Length** | 3-20 steps per episode (task-dependent difficulty) |
| **Total Episodes** | 5 (one per task) |
| **Grading** | Gaussian boundaries, delta neutrality, PnL-weighted |

---

## 📚 5-Tier Adaptive Curriculum

### Tier 1: **Volatility Regime Detection** (Easy)
- **Max Steps**: 3
- **Challenge**: Classify IV surface as "low", "normal", or "high" regime
- **Skills**: Pattern recognition, surface reading

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
│                    LLM Agent (inference.py)                  │
│  - Parses IV surface ASCII tables                           │
│  - Outputs JSON actions with reasoning                      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (OpenAI-compatible API)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (vsr_env/server/)                │
│  - /reset, /step, /state endpoints                          │
│  - WebSocket support at /ws                                 │
│  - OpenAPI docs at /docs                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ Internal call
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           VSREnvironment (vsr_environment.py)                │
│  - Task dispatcher                                          │
│  - State management (VSRState)                              │
│  - Telemetry tracking                                       │
└────────────────────────┬────────────────────────────────────┘
                         │ Dispatches to
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Task Handlers (vsr_env/tasks/*.py)                    │
│  - vol_regime_detection.py (regime classification)          │
│  - delta_hedging.py (neutrality tracking)                   │
│  - earnings_vol_crush.py (event timing)                     │
│  - gamma_scalping.py (dynamic hedging)                      │
│  - vega_gamma_stress.py (dual-derivative optimization)      │
└────────────────────────┬────────────────────────────────────┘
                         │ Calls
                         ▼
┌─────────────────────────────────────────────────────────────┐
│      RewardComputer (vsr_env/reward/reward_computer.py)      │
│  - Per-step reward decomposition                            │
│  - Gaussian boundary scoring                                │
│  - Reasoning quality evaluation                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run unit tests
pip install pytest
pytest tests/ -v

# Validate OpenEnv spec compliance
openenv validate
```

**Test Coverage**:
- Environment initialization and state management
- Grading algorithms (Gaussian boundaries, neutrality scoring)
- Reward decomposition (component isolation)
- Task-specific logic (earnings crush timing, gamma scalping)
- Episode boundaries and early termination

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

## 🎓 Citation

If you use VSR-Env in your research, please cite:

```bibtex
@misc{vsr-env-2025,
  title={VSR-Env: A Reinforcement Learning Benchmark for Volatility Surface Reasoning},
  author={Manan Bansal},
  year={2025},
  howpublished={Meta × PyTorch × SST OpenEnv AI Hackathon}
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