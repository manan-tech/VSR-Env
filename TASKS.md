# VSR-Env Difficulty Tasks (Adaptive Curriculum)

VSR-Env transitions away from single-mode simulations in favor of a **5-tier adaptive difficulty curriculum**. Models are scored on an ascending scale of complexity; failure on earlier tasks halts the progression.

---

## Curriculum Overview

| Tier | Task Name | Max Steps | Key Skills | Difficulty |
|---|---|---|---|---|
| 1 | **Vol Regime Detection** | 1 | IV surface classification | Easy |
| 2 | **Delta Hedging** | 5 | Greek neutrality, counter-trading | Medium |
| 3 | **Earnings Vol Crush** | 8 | Temporal prediction, event timing | Hard |
| 4 | **Gamma Scalping** | 12 | Dynamic re-hedging, profit scalping | Expert |
| 5 | **Vega/Gamma Stress** | 15 | Multi-derivative optimization | Super-Boss |

---

## Tier 1: Volatility Regime Detection

### Metadata
- **Task ID**: `vol_regime_detection`
- **Max Steps**: 1
- **Pass Threshold**: Score ≥ 0.10

### Objective
Read the raw IV Surface matrix (e.g. baseline `0.30` vs `0.10`) and classify the market state as one of:
- `"high"` — Elevated IV across most strikes/maturities
- `"normal"` — Typical IV surface
- `"low"` — Compressed IV below historical averages

### Action Space
- **Trade Action**: `hold` (no trades required)
- **Reasoning**: Must explicitly contain the detected regime word

### Evaluation Formula
```python
reward = identification_component + reasoning_component

identification_component = 0.8 if predicted == true_regime else 0.0
reasoning_component = score_reasoning_quality(...) * 0.2
```

### Success Example
```
Observation:
  IV Surface:
    85-strike: [0.32, 0.28, 0.25]  # Elevated front-month
    90-strike: [0.29, 0.26, 0.23]
    95-strike: [0.27, 0.24, 0.21]
    ...
  
Agent Reasoning: "The 30-day IV at 85-strike is 0.32, significantly above 
the typical 0.18 baseline. Front-month skew is pronounced. The regime is HIGH."

Score: 0.97
  - identification_component: 0.80 (correct regime)
  - reasoning_component: 0.17 (cites IV values, domain keywords)
```

### Failure Example
```
Agent Reasoning: "The market looks volatile."
Score: 0.15
  - identification_component: 0.0 (missing regime keyword)
  - reasoning_component: 0.15 (generic, no numeric citations)
```

---

## Tier 2: Delta Hedging

### Metadata
- **Task ID**: `delta_hedging`
- **Max Steps**: 5
- **Pass Threshold**: Score ≥ 0.30

### Objective
Maintain strict delta neutrality (|delta| < 0.05) through a market shock event at step 3. The agent must:
1. Hedge initial delta exposure
2. Survive market shock without excessive PnL loss
3. Re-hedge post-shock to maintain neutrality

### Evaluation Formula
```python
delta_improvement = max(0, (abs(old_delta) - abs(new_delta)) / abs(old_delta))
delta_reward = delta_improvement * 0.5

cost_efficiency = max(0, 0.3 - trade_cost * 0.1)
neutrality_bonus = 0.1 if abs(new_delta) < 0.05 else 0.0
reasoning_reward = score_reasoning_quality(...) * 0.2

total = delta_reward + cost_efficiency + neutrality_bonus + reasoning_reward
```

### Event Timeline
- **Step 1-2**: Agent can trade to neutralize initial delta
- **Step 3**: Market shock (spot jumps ±5-8%, delta drifts)
- **Step 4-5**: Agent must re-hedge to maintain neutrality

### Optimal Strategy
```
Step 1: Portfolio Delta = 2.4
  Action: Sell 2 ATM calls (strike=100, maturity=30d)
  New Delta: 0.08
  Reward: 0.72

Step 2: No movement
  Action: Hold
  Reward: 0.10

Step 3: Market shock, Delta spikes to 1.2
  Action: Sell 1 OTM call (strike=105, maturity=30d)
  New Delta: 0.04
  Reward: 0.88

Total Score: 0.81
```

### Failure Mode
```
Step 1: Portfolio Delta = 2.4
  Action: Hold (do nothing)
  Reward: 0.10

Step 2: Market shock occurs early, Delta = 4.1
  Action: Sell 4 ATM calls
  New Delta: 0.02
  Trade Cost: High (over-trading penalty)
  Reward: 0.42

Total Score: 0.34 (marginal pass)
```

---

## Tier 3: Earnings Vol Crush

### Metadata
- **Task ID**: `earnings_vol_crush`
- **Max Steps**: 8
- **Pass Threshold**: Score ≥ 0.40

### Objective
Position the portfolio optimally before an earnings announcement, then manage risk after the implied volatility collapse.

**Critical Event**: At **Step 6**, IV drops **40% uniformly** across all strikes and maturities (the "vol crush").

### Evaluation Formula
```python
pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.4
delta_neutrality = (1.0 - min(abs(delta) / 0.5, 1.0)) * 0.3
reasoning_reward = score_reasoning_quality(...) * 0.3

total = pnl_reward + delta_neutrality + reasoning_reward
```

### Event Timeline
- **Step 1-5**: Elevated IV, agent should liquidate vega positions
- **Step 6**: **Earnings released, IV crush occurs (-40%)**
- **Step 7-8**: Spot may jump ±5%, agent re-hedges delta

### Optimal Strategy (Short Vega)
```
Step 1: Portfolio Vega = 0.50, IV = 0.35 (elevated)
  Action: Sell ATM straddle (short vega)
  New Vega: 0.20
  Reasoning: "Pre-earnings IV spike, positioning for crush"
  Reward: 0.45

Step 5: Portfolio Vega = 0.20
  (No action, holding short vega)
  Reward: 0.12

Step 6: IV crush occurs, Vega PnL = +0.32
  Reasoning: "Earnings released, captured vol crush profit"
  Reward: 0.92

Total Score: 0.78
```

### Failure Mode (Long Vega)
```
Step 1: Portfolio Vega = 0.50, IV = 0.35
  Action: Buy more ATM straddle
  New Vega: 0.80
  Reward: 0.35

Step 6: IV crush occurs, Vega PnL = -0.48
  Reward: 0.08 (massive pnl_component loss)

Total Score: 0.22 (failed)
```

---

## Tier 4: Gamma Scalping

### Metadata
- **Task ID**: `gamma_scalping`
- **Max Steps**: 12
- **Pass Threshold**: Score ≥ 0.50

### Objective
Given a portfolio with significant positive gamma, profit from delta oscillations as the underlying spot price thrashes. The agent must:
1. Monitor delta changes from gamma exposure
2. Counter-trade delta to lock in profits ("scalping")
3. Balance trading costs vs. profit capture

### Evaluation Formula
```python
delta_neutrality = max(0, 1.0 - abs(delta) / 0.5) * 0.5
pnl_reward = sigmoid(pnl_change, scale=0.3) * 0.3
reasoning_reward = score_reasoning_quality(...) * 0.2

total = delta_neutrality + pnl_reward + reasoning_reward
```

### Market Mechanics
- Spot oscillates ±3-5% per step (high volatility regime)
- Gamma causes delta to change: `Δdelta ≈ gamma * Δspot`
- Agent must trade frequently but not excessively (cost penalty)

### Optimal Strategy
```
Step 1: Spot=100, Delta=0.0, Gamma=0.8
  Portfolio: Long ATM straddle
  Reward: 0.50

Step 2: Spot=104, Delta=3.2 (gamma * 4)
  Action: Sell 3 calls @ 100-strike
  New Delta: 0.2
  Reasoning: "Spot jumped 4%, gamma pushed delta to 3.2, scalping profit"
  Reward: 0.68

Step 3: Spot=97, Delta=-2.4 (gamma * -3)
  Action: Buy 2 calls @ 100-strike
  New Delta: 0.4
  Reasoning: "Spot dropped to 97, gamma flipped delta negative, scalping again"
  Reward: 0.62

...

Total Score: 0.75
```

### Failure Mode (Passive)
```
Step 1: Spot=100, Delta=0.0, Gamma=0.8
  Action: Hold
  Reward: 0.10

Step 2: Spot=104, Delta=3.2
  Action: Hold (not scalping)
  Reward: 0.08 (delta_neutrality drops to 0.0)

...

Total Score: 0.28 (failed)
```

---

## Tier 5: Vega/Gamma Stress (Super-Boss)

### Metadata
- **Task ID**: `vega_gamma_stress`
- **Max Steps**: 15
- **Pass Threshold**: Score ≥ 0.60

### Objective
Construct a multi-legged position over the first 10 steps that drives **both** net Vega and net Gamma explicitly to `0.0` (within tight tolerance) to survive a catastrophic dual-shock event.

**Critical Event**: At **Step 10**, a macro shock occurs:
- Spot crashes -15%
- Implied volatility spikes +60%

### Evaluation Formula
```python
vega_score = exp(-0.5 * (abs(vega) / 0.05) ** 2)
gamma_score = exp(-0.5 * (abs(gamma) / 0.02) ** 2)
vg_neutrality = (vega_score * 0.5 + gamma_score * 0.5) * 0.5

pnl_reward = sigmoid(pnl_change, scale=0.5) * 0.3
reasoning_reward = score_reasoning_quality(...) * 0.2

total = vg_neutrality + pnl_reward + reasoning_reward
```

### Why Gaussian Boundaries?
The exponential decay ensures **no slack** outside the tolerance zone:
- Vega at 0.05 (threshold) → `vega_score = exp(-0.5) = 0.61`
- Vega at 0.10 (2× threshold) → `vega_score = exp(-2.0) = 0.14`
- Vega at 0.20 (4× threshold) → `vega_score = exp(-8.0) = 0.0003`

**This forces exact mathematical optimization.**

### Optimal Strategy
```
Step 1: Portfolio Vega = 0.45, Gamma = 0.32
  Action: Sell calendar spread (short front-month vega, long back-month vega)
  New Vega: 0.22
  Reasoning: "Reducing vega exposure with calendar spread"

Step 5: Portfolio Vega = 0.08, Gamma = 0.12
  Action: Buy ATM strangle (offsets gamma)
  New Gamma: 0.05
  New Vega: 0.15 (vega increased, acceptable)

Step 8: Portfolio Vega = 0.10, Gamma = 0.03
  Action: Sell OTM straddle (fine-tunes both)
  New Vega: 0.03
  New Gamma: 0.01

Step 10: Dual shock occurs
  Vega = 0.03 → vega_score = exp(-0.18) = 0.84
  Gamma = 0.01 → gamma_score = exp(-0.125) = 0.88
  vg_neutrality = (0.84 * 0.5 + 0.88 * 0.5) * 0.5 = 0.43
  Reward: 0.85

Total Score: 0.82 (passed)
```

### Failure Mode
```
Step 1-9: Agent only hedges delta, ignores vega/gamma
  Vega = 0.42, Gamma = 0.28

Step 10: Dual shock occurs
  Portfolio PnL = -0.68 (catastrophic)
  Vega score = exp(-35.3) ≈ 0.0
  Gamma score = exp(-98.0) ≈ 0.0
  vg_neutrality = 0.0
  Reward: 0.05

Total Score: 0.18 (failed catastrophically)
```

---

## Adaptive Curriculum Logic

The inference script (`inference.py`) implements early-stopping:

```python
for task_name in TASKS:
    score = run_task(task_name)
    scores[task_name] = score
    
    # Early stop if baseline not met
    if score < SUCCESS_THRESHOLD:
        print(f"CURRICULUM HALTED: {task_name} scored {score:.2f}")
        break
```

**Why This Matters**:
- Forces agents to master fundamentals before attempting Super-Boss
- Prevents "gaming" later tasks without foundational skills

---

## Task-Specific Graders

Each task has a dedicated grader class:

| Task | Grader Class | Location |
|---|---|---|
| Vol Regime Detection | `VolRegimeDetectionGrader` | `vsr_env/tasks/vol_regime_detection.py` |
| Delta Hedging | `DeltaHedgingGrader` | `vsr_env/tasks/delta_hedging.py` |
| Earnings Vol Crush | `EarningsVolCrushGrader` | `vsr_env/tasks/earnings_vol_crush.py` |
| Gamma Scalping | `GammaScalpingGrader` | `vsr_env/tasks/gamma_scalping.py` |
| Vega/Gamma Stress | `VegaGammaStressGrader` | `vsr_env/tasks/vega_gamma_stress.py` |

All graders implement:
```python
class Grader:
    def score(self, episode_history: List[dict], state: VSRState) -> float:
        """Compute final score in [0.0, 1.0] from episode history."""
```

---

## Comparison: VSR-Env vs. Static Environments

| Feature | VSR-Env | Typical RL Env |
|---|---|---|
| **Temporal Events** | ✅ Earnings, shocks at specific steps | ❌ Stationary dynamics |
| **Multi-Dimensional State** | ✅ IV surface (8×3), Greeks, positions | ⚠️ Low-dim vectors |
| **Curriculum** | ✅ 5-tier adaptive progression | ❌ Single difficulty |
| **Reasoning Requirement** | ✅ Scored explicitly | ❌ Optional/ignored |
| **Mathematical Optimality** | ✅ Gaussian boundaries, exact targets | ⚠️ Heuristic thresholds |
| **Event Timing** | ✅ Must anticipate shocks | ❌ Reactive only |

---

## Why This Task Design Wins

### 1. **Depth Over Breadth**
5 carefully designed tasks with increasing complexity, not 20 shallow variations.

### 2. **Real-World Relevance**
Every task mirrors actual trading desk challenges:
- Earnings events (quarterly)
- Delta hedging (daily)
- Risk management (continuous)

### 3. **Mathematical Rigor**
Super-Boss is not just "harder" — it requires **dual-derivative optimization** under Gaussian constraints. This is quant-level reasoning.

---

**Ready to test the most challenging options reasoning benchmark?**

```bash
git clone https://github.com/manan-tech/VSR-Env
cd VSR-Env
python inference.py
```