# Architecture Overview: VSR-Env

VSR-Env follows a clean 3-layer architecture designed for **transparency, reproducibility, and extensibility**.

---

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM Agent (inference.py)                        │
│                                                                    │
│  • Reads IV surface ASCII tables                                  │
│  • Outputs JSON actions with reasoning                            │
│  • Maintains trajectory blotter for context                       │
└───────────────────────────┬────────────────────────────────────────┘
                            │ HTTP (OpenAI-compatible API)
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                 FastAPI Server (vsr_env/server/)                   │
│                                                                    │
│  Endpoints:                                                        │
│  • POST /reset     → Initialize episode                           │
│  • POST /step      → Execute action, return observation           │
│  • GET  /state     → Current episode state                        │
│  • GET  /docs      → OpenAPI documentation                        │
│  • WS   /ws        → WebSocket for persistent sessions            │
└───────────────────────────┬────────────────────────────────────────┘
                            │ Internal dispatch
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│            VSREnvironment (vsr_env/server/vsr_environment.py)      │
│                                                                    │
│  Core Methods:                                                     │
│  • reset(task_name, seed) → VSRObservation                        │
│  • step(action) → {observation, reward, done, info}               │
│                                                                    │
│  Responsibilities:                                                 │
│  • Task dispatching                                               │
│  • State management (VSRState)                                    │
│  • Telemetry tracking (trajectory history)                        │
│  • Input validation (Pydantic schemas)                           │
└───────────────────────────┬────────────────────────────────────────┘
                            │ Dispatches to task handler
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│              Task Handlers (vsr_env/tasks/*.py)                    │
│                                                                    │
│  Each task implements:                                             │
│  • Task-specific state transitions                                │
│  • Event triggers (earnings crush, macro shock)                   │
│  • Grader instantiation                                           │
│                                                                    │
│  Files:                                                            │
│  • vol_regime_detection.py  → Regime classification              │
│  • delta_hedging.py          → Greek neutrality                  │
│  • earnings_vol_crush.py     → Event timing, vega management     │
│  • gamma_scalping.py         → Dynamic re-hedging                │
│  • vega_gamma_stress.py      → Multi-derivative optimization     │
│  • straddle_trading.py      → Vol speculation (NEW)             │
│  • vertical_spread.py       → Directional spreads (NEW)         │
└───────────────────────────┬────────────────────────────────────────┘
                            │ Uses strategy objects
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│          Strategy Layer (vsr_env/strategies/)                      │
│                                                                    │
│  Multi-leg strategy support for realistic trading:                │
│  • base.py          → OptionStrategy abstract base class          │
│  • straddle.py      → ATM straddle (call + put, same strike)      │
│  • strangle.py      → OTM strangle (different strikes)            │
│  • spread.py        → Vertical and Calendar spreads               │
│                                                                    │
│  Features:                                                         │
│  • Atomic multi-leg execution                                     │
│  • Strategy-level Greek aggregation                               │
│  • Payoff and breakeven computation                               │
└───────────────────────────┬────────────────────────────────────────┘
                            │ Calls for reward computation
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│       RewardComputer (vsr_env/reward/reward_computer.py)           │
│                                                                    │
│  Methods:                                                          │
│  • compute_vol_regime_reward()                                     │
│  • compute_delta_hedging_reward()                                  │
│  • compute_earnings_crush_reward()                                 │
│  • compute_gamma_scalping_reward()                                 │
│  • compute_vega_gamma_stress_reward()                              │
│                                                                    │
│  Returns:                                                          │
│  • VSRReward (total + component breakdown)                        │
│  • Injected into info["reward_components"]                        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **VSRState** (`models.py`)

Immutable state container holding:
- `spot_price` (float): Current underlying price
- `iv_surface` (8×3 array): Implied volatility grid
- `portfolio` (Portfolio object): Holdings, Greeks, PnL
- `timestamp` (int): Episode step counter
- `task_name` (str): Current task identifier

**Why Immutable?**
- Guarantees reproducibility (no hidden mutations)
- Enables trajectory replay (episode history is immutable log)
- Simplifies debugging (state at step N is frozen)

---

### 2. **VSRObservation** (`models.py`)

Pydantic model exposed to agents:

```python
class VSRObservation(BaseModel):
    iv_surface: List[List[float]]      # 8 strikes × 3 maturities
    spot_price: float                  # Current underlying
    portfolio_greeks: Dict[str, float] # {delta, gamma, vega, theta}
    portfolio_pnl: float              # Cumulative P&L
    portfolio_positions: List[dict]   # Open positions
    market_sentiment: float           #[-1.0, 1.0] indicator
    step_number: int                  # Current step
    steps_remaining: int              # Steps left in episode
    task_name: str                    # Task identifier
    task_description: str             # Human-readable objective
    last_action_error: Optional[str]  # Validation errors
```

**Design Principle**: Include *everything* an agent needs to reason, nothing more.

---

### 3. **VSRAction** (`models.py`)

Agent's discrete action specification:

```python
class VSRAction(BaseModel):
    selected_strike: int      # Index 0-7 into STRIKES array
    selected_maturity: int    # Index 0-2 into MATURITIES array
    direction: TradeDirection # buy/sell/hold
    quantity: float           # Contracts [0.0, 10.0]
    reasoning: str            # Free-text explanation
```

**Validation** (Pydantic):
- Strike index must be 0-7
- Maturity index must be 0-2
- Quantity must be 0.0-10.0
- Direction must be enum value

---

### 4. **VSREnvironment** (`server/vsr_environment.py`)

Main orchestration class implementing OpenEnv standard:

```python
class VSREnvironment:
    def reset(self, task_name: str, seed: Optional[int] = None) -> VSRObservation:
        """Initialize episode with optional seed for reproducibility."""
        
    def step(self, action: VSRAction) -> dict:
        """
        Execute action, transition state, compute reward.
        
        Returns:
            {
                "observation": VSRObservation,
                "reward": float,
                "done": bool,
                "info": {
                    "grader_score": float,
                    "reward_components": dict,
                    "trajectory_history": List[str]
                }
            }
        """
```

**Wiring Tasks to Graders**:
```python
TASK_CONFIG = {
    "vol_regime_detection": {
        "max_steps": 1,
        "task_handler": VolRegimeDetectionTask,
        "grader_class": VolRegimeDetectionGrader,
    },
    "delta_hedging": {
        "max_steps": 5,
        "task_handler": DeltaHedgingTask,
        "grader_class": DeltaHedgingGrader,
    },
    # ... more tasks
}
```

---

### 5. **RewardComputer** (`reward/reward_computer.py`)

Stateless reward calculation module:

```python
class RewardComputer:
    def compute_vega_gamma_stress_reward(
        self, action, state, observation, prev_pnl
    ) -> VSRReward:
        # Gaussian boundary scoring
        vega_score = math.exp(-0.5 * (abs(vega) / 0.05) ** 2)
        gamma_score = math.exp(-0.5 * (abs(gamma) / 0.02) ** 2)
        vg_neutrality = (vega_score * 0.5 + gamma_score * 0.5) * 0.5
        
        pnl_reward = sigmoid(pnl_change, scale=0.5) * 0.3
        reasoning_reward = score_reasoning_quality(...) * 0.2
        
        return VSRReward(
            total=vg_neutrality + pnl_reward + reasoning_reward,
            greek_component=vg_neutrality,
            pnl_component=pnl_reward,
            reasoning_component=reasoning_reward,
        )
```

**Why Separate Module?**
- Testable in isolation (unit tests for each formula)
- Swap-able grading logic (can iterate without changing task logic)
- Transparent to judges (all formulas in one file)

---

### 6. **Telemetry Tracker** (`server/telemetry.py`)

Immutable trajectory logger:

```python
class TelemetryTracker:
    def __init__(self):
        self.episode_history: List[dict] = []
    
    def log_step(self, action, observation, reward, info):
        """Append step to immutable log."""
        record = {
            "step": observation.step_number,
            "action": action.dict(),
            "observation": observation.dict(),
            "reward": reward,
            "info": info,
        }
        self.episode_history.append(record)
```

**Use Cases**:
- Replay episodes for debugging
- Export to JSON for offline analysis
- Inject trajectory context into LLM prompts (supercharges reasoning)

---

## Data Flow: Step-by-Step

### Step 1: Agent observes environment

```
POST /reset
{
    "task_name": "delta_hedging",
    "seed": 123
}

Response:
{
    "observation": {
        "iv_surface": [[0.28, 0.25, 0.23], ...],
        "spot_price": 100.45,
        "portfolio_greeks": {"delta": 2.4, "gamma": 0.5, ...},
        ...
    }
}
```

### Step 2: Agent reasons and acts

```
Agent (LLM) receives observation → builds prompt → outputs JSON

{
    "strike_idx": 4,
    "maturity_idx": 0,
    "direction": "sell",
    "quantity": 2.0,
    "reasoning": "Portfolio delta 2.4 needs neutralization..."
}
```

### Step 3: Environment executes and grades

```
POST /step
{
    "action": {...}
}

Internal Flow:
1. Validate action (Pydantic)
2. Apply trade to portfolio (update Greeks)
3. Trigger task-specific events (if step == event_step)
4. Call RewardComputer.compute_delta_hedging_reward()
5. Log to TelemetryTracker

Response:
{
    "observation": {/* updated state */},
    "reward": 0.72,
    "done": false,
    "info": {
        "reward_components": {
            "greek_component": 0.45,
            "pnl_component": 0.18,
            "reasoning_component": 0.09,
            "total": 0.72
        },
        "grader_score": 0.68
    }
}
```

### Step 4: Episode completes

```
When done=true:

GET /state

Response:
{
    "episode_history": [...],// Full trajectory log
    "grader_score": 0.75,    // Final metric
    "task_name": "delta_hedging"
}
```

---

## File Structure

```
vsr_env/
├── __init__.py                          # Package exports
├── models.py                            # VSRState, VSRAction, VSRObservation, VSRReward
├── client.py                            # OpenEnv HTTP/WebSocket client
├── engine/                              # Market simulation
│   ├── __init__.py
│   ├── market_sim.py                    # Brownian motion, IV surface generation
│   ├── option_chain.py                  # Option pricing (Black-Scholes)
│   └── portfolio.py                     # Position tracking, Greek calc
├── reward/
│   ├── __init__.py
│   └── reward_computer.py               # All grading formulas
├── tasks/
│   ├── __init__.py
│   ├── vol_regime_detection.py          # Tier 1 task + grader
│   ├── delta_hedging.py                 # Tier 2 task + grader
│   ├── straddle_trading.py              # Tier 4 task + grader
│   ├── earnings_vol_crush.py            # Tier 5 task + grader
│   ├── gamma_scalping.py                # Tier 6 task + grader
│   └── vega_gamma_stress.py             # Tier 7 task + grader
├── server/
│   ├── __init__.py
│   ├── app.py                           # FastAPI routes
│   ├── vsr_environment.py               # Main environment class
│   └── telemetry.py                     # Trajectory logging
└── tests/
    ├── test_env.py                      # Core environment tests
    ├── test_grading.py                  # Reward formula tests
    └── test_integration.py              # End-to-end episode tests
```

---

## Inference Script Architecture

`inference.py` is competition-grade, not just a demo:

### Key Features

1. **Structured STDOUT logging** (OpenEnv standard):
   ```
   [START] task=delta_hedging env=vsr_env model=llama-3.1-8b-instant
   [STEP] step=1 action=sell(4,0,2.0) reward=0.72 done=false error=null
   [END] success=true steps=5 score=0.81 rewards=0.72,0.10,0.88,0.52,0.65
   ```

2. **Trajectory Blotter**: Injects diagnostic blocks into LLM context:
   ```
   Step 3 Result:
     Action: sell(4, 0, 2.0)
     Your Logic: "Delta 2.4 needs neutralization"
     Reward: 0.72 (greek=0.45, pnl=0.18, reasoning=0.09)
     Portfolio State Shift:
       Delta: 2.4 -> 0.08
   ```

3. **Robust JSON parsing**:
   - Handles truncated responses (max_tokens limit)
   - Extracts from markdown code blocks
   - Repairs malformed JSON with brace-matching

4. **Rate-limit handling**:
   - Sleeps between LLM calls (Groq free tier: ~30 req/min)
   - Retries on empty responses
   - Exponential backoff on API errors

---

## Testing Strategy

### Unit Tests (`tests/test_env.py`)

```python
def test_delta_hedging_grader_perfect_neutralization():
    """Perfect delta neutralization gives high score."""
    env = VSREnvironment()
    obs = env.reset("delta_hedging", seed=123)
    
    # Agent hedges perfectly
    action = VSRAction(strike=4, maturity=0, direction="sell", quantity=2.4)
    result = env.step(action)
    
    # Check reward decomposition
    assert result["info"]["reward_components"]["greek_component"] > 0.4
    assert result["reward_components"]["total"] > 0.7
```

### Integration Tests (`tests/test_integration.py`)

```python
def test_full_episode_trajectory():
    """Run complete delta_hedging episode, verify grader score."""
    env = VSREnvironment()
    obs = env.reset("delta_hedging", seed=456)
    
    # Run to completion
    for _ in range(5):
        action = random_action()
        result = env.step(action)
        if result["done"]:
            break
    
    # Check grader score is in valid range
    assert 0.0 <= result["info"]["grader_score"] <= 1.0
```

---

## Why This Architecture Wins

### 1. **Separation of Concerns**
- **Server**: HTTP/websocket handling
- **Environment**: State management, task dispatch
- **Tasks**: Domain-specific logic
- **Reward**: Mathematical grading formulas

**Judges can read one file and understand one concern.**

### 2. **OpenEnv Compliance**
- Standard `/reset`, `/step`, `/state` endpoints
- Pydantic models for all payloads
- WebSocket support for persistent sessions
- OpenAPI documentation at `/docs`

### 3. **Reproducibility**
- Seed-based initialization
- Immutable state objects
- Trajectory logging
- Deterministic grading

### 4. **Extensibility**
Add a new task in 3 steps:
1. Create `tasks/new_task.py` with task handler + grader
2. Add entry to `TASK_CONFIG` in `vsr_environment.py`
3. Add reward method to `RewardComputer`

---

## Deployment

### Local Development
```bash
uvicorn vsr_env.server.app:app --reload --port 8000
```

### Docker (Production)
```bash
docker build -t vsr-env:latest .
docker run -p 8000:8000 vsr-env:latest
```

### Hugging Face Spaces
Already deployed at: [MananBansal/VSR-Env](https://huggingface.co/spaces/MananBansal/VSR-Env)

---

**Architecture Philosophy**: Build for judges, not just agents. Every component is transparent, testable, and documented.