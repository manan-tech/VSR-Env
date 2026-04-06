# Goal Description

Expand the VSR-Env by adding the final two advanced tasks: **Gamma-Scalping** and **Vega-Gamma Stress**. This will bring the total task count to precisely 5, covering every level of intelligence evaluation.

## User Review Required

> [!WARNING]  
> **Strategic Selection of the Final 2 Tasks**
> We have 4 proposals on the table. To protect the integrity of the math engine before the deadline, I have selected the two that flawlessly reuse our existing Black-Scholes `OptionChain` while explicitly rejecting the structurally dangerous ones:
> 
> 1. ✅ **ACCEPT: Gamma-Scalping (Expert)** - Seamlessly reuses existing delta calculation but initializes with massive Gamma, forcing rapid re-hedging.
> 2. ✅ **ACCEPT: Vega-Gamma Stress (Super-Boss)** - Perfectly utilizes our existing second-order Greek calculations (`portfolio_vega`) and triggers a sudden Volatility Regime Shift mid-episode.
> 3. ❌ **REJECT: Skew Trading** - Redundant. The agent already identifies skew anomalies in `iv_reading`. Also, executing multi-leg Butterfly Spreads requires rewriting the `VSRAction` schema.
> 4. ❌ **REJECT: Multi-Asset Dispersion** - Dangerous. Attempting to add multi-asset correlation matrices hours before submission will likely destroy the container's 98/100 readiness.
> 
> **Decision:** We will implement only `gamma_scalping` and `vega_gamma_stress`. Please confirm this approach.

## Proposed Changes

We will introduce both tasks by creating their specific graders and wiring them to the API.

### Task Modules
#### [NEW] `vsr_env/tasks/gamma_scalping.py`(file:///Users/mananbansal/Desktop/meta/vsr_env/tasks/gamma_scalping.py)
- Create `GammaScalpingTask`.
- Start agent with a Long ATM Straddle to inject extreme Gamma. Focus grading entirely on the ability to constantly squash the rapidly drifting Delta.

#### [NEW] `vsr_env/tasks/vega_gamma_stress.py`(file:///Users/mananbansal/Desktop/meta/vsr_env/tasks/vega_gamma_stress.py)
- Create `VegaGammaStressTask`.
- Start agent with a delta-hedged position in a low-volatility regime. At step `3`, trigger a severe Volatility Spike. Grade heavily on the agent's ability to neutralize BOTH the Delta gap and the sudden Vega exposure before step `8`.

### Core Orchestration
#### [MODIFY] `vsr_env/server/vsr_environment.py`(file:///Users/mananbansal/Desktop/meta/vsr_env/server/vsr_environment.py)
- Register `gamma_scalping` and `vega_gamma_stress` into the task dispatcher.
#### [MODIFY] `vsr_env/reward/reward_computer.py`(file:///Users/mananbansal/Desktop/meta/vsr_env/reward/reward_computer.py)
- Add reward logic that incorporates `portfolio_vega` and penalizes both Delta and Vega imbalances.

### UI & Inference Formatting
#### [MODIFY] `openenv.yaml`(file:///Users/mananbansal/Desktop/meta/openenv.yaml), `inference.py`(file:///Users/mananbansal/Desktop/meta/inference.py), `index.html`(file:///Users/mananbansal/Desktop/meta/vsr_env/server/index.html)
- Add both tasks to the arrays, update MAX_STEPS definitions, and append to the UI dropdown selector so judges can play all 5 modes.

## Open Questions

1. Do you approve selecting **Gamma-Scalping** and **Vega-Gamma Stress** while dropping the other two? It guarantees submission safety.

## Verification Plan

### Automated Tests
- Restart the Docker container locally.
- Run `inference.py` for both new tasks individually to test grading calculations.
### Manual Verification
- Use the Web UI to play `vega_gamma_stress`. Ensure the Volatility numbers literally spike at step 3, forcing a frantic manual re-hedge of the portfolio Greeks.
