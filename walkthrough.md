# VSR-Env: Comprehensive Technical Walkthrough

This walkthrough provides a deep dive into the **Volatility Surface Reasoning Environment (VSR-Env)**, detailing its mathematical foundations, reward mechanisms, and containerized deployment strategy.

---

## The VSR-Env Advantage: Moving Beyond P&L

VSR-Env bridges the gap between pure quantitative finance and high-level LLM reasoning. Unlike standard trading environments that focus solely on P&L, VSR-Env requires agents to **synthesize complex, high-dimensional volatility data into coherent trade theses**. 

It challenges agents to move beyond simple "buy low, sell high" logic into the realm of multi-asset Greeks management, regime-shift adaptation, and qualitative reasoning—critical skills for next-generation AI financial assistants that are rarely captured in existing benchmarks.

---

## 1. Mathematical Foundations of Simulation

The environment simulates a realistic options market using stochastic processes for both the underlying asset and its volatility.

### A. Spot Price: Geometric Brownian Motion (GBM)
The underlying asset price $S_t$ is modeled using GBM:
$$dS_t = \mu S_t dt + \sigma_t S_t dW_t$$
- **Drift ($\mu$)**: $0.0$ (Risk-neutral).
- **Diffusion ($\sigma_t$)**: The instantaneous volatility $\sqrt{V_t}$.
- **Implementation**: See [market_sim.py](file:///Users/mananbansal/Desktop/meta/vsr_env/engine/market_sim.py) line 33.

### B. Variance: Ornstein-Uhlenbeck (OU) Process
Volatility clustering and mean reversion are simulated using an OU process for variance $V_t$:
$$dV_t = \theta (\bar{V} - V_t) dt + \xi dW_{V,t}$$
- **Mean Reversion ($\theta$)**: $0.1$.
- **Long-term Mean ($\bar{V}$)**: $0.04$ (20% annualized vol).
- **Vol-of-Vol ($\xi$)**: $0.01$.
- **Implementation**: See [market_sim.py](file:///Users/mananbansal/Desktop/meta/vsr_env/engine/market_sim.py) line 46.

### C. Options Pricing: Black-Scholes-Merton
The [OptionChainEngine](file:///Users/mananbansal/Desktop/meta/vsr_env/engine/option_chain.py) implements vectorized Black-Scholes for pricing and Greeks:
- **Pricing**: $C = S N(d_1) - K e^{-rT} N(d_2)$
- **Volatility Crush**: Modeled as a sudden jump process $-\Delta \sigma$ in the variance process.
- **Delta**: $\Delta = N(d_1)$ (Call), $\Delta = N(d_1) - 1$ (Put).
- **Greeks**: Includes Gamma (2nd order), Vega (Vol sensitivity), and Theta (Time decay).

---

## 2. Advanced Reward Mechanisms

VSR-Env uses a multi-faceted reward function to evaluate both quantitative performance and qualitative reasoning.

### A. Reasoning Quality Scoring
The environment scores the `reasoning` field provided by the agent in its [VSRAction](file:///Users/mananbansal/Desktop/meta/vsr_env/models.py):
- **Domain Keywords (40%)**: Credits for using terms like "delta", "hedge", "arbitrage", etc.
- **Numeric Consistency (60%)**: Verifies that the agent correctly identifies current market values (Spot, IV, Portfolio Delta) in its rationale.
- **Length Penalty**: Penalizes trivial or empty reasoning strings.

### B. Task-Specific Rewards
- **Delta Hedging**: Rewards P&L efficiency and maintaining a delta-neutral state $(| \Delta | < 0.05)$ during market shocks.
- **Earnings Vol Crush**: Focused on **Vega positioning** and sigmoid-normalized P&L outcomes after the 30-50% IV drop.
- **Gamma Scalping**: Rewards high-gamma traders who re-hedge delta precisely to lock in convexity profits during spot price oscillations.

---

## 3. Dockerization & Build Strategy

The project is containerized for portability and compliance with the OpenEnv specification.

### Dockerfile Breakdown
- **Base Image**: `python:3.11-slim`.
- **System Dependencies**: Installs `curl` for healthchecks.
- **Build Layers**: Optimized caching by copying `requirements.txt` before the application code.
- **Healthcheck**: Uses `curl` to monitor the `/health` endpoint of the FastAPI server.
- **Runtime**: Runs `uvicorn` to serve the environment via a REST API.

---

## 4. Architectural Model (The "Database")

The environment state is managed through a hierarchy of Pydantic models:

| Component | Model Class | Role |
| :--- | :--- | :--- |
| **Actions** | `VSRAction` | Agent decisions (strike, maturity, direction, reasoning). |
| **Observations** | `VSRObservation` | Filtered state visible to the agent (includes `expected_outcome`). |
| **State** | `VSRState` | Internal ground truth (incl. `vol_crush_step`, `initial_theta`, `expected_outcome`). |
| **Rewards** | `VSRReward` | Structured feedback with component breakdown. |

---

> [!TIP]
> For a quick start, check the [QUICKSTART.md](file:///Users/mananbansal/Desktop/meta/QUICKSTART.md) or run the [test_local.sh](file:///Users/mananbansal/Desktop/meta/test_local.sh) script.

> [!NOTE]
> All simulation parameters (strikes, maturities, interest rates) are configurable in settings and engine constants.
