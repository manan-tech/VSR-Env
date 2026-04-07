# Reward Heuristics & Deterministic Architecture

VSR-Env evaluates actions using a deterministic, heuristic-driven grading module rather than an arbitrary ML critic model. The reward logic ensures that agents succeed purely based on their mathematical mastery of option derivatives.

## Architecture: Reward V2 Implementation
The transition to "Reward V2" fundamentally restructured the reward pipeline. 

Previously, `env.step()` calculated a generalized PnL metric. Under Reward V2, tasks define their own specific Sub-Graders inside `vsr_env/reward/reward_computer.py`. A global unified `RewardOutcome` payload is returned each step, breaking out `action_reward`, `pnl_reward`, and `greek_penalty`.

Crucially, **Reward Components are injected directly into the Info Payload**. The returned `info` dictionary now contains:
```json
"reward_components": {
    "action_reward": 0.5,
    "pnl_reward": -0.1,
    "greek_penalty": 0.0,
    "total": 0.4
}
```
This is a critical UI/UX advancement because it enables our `inference.py` "Trajectory Blotter" to dump exact breakdown numbers to the LLM step-by-step, dramatically increasing reasoning performance.

## Core Grading Strategies

### 1. The PnL Baseline
The lowest unified metric across all multi-turn tasks. Positive fluctuations to the internal `VSRState.bank_cash` increment the `pnl_reward`.

### 2. Greek Centering (Delta Neutrality)
Delta Hedging tasks calculate reward inversely proportional to the absolute Delta.
`reward = max(0.0, 1.0 - abs(net_delta) * delta_penalty_multiplier)`

### 3. Vega-Gamma Dual Bound Matrix (Super-Boss)
The hardest task deploys aggressive Standard Deviation parameters. When neutralizing a portfolio for the `vega_gamma_stress` task, the score is graded based on Gaussian boundaries:
`vega_score = np.exp(-0.5 * (avg_vega / 0.05) ** 2)`
`gamma_score = np.exp(-0.5 * (avg_gamma / 0.05) ** 2)`

This means the agent *must* drive its net Vega and net Gamma explicitly to `0.0`. If it drifts outside the tight +/- 0.05 bounds, the Gaussian curve rapidly drops its multiplier to 0.0, aggressively tanking the unified score regardless of its raw theoretical PnL.
