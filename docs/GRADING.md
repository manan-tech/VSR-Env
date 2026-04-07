# VSR-Env Grading Transparency

This document outlines the exact mechanics behind the grading heuristics and reward compute functions for all 5 tasks in the VSR-Env. We believe in full transparency for researchers evaluating LLM decision-making capabilities.

## Reasoning Quality Rubric (Universal)

Many tasks dedicate up to **20% of their step or final score** to the `ReasoningQualityRubric`.
- **Keyword Hits (0.4 max):** Searches for domain-specific vocabulary (e.g., "delta", "vega", "hedge" "convexity", "crash").
- **Numeric Consistency (0.6 max):** Validates if the agent correctly cites the current Spot Price (0.25 max), Implied Volatilities (0.25 max), and Delta states (0.1 max).
- **Penalty (-0.3 penalty):** If reasoning length is extremely short (≤ 20 characters), the score is hit with a massive reduction.

---

## 1. Vol Regime Detection (Easy)

**Objective:** Identify the market regime from base variance.
- **Grader:** `ExactMatchRubric`
- **Methodology:** The environment sets a variance of exactly 0.01 (Low), 0.04 (Normal), or 0.09 (High). The model must emit an action holding exactly 0 quantity. The target regime string is extracted from the `expected_outcome` ground truth.
- **Score:** 1.0 if the target (`"low"`, `"normal"`, or `"high"`) is found explicitly within the `reasoning` payload. 0.0 otherwise.

---

## 2. Delta Hedging (Medium)

**Objective:** Maintain a delta-neutral portfolio before and through a random spot/IV regime shift.
- **Grader Structure:** Focuses on pre-shock neutralization (0.7) and cost efficiency (0.3).
- **Per-Step Reward Math:**
  - `delta_improvement` (0.5 max): `(abs(prev_delta) - abs(current_delta))` bounds scaled into [0, 0.5].
  - `cost_efficiency` (0.3 max): Positive change in P&L yields up to 0.3.
  - `neutrality_bonus` (0.1 max): Applied if absolute delta is tightly neutralized (`abs(current_delta) < 0.05`).
  - `reasoning` (0.2 max).
- **Final Episode Score:** Validates the aggregate sequence across the Random Shift boundaries.

---

## 3. Earnings Vol Crush (Hard)

**Objective:** Position for a massive short volatility swing.
- **Grader Structure:** Focuses on predicting the vol crush via short Vega (0.4) + Re-hedging Delta Post-Crush (0.35) + Absolute PnL survival (0.25).
- **Key Mechanics:**
  - **Proximity:** A floating variable (`earnings_proximity`) decays monotonically from 1.0 to 0.0 leading up to the exact random step where the crush triggers.
  - **The Crush:** Multiplies variance by `0.5 - 0.7` causing a 30-50% IV decimation.
  - **Bonus:** `0.5` instantaneous per-step reward bonus if the agent holds a deep short Vega position (`< -0.01`) precisely before the crush triggers.

---

## 4. Gamma Scalping (Expert)

**Objective:** Extract extrinsic value from oscillating spot movements while bleeding Theta.
- **Grader Structure:** Oscillation Re-hedge Quality (0.4) + PnL Extracted Above Theta bleed (0.35) + Action Timing (0.25).
- **Key Mechanics:**
  - The market simulator forcefully injects a `±2.5%` spot swing *on top* of traditional GBM drift globally.
  - Returns are highly predicated on the mathematical correlation linking local spot moves to the agent's directional hedging quantities.

---

## 5. Vega/Gamma Stress (Super-Boss)

**Objective:** Survive a catastrophic dual market crash.
- **Grader Structure:** Vega/Gamma bounds (0.5) + PnL Crash Survival (0.3) + Reasoning Quality (0.2).
- **Key Mechanics:**
  - Agent initializes deeply exposed (-5.0 Straddle Contracts), mapping to massively negative Vega and Gamma.
  - A dual-shock is triggered mid-episode causing Spot Price to crater by 15-20% *while simultaneously* spiking IV by 300%-500%.
  - **SD Bound Math:** A Gaussian decay penalty (`np.exp(-0.5 * (avg_greek / threshold)**2)`) enforces extreme stringency. The agent's trajectory mean must firmly nestle near 0 inside tight SD bounds on both Vega and Gamma before the shock triggers.

*All raw calculations clamp output metrics strictly into the canonical OpenEnv Hackathon [0.0, 1.0] range.*