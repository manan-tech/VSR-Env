# VSR-Env Difficulty Tasks (Curriculum)

VSR-Env transitions away from single-mode simulations in favor of a 5-tier adaptive difficulty curriculum. Models are scored on an ascending scale of complexity; failure on earlier tasks halts the progression.

## 1. Easy: Volatility Regime Detection (`vol_regime_detection`)
- **Max Steps**: 1
- **Skill Tested**: Analytical classification of grid numerics.
- **Goal**: Read the raw IV Surface matrix (e.g. baseline `0.30` vs `0.10`). Do not initiate any trades (set action to `hold`). Set the reasoning to explicitly diagnose if the market state is exhibiting `"high"`, `"normal"`, or `"low"` variance.

## 2. Medium: Delta Hedging (`delta_hedging`)
- **Max Steps**: 5
- **Skill Tested**: 1D Numerical tracking and counter-balancing.
- **Goal**: Maintain strict delta neutrality. Standard option derivatives naturally gain or lose directional exposure. The LLM must monitor its net portfolio `Delta` and offset the exposure by actively trading Call or Put options.

## 3. Hard: Earnings Vol Crush (`earnings_vol_crush`)
- **Max Steps**: 8
- **Skill Tested**: Temporal prediction and Vega exposure minimization.
- **Goal**: The scenario seeds a known earnings terminal sequence at Step 6. At Step 6, implied volatility collapses uniformly by 40% (the "Crush"). Any net long Vega positions held into Step 6 will hemorrhage PnL. The LLM must intelligently liquidate Vega-heavy positions before the crush whilst surviving Brownian noise leading up to the report.

## 4. Expert: Gamma Scalping (`gamma_scalping`)
- **Max Steps**: 12
- **Skill Tested**: Complex multi-legged arbitrage in non-stationary noise.
- **Goal**: Given a portfolio heavily skewed towards positive Gamma, the underlying spot price will wildly thrash over the duration of the episode. High Gamma dictates that Delta changes rapidly. As Delta shifts violently, the LLM must perpetually counter-trade Delta ("scalping" the fluctuations) to lock in realized profits, returning a net positive PnL by term-end.

## 5. Super-Boss: Vega/Gamma Stress (`vega_gamma_stress`)
- **Max Steps**: 15
- **Skill Tested**: Advanced dual-derivative institutional hedging under catastrophic tail-risk events.
- **Goal**: The environment injects an anomalous macro volatility shock combined with structural price failure. The LLM must construct a multi-legged position over the first 5 steps that drives **both** net Vega and net Gamma explicitly to `0.0` (+/- 0.05 tolerance threshold) to survive the incoming Gaussian shock. The grading heuristic dictates that deviation exponentially penalizes the score.
