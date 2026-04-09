# Multi-Leg Options Strategies

This document provides comprehensive documentation for VSR-Env's multi-leg options strategy support, enabling agents to trade complex positions like straddles, strangles, and spreads.

## Overview

VSR-Env supports **atomic multi-leg execution**, allowing agents to construct entire strategies in a single action rather than executing legs sequentially. This matches real trading desk operations and provides cleaner learning signals.

## Supported Strategies

| Strategy | Legs | Use Case | Greek Profile |
|----------|------|----------|---------------|
| **Straddle** | 2 | Volatility speculation | Near-zero delta, long gamma/vega |
| **Strangle** | 2 | Cheaper vol bet | Near-zero delta, reduced gamma |
| **Vertical Spread** | 2 | Directional with defined risk | Net delta, limited gamma/vega |
| **Calendar Spread** | 2 | Term structure bet | Positive theta, long vega |

## Action API

### Multi-Leg Action Structure

```python
from vsr_env.models import VSRAction, StrategyType, StrategyLeg

# Multi-leg action example
action = VSRAction(
    strategy_type=StrategyType.STRADDLE,
    legs=[
        StrategyLeg(
            strike_idx=4,        # ATM (100.0)
            maturity_idx=1,      # 90-day
            option_type="call",
            direction="buy",
            quantity=1.0
        ),
        StrategyLeg(
            strike_idx=4,        # Same strike
            maturity_idx=1,      # Same expiry
            option_type="put",
            direction="buy",
            quantity=1.0
        ),
    ],
    reasoning="Long straddle expecting vol expansion"
)
```

### Single-Leg Action (Backward Compatible)

```python
# Single-leg with explicit option_type
action = VSRAction(
    selected_strike=4,
    selected_maturity=1,
    direction=TradeDirection.BUY,
    option_type="put",  # NEW: can specify put explicitly
    quantity=2.0,
    reasoning="Buying puts for downside protection"
)
```

---

## Strategy Details

### Straddle

**Definition**: Long call + Long put at same strike and expiry (or short both).

**When to Use**:
- Expect high realized volatility (long straddle)
- Expect low realized volatility (short straddle)
- Earnings announcements, FDA decisions, central bank events

**Greek Profile** (long straddle):
| Greek | Value | Interpretation |
|-------|-------|----------------|
| Delta | ≈ 0 | Direction-neutral |
| Gamma | + | Profits from large moves |
| Vega | + | Profits from IV increase |
| Theta | - | Time decay cost |

**Payoff Diagram**:
```
P&L
  │    ╱╲
  │   ╱  ╲
  │  ╱    ╲
──┼─┼──────┼── Spot
  │ K-prem K+prem
  │ ╲      ╱
  │  ╲____╱
```

**Max Profit**: Unlimited (long), Premium received (short)
**Max Loss**: Premium paid (long), Unlimited (short)
**Breakevens**: Strike ± Premium

**Example Action**:
```python
VSRAction(
    strategy_type=StrategyType.STRADDLE,
    legs=[
        StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
        StrategyLeg(strike_idx=4, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
    ],
    reasoning="Long ATM straddle before earnings, IV=25%, expecting realized vol >30%"
)
```

---

### Strangle

**Definition**: Long OTM call + Long OTM put at different strikes, same expiry.

**When to Use**:
- Cheaper alternative to straddle
- Expect very large moves
- IV is elevated, reducing straddle value

**Greek Profile** (long strangle):
| Greek | Value | Interpretation |
|-------|-------|----------------|
| Delta | ≈ 0 | Direction-neutral |
| Gamma | + (lower) | Profits from extreme moves |
| Vega | + | Long vol exposure |
| Theta | - | Time decay (lower than straddle) |

**Payoff Diagram**:
```
P&L
  │        ╱╲
  │       ╱  ╲
  │      ╱    ╲
──┼─────┼──────┼───┼── Spot
  │     K1    K2  K3
  │╲          ╱
  │ ╲________╱
```

**Max Profit**: Unlimited (long), Premium received (short)
**Max Loss**: Premium paid (long)
**Breakevens**: Lower strike - Premium, Upper strike + Premium

**Example Action**:
```python
VSRAction(
    strategy_type=StrategyType.STRANGLE,
    legs=[
        StrategyLeg(strike_idx=6, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),  # 105 strike
        StrategyLeg(strike_idx=2, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),   # 90 strike
    ],
    reasoning="Long strangle, OTM call and put for cheap vol exposure"
)
```

---

### Vertical Spread

**Definition**: Two options of same type (both calls or both puts), same expiry, different strikes.

**Types**:
- **Bull Call Spread**: Buy lower strike call, sell higher strike call
- **Bear Put Spread**: Buy higher strike put, sell lower strike put
- **Bear Call Spread** (credit): Sell lower strike call, buy higher strike call
- **Bull Put Spread** (credit): Sell higher strike put, buy lower strike put

**When to Use**:
- Bull call spread: Bullish with capped upside
- Bear put spread: Bearish with capped downside
- Credit spreads: Neutral to slightly directional, income generation

**Greek Profile** (debit spread):
| Greek | Value | Interpretation |
|-------|-------|----------------|
| Delta | + or - | Directional bias |
| Gamma | ~ 0 | Limited gamma |
| Vega | ~ 0 | Minimal vol exposure |
| Theta | + or - | Depends on spread type |

**Payoff Diagram** (Bull Call Spread):
```
P&L
  │          ___
  │         /
  │        /
──┼───────┼──────┼── Spot
  │      K1     K2
  │______╲
```

**Max Profit**: Width - Debit (debit spread), Credit received (credit spread)
**Max Loss**: Debit paid (debit spread), Width - Credit (credit spread)
**Breakeven**: Lower strike + Debit (bull call)

**Example Action**:
```python
VSRAction(
    strategy_type=StrategyType.VERTICAL_SPREAD,
    legs=[
        StrategyLeg(strike_idx=3, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),  # 97.5
        StrategyLeg(strike_idx=5, maturity_idx=1, option_type="call", direction="sell", quantity=1.0),  # 102.5
    ],
    reasoning="Bull call spread, expecting moderate upside to 102.5"
)
```

---

### Calendar Spread

**Definition**: Same strike, same option type, different expiries. Buy longer-dated, sell shorter-dated.

**When to Use**:
- Expect realized vol to be lower than implied
- Betting on term structure normalization
- Low-cost vol exposure

**Greek Profile** (long calendar):
| Greek | Value | Interpretation |
|-------|-------|----------------|
| Delta | ≈ 0 | Neutral if ATM |
| Gamma | - | Short near-term gamma |
| Vega | + | Long vega |
| Theta | + | Time decay favorable |

**Max Profit**: Maximum when stock at strike at near-term expiry
**Breakeven**: Complex, depends on forward vol

**Example Action**:
```python
VSRAction(
    strategy_type=StrategyType.CALENDAR_SPREAD,
    legs=[
        StrategyLeg(strike_idx=4, maturity_idx=2, option_type="call", direction="buy", quantity=1.0),  # 180-day
        StrategyLeg(strike_idx=4, maturity_idx=0, option_type="call", direction="sell", quantity=1.0),  # 30-day
    ],
    reasoning="Calendar spread, betting on low realized vol and term structure flattening"
)
```

---

## Strategy-Level P&L and Greeks

### Observation Fields

Multi-leg positions are tracked at both the position level and strategy level:

```python
class VSRObservation(BaseModel):
    # ... existing fields ...
    active_strategies: List[StrategyInfo]  # NEW

class StrategyInfo(BaseModel):
    strategy_id: str
    strategy_type: str
    net_greeks: Dict[str, float]
    unrealized_pnl: float
    legs_summary: str
```

### Portfolio Functions

```python
from vsr_env.engine.portfolio import (
    add_strategy,           # Add multi-leg strategy atomically
    get_positions_by_strategy,  # Get all legs of a strategy
    compute_strategy_pnl,   # Strategy-level P&L
    compute_strategy_greeks,  # Strategy-level Greeks
    close_strategy,         # Close all legs atomically
    get_active_strategies,  # List active strategy IDs
)
```

---

## Tasks Using Multi-Leg Strategies

### Straddle Trading Task

**Difficulty**: Hard (Tier 4)
**Max Steps**: 13

**Objective**: Analyze IV levels and decide whether to buy or sell straddle.

**Scoring**:
- Direction correctness: 30%
- Entry timing: 20%
- P&L realized: 30%
- Risk management: 20%

```python
from vsr_env.tasks.straddle_trading import StraddleTradingTask, StraddleTradingGrader

task = StraddleTradingTask()
grader = StraddleTradingGrader()
```

### Vertical Spread Task

**Difficulty**: Medium (Tier 2)
**Max Steps**: 8

**Objective**: Construct appropriate spread for directional bet.

**Scoring**:
- Correct spread direction: 25%
- Strike selection: 25%
- Entry price: 20%
- Exit timing: 30%

```python
from vsr_env.tasks.vertical_spread import VerticalSpreadTask, VerticalSpreadGrader

task = VerticalSpreadTask()
grader = VerticalSpreadGrader()
```

---

## Efficiency Bonus

Using atomic multi-leg actions provides a reward bonus:

```python
# Sequential single-leg actions (no bonus)
action1 = VSRAction(selected_strike=4, ..., reasoning="Buy call")
action2 = VSRAction(selected_strike=4, ..., reasoning="Buy put")

# Atomic multi-leg action (+10% efficiency bonus)
action = VSRAction(
    strategy_type=StrategyType.STRADDLE,
    legs=[...],
    reasoning="Long straddle"
)
```

---

## Strike and Maturity Reference

### STRIKES Array
```python
STRIKES = [85.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0]
# Indices:  0     1     2     3      4       5       6      7
```

### MATURITIES Array
```python
MATURITIES = [30/365, 90/365, 180/365]  # In years
# Indices:      0        1        2
```

### ATM Strike
Index 4 = 100.0 (at-the-money when spot = 100)

---

## Best Practices

### Choosing Strategy Type

| Market Outlook | Recommended Strategy |
|----------------|---------------------|
| High realized vol expected | Long straddle |
| Very high moves expected | Long strangle |
| Moderate directional bias | Vertical spread |
| Directional with income focus | Credit spread |
| Term structure opportunity | Calendar spread |

### Position Sizing Guidelines

- **Straddles**: 1-2 contracts per trade (high gamma/vega exposure)
- **Strangles**: 1-2 contracts (wider breakevens)
- **Spreads**: Can size larger due to defined risk
- **Calendars**: 1-2 contracts (term structure timing is key)

### Exit Strategies

- **Long straddle**: Close when P&L reaches 50% of max profit, or after event
- **Short straddle**: Close at 50% profit of premium received
- **Vertical spreads**: Trade around the strikes, close near expiry
- **Calendar**: Close near-term expiry, manage roll if needed

---

## Testing Multi-Leg Strategies

Run the test suite:

```bash
pytest tests/test_multi_leg_strategies.py -v
```

Current test coverage:
- 40 tests passing
- Strategy class validation
- Action model extension
- Portfolio multi-leg support
- Integration tests

---

## Future Enhancements

Planned strategy types:
- **Iron Condor**: 4-leg income strategy
- **Butterfly**: 3-leg neutral strategy
- **Diagonal Spread**: Different strike and expiry
- **Ratio Spread**: Unequal leg quantities

---

## References

- Natenberg, S. (1994). *Option Volatility and Pricing*
- Hull, J. (2017). *Options, Futures, and Other Derivatives*
- CBOE Options Institute: [Strategy Discussions](https://www.cboe.com/education/)

---

*VSR-Env Multi-Leg Strategy Support - Enabling realistic options trading for RL agents*