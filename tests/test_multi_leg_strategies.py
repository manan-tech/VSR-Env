"""Tests for multi-leg strategy support in VSR-Env."""

import pytest

from vsr_env.models import (
    StrategyLeg,
    StrategyType,
    StrategyInfo,
    TradeDirection,
    VSRAction,
    VSRObservation,
    VSRState,
)
from vsr_env.strategies import Straddle, Strangle, VerticalSpread, CalendarSpread
from vsr_env.strategies.base import OptionStrategy


# ============================================================================
# Strategy Class Tests
# ============================================================================


class TestStraddle:
    """Tests for Straddle strategy class."""

    def test_straddle_creation_valid_legs(self):
        """Straddle can be created with valid legs."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)
        assert straddle.get_strategy_type() == "straddle"
        assert len(straddle.legs) == 2

    def test_straddle_creation_invalid_legs_different_strike(self):
        """Straddle with different strikes raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="same strike"):
            Straddle(legs)

    def test_straddle_creation_invalid_legs_different_maturity(self):
        """Straddle with different maturities raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 0, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="same maturity"):
            Straddle(legs)

    def test_straddle_creation_invalid_missing_put(self):
        """Straddle with two calls raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="one call and one put"):
            Straddle(legs)

    def test_straddle_is_long(self):
        """Straddle correctly identifies long vs short."""
        legs_long = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle_long = Straddle(legs_long)
        assert straddle_long.is_long() is True

        legs_short = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "sell", "quantity": 1.0},
        ]
        straddle_short = Straddle(legs_short)
        assert straddle_short.is_long() is False

    def test_straddle_compute_payoff_atm(self):
        """Straddle payoff calculation at different spot prices."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)
        # Strike index 4 = 100.0

        # At the money: payoff is 0
        payoff_atm = straddle.compute_payoff(100.0)
        assert payoff_atm == 0.0

        # Above strike: call has value
        payoff_above = straddle.compute_payoff(105.0)
        assert payoff_above == 5.0

        # Below strike: put has value
        payoff_below = straddle.compute_payoff(95.0)
        assert payoff_below == 5.0

    def test_straddle_get_description(self):
        """Straddle generates readable description."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)
        desc = straddle.get_description()
        assert "STRADDLE" in desc
        assert "Leg 1" in desc
        assert "Leg 2" in desc


class TestStrangle:
    """Tests for Strangle strategy class."""

    def test_strangle_creation_valid_legs(self):
        """Strangle can be created with OTM options."""
        legs = [
            {"strike_idx": 6, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 2, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        strangle = Strangle(legs)
        assert strangle.get_strategy_type() == "strangle"

    def test_strangle_creation_invalid_same_strike(self):
        """Strangle with same strike raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="different strikes"):
            Strangle(legs)

    def test_strangle_call_above_put(self):
        """Strangle requires call strike above put strike."""
        legs = [
            {"strike_idx": 2, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 6, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="call strike must be above put strike"):
            Strangle(legs)


class TestVerticalSpread:
    """Tests for VerticalSpread strategy class."""

    def test_vertical_spread_creation_valid_legs(self):
        """Vertical spread can be created with same option type, different strikes."""
        legs = [
            {"strike_idx": 3, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        spread = VerticalSpread(legs)
        assert spread.get_strategy_type() == "vertical_spread"

    def test_vertical_spread_invalid_different_option_types(self):
        """Vertical spread with call and put raises ValueError."""
        legs = [
            {"strike_idx": 3, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "put", "direction": "sell", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="same option type"):
            VerticalSpread(legs)

    def test_vertical_spread_invalid_same_strikes(self):
        """Vertical spread with same strike raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="different strikes"):
            VerticalSpread(legs)

    def test_vertical_spread_invalid_same_direction(self):
        """Vertical spread requires opposite directions."""
        legs = [
            {"strike_idx": 3, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="opposite directions"):
            VerticalSpread(legs)

    def test_vertical_spread_is_bull(self):
        """Bull spread correctly identified."""
        # Bull call spread: buy lower strike, sell higher strike
        legs = [
            {"strike_idx": 3, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        spread = VerticalSpread(legs)
        assert spread.is_bull_spread() is True

    def test_vertical_spread_is_debit(self):
        """Debit spread correctly identified."""
        # Debit call spread: buy lower strike call
        legs = [
            {"strike_idx": 3, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 5, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        spread = VerticalSpread(legs)
        assert spread.is_debit_spread() is True


class TestCalendarSpread:
    """Tests for CalendarSpread strategy class."""

    def test_calendar_spread_creation_valid_legs(self):
        """Calendar spread can be created with same strike, different maturities."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 2, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 0, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        calendar = CalendarSpread(legs)
        assert calendar.get_strategy_type() == "calendar_spread"

    def test_calendar_spread_invalid_different_strikes(self):
        """Calendar spread with different strikes raises ValueError."""
        legs = [
            {"strike_idx": 3, "maturity_idx": 2, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 0, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="same strike"):
            CalendarSpread(legs)

    def test_calendar_spread_invalid_same_maturity(self):
        """Calendar spread with same maturity raises ValueError."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        with pytest.raises(ValueError, match="different maturities"):
            CalendarSpread(legs)

    def test_calendar_spread_is_long(self):
        """Long calendar correctly identified."""
        # Long calendar: buy longer-dated, sell shorter-dated
        legs = [
            {"strike_idx": 4, "maturity_idx": 2, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 0, "option_type": "call", "direction": "sell", "quantity": 1.0},
        ]
        calendar = CalendarSpread(legs)
        assert calendar.is_long_calendar() is True


class TestStrategyGreeks:
    """Tests for strategy Greek aggregation."""

    def test_straddle_net_greeks(self):
        """Straddle computes net Greeks correctly."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        # Mock Greek data
        greek_data = [
            {"delta": 0.5, "gamma": 0.08, "vega": 0.15, "theta": -0.02},
            {"delta": -0.5, "gamma": 0.08, "vega": 0.15, "theta": -0.02},
        ]
        net_greeks = straddle.get_net_greeks(greek_data)

        # Delta should be near 0 for straddle
        assert abs(net_greeks["delta"]) < 0.01
        # Gamma should be doubled
        assert abs(net_greeks["gamma"] - 0.16) < 0.01

    def test_strategy_greek_data_length_mismatch(self):
        """Greek data length must match leg count."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        greek_data = [{"delta": 0.5}]  # Only one leg's data

        with pytest.raises(ValueError, match="must match"):
            straddle.get_net_greeks(greek_data)


class TestStrategyPnL:
    """Tests for strategy P&L computation."""

    def test_straddle_compute_pnl(self):
        """Straddle P&L computation."""
        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        entry_prices = [5.0, 4.5]  # Call and put entry prices
        current_prices = [5.5, 4.0]  # Current prices

        pnl = straddle.compute_pnl(entry_prices, current_prices)
        # Call P&L: (5.5 - 5.0) = +0.5
        # Put P&L: (4.0 - 4.5) = -0.5
        # Total: 0.0
        assert abs(pnl) < 0.01


# ============================================================================
# Action Model Tests
# ============================================================================


class TestVSRActionMultiLeg:
    """Tests for multi-leg VSRAction."""

    def test_single_leg_action_backward_compat(self):
        """Single-leg actions work with default values."""
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.BUY,
            quantity=2.0,
            reasoning="Buying ATM call",
        )
        assert action.selected_strike == 4
        assert action.direction == TradeDirection.BUY
        assert action.option_type == "call"  # Default
        assert action.strategy_type is None
        assert action.legs is None

    def test_multi_leg_action_straddle(self):
        """Multi-leg action with straddle strategy."""
        action = VSRAction(
            strategy_type=StrategyType.STRADDLE,
            legs=[
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
            ],
            reasoning="Long straddle for vol expansion",
        )
        assert action.strategy_type == StrategyType.STRADDLE
        assert len(action.legs) == 2

    def test_multi_leg_action_invalid_leg_count(self):
        """Invalid leg count for strategy type raises error."""
        with pytest.raises(ValueError, match="requires 2 legs"):
            VSRAction(
                strategy_type=StrategyType.STRADDLE,
                legs=[
                    StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
                ],
                reasoning="Incomplete straddle",
            )

    def test_option_type_field_required(self):
        """Option type can be specified for single-leg."""
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.SELL,
            option_type="put",
            quantity=1.0,
            reasoning="Selling ATM put",
        )
        assert action.option_type == "put"

    def test_strategy_type_enforces_correct_leg_count(self):
        """Each strategy type requires correct number of legs."""
        # Iron condor requires 4 legs
        with pytest.raises(ValueError, match="requires 4 legs"):
            VSRAction(
                strategy_type=StrategyType.IRON_CONDOR,
                legs=[
                    StrategyLeg(strike_idx=1, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
                    StrategyLeg(strike_idx=2, maturity_idx=1, option_type="put", direction="sell", quantity=1.0),
                ],
                reasoning="Incomplete iron condor",
            )

    def test_action_serialization_multi_leg(self):
        """Multi-leg action can be serialized to dict."""
        action = VSRAction(
            strategy_type=StrategyType.STRADDLE,
            legs=[
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
            ],
            reasoning="Long straddle",
        )
        data = action.model_dump()
        assert data["strategy_type"] == "straddle"
        assert len(data["legs"]) == 2

    def test_action_deserialization_multi_leg(self):
        """Multi-leg action can be created from dict."""
        data = {
            "strategy_type": "straddle",
            "legs": [
                {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
                {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
            ],
            "reasoning": "Long straddle from dict",
        }
        action = VSRAction(**data)
        assert action.strategy_type == StrategyType.STRADDLE


# ============================================================================
# Portfolio Multi-Leg Tests
# ============================================================================


class TestPortfolioMultiLeg:
    """Tests for portfolio multi-leg strategy support."""

    def test_add_strategy_creates_all_legs(self):
        """Adding a strategy creates all positions."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, get_positions_by_strategy
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        add_strategy(state, straddle, engine)

        assert len(state.positions) == 2

    def test_strategy_legs_share_strategy_id(self):
        """All legs of a strategy share the same strategy_id."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, get_positions_by_strategy
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)
        strategy_id = straddle.strategy_id

        add_strategy(state, straddle, engine)

        for pos in state.positions:
            assert pos.get("strategy_id") == strategy_id

    def test_get_active_strategies(self):
        """Can retrieve list of active strategy IDs."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, get_active_strategies
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        add_strategy(state, straddle, engine)

        active = get_active_strategies(state)
        assert len(active) == 1
        assert straddle.strategy_id in active

    def test_compute_strategy_pnl(self):
        """Can compute P&L for a specific strategy."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, compute_strategy_pnl
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        add_strategy(state, straddle, engine)

        pnl = compute_strategy_pnl(state, straddle.strategy_id, engine)
        # P&L should be 0 at entry
        assert isinstance(pnl, float)

    def test_compute_strategy_greeks(self):
        """Can compute Greeks for a specific strategy."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, compute_strategy_greeks
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        add_strategy(state, straddle, engine)

        greeks = compute_strategy_greeks(state, straddle.strategy_id, engine)
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_close_strategy_removes_positions(self):
        """Closing a strategy removes all its positions."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, close_strategy
        from vsr_env.models import VSRState

        state = VSRState()
        engine = OptionChainEngine()

        legs = [
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
            {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0},
        ]
        straddle = Straddle(legs)

        add_strategy(state, straddle, engine)
        assert len(state.positions) == 2

        close_strategy(state, straddle.strategy_id, engine)
        assert len(state.positions) == 0


# ============================================================================
# Observation Model Tests
# ============================================================================


class TestVSRObservationStrategies:
    """Tests for observation with strategy info."""

    def test_observation_includes_active_strategies(self):
        """Observation can include active strategies."""
        obs = VSRObservation(
            iv_surface=[[0.2] * 3 for _ in range(8)],
            spot_price=100.0,
            portfolio_greeks={"delta": 0.0, "gamma": 0.16, "vega": 0.3, "theta": -0.04},
            active_strategies=[
                StrategyInfo(
                    strategy_id="straddle_123",
                    strategy_type="straddle",
                    net_greeks={"delta": 0.0, "gamma": 0.16, "vega": 0.3, "theta": -0.04},
                    unrealized_pnl=0.0,
                    legs_summary="Long ATM straddle",
                )
            ],
        )
        assert len(obs.active_strategies) == 1
        assert obs.active_strategies[0].strategy_type == "straddle"

    def test_observation_default_empty_strategies(self):
        """Observation defaults to empty strategies list."""
        obs = VSRObservation(
            iv_surface=[[0.2] * 3 for _ in range(8)],
            spot_price=100.0,
            portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        )
        assert obs.active_strategies == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestMultiLegIntegration:
    """Integration tests for multi-leg strategy workflow."""

    def test_full_straddle_workflow(self):
        """Complete workflow: create action → add to portfolio → compute Greeks."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy, compute_strategy_greeks
        from vsr_env.models import VSRState

        # 1. Create multi-leg action
        action = VSRAction(
            strategy_type=StrategyType.STRADDLE,
            legs=[
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
                StrategyLeg(strike_idx=4, maturity_idx=1, option_type="put", direction="buy", quantity=1.0),
            ],
            reasoning="Long straddle for earnings event",
        )

        # 2. Create strategy from action
        from vsr_env.strategies import create_strategy_from_action
        strategy = create_strategy_from_action(action)

        # 3. Add to portfolio
        state = VSRState()
        engine = OptionChainEngine()
        add_strategy(state, strategy, engine)

        # 4. Verify
        assert len(state.positions) == 2
        greeks = compute_strategy_greeks(state, strategy.strategy_id, engine)
        assert "delta" in greeks

    def test_vertical_spread_workflow(self):
        """Vertical spread workflow."""
        from vsr_env.engine.option_chain import OptionChainEngine
        from vsr_env.engine.portfolio import add_strategy
        from vsr_env.models import VSRState

        action = VSRAction(
            strategy_type=StrategyType.VERTICAL_SPREAD,
            legs=[
                StrategyLeg(strike_idx=3, maturity_idx=1, option_type="call", direction="buy", quantity=1.0),
                StrategyLeg(strike_idx=5, maturity_idx=1, option_type="call", direction="sell", quantity=1.0),
            ],
            reasoning="Bull call spread for directional bet",
        )

        from vsr_env.strategies import create_strategy_from_action
        strategy = create_strategy_from_action(action)

        state = VSRState()
        engine = OptionChainEngine()
        add_strategy(state, strategy, engine)

        assert len(state.positions) == 2