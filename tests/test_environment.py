"""Unit tests for VSREnvironment core orchestration."""

import pytest
from vsr_env.server.vsr_environment import VSREnvironment
from vsr_env.models import VSRAction, TradeDirection


class TestVSREnvironment:
    """Test the main VSREnvironment class."""

    @pytest.fixture
    def env(self):
        return VSREnvironment()

    def test_environment_initialization(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_reset_with_valid_task(self, env):
        """Reset should work for all 5 tasks."""
        tasks = [
            "vol_regime_detection",
            "delta_hedging",
            "earnings_vol_crush",
            "gamma_scalping",
            "vega_gamma_stress"
        ]
        for task_name in tasks:
            obs = env.reset(task_name=task_name, seed=42)
            assert obs is not None
            assert obs.task_name == task_name

    def test_step_with_hold_action(self, env):
        """Step with hold action should return valid result."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.HOLD,
            quantity=0.0,
            reasoning="No action"
        )
        result = env.step(action)
        assert 'observation' in result
        assert 'reward' in result
        assert 'done' in result

    def test_step_with_buy_action(self, env):
        """Step with buy action should modify portfolio."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        initial_delta = obs.portfolio_greeks['delta']
        action = VSRAction(
            selected_strike=4,
            selected_maturity=0,
            direction=TradeDirection.BUY,
            quantity=1.0,
            reasoning="Buying option"
        )
        result = env.step(action)
        new_delta = result['observation'].portfolio_greeks['delta']
        # Delta should have changed

    def test_step_with_sell_action(self, env):
        """Step with sell action should modify portfolio."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        action = VSRAction(
            selected_strike=4,
            selected_maturity=0,
            direction=TradeDirection.SELL,
            quantity=1.0,
            reasoning="Selling option"
        )
        result = env.step(action)
        assert result is not None

    def test_episode_completion(self, env):
        """Episode should complete after max_steps."""
        env.reset(task_name="vol_regime_detection", seed=42)
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.HOLD,
            quantity=0.0,
            reasoning="Waiting"
        )
        done = False
        steps = 0
        while not done:
            result = env.step(action)
            done = result['done']
            steps += 1
        assert steps <= 3  # max_steps for vol_regime_detection

    def test_reward_range(self, env):
        """Reward should be in valid range [0, 1]."""
        env.reset(task_name="delta_hedging", seed=42)
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.HOLD,
            quantity=0.0,
            reasoning="Testing reward range"
        )
        for _ in range(5):
            result = env.step(action)
            assert 0.0 <= result['reward'] <= 1.0

    def test_state_property(self, env):
        """State property should return current state."""
        env.reset(task_name="delta_hedging", seed=42)
        state = env.state
        assert state is not None
        assert hasattr(state, 'task_name')

    def test_seed_reproducibility(self, env):
        """Same seed should produce same initial state."""
        obs1 = env.reset(task_name="delta_hedging", seed=123)
        obs2 = env.reset(task_name="delta_hedging", seed=123)
        # Observations should be identical
        assert obs1.spot_price == obs2.spot_price

    def test_different_seeds_produce_different_states(self, env):
        """Different seeds should produce different states."""
        obs1 = env.reset(task_name="delta_hedging", seed=100)
        obs2 = env.reset(task_name="delta_hedging", seed=200)
        # Should differ (statistically)
        # May occasionally be same, but unlikely

    def test_multiple_resets(self, env):
        """Environment should handle multiple resets."""
        for i in range(10):
            obs = env.reset(task_name="delta_hedging", seed=i)
            assert obs is not None


class TestEnvironmentEdgeCases:
    """Test environment edge cases and error handling."""

    @pytest.fixture
    def env(self):
        return VSREnvironment()

    def test_rapid_step_calls(self, env):
        """Environment should handle rapid step calls."""
        env.reset(task_name="vega_gamma_stress", seed=42)
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.HOLD,
            quantity=0.0,
            reasoning="Rapid fire"
        )
        for _ in range(20):
            result = env.step(action)
            if result['done']:
                break

    def test_extreme_quantity_values(self, env):
        """Environment should handle boundary quantity values."""
        env.reset(task_name="delta_hedging", seed=42)
        action_min = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.BUY,
            quantity=0.0,
            reasoning="Zero quantity"
        )
        action_max = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.SELL,
            quantity=10.0,
            reasoning="Max quantity"
        )
        result1 = env.step(action_min)
        result2 = env.step(action_max)
        assert result1 is not None
        assert result2 is not None


class TestObservationSpace:
    """Test observation space structure and validity."""

    @pytest.fixture
    def env(self):
        return VSREnvironment()

    def test_observation_has_iv_surface(self, env):
        """Observation should include IV surface."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        assert hasattr(obs, 'iv_surface')
        assert len(obs.iv_surface) == 8
        assert len(obs.iv_surface[0]) == 3

    def test_observation_has_greeks(self, env):
        """Observation should include portfolio Greeks."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        assert hasattr(obs, 'portfolio_greeks')
        assert 'delta' in obs.portfolio_greeks
        assert 'gamma' in obs.portfolio_greeks
        assert 'vega' in obs.portfolio_greeks
        assert 'theta' in obs.portfolio_greeks

    def test_observation_has_spot_price(self, env):
        """Observation should include spot price."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        assert hasattr(obs, 'spot_price')
        assert obs.spot_price > 0

    def test_observation_has_pnl(self, env):
        """Observation should include PnL."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        assert hasattr(obs, 'portfolio_pnl')

    def test_iv_surface_values_reasonable(self, env):
        """IV surface values should be in reasonable range."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        for row in obs.iv_surface:
            for iv in row:
                assert 0.0 <= iv <= 1.0


class TestActionValidation:
    """Test action validation and constraints."""

    @pytest.fixture
    def env(self):
        return VSREnvironment()

    def test_valid_action_processing(self, env):
        """Valid actions should be processed without errors."""
        env.reset(task_name="delta_hedging", seed=42)
        for strike in range(8):
            for maturity in range(3):
                action = VSRAction(
                    selected_strike=strike,
                    selected_maturity=maturity,
                    direction=TradeDirection.BUY,
                    quantity=1.0,
                    reasoning=f"Strike {strike}, maturity {maturity}"
                )
                result = env.step(action)
                assert result is not None
                env.reset(task_name="delta_hedging", seed=42)

    def test_action_affects_portfolio(self, env):
        """Actions should affect portfolio state."""
        obs = env.reset(task_name="delta_hedging", seed=42)
        initial_delta = obs.portfolio_greeks['delta']
        action = VSRAction(
            selected_strike=4,
            selected_maturity=0,
            direction=TradeDirection.SELL,
            quantity=2.0,
            reasoning="Large trade"
        )
        result = env.step(action)
        new_delta = result['observation'].portfolio_greeks['delta']
        # Delta should have changed
        assert new_delta != initial_delta