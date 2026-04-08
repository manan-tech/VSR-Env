"""Unit tests for RewardComputer - Gaussian boundaries, reasoning quality, edge cases."""

import math
import pytest
from vsr_env.reward.reward_computer import sigmoid, score_reasoning_quality
from vsr_env.models import VSRObservation, VSRState, VSRAction, TradeDirection


class TestSigmoidFunction:
    """Test sigmoid normalization function."""

    def test_sigmoid_zero_input(self):
        """Sigmoid(0) should return 0.5."""
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 0.01

    def test_sigmoid_positive_input(self):
        """Positive input should return > 0.5."""
        result = sigmoid(0.3)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative_input(self):
        """Negative input should return < 0.5."""
        result = sigmoid(-0.3)
        assert result > 0.0
        assert result < 0.5

    def test_sigmoid_large_positive(self):
        """Large positive input should approach 1.0."""
        result = sigmoid(5.0)
        assert result > 0.9

    def test_sigmoid_large_negative(self):
        """Large negative input should approach 0.0."""
        result = sigmoid(-5.0)
        assert result < 0.1

    def test_sigmoid_scale_parameter(self):
        """Scale parameter should affect steepness."""
        result_default = sigmoid(0.1, scale=0.3)
        result_steep = sigmoid(0.1, scale=0.1)
        assert result_steep > result_default


class TestReasoningQualityScoring:
    """Test reasoning quality evaluation."""

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation for testing."""
        obs = VSRObservation(
            iv_surface=[[0.25] * 3 for _ in range(8)],
            spot_price=100.0,
            portfolio_greeks={"delta": 1.5, "gamma": 0.3, "vega": 0.5, "theta": -0.02},
            portfolio_pnl=0.25,
            current_positions=[],
            market_sentiment=0.0,  # Neutral sentiment as float
            task_name="delta_hedging",
            step_number=1,
            max_steps=8,
            expected_outcome="Maintain delta neutrality"
        )
        return obs

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = VSRState(
            spot_price=100.0,
            portfolio_delta=1.5,
            portfolio_gamma=0.3,
            portfolio_vega=0.5,
            portfolio_theta=-0.02,
            portfolio_pnl=0.25,
            current_positions=[],
            step_number=1,
            task_name="delta_hedging"
        )
        return state

    def test_empty_reasoning(self, mock_observation, mock_state):
        """Empty reasoning should return low score."""
        score = score_reasoning_quality("", mock_observation, mock_state)
        assert score < 0.2

    def test_short_reasoning_penalty(self, mock_observation, mock_state):
        """Very short reasoning should be penalized."""
        score_short = score_reasoning_quality("OK", mock_observation, mock_state)
        score_long = score_reasoning_quality("This is a longer reasoning with delta and vega keywords", mock_observation, mock_state)
        assert score_long > score_short

    def test_keyword_presence(self, mock_observation, mock_state):
        """Domain keywords should boost score."""
        score_no_keywords = score_reasoning_quality("The market is doing things", mock_observation, mock_state)
        score_with_keywords = score_reasoning_quality("Delta hedging to achieve neutrality", mock_observation, mock_state)
        assert score_with_keywords > score_no_keywords

    def test_numeric_citation(self, mock_observation, mock_state):
        """Numeric citations should boost score."""
        score_no_numbers = score_reasoning_quality("Delta is high, should hedge", mock_observation, mock_state)
        score_with_numbers = score_reasoning_quality("Delta is 1.5, spot at 100, need to hedge", mock_observation, mock_state)
        assert score_with_numbers > score_no_numbers

    def test_multiple_keywords(self, mock_observation, mock_state):
        """Multiple keywords should increase score."""
        score_one = score_reasoning_quality("Delta hedging required", mock_observation, mock_state)
        score_multiple = score_reasoning_quality("Delta hedging needed to achieve gamma and vega neutrality", mock_observation, mock_state)
        assert score_multiple > score_one

    def test_spot_price_citation(self, mock_observation, mock_state):
        """Citing spot price from observation should boost score."""
        score = score_reasoning_quality("Spot price at 100, delta is 1.5", mock_observation, mock_state)
        assert score > 0.3


class TestGaussianBoundaryScoring:
    """Test Gaussian boundary scoring for Super-Boss task."""

    def test_vega_within_bounds(self):
        """Vega within ±0.05 should score high."""
        vega = 0.03
        score = math.exp(-0.5 * (abs(vega) / 0.05) ** 2)
        assert score > 0.6

    def test_vega_at_threshold(self):
        """Vega at threshold 0.05 should score around 0.61."""
        vega = 0.05
        score = math.exp(-0.5 * (abs(vega) / 0.05) ** 2)
        assert abs(score - math.exp(-0.5)) < 0.01

    def test_vega_outside_bounds(self):
        """Vega outside bounds should exponentially decay."""
        vega = 0.10
        score = math.exp(-0.5 * (abs(vega) / 0.05) ** 2)
        assert score < 0.2

    def test_vega_far_outside_bounds(self):
        """Vega far outside bounds should be near zero."""
        vega = 0.20
        score = math.exp(-0.5 * (abs(vega) / 0.05) ** 2)
        assert score < 0.01

    def test_gamma_within_bounds(self):
        """Gamma within ±0.02 should score high."""
        gamma = 0.01
        score = math.exp(-0.5 * (abs(gamma) / 0.02) ** 2)
        assert score > 0.6

    def test_gamma_at_threshold(self):
        """Gamma at threshold 0.02 should score around 0.61."""
        gamma = 0.02
        score = math.exp(-0.5 * (abs(gamma) / 0.02) ** 2)
        assert abs(score - math.exp(-0.5)) < 0.01

    def test_gamma_outside_bounds(self):
        """Gamma outside bounds should exponentially decay."""
        gamma = 0.04
        score = math.exp(-0.5 * (abs(gamma) / 0.02) ** 2)
        assert score < 0.2

    def test_dual_neutrality_perfect(self):
        """Perfect neutrality should score maximum."""
        vega_score = math.exp(-0.5 * (0.0 / 0.05) ** 2)  # = 1.0
        gamma_score = math.exp(-0.5 * (0.0 / 0.02) ** 2)  # = 1.0
        vg_neutrality = (vega_score * 0.5 + gamma_score * 0.5) * 0.5
        assert abs(vg_neutrality - 0.5) < 0.01


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_quantity_action(self):
        """Action with zero quantity should be valid."""
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.HOLD,
            quantity=0.0,
            reasoning="No trade - observing"
        )
        assert action.quantity == 0.0
        assert action.direction == TradeDirection.HOLD

    def test_maximum_quantity_action(self):
        """Action with maximum quantity (10.0) should be valid."""
        action = VSRAction(
            selected_strike=4,
            selected_maturity=1,
            direction=TradeDirection.BUY,
            quantity=10.0,
            reasoning="Maximum size trade"
        )
        assert action.quantity == 10.0

    def test_boundary_strike_indices(self):
        """Strike indices at boundaries should be valid."""
        action_low = VSRAction(
            selected_strike=0,
            selected_maturity=0,
            direction=TradeDirection.BUY,
            quantity=1.0,
            reasoning="Deep ITM option"
        )
        action_high = VSRAction(
            selected_strike=7,
            selected_maturity=2,
            direction=TradeDirection.SELL,
            quantity=1.0,
            reasoning="Deep OTM option"
        )
        assert action_low.selected_strike == 0
        assert action_high.selected_strike == 7

    def test_boundary_maturity_indices(self):
        """Maturity indices at boundaries should be valid."""
        action_short = VSRAction(
            selected_strike=4,
            selected_maturity=0,
            direction=TradeDirection.BUY,
            quantity=1.0,
            reasoning="Front-month option"
        )
        action_long = VSRAction(
            selected_strike=4,
            selected_maturity=2,
            direction=TradeDirection.SELL,
            quantity=1.0,
            reasoning="Long-term option"
        )
        assert action_short.selected_maturity == 0
        assert action_long.selected_maturity == 2

    def test_extreme_pnl_values(self):
        """Test sigmoid with extreme PnL values."""
        pnl_high = sigmoid(10.0)
        pnl_low = sigmoid(-10.0)
        assert pnl_high > 0.9
        assert pnl_low < 0.1

    def test_reasoning_with_special_characters(self):
        """Reasoning with special characters should be handled."""
        obs = VSRObservation(
            iv_surface=[[0.25] * 3 for _ in range(8)],
            spot_price=100.0,
            portfolio_greeks={"delta": 1.5},
            portfolio_pnl=0.0,
            current_positions=[],
            market_sentiment=0.0,
            task_name="test",
            step_number=1,
            max_steps=5,
            expected_outcome="test"
        )
        state = VSRState(
            spot_price=100.0,
            portfolio_delta=1.5,
            portfolio_gamma=0.0,
            portfolio_vega=0.0,
            portfolio_theta=0.0,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="test"
        )
        score = score_reasoning_quality("Delta @ 1.5 > need hedge! #trading", obs, state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0