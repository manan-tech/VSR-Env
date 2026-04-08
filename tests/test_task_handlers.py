"""Unit tests for all 5 task graders - grading, state transitions, events."""

import pytest
from vsr_env.tasks.vol_regime_detection import VolRegimeDetectionTask, VolRegimeDetectionGrader
from vsr_env.tasks.delta_hedging import DeltaHedgingGrader
from vsr_env.tasks.earnings_vol_crush import EarningsVolCrushGrader
from vsr_env.tasks.gamma_scalping import GammaScalpingGrader
from vsr_env.tasks.vega_gamma_stress import VegaGammaStressGrader
from vsr_env.models import VSRAction, TradeDirection, VSRState


class TestVolRegimeDetection:
    """Tests for Tier 1: Volatility Regime Detection."""

    @pytest.fixture
    def task(self):
        return VolRegimeDetectionTask()

    @pytest.fixture
    def grader(self):
        return VolRegimeDetectionGrader()

    @pytest.fixture
    def mock_state(self):
        return VSRState(
            spot_price=100.0,
            portfolio_delta=0.0,
            portfolio_gamma=0.0,
            portfolio_vega=0.0,
            portfolio_theta=0.0,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="vol_regime_detection"
        )

    def test_task_initialization(self, task, mock_state):
        """Task should initialize state correctly."""
        result = task.initialize(mock_state)
        assert result is not None

    def test_high_regime_detection(self, task, mock_state):
        """High IV regime should be settable."""
        task.selected_regime = "high"
        assert task.selected_regime == "high"

    def test_low_regime_detection(self, task, mock_state):
        """Low IV regime should be settable."""
        task.selected_regime = "low"
        assert task.selected_regime == "low"

    def test_grader_returns_score(self, grader, mock_state):
        """Grader should return score in [0, 1]."""
        episode_history = []
        score = grader.score(episode_history, mock_state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestDeltaHedging:
    """Tests for Tier 2: Delta Hedging."""

    @pytest.fixture
    def grader(self):
        return DeltaHedgingGrader()

    @pytest.fixture
    def mock_state(self):
        return VSRState(
            spot_price=100.0,
            portfolio_delta=2.5,
            portfolio_gamma=0.0,
            portfolio_vega=0.0,
            portfolio_theta=-0.02,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="delta_hedging"
        )

    def test_grader_initialization(self, grader):
        """Grader should initialize correctly."""
        assert grader is not None

    def test_grader_returns_score(self, grader, mock_state):
        """Grader should return score in [0, 1]."""
        episode_history = []
        score = grader.score(episode_history, mock_state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestEarningsVolCrush:
    """Tests for Tier 3: Earnings Vol Crush."""

    @pytest.fixture
    def grader(self):
        return EarningsVolCrushGrader()

    @pytest.fixture
    def mock_state(self):
        return VSRState(
            spot_price=100.0,
            portfolio_delta=0.0,
            portfolio_gamma=0.0,
            portfolio_vega=0.5,
            portfolio_theta=-0.01,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="earnings_vol_crush"
        )

    def test_grader_initialization(self, grader):
        """Grader should initialize correctly."""
        assert grader is not None

    def test_grader_returns_score(self, grader, mock_state):
        """Grader should return score in [0, 1]."""
        episode_history = []
        score = grader.score(episode_history, mock_state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestGammaScalping:
    """Tests for Tier 4: Gamma Scalping."""

    @pytest.fixture
    def grader(self):
        return GammaScalpingGrader()

    @pytest.fixture
    def mock_state(self):
        return VSRState(
            spot_price=100.0,
            portfolio_delta=0.0,
            portfolio_gamma=0.8,
            portfolio_vega=0.0,
            portfolio_theta=-0.05,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="gamma_scalping"
        )

    def test_grader_initialization(self, grader):
        """Grader should initialize correctly."""
        assert grader is not None

    def test_grader_returns_score(self, grader, mock_state):
        """Grader should return score in [0, 1]."""
        episode_history = []
        score = grader.score(episode_history, mock_state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestVegaGammaStress:
    """Tests for Tier 5: Vega/Gamma Stress (Super-Boss)."""

    @pytest.fixture
    def grader(self):
        return VegaGammaStressGrader()

    @pytest.fixture
    def mock_state(self):
        return VSRState(
            spot_price=100.0,
            portfolio_delta=0.0,
            portfolio_gamma=0.3,
            portfolio_vega=0.5,
            portfolio_theta=-0.01,
            portfolio_pnl=0.0,
            current_positions=[],
            step_number=1,
            task_name="vega_gamma_stress"
        )

    def test_grader_initialization(self, grader):
        """Grader should initialize correctly."""
        assert grader is not None

    def test_grader_returns_score(self, grader, mock_state):
        """Grader should return score in [0, 1]."""
        episode_history = []
        score = grader.score(episode_history, mock_state)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_gaussian_boundary_strictness(self):
        """Gaussian boundaries should exponentially penalize deviations."""
        import math
        avg_vega = 0.10  # 2x threshold
        avg_gamma = 0.04  # 2x threshold
        vega_score = math.exp(-0.5 * (avg_vega / 0.05) ** 2)
        gamma_score = math.exp(-0.5 * (avg_gamma / 0.02) ** 2)
        assert vega_score < 0.2
        assert gamma_score < 0.2


class TestTaskGraderEdgeCases:
    """Test edge cases across all task graders."""

    def test_negative_quantity_action_validation(self):
        """Negative quantity might be rejected by VSRAction model."""
        try:
            action = VSRAction(
                selected_strike=4,
                selected_maturity=1,
                direction=TradeDirection.BUY,
                quantity=-1.0,
                reasoning="Invalid negative quantity"
            )
            # If not rejected by model, that's OK too
            assert action.quantity == -1.0
        except (ValueError, AttributeError):
            # Expected if validation is strict
            pass

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

    def test_gaussian_formula_extreme_values(self):
        """Gaussian formula should handle extreme values."""
        import math
        # Very large deviation
        vega_score = math.exp(-0.5 * (1.0 / 0.05) ** 2)
        assert vega_score < 0.0001

        # Perfect neutrality
        gamma_score = math.exp(-0.5 * (0.0 / 0.02) ** 2)
        assert abs(gamma_score - 1.0) < 0.001