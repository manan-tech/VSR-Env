"""VSR Environment core implementation.

Orchestrates all engine components (pricing, market sim, portfolio, rewards)
into a complete OpenEnv-compliant environment with reset/step/state interface.
"""

import uuid
import random
from typing import Optional, Dict, Any, List

import numpy as np

from vsr_env.models import (
    TradeDirection,
    VSRAction,
    VSRObservation,
    VSRReward,
    VSRState,
)
from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.engine.market_sim import (
    advance_market,
    trigger_regime_shift,
    trigger_vol_crush,
    inject_oscillation,
)
from vsr_env.engine.portfolio import (
    add_position,
    compute_portfolio_greeks,
    update_positions_on_market_move,
)
from vsr_env.reward.reward_computer import RewardComputer
from vsr_env.tasks.delta_hedging import DeltaHedgingTask, DeltaHedgingGrader
from vsr_env.tasks.earnings_vol_crush import (
    EarningsVolCrushTask,
    EarningsVolCrushGrader,
)
from vsr_env.tasks.gamma_scalping import GammaScalpingTask, GammaScalpingGrader
from vsr_env.tasks.vol_regime_detection import (
    VolRegimeDetectionTask,
    VolRegimeDetectionGrader,
)

# Task configurations
TASK_CONFIG = {
    "vol_regime_detection": {
        "max_steps": 1,
        "task_class": VolRegimeDetectionTask,
        "grader_class": VolRegimeDetectionGrader,
    },
    "delta_hedging": {
        "max_steps": 5,
        "task_class": DeltaHedgingTask,
        "grader_class": DeltaHedgingGrader,
    },
    "earnings_vol_crush": {
        "max_steps": 8,
        "task_class": EarningsVolCrushTask,
        "grader_class": EarningsVolCrushGrader,
    },
    "gamma_scalping": {
        "max_steps": 10,
        "task_class": GammaScalpingTask,
        "grader_class": GammaScalpingGrader,
    },
}


def validate_action(action: VSRAction) -> Optional[str]:
    """Validate an action before execution.

    Returns None if valid, error string if invalid.
    """
    if not (0 <= action.selected_strike <= 7):
        return "Invalid strike index"
    if not (0 <= action.selected_maturity <= 2):
        return "Invalid maturity index"
    if action.quantity < 0:
        return "Quantity must be non-negative"
    if action.quantity > 10.0:
        return "Quantity exceeds maximum of 10 contracts"
    if action.direction not in (
        TradeDirection.BUY,
        TradeDirection.SELL,
        TradeDirection.HOLD,
    ):
        return "Invalid direction"
    if action.direction == TradeDirection.HOLD and action.quantity != 0:
        return "Hold action must have zero quantity"
    return None


class VSREnvironment:
    """Volatility Surface Reasoning Environment.

    Simulates options portfolio management with 3 tasks:
    - delta_hedging (medium): Neutralize portfolio delta through market shock
    - earnings_vol_crush (hard): Position for and recover from earnings vol crush
    - gamma_scalping (expert): Profit from gamma scalping through spot oscillations
    """

    def __init__(self):
        self.engine = OptionChainEngine()
        self.reward_computer = RewardComputer()
        self._state = VSRState()
        self._rng: Optional[np.random.RandomState] = None
        self._episode_history: List[Dict[str, Any]] = []
        self._iv_surface: List[List[float]] = []
        self._current_task = None
        self._current_grader = None
        self._prev_delta: float = 0.0
        self._prev_pnl: float = 0.0

    def reset(self, task_name: str = "delta_hedging", seed: int = 42) -> VSRObservation:
        """Initialize a new episode for the specified task.

        Args:
            task_name: One of 'delta_hedging', 'earnings_vol_crush', 'gamma_scalping'
            seed: Random seed for reproducibility

        Returns:
            Initial VSRObservation
        """
        # Validate task name
        if task_name not in TASK_CONFIG:
            task_name = "delta_hedging"

        config = TASK_CONFIG[task_name]

        # Seed RNG for reproducibility
        self._rng = np.random.RandomState(seed)
        random.seed(seed)

        # Create fresh state
        self._state = VSRState(
            episode_id=str(uuid.uuid4()),
            task_name=task_name,
            step_count=0,
            spot_price=100.0 + self._rng.uniform(-5, 5),
            variance=0.04 + self._rng.uniform(-0.01, 0.01),
            regime="normal",
        )

        self._episode_history = []
        self._prev_delta = 0.0
        self._prev_pnl = 0.0

        # Create task and grader instances
        self._current_task = config["task_class"]()
        self._current_grader = config["grader_class"]()

        # Task-specific initialization
        mispriced_cells = self._current_task.initialize(self._state, self._rng)

        # Store mispriced_cells in state for grading
        self._state.mispriced_cells = mispriced_cells

        # Inject expected outcome per task
        if task_name == "delta_hedging":
            self._state.expected_outcome = (
                "Agent must neutralize delta to within +/- 0.05 before market shock."
            )
        elif task_name == "earnings_vol_crush":
            self._state.expected_outcome = (
                "Agent must hold negative vega position before vol crush event."
            )
        elif task_name == "gamma_scalping":
            self._state.expected_outcome = "Agent must re-hedge delta when price jumps while maintaining long gamma."

        # Generate IV surface
        self._iv_surface = self.engine.generate_iv_surface(
            S=self._state.spot_price,
            rng=self._rng,
            base_vol=np.sqrt(self._state.variance),
            mispriced_cells=mispriced_cells if mispriced_cells else None,
        )

        # Store previous delta/pnl for reward computation
        self._prev_delta = self._state.portfolio_delta
        self._prev_pnl = self._state.portfolio_pnl

        return self._make_observation(config)

    def step(self, action: VSRAction) -> Dict[str, Any]:
        """Execute one step.

        Args:
            action: Agent's VSRAction

        Returns:
            Dict with 'observation', 'reward', 'done', 'info'
        """
        self._state.step_count += 1
        config = TASK_CONFIG[self._state.task_name]

        # Store previous values for reward computation
        prev_delta = self._state.portfolio_delta
        prev_pnl = self._state.portfolio_pnl

        # Validate action
        error = validate_action(action)

        # Execute action if valid
        if error is None and action.direction != TradeDirection.HOLD:
            add_position(
                state=self._state,
                strike_idx=action.selected_strike,
                maturity_idx=action.selected_maturity,
                direction=action.direction.value,
                quantity=action.quantity,
                engine=self.engine,
            )
            # Update portfolio after adding position
            update_positions_on_market_move(self._state, self.engine)

        # Advance market
        if self._rng is not None:
            # Check for task-specific market events

            # Delta hedging: trigger regime shift at step 2 or 3
            if (
                self._state.task_name == "delta_hedging"
                and hasattr(self._state, "regime_shift_step")
                and self._state.step_count == self._state.regime_shift_step
            ):
                trigger_regime_shift(self._state, self._rng)

            # Earnings vol crush: trigger vol crush at step 3-6
            if (
                self._state.task_name == "earnings_vol_crush"
                and hasattr(self._state, "vol_crush_step")
                and self._state.step_count == self._state.vol_crush_step
            ):
                trigger_vol_crush(self._state, self._rng)

            # Gamma scalping: inject oscillation every step
            if self._state.task_name == "gamma_scalping":
                inject_oscillation(self._state, self._rng, magnitude=0.025)

            # Standard market advance
            advance_market(self._state, self._rng)

        # Update positions after market move
        if self._state.positions:
            update_positions_on_market_move(self._state, self.engine)

        # Regenerate IV surface at new market conditions
        mispriced_cells = (
            self._state.mispriced_cells if self._state.mispriced_cells else None
        )
        self._iv_surface = self.engine.generate_iv_surface(
            S=self._state.spot_price,
            rng=self._rng,
            base_vol=np.sqrt(max(self._state.variance, 0.01)),
            mispriced_cells=mispriced_cells,
        )

        # Build observation (needed for reward computation)
        obs = self._make_observation(config, error)

        # Compute reward
        reward = self._compute_reward(action, obs, prev_delta, prev_pnl, error)

        # Check if episode is done
        done = self._state.step_count >= config["max_steps"]

        # Record step in history
        self._episode_history.append(
            {
                "action": action,
                "observation": obs,
                "reward": reward,
            }
        )

        # Compute grader score on episode end
        info: Dict[str, Any] = {}
        if done and self._current_grader is not None:
            grader_score = self._current_grader.score(
                self._episode_history, self._state
            )
            info["grader_score"] = float(min(max(grader_score, 0.0), 1.0))

        return {
            "observation": obs,
            "reward": float(reward.total),
            "done": done,
            "info": info,
        }

    @property
    def state(self) -> VSRState:
        """Return current internal state."""
        return self._state

    def _make_observation(
        self, config: dict, error: Optional[str] = None
    ) -> VSRObservation:
        """Build VSRObservation from current state."""
        # Compute portfolio greeks
        if self._state.positions:
            greeks = compute_portfolio_greeks(self._state, self.engine)
        else:
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        # Generate market sentiment from variance
        sentiment = (self._state.variance - 0.04) / 0.04  # normalized around baseline
        sentiment = max(-1.0, min(1.0, sentiment))

        return VSRObservation(
            iv_surface=self._iv_surface,
            spot_price=round(self._state.spot_price, 4),
            portfolio_greeks=greeks,
            portfolio_pnl=round(self._state.portfolio_pnl, 4),
            portfolio_positions=[
                {k: round(v, 4) if isinstance(v, float) else v for k, v in pos.items()}
                for pos in self._state.positions
            ],
            market_sentiment=round(sentiment, 4),
            step_number=self._state.step_count,
            steps_remaining=config["max_steps"] - self._state.step_count,
            task_name=self._state.task_name,
            task_description=(
                self._current_task.get_description() if self._current_task else ""
            ),
            last_action_error=error,
            expected_outcome=self._state.expected_outcome,
        )

    def _compute_reward(
        self,
        action: VSRAction,
        obs: VSRObservation,
        prev_delta: float,
        prev_pnl: float,
        error: Optional[str],
    ) -> VSRReward:
        """Compute reward based on current task."""
        if error is not None:
            # Invalid action gets zero reward
            return VSRReward(total=0.0)

        task = self._state.task_name

        if task == "delta_hedging":
            trade_cost = (
                abs(action.quantity) * 0.01
                if action.direction != TradeDirection.HOLD
                else 0.0
            )
            return self.reward_computer.compute_delta_hedging_reward(
                action, self._state, obs, prev_delta, trade_cost
            )
        elif task == "earnings_vol_crush":
            return self.reward_computer.compute_earnings_crush_reward(
                action, self._state, obs, prev_pnl
            )
        elif task == "gamma_scalping":
            return self.reward_computer.compute_gamma_scalping_reward(
                action, self._state, obs, prev_delta, prev_pnl
            )
        else:
            return VSRReward(total=0.0)
