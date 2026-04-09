"""Multi-leg options strategy implementations."""

from vsr_env.models import StrategyType
from vsr_env.strategies.base import OptionStrategy
from vsr_env.strategies.straddle import Straddle
from vsr_env.strategies.strangle import Strangle
from vsr_env.strategies.spread import CalendarSpread, VerticalSpread

__all__ = [
    "OptionStrategy",
    "Straddle",
    "Strangle",
    "VerticalSpread",
    "CalendarSpread",
    "create_strategy_from_action",
]


def create_strategy_from_action(action):
    """Factory function to create a strategy from a VSRAction.

    Args:
        action: VSRAction with strategy_type and legs

    Returns:
        Appropriate OptionStrategy subclass instance

    Raises:
        ValueError: If strategy_type is not supported
    """
    if action.strategy_type is None or action.legs is None:
        raise ValueError("Action does not contain multi-leg strategy data")

    strategy_type = action.strategy_type

    if strategy_type == StrategyType.STRADDLE:
        return Straddle.from_legs(action.legs)
    elif strategy_type == StrategyType.STRANGLE:
        return Strangle.from_legs(action.legs)
    elif strategy_type == StrategyType.VERTICAL_SPREAD:
        return VerticalSpread.from_legs(action.legs)
    elif strategy_type == StrategyType.CALENDAR_SPREAD:
        return CalendarSpread.from_legs(action.legs)
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")