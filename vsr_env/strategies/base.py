"""Base class for multi-leg options strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class OptionStrategy(ABC):
    """Abstract base class for multi-leg options strategies.

    Each strategy knows its leg composition, Greek profile, and P&L characteristics.
    Implementations include Straddle, Strangle, VerticalSpread, CalendarSpread.

    Attributes:
        legs: List of leg dictionaries with strike_idx, maturity_idx, option_type, direction, quantity
        strategy_id: Unique identifier for this strategy instance
    """

    def __init__(self, legs: List[Dict[str, Any]], strategy_id: Optional[str] = None):
        """Initialize strategy with legs.

        Args:
            legs: List of leg dictionaries
            strategy_id: Optional unique identifier
        """
        self._validate_legs(legs)
        self.legs = legs
        self.strategy_id = strategy_id or self._generate_id()

    @abstractmethod
    def _validate_legs(self, legs: List[Dict[str, Any]]) -> None:
        """Validate that legs form a valid strategy configuration.

        Args:
            legs: List of leg dictionaries

        Raises:
            ValueError: If legs are invalid for this strategy type
        """
        pass

    @abstractmethod
    def get_strategy_type(self) -> str:
        """Return the strategy type name.

        Returns:
            Strategy type string (e.g., "straddle", "vertical_spread")
        """
        pass

    @abstractmethod
    def compute_payoff(self, spot_at_expiry: float) -> float:
        """Compute payoff at expiration for given spot price.

        Args:
            spot_at_expiry: Underlying price at expiration

        Returns:
            Total payoff across all legs
        """
        pass

    @abstractmethod
    def get_max_profit(self) -> Optional[float]:
        """Get maximum possible profit.

        Returns:
            Maximum profit or None if unlimited
        """
        pass

    @abstractmethod
    def get_max_loss(self) -> Optional[float]:
        """Get maximum possible loss.

        Returns:
            Maximum loss or None if unlimited
        """
        pass

    @abstractmethod
    def get_breakevens(self) -> List[float]:
        """Get breakeven points.

        Returns:
            List of breakeven spot prices
        """
        pass

    def get_description(self) -> str:
        """Get human-readable description of the strategy.

        Returns:
            Description string
        """
        lines = [f"{self.get_strategy_type().upper()} Strategy ({self.strategy_id}):"]
        for i, leg in enumerate(self.legs):
            direction = leg["direction"]
            qty = leg["quantity"]
            opt_type = leg["option_type"]
            strike_idx = leg["strike_idx"]
            mat_idx = leg["maturity_idx"]
            lines.append(
                f"  Leg {i+1}: {direction.upper()} {qty} {opt_type.upper()} "
                f"(strike_idx={strike_idx}, maturity_idx={mat_idx})"
            )
        return "\n".join(lines)

    def get_net_greeks(self, greek_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate Greeks across all legs.

        Args:
            greek_data: List of Greek dictionaries for each leg (delta, gamma, vega, theta)

        Returns:
            Dict with net delta, gamma, vega, theta
        """
        if len(greek_data) != len(self.legs):
            raise ValueError(
                f"Greek data length ({len(greek_data)}) must match legs ({len(self.legs)})"
            )

        net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        for leg, greeks in zip(self.legs, greek_data):
            # Adjust sign based on direction
            sign = 1.0 if leg["direction"] == "buy" else -1.0
            qty = leg["quantity"]

            for greek in ["delta", "gamma", "vega", "theta"]:
                net[greek] += sign * qty * greeks.get(greek, 0.0)

        return net

    def _generate_id(self) -> str:
        """Generate a unique strategy ID.

        Returns:
            Unique ID string
        """
        import uuid

        return f"{self.get_strategy_type()}_{uuid.uuid4().hex[:8]}"

    def compute_pnl(self, entry_prices: List[float], current_prices: List[float]) -> float:
        """Compute P&L from entry to current prices.

        Args:
            entry_prices: Entry prices for each leg
            current_prices: Current prices for each leg

        Returns:
            Total unrealized P&L
        """
        if len(entry_prices) != len(self.legs) or len(current_prices) != len(self.legs):
            raise ValueError("Price lists must match leg count")

        pnl = 0.0
        for leg, entry, current in zip(self.legs, entry_prices, current_prices):
            sign = 1.0 if leg["direction"] == "buy" else -1.0
            qty = leg["quantity"]
            pnl += sign * qty * (current - entry)

        return pnl

    @classmethod
    @abstractmethod
    def from_legs(cls, legs: List[Any]) -> "OptionStrategy":
        """Create strategy instance from StrategyLeg objects.

        Args:
            legs: List of StrategyLeg Pydantic models

        Returns:
            Strategy instance
        """
        pass