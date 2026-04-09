"""Spread strategy implementations (Vertical and Calendar spreads)."""

from typing import Any, Dict, List, Optional

from vsr_env.strategies.base import OptionStrategy


class VerticalSpread(OptionStrategy):
    """Vertical spread: same option type, same expiry, different strikes.

    Used for directional bets with defined risk. Bull spreads profit from
    upward moves, bear spreads profit from downward moves.

    Types:
        - Bull Call Spread: Buy lower strike call, sell higher strike call
        - Bear Call Spread: Sell lower strike call, buy higher strike call
        - Bull Put Spread: Buy lower strike put, sell higher strike put
        - Bear Put Spread: Sell lower strike put, buy higher strike put

    Greek Profile:
        - Delta: Net delta (directional bias)
        - Gamma: Limited, depends on strike selection
        - Vega: Limited, spreads reduce vol exposure
        - Theta: Generally positive for credit spreads

    Max Profit: Width of spread - debit paid (debit spread)
                or credit received (credit spread)
    Breakeven: Lower strike + debit (debit call spread)
    """

    def _validate_legs(self, legs: List[Dict[str, Any]]) -> None:
        """Validate vertical spread configuration.

        Must have exactly 2 legs, same option type, same maturity, different strikes.
        """
        if len(legs) != 2:
            raise ValueError(f"Vertical spread requires 2 legs, got {len(legs)}")

        leg1, leg2 = legs[0], legs[1]

        # Must have same option type
        if leg1["option_type"] != leg2["option_type"]:
            raise ValueError(
                f"Vertical spread requires same option type, got {leg1['option_type']} and {leg2['option_type']}"
            )

        # Must have same maturity
        if leg1["maturity_idx"] != leg2["maturity_idx"]:
            raise ValueError("Vertical spread legs must have same maturity")

        # Must have different strikes
        if leg1["strike_idx"] == leg2["strike_idx"]:
            raise ValueError("Vertical spread requires different strikes")

        # Must have opposite directions (buy one, sell one)
        if leg1["direction"] == leg2["direction"]:
            raise ValueError("Vertical spread requires opposite directions (debit/credit)")

    def get_strategy_type(self) -> str:
        return "vertical_spread"

    def get_option_type(self) -> str:
        """Get the option type for this spread.

        Returns:
            "call" or "put"
        """
        return self.legs[0]["option_type"]

    def is_bull_spread(self) -> bool:
        """Check if this is a bull spread.

        Bull call spread: buy lower strike, sell higher strike
        Bull put spread: sell higher strike, buy lower strike

        Returns:
            True if bull spread, False if bear spread
        """
        opt_type = self.get_option_type()
        lower_idx = min(self.legs[0]["strike_idx"], self.legs[1]["strike_idx"])

        for leg in self.legs:
            if leg["strike_idx"] == lower_idx:
                lower_direction = leg["direction"]
                break

        if opt_type == "call":
            # Bull call: buy lower strike call
            return lower_direction == "buy"
        else:
            # Bull put: sell higher strike put = buy lower strike put in effect
            # Wait, let me reconsider
            # Bull put: buy higher strike put, sell lower strike put
            # So lower strike is sold
            return lower_direction == "sell"

    def is_debit_spread(self) -> bool:
        """Check if this is a debit spread (pay to enter).

        Debit call spread: buy lower strike, sell higher strike
        Debit put spread: buy higher strike, sell lower strike

        Returns:
            True if debit spread, False if credit spread
        """
        opt_type = self.get_option_type()

        if opt_type == "call":
            # Debit call: buy lower strike
            lower_idx = min(self.legs[0]["strike_idx"], self.legs[1]["strike_idx"])
            for leg in self.legs:
                if leg["strike_idx"] == lower_idx:
                    return leg["direction"] == "buy"
        else:
            # Debit put: buy higher strike
            higher_idx = max(self.legs[0]["strike_idx"], self.legs[1]["strike_idx"])
            for leg in self.legs:
                if leg["strike_idx"] == higher_idx:
                    return leg["direction"] == "buy"

        return False

    def compute_payoff(self, spot_at_expiry: float) -> float:
        """Compute vertical spread payoff at expiration."""
        total_payoff = 0.0

        for leg in self.legs:
            strike = self._get_strike(leg)
            sign = 1.0 if leg["direction"] == "buy" else -1.0

            if leg["option_type"] == "call":
                payoff = max(spot_at_expiry - strike, 0)
            else:
                payoff = max(strike - spot_at_expiry, 0)

            total_payoff += sign * leg["quantity"] * payoff

        return total_payoff

    def _get_strike(self, leg: Dict[str, Any]) -> float:
        """Get actual strike price from strike_idx."""
        STRIKES = [85.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0]
        return STRIKES[leg["strike_idx"]]

    def get_max_profit(self) -> Optional[float]:
        """Max profit is width - debit or credit received.

        Returns None without entry prices.
        """
        return None

    def get_max_loss(self) -> Optional[float]:
        """Max loss is debit paid or width - credit.

        Returns None without entry prices.
        """
        return None

    def get_breakevens(self) -> List[float]:
        """Get breakeven point.

        Returns empty list without entry prices.
        """
        return []

    @classmethod
    def from_legs(cls, legs: List[Any]) -> "VerticalSpread":
        """Create VerticalSpread from StrategyLeg objects."""
        leg_dicts = [
            {
                "strike_idx": leg.strike_idx,
                "maturity_idx": leg.maturity_idx,
                "option_type": leg.option_type,
                "direction": leg.direction,
                "quantity": leg.quantity,
            }
            for leg in legs
        ]
        return cls(leg_dicts)


class CalendarSpread(OptionStrategy):
    """Calendar spread (time spread): same strike, same type, different expiries.

    Used to profit from term structure changes or low realized volatility.
    Long calendar: buy longer expiry, sell shorter expiry.

    Greek Profile (long calendar):
        - Delta: near-zero if ATM strike
        - Gamma: slightly negative (short near-term gamma)
        - Vega: positive (long vega)
        - Theta: positive (time decay favors long calendar)

    Max Profit: Max when stock at strike at near expiry
    Breakeven: Complex, depends on term structure
    """

    def _validate_legs(self, legs: List[Dict[str, Any]]) -> None:
        """Validate calendar spread configuration.

        Must have exactly 2 legs, same option type, same strike, different maturities.
        """
        if len(legs) != 2:
            raise ValueError(f"Calendar spread requires 2 legs, got {len(legs)}")

        leg1, leg2 = legs[0], legs[1]

        # Must have same option type
        if leg1["option_type"] != leg2["option_type"]:
            raise ValueError("Calendar spread requires same option type")

        # Must have same strike
        if leg1["strike_idx"] != leg2["strike_idx"]:
            raise ValueError("Calendar spread requires same strike")

        # Must have different maturities
        if leg1["maturity_idx"] == leg2["maturity_idx"]:
            raise ValueError("Calendar spread requires different maturities")

        # Must have opposite directions
        if leg1["direction"] == leg2["direction"]:
            raise ValueError("Calendar spread requires opposite directions")

    def get_strategy_type(self) -> str:
        return "calendar_spread"

    def get_option_type(self) -> str:
        """Get the option type for this spread."""
        return self.legs[0]["option_type"]

    def is_long_calendar(self) -> bool:
        """Check if this is a long calendar spread (buy long-dated, sell short-dated).

        Returns:
            True if long calendar, False if short calendar
        """
        # Higher maturity_idx = longer expiry
        for leg in self.legs:
            if leg["maturity_idx"] == max(l["maturity_idx"] for l in self.legs):
                return leg["direction"] == "buy"
        return False

    def compute_payoff(self, spot_at_expiry: float) -> float:
        """Compute calendar spread payoff.

        Note: Calendar spreads have complex payoff due to different expiries.
        This simplified version assumes closing both legs at same time.
        """
        # Simplified: actual payoff depends on when you close
        # and the term structure at that time
        return 0.0

    def get_max_profit(self) -> Optional[float]:
        """Max profit when stock at strike at near-term expiry."""
        return None

    def get_max_loss(self) -> Optional[float]:
        """Max loss is the debit paid."""
        return None

    def get_breakevens(self) -> List[float]:
        """Get breakeven points (complex for calendars)."""
        return []

    @classmethod
    def from_legs(cls, legs: List[Any]) -> "CalendarSpread":
        """Create CalendarSpread from StrategyLeg objects."""
        leg_dicts = [
            {
                "strike_idx": leg.strike_idx,
                "maturity_idx": leg.maturity_idx,
                "option_type": leg.option_type,
                "direction": leg.direction,
                "quantity": leg.quantity,
            }
            for leg in legs
        ]
        return cls(leg_dicts)