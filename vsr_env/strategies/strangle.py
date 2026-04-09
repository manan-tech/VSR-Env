"""Strangle strategy implementation."""

from typing import Any, Dict, List, Optional

from vsr_env.strategies.base import OptionStrategy


class Strangle(OptionStrategy):
    """OTM strangle: long OTM call + long OTM put, same expiry, different strikes.

    Cheaper alternative to straddle for volatility speculation. Wider breakevens
    but lower cost.

    Greek Profile (long):
        - Delta: near-zero (neutral)
        - Gamma: positive (long gamma), lower than straddle
        - Vega: positive (long vol)
        - Theta: negative (time decay)

    Max Profit (long): unlimited (stock goes to 0 or infinity)
    Max Loss (long): premium paid
    Breakevens: lower_strike - premium, upper_strike + premium

    The put strike (lower) and call strike (upper) should both be OTM.
    """

    def _validate_legs(self, legs: List[Dict[str, Any]]) -> None:
        """Validate strangle configuration.

        Must have exactly 2 legs, different strikes, same maturity,
        one call and one put.
        """
        if len(legs) != 2:
            raise ValueError(f"Strangle requires 2 legs, got {len(legs)}")

        leg1, leg2 = legs[0], legs[1]

        # Must have different strikes (OTM positions)
        if leg1["strike_idx"] == leg2["strike_idx"]:
            raise ValueError("Strangle legs must have different strikes (OTM)")

        # Must have same maturity
        if leg1["maturity_idx"] != leg2["maturity_idx"]:
            raise ValueError("Strangle legs must have same maturity")

        # Must have one call and one put
        opt_types = {leg1["option_type"], leg2["option_type"]}
        if opt_types != {"call", "put"}:
            raise ValueError("Strangle requires one call and one put")

        # Directions should match
        if leg1["direction"] != leg2["direction"]:
            raise ValueError("Strangle legs must have same direction")

        # Validate call is above put (standard strangle)
        call_leg = leg1 if leg1["option_type"] == "call" else leg2
        put_leg = leg1 if leg1["option_type"] == "put" else leg2

        if call_leg["strike_idx"] <= put_leg["strike_idx"]:
            raise ValueError(
                "Strangle call strike must be above put strike (OTM structure)"
            )

    def get_strategy_type(self) -> str:
        return "strangle"

    def is_long(self) -> bool:
        """Check if this is a long strangle."""
        return self.legs[0]["direction"] == "buy"

    def compute_payoff(self, spot_at_expiry: float) -> float:
        """Compute strangle payoff at expiration."""
        total_payoff = 0.0
        sign = 1.0 if self.is_long() else -1.0

        for leg in self.legs:
            strike = self._get_strike(leg)
            if leg["option_type"] == "call":
                # Call payoff: max(S - K, 0)
                total_payoff += sign * leg["quantity"] * max(spot_at_expiry - strike, 0)
            else:
                # Put payoff: max(K - S, 0)
                total_payoff += sign * leg["quantity"] * max(strike - spot_at_expiry, 0)

        return total_payoff

    def _get_strike(self, leg: Dict[str, Any]) -> float:
        """Get actual strike price from strike_idx."""
        STRIKES = [85.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0]
        return STRIKES[leg["strike_idx"]]

    def get_call_strike_idx(self) -> int:
        """Get the strike index of the call leg."""
        for leg in self.legs:
            if leg["option_type"] == "call":
                return leg["strike_idx"]
        raise ValueError("No call leg found")

    def get_put_strike_idx(self) -> int:
        """Get the strike index of the put leg."""
        for leg in self.legs:
            if leg["option_type"] == "put":
                return leg["strike_idx"]
        raise ValueError("No put leg found")

    def get_max_profit(self) -> Optional[float]:
        """Max profit for long strangle is unlimited."""
        if self.is_long():
            return None  # Unlimited
        else:
            return None  # Premium received, unknown without entry prices

    def get_max_loss(self) -> Optional[float]:
        """Max loss for long strangle is premium paid."""
        if self.is_long():
            return None  # Premium paid, unknown without entry prices
        else:
            return None  # Unlimited

    def get_breakevens(self) -> List[float]:
        """Get breakeven points.

        Returns empty list without premium information.
        """
        return []

    @classmethod
    def from_legs(cls, legs: List[Any]) -> "Strangle":
        """Create Strangle from StrategyLeg objects."""
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