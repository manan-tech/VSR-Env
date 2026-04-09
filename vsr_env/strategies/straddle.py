"""Straddle strategy implementation."""

from typing import Any, Dict, List, Optional

from vsr_env.strategies.base import OptionStrategy


class Straddle(OptionStrategy):
    """ATM straddle: long call + long put at same strike and expiry.

    Used for volatility speculation. Long straddle profits from large moves
    in either direction. Short straddle profits from low realized volatility.

    Greek Profile (long):
        - Delta: near-zero (neutral)
        - Gamma: positive (long gamma)
        - Vega: positive (long vol)
        - Theta: negative (time decay)

    Max Profit (long): unlimited (stock goes to 0 or infinity)
    Max Loss (long): premium paid
    Breakevens: strike ± premium
    """

    def _validate_legs(self, legs: List[Dict[str, Any]]) -> None:
        """Validate straddle configuration.

        Must have exactly 2 legs, same strike, same maturity, one call and one put.
        """
        if len(legs) != 2:
            raise ValueError(f"Straddle requires 2 legs, got {len(legs)}")

        leg1, leg2 = legs[0], legs[1]

        # Must have same strike and maturity
        if leg1["strike_idx"] != leg2["strike_idx"]:
            raise ValueError("Straddle legs must have same strike")

        if leg1["maturity_idx"] != leg2["maturity_idx"]:
            raise ValueError("Straddle legs must have same maturity")

        # Must have one call and one put
        opt_types = {leg1["option_type"], leg2["option_type"]}
        if opt_types != {"call", "put"}:
            raise ValueError("Straddle requires one call and one put")

        # Directions should match (both buy or both sell)
        if leg1["direction"] != leg2["direction"]:
            raise ValueError("Straddle legs must have same direction")

    def get_strategy_type(self) -> str:
        return "straddle"

    def is_long(self) -> bool:
        """Check if this is a long straddle.

        Returns:
            True if long straddle, False if short
        """
        return self.legs[0]["direction"] == "buy"

    def compute_payoff(self, spot_at_expiry: float) -> float:
        """Compute straddle payoff at expiration.

        Payoff = max(S-K, 0) for call + max(K-S, 0) for put
        (adjusted for long/short)
        """
        # Get strike price (using strike_idx, assume STRIKES array is available)
        # For now, compute relative payoff using strike index
        total_payoff = 0.0
        sign = 1.0 if self.is_long() else -1.0

        for leg in self.legs:
            if leg["option_type"] == "call":
                # Call payoff: max(S - K, 0)
                total_payoff += sign * leg["quantity"] * max(spot_at_expiry - self._get_strike(leg), 0)
            else:
                # Put payoff: max(K - S, 0)
                total_payoff += sign * leg["quantity"] * max(self._get_strike(leg) - spot_at_expiry, 0)

        return total_payoff

    def _get_strike(self, leg: Dict[str, Any]) -> float:
        """Get actual strike price from strike_idx.

        This is a placeholder - real implementation would look up from STRIKES array.
        """
        STRIKES = [85.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0]
        return STRIKES[leg["strike_idx"]]

    def get_max_profit(self) -> Optional[float]:
        """Max profit for long straddle is unlimited. For short, it's the premium."""
        if self.is_long():
            return None  # Unlimited
        else:
            # Short straddle max profit = premium received
            # Would need entry prices to compute
            return None  # Unknown without entry prices

    def get_max_loss(self) -> Optional[float]:
        """Max loss for long straddle is premium paid. For short, it's unlimited."""
        if self.is_long():
            # Long straddle max loss = premium paid
            return None  # Unknown without entry prices
        else:
            return None  # Unlimited

    def get_breakevens(self) -> List[float]:
        """Get breakeven points: strike ± premium.

        Returns empty list without premium information.
        """
        # Would need entry prices to compute exact breakevens
        return []

    @classmethod
    def from_legs(cls, legs: List[Any]) -> "Straddle":
        """Create Straddle from StrategyLeg objects.

        Args:
            legs: List of StrategyLeg Pydantic models

        Returns:
            Straddle instance
        """
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