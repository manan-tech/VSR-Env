"""Black-Scholes pricing engine for option chain calculations.

This module provides vectorized NumPy/SciPy implementations of:
- Black-Scholes option pricing (calls and puts)
- Greeks computation (delta, gamma, vega, theta)
- IV surface generation with skew and term structure

All computations use vectorized operations for CPU efficiency.
"""

import numpy as np
from scipy.stats import norm


class OptionChainEngine:
    """Engine for Black-Scholes pricing and Greeks computation.

    Computes prices and Greeks for an 8×3 option chain (8 strikes × 3 maturities)
    using vectorized NumPy operations.

    Attributes:
        STRIKES: Array of 8 strike prices around ATM
        MATURITIES: Array of 3 times to maturity in years
        r: Risk-free interest rate
    """

    # Strike prices around ATM (85 to 110)
    STRIKES = np.array([85.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0])

    # Times to maturity in years (1M, 3M, 6M)
    MATURITIES = np.array([30 / 365, 90 / 365, 180 / 365])

    def __init__(self, r: float = 0.05):
        """Initialize the option chain engine.

        Args:
            r: Risk-free interest rate (default 0.05 = 5%)
        """
        self.r = r

    def bs_price(
        self,
        S: float,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        option_type: str = "call",
    ) -> np.ndarray:
        """Compute Black-Scholes option prices using vectorized operations.

        Formula:
            d1 = (log(S/K) + (r + 0.5*σ²)*T) / (σ*sqrt(T))
            d2 = d1 - σ*sqrt(T)
            Call = S*N(d1) - K*exp(-r*T)*N(d2)
            Put = K*exp(-r*T)*N(-d2) - S*N(-d1)

        Args:
            S: Spot price (scalar)
            K: Strike prices (array, shape matches sigma)
            T: Times to maturity (array, shape matches sigma)
            sigma: Volatilities (array)
            option_type: 'call' or 'put'

        Returns:
            Array of option prices matching input shape
        """
        # Compute d1 and d2
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def delta(
        self,
        S: float,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        option_type: str = "call",
    ) -> np.ndarray:
        """Compute option delta (first derivative with respect to spot).

        Formula:
            Call delta = N(d1)
            Put delta = N(d1) - 1

        Args:
            S: Spot price (scalar)
            K: Strike prices (array)
            T: Times to maturity (array)
            sigma: Volatilities (array)
            option_type: 'call' or 'put'

        Returns:
            Array of delta values
        """
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1

    def gamma(
        self,
        S: float,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """Compute option gamma (second derivative with respect to spot).

        Formula:
            gamma = N'(d1) / (S * σ * sqrt(T))

        Same for calls and puts.

        Args:
            S: Spot price (scalar)
            K: Strike prices (array)
            T: Times to maturity (array)
            sigma: Volatilities (array)

        Returns:
            Array of gamma values
        """
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega(
        self,
        S: float,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """Compute option vega (derivative with respect to volatility).

        Formula:
            vega = S * N'(d1) * sqrt(T) / 100

        Returns vega per 1% change in volatility.
        Same for calls and puts.

        Args:
            S: Spot price (scalar)
            K: Strike prices (array)
            T: Times to maturity (array)
            sigma: Volatilities (array)

        Returns:
            Array of vega values (per 1% vol change)
        """
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    def theta(
        self,
        S: float,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        option_type: str = "call",
    ) -> np.ndarray:
        """Compute option theta (time decay).

        Formula:
            theta = -(S * N'(d1) * σ) / (2 * sqrt(T)) - r * K * exp(-r*T) * N(d2)  [call]
            theta = -(S * N'(d1) * σ) / (2 * sqrt(T)) + r * K * exp(-r*T) * N(-d2) [put]

        Returns theta per day (divided by 365).

        Args:
            S: Spot price (scalar)
            K: Strike prices (array)
            T: Times to maturity (array)
            sigma: Volatilities (array)
            option_type: 'call' or 'put'

        Returns:
            Array of theta values (per day)
        """
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # First term: -(S * N'(d1) * σ) / (2 * sqrt(T))
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == "call":
            term2 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            term2 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)

        # Return per day
        return (term1 + term2) / 365

    def implied_vol(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Solve for implied volatility from market price.

        Uses a three-stage approach for numerical stability:
        1. Newton-Raphson (fast convergence when vega is non-zero)
        2. Brent's method (guaranteed convergence, no derivative needed)
        3. Intrinsic volatility estimate (fallback when all else fails)

        Args:
            market_price: Observed market price of the option
            S: Spot price
            K: Strike price
            T: Time to maturity in years
            option_type: 'call' or 'put'
            tol: Convergence tolerance (default 1e-6)
            max_iter: Maximum Newton-Raphson iterations (default 100)

        Returns:
            Implied volatility in decimal form (e.g., 0.20 for 20%)
        """
        # Stage 1: Newton-Raphson method
        sigma = 0.2  # Initial guess per Requirement 8.1

        for _ in range(max_iter):
            # Compute price at current sigma
            price = self.bs_price(
                S, np.array([K]), np.array([T]), np.array([sigma]), option_type
            )[0]

            # Compute vega (undo the /100 scaling from vega method)
            v = self.vega(S, np.array([K]), np.array([T]), np.array([sigma]))[0] * 100

            # Check for near-zero vega - switch to Brent's method
            if abs(v) < 1e-8:
                return self._implied_vol_brent(market_price, S, K, T, option_type, tol)

            # Check convergence
            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            # Newton-Raphson update
            sigma = sigma - diff / v

            # Clamp to reasonable range per Requirement 8.4
            sigma = np.clip(sigma, 0.01, 5.0)

        # If Newton-Raphson didn't converge, try Brent's method
        return self._implied_vol_brent(market_price, S, K, T, option_type, tol)

    def _implied_vol_brent(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str,
        tol: float,
    ) -> float:
        """Brent's method fallback for IV solving.

        Used when vega is near zero (deep ITM/OTM options) or
        Newton-Raphson fails to converge.

        Args:
            market_price: Observed market price
            S: Spot price
            K: Strike price
            T: Time to maturity in years
            option_type: 'call' or 'put'
            tol: Convergence tolerance

        Returns:
            Implied volatility
        """
        from scipy.optimize import brentq

        def objective(sigma: float) -> float:
            """Objective function: price(sigma) - market_price = 0"""
            return (
                self.bs_price(
                    S, np.array([K]), np.array([T]), np.array([sigma]), option_type
                )[0]
                - market_price
            )

        try:
            # Search range [0.01, 5.0] per Requirement 8.2
            return brentq(objective, 0.01, 5.0, xtol=tol, maxiter=100)
        except (ValueError, RuntimeError):
            # Brent's method failed - return intrinsic volatility estimate
            return self._implied_vol_intrinsic(S, K, T)

    def _implied_vol_intrinsic(self, S: float, K: float, T: float) -> float:
        """Intrinsic volatility fallback when numerical methods fail.

        Provides a rough estimate based on log-moneyness.

        Formula: max(0.05, min(abs(log(S/K)) / sqrt(T) * 0.5, 3.0))

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity in years

        Returns:
            Estimated implied volatility
        """
        # Per Requirement 8.5
        intrinsic_vol = abs(np.log(S / K)) / np.sqrt(T) * 0.5
        return max(0.05, min(intrinsic_vol, 3.0))

    def generate_iv_surface(
        self,
        S: float,
        rng: np.random.RandomState,
        base_vol: float = 0.2,
        skew: float = -0.02,
        term_slope: float = 0.01,
        mispriced_cells: list | None = None,
    ) -> list[list[float]]:
        """Generate realistic IV surface with volatility smile/skew and term structure.

        Creates an 8×3 implied volatility matrix with:
        - Base volatility level
        - Skew (volatility smile effect)
        - Term structure (maturity-dependent volatility)
        - Gaussian noise for realism
        - Optional deliberate mispricings for tasks

        Args:
            S: Current spot price
            rng: Seeded numpy RandomState for reproducibility
            base_vol: Base volatility level (ATM), default 0.2
            skew: Skew coefficient (negative = put skew), default -0.02
            term_slope: Term structure slope (positive = upward sloping), default 0.01
            mispriced_cells: List of ((strike_idx, mat_idx), direction, magnitude)
                where direction is "over" or "under"

        Returns:
            8×3 list of implied volatilities

        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7
        """
        surface = np.zeros((8, 3))

        for i, K in enumerate(self.STRIKES):
            for j, T in enumerate(self.MATURITIES):
                # Log-moneyness normalized by sqrt(T) per Requirement 9.2
                moneyness = np.log(K / S) / np.sqrt(T)

                # Base IV with skew (smile effect)
                iv = base_vol + skew * moneyness

                # Term structure per Requirement 9.3
                iv += term_slope * np.sqrt(T)

                # Add Gaussian noise with std 0.005 per Requirement 9.4
                iv += rng.normal(0, 0.005)

                # Clamp to minimum 0.05 per Requirement 9.5
                surface[i, j] = max(0.05, iv)

        # Inject deliberate mispricings per Requirement 9.6
        if mispriced_cells:
            for (strike_idx, mat_idx), direction, magnitude in mispriced_cells:
                if direction == "over":
                    surface[strike_idx, mat_idx] += magnitude
                else:  # "under"
                    surface[strike_idx, mat_idx] -= magnitude
                # Ensure minimum 0.05 after mispricing
                surface[strike_idx, mat_idx] = max(0.05, surface[strike_idx, mat_idx])

        # Return as list of lists per Requirement 9.7
        return surface.tolist()


def inject_mispricings(
    rng: np.random.RandomState,
    num_mispricings: int = 2,
) -> list[tuple[tuple[int, int], str, float]]:
    """Generate mispricing specifications for IV reading task.

    Creates random mispriced cells with:
    - Non-adjacent positions (avoid overlapping mispricings)
    - Random direction (overpriced or underpriced)
    - Random magnitude (3-8 vol points)

    Args:
        rng: Seeded numpy RandomState for reproducibility
        num_mispricings: Number of mispriced cells to generate, default 2

    Returns:
        List of ((strike_idx, maturity_idx), direction, magnitude) tuples
        where direction is "over" or "under" and magnitude is in [0.03, 0.08]

    Requirements: 3.2
    """
    cells = []
    used_positions = set()

    for _ in range(num_mispricings):
        # Find a cell that's not adjacent to existing mispricings
        while True:
            strike_idx = rng.randint(0, 8)
            mat_idx = rng.randint(0, 3)

            # Check if this position is available (not adjacent to used positions)
            if (strike_idx, mat_idx) not in used_positions:
                # Mark this cell and all adjacent cells as used
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = strike_idx + di, mat_idx + dj
                        if 0 <= ni < 8 and 0 <= nj < 3:
                            used_positions.add((ni, nj))
                break

        # Random direction: "over" or "under" per Requirement 3.2
        direction = rng.choice(["over", "under"])

        # Random magnitude between 0.03 and 0.08 per Requirement 3.2
        magnitude = rng.uniform(0.03, 0.08)

        cells.append(((strike_idx, mat_idx), direction, magnitude))

    return cells
