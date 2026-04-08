"""Grading rubrics for VSR-Env.

Provides standardized rubrics for evaluating agent performance
in alignment with OpenEnv RFC 004 capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseRubric(ABC):
    """Abstract base class for all grading rubrics."""

    @abstractmethod
    def score(self, actual: Any, expected: Any) -> float:
        """Calculate a score for the actual outcome against the expected.

        Args:
            actual: The actual outcome or agent response.
            expected: The expected outcome or target value.

        Returns:
            Score in the range [0.01, 0.99].
        """
        pass


class ExactMatchRubric(BaseRubric):
    """Scores 0.99 if actual strictly matches expected, else 0.01."""

    def score(self, actual: Any, expected: Any) -> float:
        return 0.99 if actual == expected else 0.01


class ReasoningQualityRubric(BaseRubric):
    """Scores an agent's reasoning based on domain-specific keyword coverage."""

    def __init__(self, keywords: list[str] = None):
        self.keywords = keywords or [
            "delta",
            "hedge",
            "neutral",
            "skew",
            "smile",
            "regime",
            "overpriced",
            "underpriced",
            "moneyness",
            "vega",
            "gamma",
            "theta",
            "volatility",
            "arbitrage",
            "mispricing",
        ]

    def score(self, actual: str, expected: str = None) -> float:
        """Score based on presence of keywords and length.

        Args:
            actual: the agent's reasoning text.
            expected: Not strictly used here, provided for BaseRubric signature.
        """
        text = str(actual).lower()
        if len(text) <= 20:
            return 0.01

        keyword_hits = sum(1 for kw in self.keywords if kw in text)
        keyword_score = min(keyword_hits / 4.0, 1.0)

        return keyword_score
