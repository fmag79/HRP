"""
Advisory service for HRP.

Transforms validated ML predictions into actionable, plain-English
trading recommendations with track record tracking and portfolio construction.
"""

from hrp.advisory.explainer import RecommendationExplainer
from hrp.advisory.portfolio_constructor import PortfolioConstructor, PortfolioConstraints
from hrp.advisory.recommendation_engine import Recommendation, RecommendationEngine
from hrp.advisory.safeguards import CircuitBreaker, PreTradeChecks
from hrp.advisory.track_record import TrackRecordTracker

__all__ = [
    "CircuitBreaker",
    "PortfolioConstraints",
    "PortfolioConstructor",
    "PreTradeChecks",
    "Recommendation",
    "RecommendationEngine",
    "RecommendationExplainer",
    "TrackRecordTracker",
]
