"""
Recommendation Agent — Weekly recommendation generation.

Runs the full advisory pipeline:
1. Review and close stale open recommendations
2. Generate new weekly recommendations
3. Update track record
4. Send email digest
5. Log lineage events
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

from loguru import logger

from hrp.agents.base import ResearchAgent
from hrp.agents.jobs import DataRequirement
from hrp.research.lineage import EventType


class RecommendationAgent(ResearchAgent):
    """
    Weekly recommendation generation agent.

    Schedule: Sunday evening (prepares Monday recommendations)

    Pipeline:
    1. Load user profile (or use defaults)
    2. Run pre-trade safety checks
    3. Review and close stale open recommendations
    4. Generate new recommendations from deployed models
    5. Update track record
    6. Send email digest (if configured)
    7. Log lineage events
    """

    def __init__(self):
        super().__init__(
            job_id="recommendation-agent",
            actor="agent:recommendation",
            data_requirements=[
                DataRequirement(
                    table="prices",
                    min_rows=100,
                    max_age_days=3,
                    description="Recent price data",
                ),
                DataRequirement(
                    table="model_deployments",
                    min_rows=1,
                    description="At least one deployed model",
                ),
            ],
            max_retries=2,
        )

    def execute(self) -> dict[str, Any]:
        """Run the weekly recommendation pipeline."""
        as_of_date = date.today()
        logger.info(f"RecommendationAgent starting for {as_of_date}")

        # Load risk tolerance from env or default
        risk_tolerance = int(os.getenv("HRP_ADVISORY_RISK_TOLERANCE", "3"))

        # Initialize advisory components
        from hrp.advisory.explainer import RecommendationExplainer
        from hrp.advisory.recommendation_engine import RecommendationEngine
        from hrp.advisory.safeguards import CircuitBreaker, PreTradeChecks
        from hrp.advisory.track_record import TrackRecordTracker

        pre_trade = PreTradeChecks(self.api)
        explainer = RecommendationExplainer()
        engine = RecommendationEngine(
            api=self.api,
            explainer=explainer,
            pre_trade_checks=pre_trade,
        )
        tracker = TrackRecordTracker(self.api)
        breaker = CircuitBreaker(self.api)

        results: dict[str, Any] = {
            "date": str(as_of_date),
            "recommendations_generated": 0,
            "recommendations_closed": 0,
            "circuit_breaker": False,
            "digest_sent": False,
        }

        # 1. Check circuit breaker
        should_halt, halt_reason = breaker.should_halt(as_of_date)
        if should_halt:
            logger.warning(f"Circuit breaker activated: {halt_reason}")
            self._log_agent_event(
                EventType.CIRCUIT_BREAKER_ACTIVATED,
                {"reason": halt_reason},
            )
            results["circuit_breaker"] = True
            results["halt_reason"] = halt_reason
            return results

        # 2. Run pre-trade checks
        checks = pre_trade.run_all_checks(as_of_date)
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            for check in failed_checks:
                logger.warning(f"Pre-trade check failed: {check.check_name} — {check.message}")
            if any(c.severity == "error" for c in failed_checks):
                results["failed_checks"] = [c.message for c in failed_checks]
                return results

        # 3. Review and close stale open recommendations
        updates = engine.review_open_recommendations(as_of_date)
        closed_count = sum(1 for u in updates if u.status != "active")
        results["recommendations_closed"] = closed_count
        logger.info(f"Reviewed {len(updates)} open recommendations, closed {closed_count}")

        # 4. Generate new recommendations
        recommendations = engine.generate_weekly_recommendations(
            as_of_date=as_of_date,
            risk_tolerance=risk_tolerance,
        )
        results["recommendations_generated"] = len(recommendations)
        results["symbols"] = [r.symbol for r in recommendations]

        # 5. Log lineage event
        if recommendations:
            self._log_agent_event(
                EventType.RECOMMENDATION_GENERATED,
                {
                    "count": len(recommendations),
                    "batch_id": recommendations[0].batch_id,
                    "symbols": [r.symbol for r in recommendations],
                    "confidences": [r.confidence for r in recommendations],
                },
            )

        # 6. Update track record
        try:
            summary = tracker.compute_track_record(
                start_date=date(2020, 1, 1), end_date=as_of_date
            )
            tracker.persist_track_record(summary)
            results["track_record_win_rate"] = summary.win_rate
            results["track_record_excess_return"] = summary.excess_return
        except Exception as e:
            logger.warning(f"Track record update failed: {e}")

        # 7. Send email digest (if configured)
        notification_email = os.getenv("NOTIFICATION_EMAIL")
        if notification_email and recommendations:
            try:
                from hrp.advisory.digest import WeeklyDigest
                report = tracker.generate_weekly_report(as_of_date)
                digest = WeeklyDigest()
                content = digest.generate(report)
                sent = digest.send(content, notification_email)
                results["digest_sent"] = sent
            except Exception as e:
                logger.warning(f"Email digest failed: {e}")

        logger.info(
            f"RecommendationAgent complete: {len(recommendations)} new, "
            f"{closed_count} closed"
        )
        return results
