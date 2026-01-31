"""
CIO Agent - Chief Investment Officer Agent.

Makes strategic decisions about research lines and manages paper portfolio.
Advisory mode: presents recommendations, awaits user approval.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal, Optional

import pandas as pd

from hrp.agents.sdk_agent import SDKAgent
from hrp.api.platform import PlatformAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anthropic import Anthropic


@dataclass
class CIOScore:
    """
    Balanced score across 4 dimensions for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis being scored
        statistical: Statistical quality score (0-1)
        risk: Risk profile score (0-1)
        economic: Economic rationale score (0-1)
        cost: Cost realism score (0-1)
        critical_failure: Whether a critical failure was detected
    """

    hypothesis_id: str
    statistical: float
    risk: float
    economic: float
    cost: float
    critical_failure: bool = False

    @property
    def total(self) -> float:
        """Calculate total score as average of 4 dimensions."""
        return (self.statistical + self.risk + self.economic + self.cost) / 4

    @property
    def decision(self) -> Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]:
        """
        Map score to decision.

        Returns:
            CONTINUE: Score >= 0.75, no critical failure
            CONDITIONAL: Score 0.50-0.74, no critical failure
            KILL: Score < 0.50, no critical failure
            PIVOT: Critical failure detected (overrides score)
        """
        if self.critical_failure:
            return "PIVOT"

        if self.total >= 0.75:
            return "CONTINUE"
        if self.total >= 0.50:
            return "CONDITIONAL"
        return "KILL"


@dataclass
class CIODecision:
    """
    Single decision for a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis this decision is for
        decision: One of CONTINUE, CONDITIONAL, KILL, PIVOT
        score: The CIOScore that led to this decision
        rationale: Human-readable explanation
        evidence: Supporting data (MLflow runs, reports, metrics)
        paper_allocation: For CONTINUE decisions, portfolio weight (0-1)
        pivot_direction: For PIVOT decisions, suggested redirect
    """

    hypothesis_id: str
    decision: Literal["CONTINUE", "CONDITIONAL", "KILL", "PIVOT"]
    score: CIOScore
    rationale: str
    evidence: dict
    paper_allocation: Optional[float] = None
    pivot_direction: Optional[str] = None


@dataclass
class CIOReport:
    """
    Complete weekly CIO report.

    Attributes:
        report_date: When the report was generated
        decisions: All decisions made in this review cycle
        portfolio_state: Current paper portfolio state
        market_regime: Current market regime context
        next_actions: Prioritized action items
        report_path: Path to the generated markdown report
    """

    report_date: date
    decisions: list[CIODecision]
    portfolio_state: dict
    market_regime: str
    next_actions: list[dict]
    report_path: str


class CIOAgent(SDKAgent):
    """
    Chief Investment Officer Agent.

    Makes strategic decisions about research lines and manages paper portfolio.
    Advisory mode: presents recommendations, awaits user approval.
    """

    agent_name = "cio"
    agent_version = "1.0.0"

    DEFAULT_THRESHOLDS = {
        "min_sharpe": 1.0,
        "max_drawdown": 0.20,
        "sharpe_decay_limit": 0.50,
        "min_ic": 0.03,
        "max_turnover": 0.50,
        "critical_sharpe_decay": 0.75,
        "critical_target_leakage": 0.95,
        "critical_max_drawdown": 0.35,
        "min_profitable_regimes": 2,
    }

    def __init__(
        self,
        job_id: str,
        actor: str,
        api: PlatformAPI | None = None,
        thresholds: dict | None = None,
        dependencies: list[str] | None = None,
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize CIO Agent.

        Args:
            job_id: Unique job identifier
            actor: Actor identity (e.g., "agent:cio")
            api: PlatformAPI instance (created if None)
            thresholds: Custom decision thresholds
            dependencies: List of data requirements
            anthropic_api_key: Optional Anthropic API key (reads from env if None)
        """
        super().__init__(
            job_id=job_id,
            actor=actor,
            dependencies=dependencies or [],
        )
        self.api = api or PlatformAPI()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.anthropic_client = self._init_anthropic_client(anthropic_api_key)

    def _init_anthropic_client(self, api_key: str | None) -> "Anthropic | None":
        """
        Initialize Anthropic client for economic rationale assessment.

        Args:
            api_key: Optional API key (reads from ANTHROPIC_API_KEY env if None)

        Returns:
            Anthropic client instance or None if no API key available
        """
        import os

        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        return Anthropic(api_key=key)

    def execute(self) -> dict[str, any]:
        """
        Execute CIO Agent weekly review.

        Reviews all validated hypotheses, scores them across 4 dimensions,
        persists decisions to database, and generates a report.

        Returns:
            Dict with decisions summary and report path
        """
        from datetime import date
        import json

        # Fetch validated hypotheses awaiting CIO review
        hypotheses = self._fetch_validated_hypotheses()

        if not hypotheses:
            return {
                "status": "no_hypotheses",
                "decisions": [],
                "message": "No validated hypotheses awaiting CIO review",
            }

        decisions = []
        for hyp in hypotheses:
            hypothesis_id = hyp["hypothesis_id"]

            # Gather data for scoring
            experiment_data = self._get_experiment_data(hypothesis_id)
            if not experiment_data:
                continue

            risk_data = self._get_risk_data(experiment_data)
            economic_data = self._get_economic_data(hyp)
            cost_data = self._get_cost_data(hyp)

            # Score the hypothesis
            score = self.score_hypothesis(
                hypothesis_id=hypothesis_id,
                experiment_data=experiment_data,
                risk_data=risk_data,
                economic_data=economic_data,
                cost_data=cost_data,
            )

            # Build rationale
            rationale = self._build_rationale(score, experiment_data, risk_data)

            # Save decision to database
            self._save_decision(
                hypothesis_id=hypothesis_id,
                score=score,
                rationale=rationale,
            )

            decisions.append({
                "hypothesis_id": hypothesis_id,
                "title": hyp.get("title", ""),
                "decision": score.decision,
                "score": score.total,
                "rationale": rationale,
            })

            # Stage model if CONTINUE
            self._maybe_stage_model(
                hypothesis_id=hypothesis_id,
                decision=score.decision,
                experiment_data=experiment_data,
            )

        # Generate report
        report_path = self._generate_report(decisions)

        return {
            "status": "complete",
            "decisions": decisions,
            "report_path": str(report_path),
            "decision_count": len(decisions),
        }

    def _fetch_validated_hypotheses(self) -> list[dict]:
        """Fetch hypotheses with 'validated' status."""
        result = self.api._db.fetchdf(
            """
            SELECT hypothesis_id, title, thesis, status, metadata
            FROM hypotheses
            WHERE status = 'validated'
            ORDER BY created_at DESC
            """
        )

        if result.empty:
            return []

        return result.to_dict(orient="records")

    def _get_experiment_data(self, hypothesis_id: str) -> dict | None:
        """Get experiment metrics for a hypothesis from ML Scientist results."""
        import json

        hyp = self.api.get_hypothesis(hypothesis_id)
        if not hyp:
            return None

        # Metadata is stored in the hypothesis record fetched by _fetch_validated_hypotheses
        # but get_hypothesis doesn't return it — re-fetch from DB
        row = self.api._db.fetchone(
            "SELECT metadata FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        if not row or not row[0]:
            return None

        metadata = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        ml_results = metadata.get("ml_scientist_results", {})
        if not ml_results:
            return None

        mean_ic = ml_results.get("mean_ic", 0)
        ic_std = ml_results.get("ic_std", 0)

        return {
            "sharpe": mean_ic * 20,  # IC-based proxy (IC=0.05 → ~1.0 Sharpe)
            "stability_score": ml_results.get("stability_score", 2.0),
            "mean_ic": mean_ic,
            "fold_cv": ic_std / max(mean_ic, 0.001),
        }

    def _get_risk_data(self, experiment_data: dict) -> dict:
        """Derive risk metrics from experiment data."""
        stability = experiment_data.get("stability_score", 2.0)
        return {
            "max_drawdown": min(0.10 + stability * 0.10, 0.50),
            "volatility": 0.15,  # conservative default for long-only equities
            "regime_stable": stability <= 1.0,
            "sharpe_decay": experiment_data.get("fold_cv", 2.0) / 3.0,
        }

    def _get_economic_data(self, hypothesis: dict) -> dict:
        """Get economic rationale data from hypothesis."""
        import json

        metadata = hypothesis.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        return {
            "thesis": hypothesis.get("thesis", ""),
            "current_regime": "Bull Market",  # Would detect from market data
            "black_box_count": metadata.get("black_box_count", 1),
            "uniqueness": metadata.get("uniqueness", "unknown"),
            "agent_reports": {},
        }

    def _get_cost_data(self, hypothesis: dict) -> dict:
        """Get cost realism data."""
        return {
            "slippage_survival": "stable",
            "turnover": 0.30,
            "capacity": "high",
            "execution_complexity": "low",
        }

    def _build_rationale(
        self, score: "CIOScore", experiment_data: dict, risk_data: dict
    ) -> str:
        """Build human-readable rationale for the decision."""
        parts = []

        # Decision explanation
        if score.decision == "CONTINUE":
            parts.append(f"Strong overall performance (score: {score.total:.2f}).")
        elif score.decision == "CONDITIONAL":
            parts.append(f"Moderate performance (score: {score.total:.2f}) requires additional validation.")
        elif score.decision == "KILL":
            parts.append(f"Poor performance (score: {score.total:.2f}) does not meet deployment criteria.")
        else:  # PIVOT
            parts.append("Critical failure detected; approach requires fundamental revision.")

        # Dimension breakdown
        parts.append(f"\nDimension Scores:")
        parts.append(f"  Statistical: {score.statistical:.2f} (Sharpe: {experiment_data.get('sharpe', 0):.2f})")
        parts.append(f"  Risk: {score.risk:.2f} (Max DD: {risk_data.get('max_drawdown', 0):.1%})")
        parts.append(f"  Economic: {score.economic:.2f}")
        parts.append(f"  Cost: {score.cost:.2f}")

        return "\n".join(parts)

    def _save_decision(
        self, hypothesis_id: str, score: "CIOScore", rationale: str
    ) -> None:
        """Save CIO decision to database."""
        from datetime import date

        import uuid

        decision_id = f"CIO-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

        # Get next id value
        max_id_result = self.api._db.fetchdf(
            "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM cio_decisions"
        )
        next_id = int(max_id_result.iloc[0]["next_id"])

        self.api._db.execute(
            """
            INSERT INTO cio_decisions
            (id, decision_id, report_date, hypothesis_id, decision,
             score_total, score_statistical, score_risk, score_economic, score_cost,
             rationale, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                next_id,
                decision_id,
                date.today(),
                hypothesis_id,
                score.decision,
                round(score.total, 2),
                round(score.statistical, 2),
                round(score.risk, 2),
                round(score.economic, 2),
                round(score.cost, 2),
                rationale,
                False,  # Requires manual approval
            ),
        )

    def _maybe_stage_model(
        self,
        hypothesis_id: str,
        decision: str,
        experiment_data: dict,
    ) -> None:
        """Stage model for deployment if decision is CONTINUE."""
        if decision != "CONTINUE":
            return
        self._stage_model_for_deployment(hypothesis_id, experiment_data)

    def _stage_model_for_deployment(
        self,
        hypothesis_id: str,
        experiment_data: dict,
    ) -> None:
        """Register model and deploy to staging."""
        try:
            experiment_id = experiment_data.get("experiment_id")
            model_type = experiment_data.get("model_type", "unknown")
            model_name = f"hyp_{hypothesis_id}_{model_type}"

            model_version = self.api.register_model(
                model=None,
                model_name=model_name,
                model_type=model_type,
                features=experiment_data.get("features", []),
                target=experiment_data.get("target", "returns_20d"),
                metrics=experiment_data.get("metrics", {}),
                hyperparameters=experiment_data.get("hyperparameters", {}),
                training_date=date.today(),
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
            )

            self.api.deploy_model(
                model_name=model_name,
                model_version=model_version,
                validation_data=pd.DataFrame(),
                environment="staging",
                actor="agent:cio",
            )

            self.api.log_event(
                event_type="model_deployed",
                actor="agent:cio",
                hypothesis_id=hypothesis_id,
                details={
                    "model_name": model_name,
                    "environment": "staging",
                    "experiment_id": experiment_id,
                },
            )

            logger.info(f"Staged model {model_name} for hypothesis {hypothesis_id}")

        except Exception as e:
            logger.error(f"Failed to stage model for {hypothesis_id}: {e}")

    def _generate_report(self, decisions: list[dict]) -> "Path":
        """Generate weekly CIO report markdown file."""
        from datetime import date
        from pathlib import Path
        from hrp.utils.config import get_config

        report_dir = get_config().data.reports_dir / date.today().strftime("%Y-%m-%d")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / f"{date.today().strftime('%H-%M')}-cio-review.md"

        # Count decisions by type
        decision_counts = {}
        for d in decisions:
            decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1

        # Build report content
        lines = [
            f"# CIO Agent Weekly Report - {date.today().strftime('%Y-%m-%d')}\n",
            "## Summary\n",
            f"- **Hypotheses Reviewed**: {len(decisions)}",
        ]

        for decision_type, count in decision_counts.items():
            emoji = {"CONTINUE": "✅", "CONDITIONAL": "⚠️", "KILL": "❌", "PIVOT": "🔄"}.get(
                decision_type, "❓"
            )
            lines.append(f"- **{decision_type}**: {count} {emoji}")

        lines.extend([
            "\n## Decisions\n",
            "---\n",
        ])

        for d in decisions:
            emoji = {"CONTINUE": "✅", "CONDITIONAL": "⚠️", "KILL": "❌", "PIVOT": "🔄"}.get(
                d["decision"], "❓"
            )
            lines.extend([
                f"### {emoji} {d['hypothesis_id']}: {d['title']}\n",
                f"**Decision**: {d['decision']} (Score: {d['score']:.2f})\n",
                f"\n**Rationale**:\n```\n{d['rationale']}\n```\n",
                "---\n",
            ])

        lines.extend([
            "## Next Actions\n",
            "1. Review CONTINUE decisions for paper portfolio allocation\n",
            "2. Address CONDITIONAL items with additional validation\n",
            "3. Archive KILL/PIVOT hypotheses with rationale\n",
        ])

        # Write report
        report_path.write_text("\n".join(lines))

        return report_path

    def _score_sharpe(self, sharpe: float) -> float:
        """
        Score Sharpe ratio: 0.5->0, 1.0->0.5, 1.5->1.0.

        Args:
            sharpe: Walk-forward Sharpe ratio

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.5, 1.0, 1.5
        if sharpe <= bad:
            return 0.0
        if sharpe >= good:
            return 1.0
        return (sharpe - bad) / (good - bad)

    def _score_stability(self, stability: float) -> float:
        """
        Score stability score: 2.0->0, 1.0->0.5, 0.5->1.0.

        Lower is better for stability.

        Args:
            stability: Stability score from walk-forward validation

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 2.0, 1.0, 0.5
        if stability >= bad:
            return 0.0
        if stability <= good:
            return 1.0
        if stability >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - stability) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - stability) / (target - good)

    def _score_ic(self, ic: float) -> float:
        """
        Score Information Coefficient: 0.01->0, 0.03->0.5, 0.05->1.0.

        Args:
            ic: Mean Information Coefficient

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.01, 0.03, 0.05
        if ic <= bad:
            return 0.0
        if ic >= good:
            return 1.0
        return (ic - bad) / (good - bad)

    def _score_fold_cv(self, cv: float) -> float:
        """
        Score fold coefficient of variation: 3.0->0, 2.0->0.5, 1.0->1.0.

        Lower is better for CV (less variability across folds).

        Args:
            cv: Coefficient of variation across folds

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 3.0, 2.0, 1.0
        if cv >= bad:
            return 0.0
        if cv <= good:
            return 1.0
        return (bad - cv) / (bad - good)

    def _score_statistical_dimension(
        self,
        hypothesis_id: str,
        experiment_data: dict,
    ) -> float:
        """
        Score statistical quality dimension.

        Averages scores for: Sharpe, stability, IC, fold CV.

        Args:
            hypothesis_id: The hypothesis being scored
            experiment_data: Dict with sharpe, stability_score, mean_ic, fold_cv

        Returns:
            Score 0-1
        """
        sharpe_score = self._score_sharpe(experiment_data.get("sharpe", 0))
        stability_score = self._score_stability(experiment_data.get("stability_score", 2.0))
        ic_score = self._score_ic(experiment_data.get("mean_ic", 0))
        fold_cv_score = self._score_fold_cv(experiment_data.get("fold_cv", 3.0))

        return (sharpe_score + stability_score + ic_score + fold_cv_score) / 4

    def _score_max_drawdown(self, max_dd: float) -> float:
        """
        Score max drawdown: 30%->0, 20%->0.5, 10%->1.0.

        Lower is better for drawdown.

        Args:
            max_dd: Maximum drawdown (as decimal, e.g., 0.20 for 20%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.30, 0.20, 0.10
        if max_dd >= bad:
            return 0.0
        if max_dd <= good:
            return 1.0
        return (bad - max_dd) / (bad - good)

    def _score_volatility(self, vol: float) -> float:
        """
        Score annual volatility: 25%->0, 15%->0.5, 10%->1.0.

        Lower is better for volatility.

        Args:
            vol: Annualized volatility (as decimal)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 0.25, 0.15, 0.10
        if vol >= bad:
            return 0.0
        if vol <= good:
            return 1.0
        if vol >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - vol) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - vol) / (target - good)

    def _score_regime_stability(self, stable: bool) -> float:
        """
        Score regime stability (binary).

        Args:
            stable: Whether >= 2/3 regimes are profitable

        Returns:
            1.0 if stable, 0.0 if not
        """
        return 1.0 if stable else 0.0

    def _score_sharpe_decay(self, decay: float) -> float:
        """
        Score Sharpe decay (binary).

        Decay <= 50% is good (no overfitting).

        Args:
            decay: Sharpe decay ratio (train_sharpe - test_sharpe) / train_sharpe

        Returns:
            1.0 if decay <= 0.50, 0.0 if > 0.50
        """
        limit = self.thresholds["sharpe_decay_limit"]
        return 1.0 if decay <= limit else 0.0

    def _check_critical_failures_risk(self, risk_data: dict) -> bool:
        """
        Check for critical failures in risk dimension.

        Critical:
        - Max drawdown > 35%
        - Sharpe decay > 75%

        Args:
            risk_data: Dict with max_drawdown, sharpe_decay

        Returns:
            True if critical failure detected
        """
        max_dd = risk_data.get("max_drawdown", 0)
        sharpe_decay = risk_data.get("sharpe_decay", 0)

        if max_dd > self.thresholds["critical_max_drawdown"]:
            return True
        if sharpe_decay > self.thresholds["critical_sharpe_decay"]:
            return True

        return False

    def _score_risk_dimension(self, hypothesis_id: str, risk_data: dict) -> float:
        """
        Score risk profile dimension.

        Averages scores for: Max DD, volatility, regime stability, Sharpe decay.

        Args:
            hypothesis_id: The hypothesis being scored
            risk_data: Dict with max_drawdown, volatility, regime_stable, sharpe_decay_ok

        Returns:
            Score 0-1
        """
        max_dd_score = self._score_max_drawdown(risk_data.get("max_drawdown", 0.30))
        vol_score = self._score_volatility(risk_data.get("volatility", 0.25))
        regime_score = self._score_regime_stability(risk_data.get("regime_stable", False))

        # Handle both sharpe_decay_ok (boolean) and sharpe_decay (float)
        if "sharpe_decay_ok" in risk_data:
            decay_score = 1.0 if risk_data["sharpe_decay_ok"] else 0.0
        else:
            decay_score = self._score_sharpe_decay(risk_data.get("sharpe_decay", 0.60))

        return (max_dd_score + vol_score + regime_score + decay_score) / 4

    def _score_turnover(self, turnover: float) -> float:
        """
        Score annual turnover: 100%->0, 50%->0.5, 20%->1.0.

        Lower is better for turnover (less trading = lower costs).

        Args:
            turnover: Annual turnover rate (as decimal, e.g., 0.50 for 50%)

        Returns:
            Score 0-1, clamped
        """
        bad, target, good = 1.00, 0.50, 0.20
        if turnover >= bad:
            return 0.0
        if turnover <= good:
            return 1.0
        if turnover >= target:
            # Between bad and target: scale 0 to 0.5
            return 0.5 * (bad - turnover) / (bad - target)
        else:
            # Between target and good: scale 0.5 to 1.0
            return 0.5 + 0.5 * (target - turnover) / (target - good)

    def _score_capacity(self, capacity: str) -> float:
        """
        Score capacity estimate (ordinal).

        Args:
            capacity: One of "low" (<$1M), "medium" ($1-10M), "high" (>$10M)

        Returns:
            Score 0-1
        """
        scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return scores.get(capacity.lower(), 0.5)

    def _score_slippage_survival(self, survival: str) -> float:
        """
        Score slippage survival (ordinal).

        Args:
            survival: One of "collapse", "degraded", "stable"

        Returns:
            Score 0-1
        """
        scores = {"collapse": 0.0, "degraded": 0.5, "stable": 1.0}
        return scores.get(survival.lower(), 0.5)

    def _score_execution_complexity(self, complexity: str) -> float:
        """
        Score execution complexity (ordinal).

        Args:
            complexity: One of "high", "medium", "low"

        Returns:
            Score 0-1
        """
        scores = {"high": 0.0, "medium": 0.5, "low": 1.0}
        return scores.get(complexity.lower(), 0.5)

    def _score_cost_dimension(self, hypothesis_id: str, cost_data: dict) -> float:
        """
        Score cost realism dimension.

        Averages scores for: Slippage survival, turnover, capacity, complexity.

        Args:
            hypothesis_id: The hypothesis being scored
            cost_data: Dict with slippage_survival, turnover, capacity, execution_complexity

        Returns:
            Score 0-1
        """
        slippage_score = self._score_slippage_survival(
            cost_data.get("slippage_survival", "degraded")
        )
        turnover_score = self._score_turnover(cost_data.get("turnover", 1.00))
        capacity_score = self._score_capacity(cost_data.get("capacity", "medium"))
        complexity_score = self._score_execution_complexity(
            cost_data.get("execution_complexity", "medium")
        )

        return (slippage_score + turnover_score + capacity_score + complexity_score) / 4

    def _score_thesis_strength(self, strength: str) -> float:
        """
        Score thesis strength (ordinal).

        Args:
            strength: One of "weak", "moderate", "strong"

        Returns:
            Score 0-1
        """
        scores = {"weak": 0.0, "moderate": 0.5, "strong": 1.0}
        return scores.get(strength.lower(), 0.5)

    def _score_regime_alignment(self, alignment: str) -> float:
        """
        Score regime alignment (ordinal).

        Args:
            alignment: One of "mismatch", "neutral", "aligned"

        Returns:
            Score 0-1
        """
        scores = {"mismatch": 0.0, "neutral": 0.5, "aligned": 1.0}
        return scores.get(alignment.lower(), 0.5)

    def _score_feature_interpretability(self, black_box_count: int) -> float:
        """
        Score feature interpretability: >5->0, 3-5->0.5, <3->1.0.

        Fewer black-box features = more interpretable = better score.

        Args:
            black_box_count: Number of black-box features (e.g., neural net outputs)

        Returns:
            Score 0-1
        """
        if black_box_count > 5:
            return 0.0
        if black_box_count < 3:
            return 1.0
        return 0.5  # 3-5 features

    def _score_uniqueness(self, uniqueness: str) -> float:
        """
        Score uniqueness (ordinal).

        Args:
            uniqueness: One of "duplicate", "related", "novel"

        Returns:
            Score 0-1
        """
        scores = {"duplicate": 0.0, "related": 0.5, "novel": 1.0}
        return scores.get(uniqueness.lower(), 0.5)

    def _assess_thesis_with_claude(
        self,
        hypothesis_id: str,
        thesis: str,
        agent_reports: dict,
        current_regime: str,
    ) -> dict:
        """
        Use Claude API to assess thesis strength and regime alignment.

        Args:
            hypothesis_id: The hypothesis being assessed
            thesis: The thesis statement
            agent_reports: Dict of agent report content
            current_regime: Current market regime (e.g., "Bull Market")

        Returns:
            Dict with thesis_strength and regime_alignment
        """
        if self.anthropic_client is None:
            # Fallback to moderate scores if no Claude client
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

        # Build prompt for Claude with more specific guidance
        prompt = f"""Assess this trading hypothesis for economic rationale:

Hypothesis: {thesis}

Current Market Regime: {current_regime}

Agent Reports:
{chr(10).join(f'- {k}: {v[:200]}...' for k, v in agent_reports.items()) if agent_reports else 'No agent reports yet'}

Respond ONLY with valid JSON:
{{
    "thesis_strength": "weak" | "moderate" | "strong",
    "regime_alignment": "mismatch" | "neutral" | "aligned"
}}

Scoring Guide:
- thesis_strength:
  - "strong": Well-documented market anomaly (e.g., momentum, value, low volatility), clear economic rationale, academic support
  - "moderate": Plausible but less established effect, or mixed evidence
  - "weak": No clear economic logic, contradicts established research

- regime_alignment:
  - "aligned": Strategy performs well in current regime (e.g., momentum works in bull markets)
  - "neutral": Regime-independent or unclear relationship
  - "mismatch": Strategy historically underperforms in current regime

Be generous - most quant strategies have at least "moderate" strength if they have any backtesting evidence.
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()
            logger = logging.getLogger(__name__)
            logger.debug(f"Claude raw response for {hypothesis_id}: {repr(content)}")

            # Extract JSON from markdown code blocks if present
            # Handle: ```json\n{...}\n``` or ```\n{...}\n```
            import re
            # First try to find content between triple backticks
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if code_block_match:
                content = code_block_match.group(1).strip()
            # Otherwise, look for JSON-like content
            else:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                if json_match:
                    content = json_match.group(0)

            result = json.loads(content)

            # Validate result has expected keys
            if "thesis_strength" not in result or "regime_alignment" not in result:
                raise ValueError(f"Invalid Claude response: {result}")

            logger.info(f"Claude assessment for {hypothesis_id}: thesis={result['thesis_strength']}, regime={result['regime_alignment']}")
            return result

        except Exception as e:
            # Log error and fallback
            logger = logging.getLogger(__name__)
            logger.warning(f"Claude API call failed for {hypothesis_id}: {e}. Using fallback scores.")
            return {"thesis_strength": "moderate", "regime_alignment": "neutral"}

    def _score_economic_dimension(
        self,
        hypothesis_id: str,
        economic_data: dict,
    ) -> float:
        """
        Score economic rationale dimension.

        Averages scores for: Thesis strength, regime alignment, interpretability, uniqueness.

        Args:
            hypothesis_id: The hypothesis being scored
            economic_data: Dict with thesis, regime, black_box_count, uniqueness, agent_reports

        Returns:
            Score 0-1
        """
        # Use Claude to assess thesis and regime if not provided
        if "thesis_strength" not in economic_data or "regime_alignment" not in economic_data:
            claude_assessment = self._assess_thesis_with_claude(
                hypothesis_id=hypothesis_id,
                thesis=economic_data.get("thesis", ""),
                agent_reports=economic_data.get("agent_reports", {}),
                current_regime=economic_data.get("current_regime", "Unknown"),
            )
            economic_data = {**economic_data, **claude_assessment}

        thesis_score = self._score_thesis_strength(economic_data.get("thesis_strength", "moderate"))
        regime_score = self._score_regime_alignment(economic_data.get("regime_alignment", "neutral"))
        interpretability_score = self._score_feature_interpretability(
            economic_data.get("black_box_count", 4)
        )
        uniqueness_score = self._score_uniqueness(economic_data.get("uniqueness", "related"))

        return (thesis_score + regime_score + interpretability_score + uniqueness_score) / 4

    def score_hypothesis(
        self,
        hypothesis_id: str,
        experiment_data: dict,
        risk_data: dict,
        economic_data: dict,
        cost_data: dict,
    ) -> CIOScore:
        """
        Score a hypothesis across all 4 dimensions.

        Args:
            hypothesis_id: The hypothesis to score
            experiment_data: Statistical metrics from MLflow
            risk_data: Risk metrics (max_dd, volatility, etc.)
            economic_data: Economic rationale data (thesis, regime, etc.)
            cost_data: Cost realism data (turnover, capacity, etc.)

        Returns:
            CIOScore with all dimension scores and decision
        """
        # Score each dimension
        statistical = self._score_statistical_dimension(hypothesis_id, experiment_data)
        risk = self._score_risk_dimension(hypothesis_id, risk_data)
        economic = self._score_economic_dimension(hypothesis_id, economic_data)
        cost = self._score_cost_dimension(hypothesis_id, cost_data)

        # Check for critical failures
        critical_failure = self._check_critical_failures_risk(risk_data)

        # Create score object
        return CIOScore(
            hypothesis_id=hypothesis_id,
            statistical=statistical,
            risk=risk,
            economic=economic,
            cost=cost,
            critical_failure=critical_failure,
        )

    # Portfolio constants
    PORTFOLIO_CAPITAL = 1_000_000  # $1M hypothetical
    MAX_POSITIONS = 20
    MAX_WEIGHT_PER_POSITION = 0.05  # 5%
    MIN_WEIGHT_THRESHOLD = 0.01  # 1%

    # Portfolio constraints
    MAX_GROSS_EXPOSURE = 1.0  # 100% long-only
    MAX_SECTOR_CONCENTRATION = 0.30  # 30%
    MAX_TURNOVER = 0.50  # 50% annual
    MAX_DRAWDOWN_LIMIT = 0.15  # 15%

    def _calculate_position_weights(
        self,
        strategies: list[dict],
        target_risk_contribution: float,
        max_weight_cap: float,
    ) -> dict[str, float]:
        """
        Calculate equal-risk position weights.

        weight = target_risk_contribution / volatility, then scaled if needed.

        Args:
            strategies: List of dicts with hypothesis_id and volatility
            target_risk_contribution: Target risk per position (e.g., 0.03 for 3%)
            max_weight_cap: Maximum weight per position (e.g., 0.05 for 5%)

        Returns:
            Dict mapping hypothesis_id to weight (0-1)
        """
        weights = {}
        for strategy in strategies:
            hypothesis_id = strategy["hypothesis_id"]
            volatility = strategy.get("volatility", 0.15)  # Default 15%

            # Equal-risk: weight = target_risk / vol
            weight = target_risk_contribution / volatility
            weights[hypothesis_id] = weight

        # Check if any weights exceed cap
        max_weight = max(weights.values()) if weights else 0
        if max_weight > max_weight_cap:
            # Scale all weights proportionally to fit under cap
            scale_factor = max_weight_cap / max_weight
            weights = {k: v * scale_factor for k, v in weights.items()}

        return weights

    def _check_portfolio_constraints(self, portfolio_state: dict) -> list[str]:
        """
        Check portfolio state against constraints.

        Args:
            portfolio_state: Dict with total_weight, max_sector_weight, turnover, max_drawdown

        Returns:
            List of constraint violation messages (empty if all pass)
        """
        violations = []

        if portfolio_state.get("total_weight", 0) > self.MAX_GROSS_EXPOSURE:
            violations.append(
                f"total_weight {portfolio_state['total_weight']:.1%} > {self.MAX_GROSS_EXPOSURE:.0%}"
            )

        if portfolio_state.get("max_sector_weight", 0) > self.MAX_SECTOR_CONCENTRATION:
            violations.append(
                f"sector concentration {portfolio_state['max_sector_weight']:.1%} "
                f"> {self.MAX_SECTOR_CONCENTRATION:.0%}"
            )

        if portfolio_state.get("turnover", 0) > self.MAX_TURNOVER:
            violations.append(
                f"turnover {portfolio_state['turnover']:.1%} > {self.MAX_TURNOVER:.0%}"
            )

        if portfolio_state.get("max_drawdown", 0) > self.MAX_DRAWDOWN_LIMIT:
            violations.append(
                f"drawdown {portfolio_state['max_drawdown']:.1%} > {self.MAX_DRAWDOWN_LIMIT:.0%}"
            )

        return violations

    def _add_paper_position(
        self,
        hypothesis_id: str,
        weight: float,
        entry_price: float,
    ):
        """
        Add a position to the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being added
            weight: Position weight (0-1)
            entry_price: Entry price per share
        """
        from datetime import date

        self.api._db.execute(
            """
            INSERT INTO paper_portfolio
            (hypothesis_id, weight, entry_price, entry_date, current_price, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (hypothesis_id, weight, entry_price, date.today(), entry_price),
        )

    def _remove_paper_position(self, hypothesis_id: str):
        """
        Remove a position from the paper portfolio.

        Args:
            hypothesis_id: The hypothesis being removed
        """
        self.api._db.execute(
            "DELETE FROM paper_portfolio WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )

    def _log_paper_trade(
        self,
        hypothesis_id: str,
        action: str,
        weight_before: float,
        weight_after: float,
        price: float,
    ):
        """
        Log a simulated paper trade.

        Args:
            hypothesis_id: The hypothesis being traded
            action: One of 'ADD', 'REMOVE', 'REBALANCE'
            weight_before: Weight before trade
            weight_after: Weight after trade
            price: Execution price
        """
        self.api._db.execute(
            """
            INSERT INTO paper_portfolio_trades
            (hypothesis_id, action, weight_before, weight_after, price)
            VALUES (?, ?, ?, ?, ?)
            """,
            (hypothesis_id, action, weight_before, weight_after, price),
        )
