"""
CIO Agent - Chief Investment Officer Agent.

Makes strategic decisions about research lines and manages paper portfolio.
Advisory mode: presents recommendations, awaits user approval.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

from hrp.agents.sdk_agent import SDKAgent
from hrp.api.platform import PlatformAPI


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
    ):
        """
        Initialize CIO Agent.

        Args:
            job_id: Unique job identifier
            actor: Actor identity (e.g., "agent:cio")
            api: PlatformAPI instance (created if None)
            thresholds: Custom decision thresholds
            dependencies: List of data requirements
        """
        super().__init__(
            job_id=job_id,
            actor=actor,
            dependencies=dependencies or [],
        )
        self.api = api or PlatformAPI()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def execute(self) -> dict[str, any]:
        """
        Execute CIO Agent logic.

        This is a placeholder implementation to satisfy the abstract base class.
        The actual weekly review logic will be implemented in later tasks.

        Returns:
            Empty dict for now
        """
        return {}

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

        # Build prompt for Claude
        prompt = f"""Assess this trading hypothesis for economic rationale:

Hypothesis: {thesis}

Current Market Regime: {current_regime}

Agent Reports:
{chr(10).join(f'- {k}: {v[:200]}...' for k, v in agent_reports.items())}

Respond ONLY with valid JSON:
{{
    "thesis_strength": "weak" | "moderate" | "strong",
    "regime_alignment": "mismatch" | "neutral" | "aligned"
}}

Criteria:
- thesis_strength: Is the economic logic sound? Does it have a clear edge?
- regime_alignment: Does this strategy suit the current market regime?
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            return json.loads(content)

        except Exception:
            # Fallback on any error
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
