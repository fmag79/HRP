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

        self.api.db.execute(
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
        self.api.db.execute(
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
        self.api.db.execute(
            """
            INSERT INTO paper_portfolio_trades
            (hypothesis_id, action, weight_before, weight_after, price)
            VALUES (?, ?, ?, ?, ?)
            """,
            (hypothesis_id, action, weight_before, weight_after, price),
        )
