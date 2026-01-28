"""Tests for CIO scoring dimensions."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from hrp.agents.cio import CIOAgent, CIOScore


class TestStatisticalScoring:
    """Test statistical dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_score_statistical_perfect(self, agent):
        """Test perfect statistical score (all metrics at target)."""
        # Mock experiment data with perfect metrics
        experiment_data = {
            "sharpe": 1.5,  # Above 1.0 target
            "stability_score": 0.5,  # At target
            "mean_ic": 0.05,  # Above 0.03 target
            "fold_cv": 1.0,  # Below 2.0 target
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # All at or above target should give ~1.0
        assert score >= 0.9
        assert score <= 1.0

    def test_score_statistical_mixed(self, agent):
        """Test mixed statistical score (some good, some bad)."""
        experiment_data = {
            "sharpe": 1.2,  # Above target
            "stability_score": 1.5,  # Above (worse than) target
            "mean_ic": 0.02,  # Below target
            "fold_cv": 2.5,  # Above (worse than) target
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # Mixed should give middle score
        assert 0.3 <= score <= 0.7

    def test_score_statistical_poor(self, agent):
        """Test poor statistical score (all metrics below target)."""
        experiment_data = {
            "sharpe": 0.4,  # Below 0.5 bad threshold
            "stability_score": 2.5,  # Above 2.0 bad threshold
            "mean_ic": 0.005,  # Below 0.01 bad threshold
            "fold_cv": 3.5,  # Above 3.0 bad threshold
        }

        score = agent._score_statistical_dimension("HYP-2026-001", experiment_data)

        # All bad should give ~0
        assert score >= 0.0
        assert score <= 0.2

    def test_score_statistical_linear_sharpe(self, agent):
        """Test Sharpe scoring is linear: 0.5->0, 1.0->0.5, 1.5->1.0."""
        # At bad threshold (0.5)
        assert agent._score_sharpe(0.5) == pytest.approx(0.0, abs=0.01)
        # At target (1.0)
        assert agent._score_sharpe(1.0) == pytest.approx(0.5, abs=0.01)
        # At good threshold (1.5)
        assert agent._score_sharpe(1.5) == pytest.approx(1.0, abs=0.01)
        # Clamp below bad
        assert agent._score_sharpe(0.3) == 0.0
        # Clamp above good
        assert agent._score_sharpe(2.0) == 1.0

    def test_score_statistical_linear_stability(self, agent):
        """Test stability scoring is linear: 2.0->0, 1.0->0.5, 0.5->1.0."""
        # Lower is better for stability
        # At bad threshold (2.0)
        assert agent._score_stability(2.0) == pytest.approx(0.0, abs=0.01)
        # At target (1.0)
        assert agent._score_stability(1.0) == pytest.approx(0.5, abs=0.01)
        # At good threshold (0.5)
        assert agent._score_stability(0.5) == pytest.approx(1.0, abs=0.01)

    def test_score_statistical_linear_ic(self, agent):
        """Test IC scoring is linear: 0.01->0, 0.03->0.5, 0.05->1.0."""
        # At bad threshold (0.01)
        assert agent._score_ic(0.01) == pytest.approx(0.0, abs=0.01)
        # At target (0.03)
        assert agent._score_ic(0.03) == pytest.approx(0.5, abs=0.01)
        # At good threshold (0.05)
        assert agent._score_ic(0.05) == pytest.approx(1.0, abs=0.01)

    def test_score_statistical_linear_fold_cv(self, agent):
        """Test fold CV scoring is linear: 3.0->0, 2.0->0.5, 1.0->1.0."""
        # Lower is better for CV
        assert agent._score_fold_cv(3.0) == pytest.approx(0.0, abs=0.01)
        assert agent._score_fold_cv(2.0) == pytest.approx(0.5, abs=0.01)
        assert agent._score_fold_cv(1.0) == pytest.approx(1.0, abs=0.01)


class TestRiskScoring:
    """Test risk dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_score_risk_perfect(self, agent):
        """Test perfect risk score (all metrics at target)."""
        risk_data = {
            "max_drawdown": 0.10,  # Below 20% target (good)
            "volatility": 0.10,  # Below 15% target (good)
            "regime_stable": True,  # Binary good
            "sharpe_decay_ok": True,  # Binary good
        }

        score = agent._score_risk_dimension("HYP-2026-001", risk_data)

        assert score >= 0.9

    def test_score_risk_mixed(self, agent):
        """Test mixed risk score."""
        risk_data = {
            "max_drawdown": 0.18,  # Near target
            "volatility": 0.14,  # Near target
            "regime_stable": True,
            "sharpe_decay_ok": False,  # Bad
        }

        score = agent._score_risk_dimension("HYP-2026-001", risk_data)

        # 3 good (including 2 linear near target) + 1 bad = middle score
        assert 0.4 <= score <= 0.7

    def test_score_risk_linear_max_dd(self, agent):
        """Test max drawdown scoring: 30%->0, 20%->0.5, 10%->1.0."""
        # At bad threshold (30%)
        assert agent._score_max_drawdown(0.30) == pytest.approx(0.0, abs=0.01)
        # At target (20%)
        assert agent._score_max_drawdown(0.20) == pytest.approx(0.5, abs=0.01)
        # At good threshold (10%)
        assert agent._score_max_drawdown(0.10) == pytest.approx(1.0, abs=0.01)

    def test_score_risk_linear_volatility(self, agent):
        """Test volatility scoring: 25%->0, 15%->0.5, 10%->1.0."""
        assert agent._score_volatility(0.25) == pytest.approx(0.0, abs=0.01)
        assert agent._score_volatility(0.15) == pytest.approx(0.5, abs=0.01)
        assert agent._score_volatility(0.10) == pytest.approx(1.0, abs=0.01)

    def test_score_risk_binary_regime(self, agent):
        """Test regime stability is binary."""
        assert agent._score_regime_stability(True) == 1.0
        assert agent._score_regime_stability(False) == 0.0

    def test_score_risk_binary_sharpe_decay(self, agent):
        """Test Sharpe decay is binary (<= 50% is good)."""
        assert agent._score_sharpe_decay(0.40) == 1.0  # Below limit
        assert agent._score_sharpe_decay(0.50) == 1.0  # At limit
        assert agent._score_sharpe_decay(0.60) == 0.0  # Above limit

    def test_check_critical_failure_risk(self, agent):
        """Test critical failure detection in risk dimension."""
        # No critical failure
        risk_data = {
            "max_drawdown": 0.20,
            "volatility": 0.15,
            "regime_stable": True,
            "sharpe_decay": 0.40,
        }
        assert agent._check_critical_failures_risk(risk_data) is False

        # Critical: Max DD > 35%
        risk_data["max_drawdown"] = 0.40
        assert agent._check_critical_failures_risk(risk_data) is True

        # Critical: Sharpe decay > 75%
        risk_data["max_drawdown"] = 0.20
        risk_data["sharpe_decay"] = 0.80
        assert agent._check_critical_failures_risk(risk_data) is True


class TestCostScoring:
    """Test cost realism dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_score_cost_perfect(self, agent):
        """Test perfect cost score (all metrics at target)."""
        cost_data = {
            "slippage_survival": "stable",  # Survives 2x slippage
            "turnover": 0.20,  # Below 50% target (good)
            "capacity": "high",  # >$10M (good)
            "execution_complexity": "low",  # Low complexity (good)
        }

        score = agent._score_cost_dimension("HYP-2026-001", cost_data)

        assert score >= 0.9

    def test_score_cost_linear_turnover(self, agent):
        """Test turnover scoring: 100%->0, 50%->0.5, 20%->1.0."""
        # At bad threshold (100%)
        assert agent._score_turnover(1.00) == pytest.approx(0.0, abs=0.01)
        # At target (50%)
        assert agent._score_turnover(0.50) == pytest.approx(0.5, abs=0.01)
        # At good threshold (20%)
        assert agent._score_turnover(0.20) == pytest.approx(1.0, abs=0.01)

    def test_score_cost_ordinal_capacity(self, agent):
        """Test capacity scoring is ordinal."""
        assert agent._score_capacity("low") == pytest.approx(0.0, abs=0.01)   # <$1M
        assert agent._score_capacity("medium") == pytest.approx(0.5, abs=0.01)  # $1-10M
        assert agent._score_capacity("high") == pytest.approx(1.0, abs=0.01)    # >$10M

    def test_score_cost_ordinal_slippage(self, agent):
        """Test slippage survival scoring is ordinal."""
        assert agent._score_slippage_survival("collapse") == pytest.approx(0.0, abs=0.01)
        assert agent._score_slippage_survival("degraded") == pytest.approx(0.5, abs=0.01)
        assert agent._score_slippage_survival("stable") == pytest.approx(1.0, abs=0.01)

    def test_score_cost_ordinal_complexity(self, agent):
        """Test execution complexity scoring is ordinal."""
        assert agent._score_execution_complexity("high") == pytest.approx(0.0, abs=0.01)
        assert agent._score_execution_complexity("medium") == pytest.approx(0.5, abs=0.01)
        assert agent._score_execution_complexity("low") == pytest.approx(1.0, abs=0.01)

    def test_score_cost_dimension(self, agent):
        """Test full cost dimension scoring."""
        cost_data = {
            "slippage_survival": "stable",
            "turnover": 0.35,  # Between target and good
            "capacity": "medium",
            "execution_complexity": "medium",
        }

        score = agent._score_cost_dimension("HYP-2026-001", cost_data)

        # Should be middle score
        assert 0.4 <= score <= 0.7


class TestEconomicScoring:
    """Test economic rationale dimension scoring."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_score_economic_dimension(self, agent):
        """Test economic dimension with mocked Claude API."""
        economic_data = {
            "thesis_strength": "strong",
            "regime_alignment": "aligned",
            "feature_interpretability": 2,  # < 3 black-box features
            "uniqueness": "novel",
        }

        with patch.object(agent, "_assess_thesis_with_claude") as mock_claude:
            mock_claude.return_value = {
                "thesis_strength": "strong",
                "regime_alignment": "aligned",
            }

            score = agent._score_economic_dimension("HYP-2026-001", economic_data)

            # All strong should give high score
            assert score >= 0.8

    def test_score_economic_ordinal_thesis(self, agent):
        """Test thesis strength scoring is ordinal."""
        assert agent._score_thesis_strength("weak") == pytest.approx(0.0, abs=0.01)
        assert agent._score_thesis_strength("moderate") == pytest.approx(0.5, abs=0.01)
        assert agent._score_thesis_strength("strong") == pytest.approx(1.0, abs=0.01)

    def test_score_economic_ordinal_regime(self, agent):
        """Test regime alignment scoring is ordinal."""
        assert agent._score_regime_alignment("mismatch") == pytest.approx(0.0, abs=0.01)
        assert agent._score_regime_alignment("neutral") == pytest.approx(0.5, abs=0.01)
        assert agent._score_regime_alignment("aligned") == pytest.approx(1.0, abs=0.01)

    def test_score_economic_linear_interpretability(self, agent):
        """Test feature interpretability: >5->0, 3-5->0.5, <3->1.0."""
        assert agent._score_feature_interpretability(7) == pytest.approx(0.0, abs=0.01)
        assert agent._score_feature_interpretability(4) == pytest.approx(0.5, abs=0.01)
        assert agent._score_feature_interpretability(2) == pytest.approx(1.0, abs=0.01)

    def test_score_economic_ordinal_uniqueness(self, agent):
        """Test uniqueness scoring is ordinal."""
        assert agent._score_uniqueness("duplicate") == pytest.approx(0.0, abs=0.01)
        assert agent._score_uniqueness("related") == pytest.approx(0.5, abs=0.01)
        assert agent._score_uniqueness("novel") == pytest.approx(1.0, abs=0.01)

    def test_assess_thesis_with_claude(self, agent):
        """Test Claude API assessment of thesis strength and regime."""
        # Mock the Claude client
        mock_client = Mock()
        agent.anthropic_client = mock_client

        # Mock API response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"thesis_strength": "strong", "regime_alignment": "aligned"}')]
        mock_client.messages.create.return_value = mock_response

        result = agent._assess_thesis_with_claude(
            hypothesis_id="HYP-2026-001",
            thesis="Momentum predicts returns",
            agent_reports={"alpha_researcher": "Strong thesis..."},
            current_regime="Bull Market",
        )

        assert result["thesis_strength"] == "strong"
        assert result["regime_alignment"] == "aligned"
