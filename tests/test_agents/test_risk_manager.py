"""Tests for RiskManager class."""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from hrp.agents.research_agents import (
    RiskManager,
    RiskVeto,
    PortfolioRiskAssessment,
    RiskManagerReport,
)


class TestRiskManagerInit:
    """Test RiskManager initialization."""

    def test_init_with_defaults(self):
        """Test RiskManager can be initialized with defaults."""
        agent = RiskManager()

        assert agent.DEFAULT_JOB_ID == "risk_manager_review"
        assert agent.ACTOR == "agent:risk-manager"
        assert agent.max_drawdown == 0.20
        assert agent.max_correlation == 0.70
        assert agent.max_sector_exposure == 0.30
        assert agent.hypothesis_ids is None

    def test_init_with_custom_limits(self):
        """Test RiskManager accepts custom risk limits."""
        agent = RiskManager(
            max_drawdown=0.15,
            max_correlation=0.60,
            max_sector_exposure=0.25,
        )

        assert agent.max_drawdown == 0.15
        assert agent.max_correlation == 0.60
        assert agent.max_sector_exposure == 0.25

    def test_init_with_hypothesis_ids(self):
        """Test RiskManager can filter to specific hypotheses."""
        agent = RiskManager(
            hypothesis_ids=["HYP-001", "HYP-002"],
        )

        assert agent.hypothesis_ids == ["HYP-001", "HYP-002"]


class TestRiskVeto:
    """Test RiskVeto dataclass."""

    def test_risk_veto_creation(self):
        """Test RiskVeto can be created."""
        veto = RiskVeto(
            hypothesis_id="HYP-001",
            veto_reason="Max drawdown too high",
            veto_type="drawdown",
            severity="critical",
            details={"max_drawdown": 0.25},
            veto_date=date.today(),
        )

        assert veto.hypothesis_id == "HYP-001"
        assert veto.veto_reason == "Max drawdown too high"
        assert veto.veto_type == "drawdown"
        assert veto.severity == "critical"
        assert veto.details["max_drawdown"] == 0.25


class TestPortfolioRiskAssessment:
    """Test PortfolioRiskAssessment dataclass."""

    def test_assessment_with_no_vetos(self):
        """Test assessment passes with no vetos."""
        assessment = PortfolioRiskAssessment(
            hypothesis_id="HYP-001",
            passed=True,
            vetos=[],
            warnings=[],
            portfolio_impact={"current_positions": 5},
            assessment_date=date.today(),
        )

        assert assessment.passed is True
        assert len(assessment.vetos) == 0

    def test_assessment_with_critical_veto(self):
        """Test assessment fails with critical veto."""
        veto = RiskVeto(
            hypothesis_id="HYP-001",
            veto_reason="Too risky",
            veto_type="limits",
            severity="critical",
            details={},
            veto_date=date.today(),
        )

        assessment = PortfolioRiskAssessment(
            hypothesis_id="HYP-001",
            passed=False,
            vetos=[veto],
            warnings=[],
            portfolio_impact={},
            assessment_date=date.today(),
        )

        assert assessment.passed is False
        assert len(assessment.vetos) == 1
        assert assessment.vetos[0].severity == "critical"

    def test_assessment_with_warning_only(self):
        """Test assessment passes with warning only."""
        veto = RiskVeto(
            hypothesis_id="HYP-001",
            veto_reason="High volatility",
            veto_type="limits",
            severity="warning",
            details={},
            veto_date=date.today(),
        )

        assessment = PortfolioRiskAssessment(
            hypothesis_id="HYP-001",
            passed=True,  # Warnings don't fail
            vetos=[veto],
            warnings=["Warning message"],
            portfolio_impact={},
            assessment_date=date.today(),
        )

        assert assessment.passed is True


class TestRiskManagerExecute:
    """Test RiskManager execute method."""

    @patch("hrp.agents.base.PlatformAPI")
    def test_execute_no_hypotheses(self, mock_api_class):
        """Test execute returns early when no hypotheses to assess."""
        mock_api = Mock()
        mock_api.list_hypotheses_with_metadata.return_value = []
        mock_api.get_paper_portfolio.return_value = []
        mock_api_class.return_value = mock_api

        agent = RiskManager()
        result = agent.run()

        assert result["status"] == "no_hypotheses"
        assert result["assessments"] == []

    @patch("hrp.agents.base.PlatformAPI")
    def test_execute_with_hypotheses(self, mock_api_class, tmp_path):
        """Test execute processes hypotheses."""
        mock_api = Mock()

        # Mock list_hypotheses_with_metadata for _get_hypotheses_to_assess
        mock_api.list_hypotheses_with_metadata.return_value = [
            {
                "hypothesis_id": "HYP-001",
                "title": "Test Strategy",
                "thesis": "Momentum works",
                "status": "validated",
                "metadata": {"validation_analyst_review": {"sharpe": 1.0, "max_drawdown": 0.15}},
            }
        ]

        # Mock get_paper_portfolio for _calculate_portfolio_impact
        mock_api.get_paper_portfolio.return_value = []

        mock_api_class.return_value = mock_api

        # Use tmp_path for research note output
        research_dir = tmp_path / "research"
        research_dir.mkdir()

        agent = RiskManager()

        with patch("hrp.utils.config.get_config") as mock_config:
            mock_config.return_value.data.research_dir = research_dir
            result = agent.run()

        # Result should contain assessments
        assert len(result["assessments"]) == 1
        assert result["assessments"][0].hypothesis_id == "HYP-001"


class TestRiskManagerCheckDrawdown:
    """Test drawdown risk checking."""

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_drawdown_pass(self, mock_api_class):
        """Test drawdown check passes when within limits."""
        agent = RiskManager(max_drawdown=0.20)

        veto = agent._check_drawdown_risk(
            "HYP-001",
            {"max_drawdown": 0.15},
        )

        assert veto is None

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_drawdown_veto(self, mock_api_class):
        """Test drawdown veto when exceeding limits."""
        agent = RiskManager(max_drawdown=0.20)

        veto = agent._check_drawdown_risk(
            "HYP-001",
            {"max_drawdown": 0.25},
        )

        assert veto is not None
        assert veto.veto_type == "drawdown"
        assert veto.severity == "critical"
        assert "25.0%" in veto.veto_reason


class TestRiskManagerCheckConcentration:
    """Test concentration risk checking."""

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_concentration_pass(self, mock_api_class):
        """Test concentration check passes with good diversification."""
        agent = RiskManager()

        vetos, warnings = agent._check_concentration_risk(
            "HYP-001",
            {"num_positions": 20, "sector_exposure": {}},
            {},
        )

        assert len(vetos) == 0

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_concentration_veto_few_positions(self, mock_api_class):
        """Test concentration veto for too few positions."""
        agent = RiskManager()

        vetos, warnings = agent._check_concentration_risk(
            "HYP-001",
            {"num_positions": 5, "sector_exposure": {}},
            {},
        )

        assert len(vetos) == 1
        assert vetos[0].veto_type == "concentration"
        assert "5 positions" in vetos[0].veto_reason

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_concentration_veto_sector_exposure(self, mock_api_class):
        """Test concentration veto for high sector exposure."""
        agent = RiskManager(max_sector_exposure=0.30)

        vetos, warnings = agent._check_concentration_risk(
            "HYP-001",
            {
                "num_positions": 20,
                "sector_exposure": {"Technology": 0.40},
            },
            {},
        )

        assert len(vetos) == 1
        assert vetos[0].veto_type == "concentration"
        assert "Technology" in vetos[0].veto_reason


class TestRiskManagerCheckRiskLimits:
    """Test risk limits checking."""

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_risk_limits_pass(self, mock_api_class):
        """Test risk limits pass for reasonable values."""
        agent = RiskManager()

        vetos = agent._check_risk_limits(
            "HYP-001",
            {"volatility": 0.15, "turnover": 0.30},
        )

        # Should be empty or only warnings
        critical_vetos = [v for v in vetos if v.severity == "critical"]
        assert len(critical_vetos) == 0

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_risk_limits_warning_high_volatility(self, mock_api_class):
        """Test risk limits warning for high volatility."""
        agent = RiskManager()

        vetos = agent._check_risk_limits(
            "HYP-001",
            {"volatility": 0.30, "turnover": 0.30},
        )

        # Should have a warning for high volatility
        assert any(v.veto_type == "limits" for v in vetos)
        assert any(v.severity == "warning" for v in vetos)

    @patch("hrp.agents.base.PlatformAPI")
    def test_check_risk_limits_warning_high_turnover(self, mock_api_class):
        """Test risk limits warning for high turnover."""
        agent = RiskManager()

        vetos = agent._check_risk_limits(
            "HYP-001",
            {"volatility": 0.15, "turnover": 0.60},
        )

        # Should have a warning for high turnover
        assert any(v.veto_type == "limits" and "turnover" in v.veto_reason.lower() for v in vetos)


class TestRiskManagerCalculatePortfolioImpact:
    """Test portfolio impact calculation."""

    @patch("hrp.agents.base.PlatformAPI")
    def test_calculate_impact_empty_portfolio(self, mock_api_class):
        """Test portfolio impact calculation with empty portfolio."""
        mock_api = Mock()
        mock_df = Mock()
        mock_df.iloc = [{"num_positions": 0, "total_weight": 0.0}]
        mock_api.execute.return_value.fetchdf.return_value = mock_df
        mock_api_class.return_value = mock_api

        agent = RiskManager()

        impact = agent._calculate_portfolio_impact(
            "HYP-001",
            {},
            {},
        )

        assert impact["current_positions"] == 0
        assert impact["new_positions"] == 1
        assert impact["weight_increase"] == 0.05

    @patch("hrp.agents.base.PlatformAPI")
    def test_calculate_impact_existing_positions(self, mock_api_class):
        """Test portfolio impact calculation with existing positions."""
        mock_api = Mock()
        # Mock get_paper_portfolio to return existing positions
        mock_api.get_paper_portfolio.return_value = [
            {"hypothesis_id": f"HYP-{i:03d}", "weight": 0.05} for i in range(5)
        ]
        mock_api_class.return_value = mock_api

        agent = RiskManager()

        impact = agent._calculate_portfolio_impact(
            "HYP-001",
            {},
            {},
        )

        assert impact["current_positions"] == 5
        assert impact["new_positions"] == 6
