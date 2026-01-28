"""Tests for paper portfolio management."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from hrp.agents.cio import CIOAgent


class TestPortfolioAllocation:
    """Test portfolio allocation calculations."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_calculate_position_weights_equal_risk(self, agent):
        """Test equal-risk position weighting."""
        # 3 strategies with different volatilities
        strategies = [
            {"hypothesis_id": "HYP-001", "volatility": 0.10},
            {"hypothesis_id": "HYP-002", "volatility": 0.15},
            {"hypothesis_id": "HYP-003", "volatility": 0.20},
        ]

        weights = agent._calculate_position_weights(
            strategies=strategies,
            target_risk_contribution=0.03,  # 3% risk per position
            max_weight_cap=0.05,  # 5% max weight
        )

        # Lower volatility = higher weight
        assert weights["HYP-001"] > weights["HYP-002"]
        assert weights["HYP-002"] > weights["HYP-003"]

        # All weights capped at 5%
        for weight in weights.values():
            assert weight <= 0.05

    def test_calculate_position_weights_respects_max_weight(self, agent):
        """Test that max weight cap is enforced."""
        strategies = [
            {"hypothesis_id": "HYP-001", "volatility": 0.05},  # Very low vol
        ]

        weights = agent._calculate_position_weights(
            strategies=strategies,
            target_risk_contribution=0.03,
            max_weight_cap=0.05,  # 5% cap
        )

        # Should be capped at 5% even though vol is very low
        assert weights["HYP-001"] == 0.05

    def test_calculate_position_weights_empty(self, agent):
        """Test empty strategy list returns empty weights."""
        weights = agent._calculate_position_weights(
            strategies=[],
            target_risk_contribution=0.03,
            max_weight_cap=0.05,
        )

        assert len(weights) == 0

    def test_check_portfolio_constraints_pass(self, agent):
        """Test portfolio constraints pass when valid."""
        portfolio_state = {
            "total_weight": 0.95,  # Below 100%
            "max_sector_weight": 0.25,  # Below 30%
            "turnover": 0.40,  # Below 50%
            "max_drawdown": 0.12,  # Below 15%
        }

        violations = agent._check_portfolio_constraints(portfolio_state)

        assert len(violations) == 0

    def test_check_portfolio_constraints_fail(self, agent):
        """Test portfolio constraints detect violations."""
        portfolio_state = {
            "total_weight": 1.10,  # Over 100%
            "max_sector_weight": 0.35,  # Over 30%
            "turnover": 0.60,  # Over 50%
            "max_drawdown": 0.18,  # Over 15%
        }

        violations = agent._check_portfolio_constraints(portfolio_state)

        assert len(violations) == 4
        assert any("total_weight" in v for v in violations)
        assert any("sector" in v for v in violations)
        assert any("turnover" in v for v in violations)
        assert any("drawdown" in v for v in violations)


class TestPortfolioOperations:
    """Test portfolio database operations."""

    @pytest.fixture
    def agent(self):
        """Create a CIOAgent for testing."""
        with patch("hrp.agents.cio.PlatformAPI"):
            return CIOAgent(job_id="test-job", actor="agent:cio-test")

    def test_add_position_to_portfolio(self, agent):
        """Test adding a position to paper portfolio."""
        with patch.object(agent.api.db, "execute") as mock_execute:
            agent._add_paper_position(
                hypothesis_id="HYP-001",
                weight=0.042,
                entry_price=150.0,
            )

            mock_execute.assert_called_once()

    def test_remove_position_from_portfolio(self, agent):
        """Test removing a position from paper portfolio."""
        with patch.object(agent.api.db, "execute") as mock_execute:
            agent._remove_paper_position(hypothesis_id="HYP-001")

            mock_execute.assert_called_once()

    def test_log_paper_trade(self, agent):
        """Test logging a simulated trade."""
        with patch.object(agent.api.db, "execute") as mock_execute:
            agent._log_paper_trade(
                hypothesis_id="HYP-001",
                action="ADD",
                weight_before=0.0,
                weight_after=0.042,
                price=150.0,
            )

            mock_execute.assert_called_once()
