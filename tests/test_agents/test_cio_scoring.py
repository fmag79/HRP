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
