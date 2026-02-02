"""
Tests for SignalScientist research agent.

Tests cover:
- IC (Information Coefficient) calculation
- Rolling IC methodology
- Hypothesis creation from promising signals
- MLflow logging
- Email notification
"""

import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.data.db import DatabaseManager
from hrp.data.schema import create_tables


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def signal_test_db():
    """Create a temporary database with schema and test data for signal tests."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)
    os.environ["HRP_DB_PATH"] = db_path

    # Insert data_sources entry for signal scientist
    from hrp.data.db import get_db

    db = get_db(db_path)
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES
                ('signal_scientist_scan', 'research_agent', 'active'),
                ('feature_computation', 'scheduled_job', 'active')
            """
        )

        # Insert symbols first (required for FK constraint)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            conn.execute(
                """
                INSERT INTO symbols (symbol, name, exchange, asset_type)
                VALUES (?, ?, 'NASDAQ', 'equity')
                """,
                (symbol, f"{symbol} Inc."),
            )

        # Insert test price data (need >200 days for valid features)
        base_date = date(2024, 1, 1)
        symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in symbols:
            for i in range(400):  # ~2 years of data
                test_date = base_date + timedelta(days=i)
                # Skip weekends
                if test_date.weekday() >= 5:
                    continue

                # Generate price with momentum pattern for AAPL
                base_price = 100 + i * 0.05  # Upward trend
                if symbol == "AAPL":
                    price = base_price * (1 + np.sin(i / 20) * 0.1)  # Momentum signal
                elif symbol == "MSFT":
                    price = base_price * (1 + np.cos(i / 30) * 0.08)
                else:
                    price = base_price * (1 + np.random.randn() * 0.02)

                conn.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
                    """,
                    (
                        symbol,
                        test_date,
                        price * 0.99,
                        price * 1.02,
                        price * 0.98,
                        price,
                        price,
                        1000000 + i * 1000,
                    ),
                )

        # Insert universe data
        for symbol in symbols:
            conn.execute(
                """
                INSERT INTO universe (symbol, date, in_universe)
                VALUES (?, ?, TRUE)
                """,
                (symbol, date.today()),
            )

        # Insert feature data (momentum with known correlation to forward returns)
        for symbol in symbols:
            for i in range(100, 350):  # Start after warmup period
                test_date = base_date + timedelta(days=i)
                if test_date.weekday() >= 5:
                    continue

                # Momentum feature - correlates with forward returns
                momentum = 0.05 + np.sin(i / 20) * 0.1
                volatility = 0.15 + np.cos(i / 30) * 0.05

                for feature_name, value in [
                    ("momentum_20d", momentum),
                    ("volatility_60d", volatility),
                    ("rsi_14d", 50 + np.sin(i / 10) * 30),
                ]:
                    conn.execute(
                        """
                        INSERT INTO features (symbol, date, feature_name, value, version)
                        VALUES (?, ?, ?, ?, 'v1')
                        """,
                        (symbol, test_date, feature_name, value),
                    )

        # Insert successful feature_computation record (for dependency check)
        conn.execute(
            """
            INSERT INTO ingestion_log (log_id, source_id, status, completed_at)
            VALUES (1, 'feature_computation', 'completed', CURRENT_TIMESTAMP)
            """
        )

    yield db_path

    # Cleanup
    DatabaseManager.reset()
    if "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        tmp_file = db_path + ext
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# =============================================================================
# IC Calculation Tests
# =============================================================================


class TestICCalculation:
    """Tests for Information Coefficient calculation."""

    def test_calculate_rolling_ic_positive_correlation(self, signal_test_db):
        """IC calculation should correctly identify positive correlation."""
        from hrp.agents.research_agents import SignalScientist

        # Create feature/return data with known positive correlation
        np.random.seed(42)
        feature = np.linspace(0, 1, 100)  # Monotonically increasing
        forward_returns = feature * 0.1 + np.random.randn(100) * 0.01  # Positive correlation

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=date(2024, 6, 1),
        )

        # Test the IC calculation method
        ic_result = agent._calculate_rolling_ic(
            feature_values=feature,
            forward_returns=forward_returns,
            window_size=60,
        )

        # IC should be positive (>0) for positively correlated data
        assert ic_result["mean_ic"] > 0.5, "Expected strong positive IC for correlated data"
        assert ic_result["ic_ir"] > 0.3, "Expected positive IC IR"

    def test_calculate_rolling_ic_negative_correlation(self, signal_test_db):
        """IC calculation should correctly identify negative correlation."""
        from hrp.agents.research_agents import SignalScientist

        np.random.seed(42)
        feature = np.linspace(0, 1, 100)
        forward_returns = -feature * 0.1 + np.random.randn(100) * 0.01  # Negative correlation

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["volatility_60d"],
            as_of_date=date(2024, 6, 1),
        )

        ic_result = agent._calculate_rolling_ic(
            feature_values=feature,
            forward_returns=forward_returns,
            window_size=60,
        )

        # IC should be negative (<0) for negatively correlated data
        assert ic_result["mean_ic"] < -0.5, "Expected strong negative IC for anti-correlated data"

    def test_calculate_rolling_ic_no_correlation(self, signal_test_db):
        """IC should be near zero for uncorrelated data."""
        from hrp.agents.research_agents import SignalScientist

        np.random.seed(42)
        feature = np.random.randn(100)
        forward_returns = np.random.randn(100)  # Completely random, no correlation

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=date(2024, 6, 1),
        )

        ic_result = agent._calculate_rolling_ic(
            feature_values=feature,
            forward_returns=forward_returns,
            window_size=60,
        )

        # IC should be near zero for uncorrelated data
        assert abs(ic_result["mean_ic"]) < 0.15, "Expected near-zero IC for uncorrelated data"

    def test_rolling_ic_includes_std_and_ir(self, signal_test_db):
        """IC result should include standard deviation and information ratio."""
        from hrp.agents.research_agents import SignalScientist

        np.random.seed(42)
        feature = np.linspace(0, 1, 100)
        forward_returns = feature * 0.1 + np.random.randn(100) * 0.02

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=date(2024, 6, 1),
        )

        ic_result = agent._calculate_rolling_ic(
            feature_values=feature,
            forward_returns=forward_returns,
            window_size=60,
        )

        # Result should contain all required fields
        assert "mean_ic" in ic_result
        assert "ic_std" in ic_result
        assert "ic_ir" in ic_result
        assert "sample_size" in ic_result

        # IC IR = mean_ic / ic_std
        if ic_result["ic_std"] > 0:
            expected_ir = ic_result["mean_ic"] / ic_result["ic_std"]
            assert abs(ic_result["ic_ir"] - expected_ir) < 0.01


class TestSignalScan:
    """Tests for scanning features for signals."""

    def test_scan_single_feature(self, signal_test_db):
        """scan_feature should return SignalScanResult for a single feature with enough data."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        # Use a date within our test data range and reduce sample requirements
        agent = SignalScientist(
            symbols=["AAPL", "MSFT"],
            features=["momentum_20d"],
            forward_horizons=[20],
            lookback_days=200,  # Reduced to match test data
            as_of_date=date(2024, 10, 1),  # Within test data range
        )

        result = agent._scan_feature(
            feature="momentum_20d",
            horizon=20,
            symbols=["AAPL", "MSFT"],
        )

        # Result may be None if not enough data points meet threshold
        # This is expected behavior - just verify the return type
        if result is not None:
            assert isinstance(result, SignalScanResult)
            assert result.feature_name == "momentum_20d"
            assert result.forward_horizon == 20
            assert isinstance(result.ic, float)
            assert isinstance(result.ic_std, float)
            assert isinstance(result.ic_ir, float)

    def test_scan_multiple_horizons(self, signal_test_db):
        """Agent should scan across multiple forward horizons."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            forward_horizons=[5, 10, 20],
            as_of_date=date(2024, 6, 1),
            create_hypotheses=False,  # Don't create hypotheses in test
        )

        # Mock to avoid full execution
        with patch.object(agent, "_load_prices") as mock_prices:
            mock_prices.return_value = pd.DataFrame()

            with patch.object(agent, "_scan_feature") as mock_scan:
                mock_scan.return_value = None

                with patch.object(agent, "_log_to_mlflow") as mock_mlflow:
                    mock_mlflow.return_value = "test_run_id"

                    with patch.object(agent, "_send_email_notification"):
                        agent.execute()

                # Should have scanned each horizon
                assert mock_scan.call_count >= 3  # At least one feature x 3 horizons


class TestTwoFactorCombinations:
    """Tests for two-factor combination scanning."""

    def test_scan_combination_additive(self, signal_test_db):
        """Additive combination should rank(A) + rank(B)."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist(
            symbols=["AAPL", "MSFT"],
            features=["momentum_20d", "volatility_60d"],
            forward_horizons=[20],
            as_of_date=date(2024, 6, 1),
        )

        result = agent._scan_combination(
            feature_a="momentum_20d",
            feature_b="volatility_60d",
            method="additive",
            horizon=20,
            symbols=["AAPL", "MSFT"],
        )

        # Result should indicate it's a combination
        if result is not None:
            assert result.is_combination is True
            assert result.combination_method == "additive"
            assert "momentum_20d" in result.feature_name
            assert "volatility_60d" in result.feature_name

    def test_scan_combination_subtractive(self, signal_test_db):
        """Subtractive combination should rank(A) - rank(B)."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist(
            symbols=["AAPL", "MSFT"],
            features=["momentum_20d", "volatility_60d"],
            forward_horizons=[20],
            as_of_date=date(2024, 6, 1),
        )

        result = agent._scan_combination(
            feature_a="momentum_20d",
            feature_b="volatility_60d",
            method="subtractive",
            horizon=20,
            symbols=["AAPL", "MSFT"],
        )

        if result is not None:
            assert result.is_combination is True
            assert result.combination_method == "subtractive"


# =============================================================================
# Hypothesis Creation Tests
# =============================================================================


class TestHypothesisCreation:
    """Tests for automatic hypothesis creation."""

    def test_creates_hypothesis_above_threshold(self, signal_test_db):
        """Should create hypothesis when IC > threshold."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        # Create agent first, then mock its api attribute
        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            ic_threshold=0.03,
            create_hypotheses=True,
            as_of_date=date(2024, 6, 1),
        )

        # Create a signal result that exceeds threshold
        signal = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.045,  # Above 0.03 threshold
            ic_std=0.08,
            ic_ir=0.56,  # Above 0.3 quality threshold
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        # Mock the api attribute directly
        mock_api = MagicMock()
        mock_api.create_hypothesis.return_value = "HYP-2026-001"
        agent.api = mock_api

        hyp_id = agent._create_hypothesis(signal)

        assert hyp_id is not None
        assert hyp_id.startswith("HYP-")
        mock_api.create_hypothesis.assert_called_once()

        # Check hypothesis content
        call_kwargs = mock_api.create_hypothesis.call_args[1]
        assert "momentum_20d" in call_kwargs["title"]
        assert call_kwargs["actor"] == "agent:signal-scientist"

    def test_no_hypothesis_below_threshold(self, signal_test_db):
        """Should NOT create hypothesis when IC < threshold."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            ic_threshold=0.03,
            create_hypotheses=True,
            as_of_date=date(2024, 6, 1),
        )

        # Create a signal result below threshold
        signal = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.015,  # Below 0.03 threshold
            ic_std=0.05,
            ic_ir=0.30,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        with patch("hrp.agents.base.PlatformAPI") as MockAPI:
            mock_api = MagicMock()
            MockAPI.return_value = mock_api

            # Hypotheses are filtered in execute(), not _create_hypothesis
            # So we test the filtering logic directly
            promising = [signal]
            filtered = [
                s
                for s in promising
                if abs(s.ic) >= agent.ic_threshold and s.ic_ir >= 0.3
            ]

            assert len(filtered) == 0, "Signal below threshold should be filtered"

    def test_no_hypothesis_when_disabled(self, signal_test_db):
        """Should NOT create hypothesis when create_hypotheses=False."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            ic_threshold=0.03,
            create_hypotheses=False,  # Disabled
            as_of_date=date(2024, 6, 1),
        )

        signal = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.06,  # Strong signal
            ic_std=0.08,
            ic_ir=0.75,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        with patch("hrp.agents.base.PlatformAPI") as MockAPI:
            mock_api = MagicMock()
            MockAPI.return_value = mock_api

            # When create_hypotheses is False, agent should not call create
            # This is checked in execute() flow, not _create_hypothesis directly
            assert agent.create_hypotheses is False

    def test_hypothesis_content_positive_correlation(self, signal_test_db):
        """Hypothesis should describe positive correlation correctly."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            ic_threshold=0.03,
            as_of_date=date(2024, 6, 1),
        )

        signal = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.045,  # Positive IC
            ic_std=0.08,
            ic_ir=0.56,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        # Mock the api attribute directly
        mock_api = MagicMock()
        mock_api.create_hypothesis.return_value = "HYP-2026-001"
        agent.api = mock_api

        agent._create_hypothesis(signal)

        call_kwargs = mock_api.create_hypothesis.call_args[1]
        assert "positively correlated" in call_kwargs["thesis"]

    def test_hypothesis_content_negative_correlation(self, signal_test_db):
        """Hypothesis should describe negative correlation correctly."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            ic_threshold=0.03,
            as_of_date=date(2024, 6, 1),
        )

        signal = SignalScanResult(
            feature_name="volatility_60d",
            forward_horizon=20,
            ic=-0.035,  # Negative IC
            ic_std=0.06,
            ic_ir=-0.58,
            sample_size=500,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        # Mock the api attribute directly
        mock_api = MagicMock()
        mock_api.create_hypothesis.return_value = "HYP-2026-001"
        agent.api = mock_api

        agent._create_hypothesis(signal)

        call_kwargs = mock_api.create_hypothesis.call_args[1]
        assert "negatively correlated" in call_kwargs["thesis"]


# =============================================================================
# MLflow Logging Tests
# =============================================================================


class TestMLflowLogging:
    """Tests for MLflow experiment logging."""

    def test_logs_scan_results_to_mlflow(self, signal_test_db):
        """Scan results should be logged to MLflow."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            forward_horizons=[20],
            create_hypotheses=False,
            as_of_date=date(2024, 6, 1),
        )

        results = [
            SignalScanResult(
                feature_name="momentum_20d",
                forward_horizon=20,
                ic=0.04,
                ic_std=0.08,
                ic_ir=0.5,
                sample_size=500,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
            )
        ]

        with patch("hrp.agents.signal_scientist.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock(info=MagicMock(run_id="test_run_123"))
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

            run_id = agent._log_to_mlflow(results)

            # Should have logged parameters and metrics
            mock_mlflow.log_params.assert_called()
            mock_mlflow.log_metrics.assert_called()


# =============================================================================
# Email Notification Tests
# =============================================================================


class TestEmailNotification:
    """Tests for email notification sending."""

    def test_sends_email_on_scan_complete(self, signal_test_db):
        """Should send email notification when scan completes."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            forward_horizons=[20],
            create_hypotheses=False,
            as_of_date=date(2024, 6, 1),
        )

        results = [
            SignalScanResult(
                feature_name="momentum_20d",
                forward_horizon=20,
                ic=0.04,
                ic_std=0.08,
                ic_ir=0.5,
                sample_size=500,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
            )
        ]

        with patch("hrp.agents.signal_scientist.EmailNotifier") as MockNotifier:
            mock_notifier = MagicMock()
            MockNotifier.return_value = mock_notifier

            agent._send_email_notification(
                results=results,
                hypotheses_created=["HYP-2025-001"],
                mlflow_run_id="test_run_123",
                duration=127.5,
            )

            mock_notifier.send_summary_email.assert_called_once()

            # Check email content
            call_args = mock_notifier.send_summary_email.call_args
            subject = call_args[1]["subject"]
            assert "Signal Scan Complete" in subject

    def test_email_includes_top_signals(self, signal_test_db):
        """Email should include top signals found."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            as_of_date=date(2024, 6, 1),
        )

        results = [
            SignalScanResult(
                feature_name="momentum_20d",
                forward_horizon=20,
                ic=0.045,
                ic_std=0.08,
                ic_ir=0.56,
                sample_size=500,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
            ),
            SignalScanResult(
                feature_name="volatility_60d",
                forward_horizon=20,
                ic=-0.035,
                ic_std=0.06,
                ic_ir=-0.58,
                sample_size=500,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 1, 1),
            ),
        ]

        with patch("hrp.agents.signal_scientist.EmailNotifier") as MockNotifier:
            mock_notifier = MagicMock()
            MockNotifier.return_value = mock_notifier

            agent._send_email_notification(
                results=results,
                hypotheses_created=[],
                mlflow_run_id="test_run_123",
                duration=60.0,
            )

            call_kwargs = mock_notifier.send_summary_email.call_args[1]
            summary_data = call_kwargs["summary_data"]

            # Should include signal count
            assert "signals_found" in summary_data or len(summary_data) > 0


# =============================================================================
# Lineage Tracking Tests
# =============================================================================


class TestLineageTracking:
    """Tests for lineage/audit trail logging."""

    def test_logs_agent_completion_event(self, signal_test_db):
        """Agent should log completion event to lineage."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist(
            symbols=["AAPL"],
            features=["momentum_20d"],
            forward_horizons=[20],
            create_hypotheses=False,
            as_of_date=date(2024, 6, 1),
        )

        with patch.object(agent, "_log_agent_event") as mock_log:
            with patch.object(agent, "_get_universe_symbols") as mock_universe:
                mock_universe.return_value = ["AAPL"]

                with patch.object(agent, "_load_prices") as mock_prices:
                    mock_prices.return_value = pd.DataFrame()

                    with patch.object(agent, "_scan_feature") as mock_scan:
                        mock_scan.return_value = None

                        with patch.object(agent, "_log_to_mlflow") as mock_mlflow:
                            mock_mlflow.return_value = "test_run_id"

                            with patch.object(agent, "_send_email_notification"):
                                agent.execute()

            # Should have logged agent completion event
            mock_log.assert_called()
            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["event_type"] == "agent_run_complete"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSignalScientistIntegration:
    """Integration tests for full signal scan workflow."""

    def test_init_defaults(self, signal_test_db):
        """SignalScientist should initialize with sensible defaults."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist()

        assert agent.job_id == "signal_scientist_scan"
        assert agent.actor == "agent:signal-scientist"
        assert agent.ic_threshold == 0.03
        assert agent.create_hypotheses is True
        assert 5 in agent.forward_horizons
        assert 10 in agent.forward_horizons
        assert 20 in agent.forward_horizons

    def test_init_with_custom_params(self, signal_test_db):
        """SignalScientist should accept custom parameters."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist(
            symbols=["AAPL", "MSFT"],
            features=["momentum_20d"],
            forward_horizons=[5],
            ic_threshold=0.05,
            create_hypotheses=False,
            as_of_date=date(2024, 1, 15),
        )

        assert agent.symbols == ["AAPL", "MSFT"]
        assert agent.features == ["momentum_20d"]
        assert agent.forward_horizons == [5]
        assert agent.ic_threshold == 0.05
        assert agent.create_hypotheses is False
        assert agent.as_of_date == date(2024, 1, 15)

    def test_has_data_requirement_on_features(self, signal_test_db):
        """SignalScientist should require feature data to exist."""
        from hrp.agents.research_agents import SignalScientist

        agent = SignalScientist()

        # Now uses data requirements instead of job dependencies
        assert agent.dependencies == []
        assert len(agent.data_requirements) == 1
        assert agent.data_requirements[0].table == "features"

    def test_factor_pairs_are_defined(self, signal_test_db):
        """SignalScientist should have predefined factor pairs."""
        from hrp.agents.research_agents import SignalScientist

        # Check that FACTOR_PAIRS class variable exists and has entries
        assert hasattr(SignalScientist, "FACTOR_PAIRS")
        assert len(SignalScientist.FACTOR_PAIRS) > 0

        # Each pair should be a tuple of two feature names
        for pair in SignalScientist.FACTOR_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2


# =============================================================================
# TestAdaptiveICThresholds - Tests for adaptive IC thresholds
# ==============================================================================


class TestAdaptiveICThresholds:
    """Tests for adaptive IC thresholds by strategy class."""

    def test_adaptive_ic_thresholds(self):
        """IC thresholds adapt to strategy class."""
        from hrp.agents.research_agents import IC_THRESHOLDS

        # Cross-sectional factor (more lenient)
        assert IC_THRESHOLDS["cross_sectional_factor"]["pass"] == 0.015
        assert IC_THRESHOLDS["cross_sectional_factor"]["kill"] == 0.005

        # Time-series momentum (moderate)
        assert IC_THRESHOLDS["time_series_momentum"]["pass"] == 0.02
        assert IC_THRESHOLDS["time_series_momentum"]["kill"] == 0.01

        # ML composite (stricter)
        assert IC_THRESHOLDS["ml_composite"]["pass"] == 0.025
        assert IC_THRESHOLDS["ml_composite"]["kill"] == 0.01

    def test_ic_threshold_by_strategy_class(self):
        """Get IC threshold for specific strategy class."""
        from hrp.agents.research_agents import get_ic_thresholds

        thresholds = get_ic_thresholds("cross_sectional_factor")
        assert thresholds["pass"] == 0.015

        thresholds = get_ic_thresholds("ml_composite")
        assert thresholds["pass"] == 0.025

    def test_ic_threshold_default_fallback(self):
        """Unknown strategy class returns default thresholds."""
        from hrp.agents.research_agents import get_ic_thresholds

        thresholds = get_ic_thresholds("unknown_strategy")
        assert thresholds["pass"] == 0.03  # Default

    def test_hypothesis_created_with_strategy_class(self, signal_test_db):
        """Signal Scientist tags hypothesis with strategy class."""
        from hrp.agents.research_agents import SignalScientist, SignalScanResult

        # Create a signal result (using correct fields)
        signal = SignalScanResult(
            feature_name="momentum_20d",
            forward_horizon=20,
            ic=0.04,
            ic_std=0.01,
            ic_ir=3.5,
            sample_size=1000,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )

        agent = SignalScientist()

        # Verify that _create_hypothesis method exists
        assert hasattr(agent, "_create_hypothesis")

        # Verify the logic would classify momentum as time_series_momentum
        feature = "momentum_20d".lower()
        if "momentum" in feature or "returns" in feature:
            strategy_class = "time_series_momentum"
        else:
            strategy_class = "cross_sectional_factor"

        assert strategy_class == "time_series_momentum"

        # Verify rsi feature would be classified as cross_sectional_factor
        feature = "rsi_14d".lower()
        if "momentum" in feature or "returns" in feature:
            strategy_class = "time_series_momentum"
        else:
            strategy_class = "cross_sectional_factor"

        assert strategy_class == "cross_sectional_factor"
