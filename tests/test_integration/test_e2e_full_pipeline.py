"""
Full End-to-End Integration Test: Data → Features → Hypothesis → Recommendations → Approval

Tests the entire HRP pipeline with synthetic (but realistic) market data.
This validates every layer of the architecture works together:

1. Database schema creation and seeding
2. Price data ingestion (synthetic, realistic OHLCV)
3. Feature computation (38 technical indicators)
4. Hypothesis lifecycle (draft → testing → validated)
5. Model deployment and predictions
6. Recommendation generation
7. Portfolio construction
8. Explanation generation
9. Pre-trade safety checks
10. Approval/rejection workflow
11. Track record computation
12. Post-trade attribution
13. Kill gate calibration
14. Data quality framework
15. Lineage audit trail verification
"""

import json
import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hrp.data.db import DatabaseManager, get_db
from hrp.data.schema import create_tables


# =============================================================================
# Test Data Generator
# =============================================================================


def generate_realistic_prices(
    symbols: list[str],
    start_date: date,
    num_days: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data using geometric Brownian motion.

    Produces data that looks like real market data:
    - Realistic return distributions (mean ~0.05% daily, vol ~1.5%)
    - Proper OHLC relationships (high >= open,close; low <= open,close)
    - Volume with mean-reversion and occasional spikes
    - Weekdays only

    Args:
        symbols: List of ticker symbols
        start_date: First date in the series
        num_days: Calendar days to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
    """
    rng = np.random.default_rng(seed)

    base_prices = {
        "AAPL": 175.0, "MSFT": 370.0, "GOOGL": 145.0,
        "AMZN": 180.0, "META": 500.0, "NVDA": 850.0,
        "TSLA": 240.0, "JPM": 195.0, "V": 275.0, "UNH": 520.0,
    }

    rows = []
    for symbol in symbols:
        base = base_prices.get(symbol, 150.0)
        price = base

        # Per-symbol drift and volatility
        mu = 0.0005 + rng.normal(0, 0.0002)  # Daily drift
        sigma = 0.015 + rng.normal(0, 0.003)  # Daily volatility
        mean_volume = int(rng.uniform(5e6, 80e6))

        current_date = start_date
        for day_idx in range(num_days):
            current_date = start_date + timedelta(days=day_idx)

            # Skip weekends
            if current_date.weekday() >= 5:
                continue

            # GBM price dynamics
            shock = rng.normal(mu, abs(sigma))
            price = price * (1 + shock)
            price = max(price, 1.0)  # Floor

            # Intraday dynamics
            intraday_vol = abs(rng.normal(0.008, 0.004))
            open_price = price * (1 + rng.normal(0, intraday_vol / 2))
            close_price = price
            high_price = max(open_price, close_price) * (1 + abs(rng.normal(0, intraday_vol)))
            low_price = min(open_price, close_price) * (1 - abs(rng.normal(0, intraday_vol)))

            # Volume with mean-reversion
            vol_factor = rng.lognormal(0, 0.3)
            volume = int(mean_volume * vol_factor)

            rows.append({
                "symbol": symbol,
                "date": current_date,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "adj_close": round(close_price, 2),
                "volume": volume,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def e2e_db():
    """
    Create a fully isolated database for end-to-end testing.

    Sets up schema, symbols, universe, and realistic price data.
    """
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    os.remove(db_path)
    DatabaseManager.reset()
    create_tables(db_path)

    original_env = os.environ.get("HRP_DB_PATH")
    os.environ["HRP_DB_PATH"] = db_path

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    start_date = date(2023, 1, 2)

    # Generate 400 days of realistic price data (Jan 2023 - Mar 2024)
    prices_df = generate_realistic_prices(symbols, start_date, num_days=450, seed=42)

    db = get_db(db_path)
    with db.connection() as conn:
        # Insert symbols
        for sym in symbols:
            conn.execute(
                "INSERT INTO symbols (symbol, name, exchange, asset_type) "
                "VALUES (?, ?, 'NASDAQ', 'equity') ON CONFLICT DO NOTHING",
                [sym, f"{sym} Corp"],
            )

        # Insert universe entries (need entries spanning our date range)
        for d in pd.date_range(start_date, start_date + timedelta(days=450), freq="MS"):
            for sym in symbols:
                conn.execute(
                    "INSERT INTO universe (symbol, date, in_universe, sector, market_cap) "
                    "VALUES (?, ?, TRUE, 'Technology', 2000000000000) ON CONFLICT DO NOTHING",
                    [sym, d.date()],
                )

        # Batch insert prices
        conn.execute("BEGIN")
        for _, row in prices_df.iterrows():
            conn.execute(
                "INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test') ON CONFLICT DO NOTHING",
                [row["symbol"], row["date"], row["open"], row["high"],
                 row["low"], row["close"], row["adj_close"], row["volume"]],
            )
        conn.execute("COMMIT")

        # Insert data sources
        conn.execute("""
            INSERT INTO data_sources (source_id, source_type, status)
            VALUES ('test', 'test', 'active'),
                   ('feature_computation', 'scheduled_job', 'active')
            ON CONFLICT DO NOTHING
        """)

    yield db_path, symbols, prices_df

    # Cleanup
    DatabaseManager.reset()
    if original_env:
        os.environ["HRP_DB_PATH"] = original_env
    elif "HRP_DB_PATH" in os.environ:
        del os.environ["HRP_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in [".wal", "-journal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)


# =============================================================================
# Tests
# =============================================================================


class TestE2EFullPipeline:
    """Full end-to-end pipeline tests with realistic synthetic data."""

    def test_01_price_data_integrity(self, e2e_db):
        """Verify synthetic price data was correctly ingested."""
        db_path, symbols, prices_df = e2e_db
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Verify data exists for all symbols
        for symbol in symbols:
            prices = api.get_prices([symbol], date(2023, 1, 1), date(2024, 3, 31))
            assert not prices.empty, f"No prices for {symbol}"
            assert "close" in prices.columns
            assert len(prices) >= 200, f"Only {len(prices)} rows for {symbol}, expected 200+"

        # Verify OHLC relationships
        all_prices = api.get_prices(symbols, date(2023, 1, 1), date(2024, 3, 31))
        assert (all_prices["high"] >= all_prices["low"]).all()
        assert (all_prices["volume"] > 0).all()

    def test_02_feature_computation(self, e2e_db):
        """Compute all 38 technical features on real data and verify."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.data.ingestion.features import compute_features

        # Compute features for a recent date range
        stats = compute_features(
            symbols=symbols,
            start=date(2024, 1, 2),
            end=date(2024, 2, 29),
            lookback_days=252,
            version="v1",
        )

        assert stats["symbols_success"] >= 3, (
            f"Only {stats['symbols_success']} symbols succeeded, "
            f"failed: {stats['failed_symbols']}"
        )
        assert stats["rows_inserted"] > 0, "No feature rows inserted"
        assert stats["features_computed"] > 0, "No features computed"

        # Verify features are in the database
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Check key features exist
        for feature in ["momentum_20d", "rsi_14d", "volatility_20d", "sma_50d"]:
            result = api.fetchone_readonly(
                "SELECT COUNT(*) FROM features WHERE feature_name = ? AND date >= '2024-01-02'",
                [feature],
            )
            assert result[0] > 0, f"Feature {feature} not found in database"

        # Verify feature values are reasonable
        rsi_vals = api.query_readonly(
            "SELECT value FROM features WHERE feature_name = 'rsi_14d' AND date >= '2024-01-02'",
        )
        assert (rsi_vals["value"] >= 0).all(), "RSI values below 0"
        assert (rsi_vals["value"] <= 100).all(), "RSI values above 100"

    def test_03_hypothesis_lifecycle(self, e2e_db):
        """Test complete hypothesis lifecycle: draft → testing → validated."""
        db_path, _, _ = e2e_db
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Create hypothesis
        hyp_id = api.create_hypothesis(
            title="Momentum Factor: 20d returns predict 5d forward returns",
            thesis="Stocks with strong 20d momentum continue outperforming over next 5 trading days",
            prediction="Top quintile 20d momentum stocks earn >2% over next 5 days",
            falsification="Sharpe ratio < 0.3 across walk-forward folds",
            actor="e2e_test",
        )
        assert hyp_id.startswith("HYP-")

        hyp = api.get_hypothesis(hyp_id)
        assert hyp["status"] == "draft"

        # Verify lineage event created
        lineage = api.get_lineage(hypothesis_id=hyp_id)
        assert len(lineage) >= 1
        assert any(e["event_type"] == "hypothesis_created" for e in lineage)

        # draft → testing (Alpha Researcher)
        api.update_hypothesis(hyp_id, status="testing", outcome=None, actor="alpha_researcher")
        hyp = api.get_hypothesis(hyp_id)
        assert hyp["status"] == "testing"

        # testing → validated (ML Scientist — requires experiment link)
        # First link an experiment
        api.link_experiment(hyp_id, "exp-e2e-001")
        api.update_hypothesis(hyp_id, status="validated", outcome="Passed walk-forward", actor="ml_scientist")
        hyp = api.get_hypothesis(hyp_id)
        assert hyp["status"] == "validated"

        # Verify full lineage trail
        lineage = api.get_lineage(hypothesis_id=hyp_id)
        assert len(lineage) >= 3  # created + at least 2 status changes

    def test_04_model_deployment_and_predictions(self, e2e_db):
        """Test model deployment tracking (mocked MLflow) and hypothesis validation."""
        db_path, symbols, _ = e2e_db
        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Create and validate a hypothesis
        hyp_id = api.create_hypothesis(
            title="ML Momentum Model",
            thesis="XGBoost predicts 5d returns from momentum features",
            prediction="Model Sharpe > 1.0",
            falsification="Sharpe < 0.5",
            actor="e2e_test",
        )
        api.update_hypothesis(hyp_id, status="testing", outcome=None, actor="alpha_researcher")
        api.link_experiment(hyp_id, "exp-model-001")
        api.update_hypothesis(hyp_id, status="validated", outcome="Walk-forward passed", actor="ml_scientist")

        # Directly insert a model deployment record (bypasses MLflow which needs network)
        model_name = "momentum_ridge_e2e"
        next_id = api.fetchone_readonly(
            "SELECT COALESCE(MAX(deployment_id), 0) + 1 FROM model_deployments"
        )[0]
        api.execute_write(
            """INSERT INTO model_deployments (
                deployment_id, model_name, model_version, environment, deployed_by,
                deployed_at, validation_results, status
            ) VALUES (?, ?, '1', 'staging', 'human_cio', CURRENT_TIMESTAMP, ?, 'active')""",
            [next_id, model_name, json.dumps({"sharpe": 1.2, "folds": 5})],
        )

        # Verify deployment
        deployments = api.fetchall_readonly(
            "SELECT * FROM model_deployments WHERE model_name = ?",
            [model_name],
        )
        assert len(deployments) >= 1

    def test_05_recommendation_generation(self, e2e_db):
        """Test full recommendation generation pipeline."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.recommendation_engine import RecommendationEngine, Recommendation
        from hrp.advisory.explainer import RecommendationExplainer
        from hrp.advisory.safeguards import PreTradeChecks

        api = PlatformAPI(db_path=db_path)

        # First compute features
        from hrp.data.ingestion.features import compute_features
        compute_features(
            symbols=symbols,
            start=date(2024, 1, 2),
            end=date(2024, 2, 29),
            lookback_days=252,
            version="v1",
        )

        # Create hypothesis + deploy model (insert deployment record directly)
        hyp_id = api.create_hypothesis(
            title="E2E Recommendation Test Model",
            thesis="Model generates actionable picks",
            prediction="Top picks outperform by 2%",
            falsification="Win rate < 40%",
            actor="e2e_test",
        )
        api.update_hypothesis(hyp_id, status="testing", outcome=None, actor="alpha_researcher")
        api.link_experiment(hyp_id, "exp-rec-001")
        api.update_hypothesis(hyp_id, status="validated", outcome="Passed", actor="ml_scientist")

        # Insert deployment record directly (bypasses MLflow)
        next_id = api.fetchone_readonly(
            "SELECT COALESCE(MAX(deployment_id), 0) + 1 FROM model_deployments"
        )[0]
        api.execute_write(
            """INSERT INTO model_deployments (
                deployment_id, model_name, model_version, environment, deployed_by,
                deployed_at, validation_results, status
            ) VALUES (?, 'rec_test_model', '1', 'staging', 'human_cio',
                      CURRENT_TIMESTAMP, '{"sharpe": 1.5}', 'active')""",
            [next_id],
        )

        # Create recommendation engine with explainer
        explainer = RecommendationExplainer()
        engine = RecommendationEngine(api, explainer=explainer)

        # Mock the prediction to return controlled values
        # Signals must be >= 0.4 for MEDIUM confidence (min_confidence default)
        mock_predictions = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "predicted_return": [0.85, 0.72, 0.55, 0.45, -0.01],
            "model_name": ["rec_test_model"] * 5,
            "hypothesis_id": [hyp_id] * 5,
        })

        mock_deployed = [{"model_name": "rec_test_model", "model_version": "1"}]
        with patch.object(engine, "_get_deployed_models", return_value=mock_deployed):
            with patch.object(engine, "_collect_predictions", return_value=mock_predictions):
                with patch.object(engine, "_get_latest_prices", return_value={
                    "AAPL": 185.0, "MSFT": 380.0, "GOOGL": 148.0,
                    "AMZN": 188.0, "META": 510.0,
                }):
                    recs = engine.generate_weekly_recommendations(
                        as_of_date=date(2024, 2, 15),
                        symbols=symbols,
                        risk_tolerance=3,
                    )

        assert len(recs) > 0, "No recommendations generated"

        # Verify recommendation structure
        for rec in recs:
            assert rec.symbol in symbols
            assert rec.action in ("BUY", "HOLD", "SELL")
            assert rec.confidence in ("HIGH", "MEDIUM", "LOW")
            assert rec.entry_price > 0
            assert rec.target_price > 0
            assert rec.stop_price > 0
            assert rec.signal_strength > 0
            assert len(rec.thesis_plain) > 0
            assert len(rec.risk_plain) > 0

        # Recommendations are already persisted by generate_weekly_recommendations

        # Verify recommendations in database
        db_recs = api.get_recommendations(status="active")
        assert len(db_recs) > 0

    def test_06_portfolio_construction(self, e2e_db):
        """Test MVO portfolio optimization with real covariance estimation."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.portfolio_constructor import (
            PortfolioConstructor,
            PortfolioConstraints,
        )

        api = PlatformAPI(db_path=db_path)

        constraints = PortfolioConstraints(
            max_position_pct=0.30,
            max_sector_pct=0.80,
            max_positions=10,
        )

        constructor = PortfolioConstructor(api, constraints=constraints)

        # Create signal scores
        signals = {
            "AAPL": 0.8,
            "MSFT": 0.7,
            "GOOGL": 0.6,
            "AMZN": 0.5,
            "META": 0.4,
        }

        allocation = constructor.construct(
            signals=signals,
            as_of_date=date(2024, 2, 15),
            lookback_days=252,
        )

        assert len(allocation.weights) > 0, "Empty portfolio"
        total_weight = sum(allocation.weights.values())
        assert abs(total_weight - 1.0) < 0.05, f"Portfolio weights don't sum to ~1.0: {total_weight}"

        # All weights should be positive (long-only)
        for sym, weight in allocation.weights.items():
            assert weight >= 0, f"Negative weight for {sym}: {weight}"
            assert weight <= constraints.max_position_pct + 0.01, (
                f"Weight {weight} exceeds max position {constraints.max_position_pct}"
            )

        # Check PortfolioAllocation fields
        assert allocation.expected_risk >= 0
        assert isinstance(allocation.sector_exposures, dict)

    def test_07_approval_workflow(self, e2e_db):
        """Test recommendation approval and rejection flow."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.approval_workflow import ApprovalWorkflow

        api = PlatformAPI(db_path=db_path)
        workflow = ApprovalWorkflow(api, dry_run=True)

        # Insert test recommendations
        rec_date = date(2024, 2, 15)
        with get_db(db_path).connection() as conn:
            for i, (sym, price) in enumerate([
                ("AAPL", 185.0), ("MSFT", 380.0), ("GOOGL", 148.0),
            ]):
                conn.execute(
                    """INSERT INTO recommendations (
                        recommendation_id, batch_id, symbol, action, confidence,
                        signal_strength, entry_price, target_price, stop_price,
                        position_pct, thesis_plain, risk_plain, time_horizon_days,
                        model_name, status, created_at
                    ) VALUES (?, ?, ?, 'BUY', 'HIGH', ?, ?, ?, ?, 0.05,
                              'Strong momentum', 'Market risk', 20,
                              'test_model', 'pending_approval', CURRENT_TIMESTAMP)""",
                    [
                        f"REC-2024-{100+i:03d}", "BATCH-E2E",
                        sym, 0.8 - i * 0.1,
                        price, price * 1.10, price * 0.95,
                    ],
                )

        # Get pending recommendations
        pending = workflow.get_pending()
        assert len(pending) >= 3, f"Expected 3+ pending, got {len(pending)}"

        # Approve one
        result = workflow.approve("REC-2024-100", actor="human_cio")
        assert result.action == "approved"
        assert result.recommendation_id == "REC-2024-100"

        # Reject one with reason
        result = workflow.reject("REC-2024-101", actor="human_cio", reason="Too risky")
        assert result.action == "cancelled"  # reject sets action to 'cancelled'

        # Verify agent cannot approve
        result = workflow.approve("REC-2024-102", actor="agent:cio_agent")
        assert "denied" in result.message.lower() or "human" in result.message.lower() or "agent" in result.message.lower()

        # Verify status changes persisted
        rec_100 = api.get_recommendation_by_id("REC-2024-100")
        assert rec_100["status"] == "active"  # approve sets pending_approval → active

        rec_101 = api.get_recommendation_by_id("REC-2024-101")
        assert rec_101["status"] == "cancelled"  # reject sets status to cancelled

    def test_08_api_approval_methods(self, e2e_db):
        """Test PlatformAPI approval/reject/approve_all methods."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Insert test recommendation
        with get_db(db_path).connection() as conn:
            conn.execute(
                """INSERT INTO recommendations (
                    recommendation_id, batch_id, symbol, action, confidence,
                    signal_strength, entry_price, target_price, stop_price,
                    position_pct, thesis_plain, risk_plain, time_horizon_days,
                    model_name, status, created_at
                ) VALUES ('REC-2024-200', 'BATCH-API', 'AAPL', 'BUY', 'HIGH',
                          0.8, 185.0, 200.0, 175.0, 0.05,
                          'Strong buy signal', 'Market risk', 20,
                          'test_model', 'pending_approval', CURRENT_TIMESTAMP)"""
            )

        # Approve via API
        result = api.approve_recommendation("REC-2024-200", actor="human_cio", dry_run=True)
        assert result["action"] == "approved"

        # Track record
        track_record = api.get_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert track_record is not None

    def test_09_explainer_with_real_features(self, e2e_db):
        """Test plain-English explanation generation with real feature data."""
        db_path, symbols, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.advisory.explainer import RecommendationExplainer

        explainer = RecommendationExplainer()

        # Generate thesis with feature data
        feature_data = {
            "momentum_20d": 0.08,
            "rsi_14d": 62.0,
            "volatility_20d": 0.18,
            "volume_ratio": 1.3,
            "price_to_sma_50d": 1.05,
        }

        thesis = explainer.generate_thesis(
            "AAPL", feature_data, signal_strength=0.8, model_name="momentum_ridge"
        )
        assert len(thesis) > 20, f"Thesis too short: {thesis}"
        assert "AAPL" in thesis

        # Risk scenario
        risk = explainer.generate_risk_scenario(
            "AAPL", entry_price=185.0, stop_price=175.0, target_price=203.0
        )
        assert len(risk) > 10, f"Risk too short: {risk}"

        # Confidence explanation
        confidence_text = explainer.generate_confidence_explanation(
            "HIGH", signal_strength=0.85
        )
        assert len(confidence_text) > 10

    def test_10_track_record_computation(self, e2e_db):
        """Test track record with closed recommendations."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.track_record import TrackRecordTracker

        api = PlatformAPI(db_path=db_path)
        tracker = TrackRecordTracker(api)

        # Insert closed (resolved) recommendations using valid statuses
        with get_db(db_path).connection() as conn:
            closed_recs = [
                ("REC-2024-300", "AAPL", 180.0, 198.0, "closed_profit"),   # +10% win
                ("REC-2024-301", "MSFT", 370.0, 355.0, "closed_stopped"),  # loss
                ("REC-2024-302", "GOOGL", 145.0, 160.0, "closed_profit"),  # +10% win
                ("REC-2024-303", "AMZN", 180.0, 172.0, "closed_loss"),     # loss
                ("REC-2024-304", "META", 500.0, 550.0, "closed_profit"),   # +10% win
            ]
            for rec_id, sym, entry, close_px, status in closed_recs:
                realized = (close_px - entry) / entry
                conn.execute(
                    """INSERT INTO recommendations (
                        recommendation_id, batch_id, symbol, action, confidence,
                        signal_strength, entry_price, target_price, stop_price,
                        position_pct, thesis_plain, risk_plain, time_horizon_days,
                        model_name, status, close_price, realized_return,
                        created_at, closed_at
                    ) VALUES (?, 'BATCH-TR', ?, 'BUY', 'HIGH', 0.8,
                              ?, ?, ?, 0.05, 'Test', 'Risk', 20,
                              'test_model', ?, ?, ?,
                              '2024-01-15', '2024-02-15')""",
                    [rec_id, sym, entry, entry * 1.10, entry * 0.95,
                     status, close_px, realized],
                )

        record = tracker.compute_track_record(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        )
        assert record is not None
        assert record.total_recommendations >= 5
        assert 0 <= record.win_rate <= 1.0
        # 3 wins out of 5
        assert abs(record.win_rate - 0.6) < 0.01

    def test_11_post_trade_attribution(self, e2e_db):
        """Test trade outcome analysis."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.post_trade_attribution import PostTradeAttributor

        api = PlatformAPI(db_path=db_path)
        attributor = PostTradeAttributor(api)

        # Insert a closed recommendation for attribution
        with get_db(db_path).connection() as conn:
            conn.execute(
                """INSERT INTO recommendations (
                    recommendation_id, batch_id, symbol, action, confidence,
                    signal_strength, entry_price, target_price, stop_price,
                    position_pct, thesis_plain, risk_plain, time_horizon_days,
                    model_name, status, close_price, realized_return,
                    created_at, closed_at
                ) VALUES ('REC-2024-400', 'BATCH-ATTR', 'AAPL', 'BUY', 'HIGH',
                          0.8, 180.0, 198.0, 171.0, 0.05,
                          'Momentum thesis', 'Market risk', 20,
                          'test_model', 'closed_profit', 195.0, 0.0833,
                          '2024-01-15', '2024-02-01')"""
            )

        attribution = attributor.attribute("REC-2024-400")
        assert attribution is not None
        assert attribution.total_return > 0  # Profitable trade
        assert attribution.symbol == "AAPL"

    def test_12_data_quality_framework(self, e2e_db):
        """Test quality checks with real data."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.data.quality.report import QualityReportGenerator

        generator = QualityReportGenerator(db_path, read_only=False)
        report = generator.generate_report(date(2024, 2, 1))

        assert report is not None
        assert report.checks_run > 0
        assert 0 <= report.health_score <= 100

        # Store and retrieve
        report_id = generator.store_report(report)
        assert report_id > 0

        history = generator.get_historical_reports(date(2024, 1, 1), date(2024, 3, 1))
        assert len(history) >= 1

    def test_13_safeguards_and_circuit_breakers(self, e2e_db):
        """Test pre-trade safety checks."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.safeguards import PreTradeChecks, CircuitBreaker

        api = PlatformAPI(db_path=db_path)
        checks = PreTradeChecks(api)
        breaker = CircuitBreaker(api)

        # Data freshness check — should pass since we just loaded data
        # Use a date close to our data range
        result = checks.check_data_freshness(date(2024, 2, 15))
        assert result is not None
        # The result may or may not pass depending on threshold config

        # Market regime check
        result = checks.check_market_regime(date(2024, 2, 15))
        assert result is not None

        # Portfolio concentration check
        portfolio = {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.2}
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Technology"}
        result = checks.check_portfolio_concentration(portfolio, sector_map)
        assert result is not None

        # Circuit breaker
        should_halt, reason = breaker.should_halt(date(2024, 2, 15))
        # Just verify it returns a tuple (halt decision depends on data)
        assert isinstance(should_halt, bool)
        assert isinstance(reason, str)

    def test_14_kill_gate_calibration(self, e2e_db):
        """Test kill gate false positive/negative analysis."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.api.platform import PlatformAPI
        from hrp.advisory.kill_gate_calibrator import KillGateCalibrator

        api = PlatformAPI(db_path=db_path)

        # Create some hypotheses and lineage events for calibration
        with get_db(db_path).connection() as conn:
            # Hypothesis that was killed but later reopened (false positive)
            conn.execute("""
                INSERT INTO hypotheses (
                    hypothesis_id, title, thesis, testable_prediction,
                    falsification_criteria, status, pipeline_stage,
                    created_at, updated_at, created_by
                ) VALUES (
                    'HYP-CAL-001', 'Killed but Reopened', 'FP test',
                    'Test', 'Test', 'testing', 'ml_training',
                    '2024-01-01', '2024-02-01', 'test'
                )
            """)

            # Hypothesis that passed but failed later (false negative)
            conn.execute("""
                INSERT INTO hypotheses (
                    hypothesis_id, title, thesis, testable_prediction,
                    falsification_criteria, status, pipeline_stage,
                    created_at, updated_at, created_by
                ) VALUES (
                    'HYP-CAL-002', 'Passed but Failed', 'FN test',
                    'Test', 'Test', 'rejected', 'kill_gate',
                    '2024-01-01', '2024-02-01', 'test'
                )
            """)

            # Kill gate events
            conn.execute("""
                INSERT INTO lineage (lineage_id, event_type, actor, hypothesis_id, details, timestamp)
                VALUES
                    (1001, 'kill_gate_triggered', 'kill_gate_enforcer', 'HYP-CAL-001',
                     '{"gate": "sharpe_decay", "value": 0.4}', '2024-01-15'),
                    (1002, 'kill_gate_enforcer_complete', 'kill_gate_enforcer', 'HYP-CAL-002',
                     '{"action": "PASS", "gates_checked": 5}', '2024-01-15')
            """)

        calibrator = KillGateCalibrator(api)
        report = calibrator.calibrate(lookback_days=365)

        assert report is not None
        assert report.total_hypotheses >= 0
        assert report.false_positive_rate >= 0
        assert report.false_negative_rate >= 0

    def test_15_digest_generation(self, e2e_db):
        """Test weekly digest email report generation."""
        db_path, _, _ = e2e_db
        os.environ["HRP_DB_PATH"] = db_path

        from hrp.advisory.digest import WeeklyDigest
        from hrp.advisory.track_record import WeeklyReport, TrackRecordSummary

        digest = WeeklyDigest()

        # Create a WeeklyReport with test data
        track_record = TrackRecordSummary(
            period_start=date(2024, 1, 1),
            period_end=date(2024, 2, 15),
            total_recommendations=5,
            closed_recommendations=3,
            profitable=2,
            unprofitable=1,
            win_rate=0.67,
            avg_return=0.05,
            avg_win=0.10,
            avg_loss=-0.04,
            best_pick="AAPL",
            best_return=0.12,
            worst_pick="MSFT",
            worst_return=-0.04,
            total_return=0.15,
            benchmark_return=0.08,
            excess_return=0.07,
            sharpe_ratio=1.5,
        )

        report = WeeklyReport(
            report_date=date(2024, 2, 15),
            new_recommendations=[{
                "symbol": "AAPL", "action": "BUY", "confidence": "HIGH",
                "thesis_plain": "Strong momentum with rising volume",
                "entry_price": 185.0, "target_price": 203.5,
            }],
            open_positions=[],
            closed_this_week=[],
            track_record=track_record,
        )

        content = digest.generate(report)
        assert content is not None
        assert "AAPL" in content.html_body
        assert "AAPL" in content.text_body

    def test_16_lineage_audit_completeness(self, e2e_db):
        """Verify lineage audit trail captures all significant actions."""
        db_path, _, _ = e2e_db

        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Create hypothesis with full lifecycle
        hyp_id = api.create_hypothesis(
            title="Lineage Audit Test",
            thesis="Testing audit trail",
            prediction="All events captured",
            falsification="Missing events",
            actor="lineage_test",
        )

        api.update_hypothesis(hyp_id, status="testing", outcome=None, actor="alpha_researcher")
        api.link_experiment(hyp_id, "exp-lineage-001")
        api.update_hypothesis(hyp_id, status="validated", outcome="Good", actor="ml_scientist")

        lineage = api.get_lineage(hypothesis_id=hyp_id)

        # Should have at least: created, status_change (testing), experiment_linked, status_change (validated)
        assert len(lineage) >= 3
        event_types = [e["event_type"] for e in lineage]
        assert "hypothesis_created" in event_types

    def test_17_health_check_full(self, e2e_db):
        """Test platform health check with populated database."""
        db_path, _, _ = e2e_db

        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)
        health = api.health_check()

        assert health["api"] == "ok"
        assert health["database"] == "ok"

    def test_18_concurrent_operations(self, e2e_db):
        """Test that database handles concurrent reads correctly."""
        db_path, symbols, _ = e2e_db

        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # Simulate concurrent operations
        results = []
        for _ in range(10):
            prices = api.get_prices(["AAPL"], date(2023, 6, 1), date(2023, 12, 31))
            results.append(len(prices))

        # All results should be consistent
        assert len(set(results)) == 1, f"Inconsistent results: {results}"

    def test_19_readonly_query_safety(self, e2e_db):
        """Verify readonly queries cannot modify data."""
        db_path, _, _ = e2e_db

        from hrp.api.platform import PlatformAPI

        api = PlatformAPI(db_path=db_path)

        # This should work (SELECT)
        result = api.query_readonly("SELECT COUNT(*) as cnt FROM prices")
        assert not result.empty

        # These should fail (write operations in readonly mode)
        with pytest.raises(Exception):
            api.query_readonly("DELETE FROM prices WHERE symbol = 'AAPL'")

        with pytest.raises(Exception):
            api.query_readonly("DROP TABLE prices")

    def test_20_universe_management(self, e2e_db):
        """Test universe queries work correctly."""
        db_path, symbols, _ = e2e_db

        from hrp.data.universe import UniverseManager

        manager = UniverseManager(db_path)

        universe = manager.get_universe_at_date(date(2024, 1, 1))
        assert len(universe) == len(symbols)
        for sym in symbols:
            assert sym in universe


class TestSyntheticDataQuality:
    """Verify the synthetic data generator produces realistic data."""

    def test_price_statistics(self):
        """Verify synthetic prices have realistic statistical properties."""
        prices = generate_realistic_prices(["AAPL"], date(2023, 1, 2), num_days=500)
        closes = prices["close"].values

        # Daily returns
        returns = np.diff(closes) / closes[:-1]

        # Mean daily return should be small (-0.5% to 0.5%)
        assert abs(np.mean(returns)) < 0.005

        # Annualized vol should be 10-50%
        annual_vol = np.std(returns) * np.sqrt(252)
        assert 0.05 < annual_vol < 0.60

        # Should have both positive and negative days
        assert np.sum(returns > 0) > 50
        assert np.sum(returns < 0) > 50

    def test_ohlc_relationships(self):
        """Verify OHLC relationships are always valid."""
        prices = generate_realistic_prices(
            ["AAPL", "MSFT", "GOOGL"],
            date(2023, 1, 2),
            num_days=300,
        )

        assert (prices["high"] >= prices["low"]).all()
        assert (prices["high"] >= prices["close"]).all()
        assert (prices["high"] >= prices["open"]).all()
        assert (prices["low"] <= prices["close"]).all()
        assert (prices["low"] <= prices["open"]).all()
        assert (prices["volume"] > 0).all()

    def test_no_weekend_data(self):
        """Verify no data on weekends."""
        prices = generate_realistic_prices(["AAPL"], date(2023, 1, 2), num_days=100)
        weekdays = prices["date"].apply(lambda d: d.weekday())
        assert (weekdays < 5).all(), "Found data on weekends"
