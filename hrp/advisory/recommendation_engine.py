"""
Recommendation engine for HRP advisory service.

Transforms validated model predictions into structured, actionable
trading recommendations with confidence levels and risk context.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.advisory.explainer import RecommendationExplainer
    from hrp.advisory.portfolio_constructor import PortfolioConstructor
    from hrp.advisory.safeguards import PreTradeChecks
    from hrp.api.platform import PlatformAPI


@dataclass
class Recommendation:
    """A single trading recommendation with full context."""

    recommendation_id: str
    symbol: str
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    signal_strength: float
    entry_price: float
    target_price: float
    stop_price: float
    position_pct: float
    thesis_plain: str
    risk_plain: str
    time_horizon_days: int
    hypothesis_id: str | None
    model_name: str
    batch_id: str
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return asdict(self)


@dataclass
class RecommendationUpdate:
    """Status update for an open recommendation."""

    recommendation_id: str
    symbol: str
    action: str
    entry_price: float
    current_price: float
    unrealized_return: float
    days_held: int
    status: str
    close_reason: str | None = None


def _next_recommendation_id(api: PlatformAPI) -> str:
    """Generate the next REC-YYYY-NNN recommendation ID."""
    year = date.today().year
    prefix = f"REC-{year}-"
    result = api.fetchone_readonly(
        "SELECT recommendation_id FROM recommendations "
        "WHERE recommendation_id LIKE ? "
        "ORDER BY recommendation_id DESC LIMIT 1",
        [f"{prefix}%"],
    )
    if result:
        last_num = int(result[0].split("-")[-1])
        return f"{prefix}{last_num + 1:03d}"
    return f"{prefix}001"


def _next_batch_id() -> str:
    """Generate a batch ID for this run."""
    return f"BATCH-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"


class RecommendationEngine:
    """Transforms validated model predictions into actionable recommendations."""

    def __init__(
        self,
        api: PlatformAPI,
        explainer: RecommendationExplainer | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        pre_trade_checks: PreTradeChecks | None = None,
    ):
        self.api = api
        self.explainer = explainer
        self.portfolio_constructor = portfolio_constructor
        self.pre_trade_checks = pre_trade_checks
        self.max_recommendations = int(
            os.getenv("HRP_ADVISORY_MAX_RECOMMENDATIONS", "5")
        )
        self.min_confidence = os.getenv("HRP_ADVISORY_MIN_CONFIDENCE", "MEDIUM")

    def generate_weekly_recommendations(
        self,
        as_of_date: date,
        symbols: list[str] | None = None,
        risk_tolerance: int = 3,
    ) -> list[Recommendation]:
        """
        Generate weekly recommendations from deployed models.

        Pipeline:
        1. Get deployed models
        2. Get latest predictions
        3. Get current prices
        4. Run pre-trade checks
        5. Rank by signal strength
        6. Apply portfolio construction
        7. Generate plain-English explanations
        8. Persist to database

        Args:
            as_of_date: Date for predictions
            symbols: Override symbol universe (default: S&P 500)
            risk_tolerance: 1-5 scale (1=conservative, 5=aggressive)

        Returns:
            List of Recommendation objects
        """
        batch_id = _next_batch_id()
        logger.info(f"Generating recommendations for {as_of_date}, batch={batch_id}")

        # 1. Get deployed models
        deployed = self._get_deployed_models()
        if not deployed:
            logger.warning("No deployed models found — skipping recommendation generation")
            return []

        # 2. Get universe if not specified
        if symbols is None:
            symbols = self._get_universe(as_of_date)
        if not symbols:
            logger.warning("Empty symbol universe — skipping")
            return []

        # 3. Collect predictions from all deployed models
        all_predictions = self._collect_predictions(deployed, symbols, as_of_date)
        if all_predictions.empty:
            logger.warning("No predictions generated — skipping")
            return []

        # 4. Run pre-trade checks
        if self.pre_trade_checks:
            check_result = self.pre_trade_checks.check_data_freshness(as_of_date)
            if not check_result.passed:
                logger.error(f"Pre-trade check failed: {check_result.message}")
                return []

        # 5. Get current prices
        prices = self._get_latest_prices(all_predictions["symbol"].unique().tolist(), as_of_date)

        # 6. Score and rank
        ranked = self._rank_predictions(all_predictions, risk_tolerance)

        # 7. Apply portfolio construction or simple top-N
        top_n = ranked.head(self.max_recommendations)

        # 8. Build recommendations
        recommendations = []
        for _, row in top_n.iterrows():
            symbol = row["symbol"]
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            signal_strength = float(row["signal_strength"])
            confidence = self._signal_to_confidence(signal_strength)

            if self._confidence_rank(confidence) < self._confidence_rank(self.min_confidence):
                continue

            # Calculate target and stop
            target_price = price * (1 + abs(signal_strength) * 0.5)
            stop_price = price * (1 - self._stop_loss_pct(risk_tolerance))
            position_pct = self._position_size(signal_strength, risk_tolerance)

            rec_id = _next_recommendation_id(self.api)

            # Generate explanations
            thesis = ""
            risk_text = ""
            if self.explainer:
                features = self._get_top_features(symbol, as_of_date)
                thesis = self.explainer.generate_thesis(
                    symbol, features, signal_strength, row.get("model_name", "")
                )
                risk_text = self.explainer.generate_risk_scenario(
                    symbol, price, stop_price, target_price
                )
            if not thesis:
                thesis = (
                    f"Model predicts positive returns for {symbol} with "
                    f"signal strength {signal_strength:.2f}."
                )
            if not risk_text:
                risk_text = (
                    f"Stop loss at ${stop_price:.2f} ({self._stop_loss_pct(risk_tolerance):.0%} "
                    f"below entry). Monitor for broad market drawdowns."
                )

            action = "BUY" if signal_strength > 0 else "SELL"
            time_horizon = 20 if abs(signal_strength) > 0.5 else 60

            rec = Recommendation(
                recommendation_id=rec_id,
                symbol=symbol,
                action=action,
                confidence=confidence,
                signal_strength=signal_strength,
                entry_price=price,
                target_price=round(target_price, 2),
                stop_price=round(stop_price, 2),
                position_pct=round(position_pct, 4),
                thesis_plain=thesis,
                risk_plain=risk_text,
                time_horizon_days=time_horizon,
                hypothesis_id=row.get("hypothesis_id"),
                model_name=row.get("model_name", ""),
                batch_id=batch_id,
            )
            recommendations.append(rec)

        # 9. Persist
        for rec in recommendations:
            self._persist_recommendation(rec)

        logger.info(
            f"Generated {len(recommendations)} recommendations in batch {batch_id}"
        )
        return recommendations

    def review_open_recommendations(self, as_of_date: date) -> list[RecommendationUpdate]:
        """Check all active recommendations and close those that hit stop/target/expiry."""
        updates = []
        active = self.api.query_readonly(
            "SELECT recommendation_id, symbol, action, entry_price, "
            "target_price, stop_price, time_horizon_days, created_at "
            "FROM recommendations WHERE status = 'active'"
        )
        if active.empty:
            return updates

        symbols = active["symbol"].unique().tolist()
        prices = self._get_latest_prices(symbols, as_of_date)

        for _, row in active.iterrows():
            rec_id = row["recommendation_id"]
            symbol = row["symbol"]
            entry_price = float(row["entry_price"])
            target_price = float(row["target_price"])
            stop_price = float(row["stop_price"])
            horizon = int(row["time_horizon_days"])
            created = pd.Timestamp(row["created_at"])
            days_held = (pd.Timestamp(as_of_date) - created).days

            current_price = prices.get(symbol)
            if current_price is None:
                continue

            unrealized_return = (current_price - entry_price) / entry_price
            status = "active"
            close_reason = None

            if current_price >= target_price:
                status = "closed_profit"
                close_reason = "target_reached"
            elif current_price <= stop_price:
                status = "closed_stopped"
                close_reason = "stop_hit"
            elif days_held >= horizon * 1.5:
                status = "expired"
                close_reason = "time_expired"

            if status != "active":
                self.close_recommendation(rec_id, current_price, close_reason)

            updates.append(
                RecommendationUpdate(
                    recommendation_id=rec_id,
                    symbol=symbol,
                    action=row["action"],
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_return=round(unrealized_return, 4),
                    days_held=days_held,
                    status=status,
                    close_reason=close_reason,
                )
            )

        return updates

    def close_recommendation(
        self, recommendation_id: str, close_price: float, reason: str
    ) -> None:
        """Close a recommendation with outcome tracking."""
        rec = self.api.fetchone_readonly(
            "SELECT entry_price FROM recommendations WHERE recommendation_id = ?",
            [recommendation_id],
        )
        if not rec:
            logger.warning(f"Recommendation {recommendation_id} not found")
            return

        entry_price = float(rec[0])
        realized_return = (close_price - entry_price) / entry_price if entry_price > 0 else 0.0

        status = "closed_profit" if realized_return > 0 else "closed_loss"
        if reason == "stop_hit":
            status = "closed_stopped"
        elif reason == "time_expired":
            status = "expired"

        self.api.execute_write(
            "UPDATE recommendations SET status = ?, closed_at = CURRENT_TIMESTAMP, "
            "close_price = ?, realized_return = ? WHERE recommendation_id = ?",
            [status, close_price, realized_return, recommendation_id],
        )
        logger.info(
            f"Closed {recommendation_id}: {status}, return={realized_return:.2%}"
        )

    # --- Private helpers ---

    def _get_deployed_models(self) -> list[dict]:
        """Get all actively deployed models."""
        df = self.api.query_readonly(
            "SELECT model_name, model_version, environment, deployed_at "
            "FROM model_deployments WHERE status = 'active' AND environment = 'production'"
        )
        if df.empty:
            return []
        return df.to_dict("records")

    def _get_universe(self, as_of_date: date) -> list[str]:
        """Get the current trading universe."""
        df = self.api.query_readonly(
            "SELECT DISTINCT symbol FROM universe "
            "WHERE date = (SELECT MAX(date) FROM universe WHERE date <= ?) "
            "AND in_universe = TRUE",
            [as_of_date],
        )
        if df.empty:
            # Fallback to symbols table
            df = self.api.query_readonly(
                "SELECT symbol FROM symbols WHERE asset_type = 'equity'"
            )
        return df["symbol"].tolist() if not df.empty else []

    def _collect_predictions(
        self, deployed: list[dict], symbols: list[str], as_of_date: date
    ) -> pd.DataFrame:
        """Collect predictions from all deployed models."""
        all_preds = []
        for model in deployed:
            try:
                preds = self.api.predict_model(
                    model_name=model["model_name"],
                    symbols=symbols,
                    as_of_date=as_of_date,
                    model_version=model.get("model_version"),
                )
                if preds is not None and not preds.empty:
                    preds["model_name"] = model["model_name"]
                    all_preds.append(preds)
            except Exception as e:
                logger.warning(
                    f"Prediction failed for {model['model_name']}: {e}"
                )
        if not all_preds:
            return pd.DataFrame()
        return pd.concat(all_preds, ignore_index=True)

    def _get_latest_prices(
        self, symbols: list[str], as_of_date: date
    ) -> dict[str, float]:
        """Get the most recent close prices for symbols."""
        prices = {}
        for symbol in symbols:
            try:
                df = self.api.get_prices([symbol], as_of_date, as_of_date)
                if not df.empty:
                    prices[symbol] = float(df.iloc[-1]["close"])
                else:
                    # Try recent 5-day window
                    lookback = as_of_date - timedelta(days=5)
                    df = self.api.get_prices([symbol], lookback, as_of_date)
                    if not df.empty:
                        prices[symbol] = float(df.iloc[-1]["close"])
            except Exception:
                logger.debug(f"Could not get price for {symbol} on {as_of_date}")
        return prices

    def _get_top_features(self, symbol: str, as_of_date: date) -> dict[str, float]:
        """Get key features for explanation generation."""
        key_features = [
            "momentum_20d", "rsi_14d", "volatility_20d",
            "price_to_sma_50d", "volume_ratio",
        ]
        features = {}
        for f in key_features:
            try:
                df = self.api.get_features([symbol], [f], as_of_date)
                if not df.empty:
                    features[f] = float(df.iloc[-1]["value"])
            except Exception:
                pass
        return features

    def _rank_predictions(
        self, predictions: pd.DataFrame, risk_tolerance: int
    ) -> pd.DataFrame:
        """Rank predictions by signal strength, adjusted for risk tolerance."""
        if "prediction" in predictions.columns:
            predictions["signal_strength"] = predictions["prediction"]
        elif "signal" in predictions.columns:
            predictions["signal_strength"] = predictions["signal"]
        else:
            # Use first numeric column as signal
            numeric_cols = predictions.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                predictions["signal_strength"] = predictions[numeric_cols[0]]
            else:
                return predictions.head(0)

        # Filter for positive signals (long-only)
        positive = predictions[predictions["signal_strength"] > 0].copy()
        positive = positive.sort_values("signal_strength", ascending=False)
        return positive

    def _signal_to_confidence(self, signal_strength: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Map signal strength to confidence level."""
        abs_signal = abs(signal_strength)
        if abs_signal >= 0.7:
            return "HIGH"
        if abs_signal >= 0.4:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _confidence_rank(confidence: str) -> int:
        """Numeric rank for confidence comparison."""
        return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(confidence, 0)

    @staticmethod
    def _stop_loss_pct(risk_tolerance: int) -> float:
        """Stop loss percentage based on risk tolerance."""
        return {1: 0.03, 2: 0.05, 3: 0.07, 4: 0.10, 5: 0.15}.get(risk_tolerance, 0.07)

    @staticmethod
    def _position_size(signal_strength: float, risk_tolerance: int) -> float:
        """Position size as fraction of portfolio."""
        base = {1: 0.03, 2: 0.05, 3: 0.07, 4: 0.08, 5: 0.10}.get(risk_tolerance, 0.07)
        return min(base * (1 + abs(signal_strength)), 0.10)

    def _persist_recommendation(self, rec: Recommendation) -> None:
        """Write recommendation to database."""
        self.api.execute_write(
            "INSERT INTO recommendations ("
            "recommendation_id, symbol, action, confidence, signal_strength, "
            "entry_price, target_price, stop_price, position_pct, "
            "thesis_plain, risk_plain, time_horizon_days, "
            "hypothesis_id, model_name, batch_id, status"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rec.recommendation_id, rec.symbol, rec.action, rec.confidence,
                rec.signal_strength, rec.entry_price, rec.target_price,
                rec.stop_price, rec.position_pct, rec.thesis_plain,
                rec.risk_plain, rec.time_horizon_days,
                rec.hypothesis_id, rec.model_name, rec.batch_id, rec.status,
            ],
        )
