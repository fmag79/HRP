"""
Approval workflow for trading recommendations.

Bridges the advisory service to the execution layer. Users approve or reject
individual recommendations; approved picks are converted to orders and
submitted to the configured broker (or logged in dry-run mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hrp.api.platform import PlatformAPI


@dataclass
class ApprovalResult:
    """Result of approving or rejecting a recommendation."""

    recommendation_id: str
    action: Literal["approved", "rejected", "cancelled"]
    actor: str
    reason: str | None = None
    order_id: str | None = None  # Set when order is generated
    message: str = ""


class ApprovalWorkflow:
    """
    Manages the approve/reject lifecycle for recommendations.

    Flow:
    1. RecommendationEngine generates picks → status = 'pending_approval' or 'active'
    2. User reviews on dashboard or via digest
    3. User approves → convert to Order → submit to broker (or dry-run)
    4. User rejects → mark as cancelled with reason
    """

    def __init__(self, api: PlatformAPI, dry_run: bool = True):
        self.api = api
        self.dry_run = dry_run

    def get_pending(self) -> pd.DataFrame:
        """Get all recommendations awaiting approval."""
        return self.api.get_recommendations(status="pending_approval")

    def submit_for_approval(self, recommendation_ids: list[str]) -> int:
        """
        Mark active recommendations as pending_approval.

        Used when the workflow requires explicit user approval before
        recommendations become tradeable.

        Returns:
            Number of recommendations moved to pending_approval.
        """
        count = 0
        for rec_id in recommendation_ids:
            rec = self.api.get_recommendation_by_id(rec_id)
            if rec is None:
                logger.warning(f"Recommendation {rec_id} not found")
                continue
            if rec["status"] != "active":
                logger.debug(f"Recommendation {rec_id} is {rec['status']}, not active")
                continue

            self.api.execute_write(
                "UPDATE recommendations SET status = 'pending_approval' "
                "WHERE recommendation_id = ?",
                [rec_id],
            )
            count += 1
            logger.info(f"Recommendation {rec_id} submitted for approval")

        return count

    def approve(
        self,
        recommendation_id: str,
        actor: str = "user",
    ) -> ApprovalResult:
        """
        Approve a recommendation for execution.

        Steps:
        1. Validate recommendation exists and is pending
        2. Update status to 'active'
        3. Generate order (if not dry-run, submit to broker)
        4. Log lineage event

        Args:
            recommendation_id: The recommendation to approve
            actor: Who approved (must not be an agent)

        Returns:
            ApprovalResult with order details
        """
        if actor.startswith("agent:"):
            return ApprovalResult(
                recommendation_id=recommendation_id,
                action="rejected",
                actor=actor,
                reason="Agents cannot approve recommendations for execution",
                message="Permission denied: only human users can approve trades",
            )

        rec = self.api.get_recommendation_by_id(recommendation_id)
        if rec is None:
            return ApprovalResult(
                recommendation_id=recommendation_id,
                action="rejected",
                actor=actor,
                reason="not_found",
                message=f"Recommendation {recommendation_id} not found",
            )

        if rec["status"] not in ("pending_approval", "active"):
            return ApprovalResult(
                recommendation_id=recommendation_id,
                action="rejected",
                actor=actor,
                reason=f"invalid_status:{rec['status']}",
                message=f"Cannot approve recommendation in '{rec['status']}' status",
            )

        # Update status to active
        self.api.execute_write(
            "UPDATE recommendations SET status = 'active' WHERE recommendation_id = ?",
            [recommendation_id],
        )

        # Generate order
        order_id = None
        if not self.dry_run:
            order_id = self._execute_order(rec)

        # Log lineage
        self._log_approval(recommendation_id, actor, "approved")

        action_msg = "approved (dry-run)" if self.dry_run else "approved and submitted"
        return ApprovalResult(
            recommendation_id=recommendation_id,
            action="approved",
            actor=actor,
            order_id=order_id,
            message=f"Recommendation {recommendation_id} {action_msg}",
        )

    def approve_all(self, actor: str = "user") -> list[ApprovalResult]:
        """
        Approve all pending recommendations at once.

        Returns:
            List of ApprovalResult for each recommendation.
        """
        pending = self.get_pending()
        if pending.empty:
            return []

        results = []
        for _, row in pending.iterrows():
            result = self.approve(row["recommendation_id"], actor)
            results.append(result)

        return results

    def reject(
        self,
        recommendation_id: str,
        actor: str = "user",
        reason: str = "",
    ) -> ApprovalResult:
        """
        Reject a recommendation.

        Args:
            recommendation_id: The recommendation to reject
            actor: Who rejected
            reason: Why it was rejected (for audit trail)

        Returns:
            ApprovalResult
        """
        rec = self.api.get_recommendation_by_id(recommendation_id)
        if rec is None:
            return ApprovalResult(
                recommendation_id=recommendation_id,
                action="rejected",
                actor=actor,
                reason="not_found",
                message=f"Recommendation {recommendation_id} not found",
            )

        if rec["status"] not in ("pending_approval", "active"):
            return ApprovalResult(
                recommendation_id=recommendation_id,
                action="rejected",
                actor=actor,
                reason=f"invalid_status:{rec['status']}",
                message=f"Cannot reject recommendation in '{rec['status']}' status",
            )

        self.api.execute_write(
            "UPDATE recommendations SET status = 'cancelled', closed_at = ? "
            "WHERE recommendation_id = ?",
            [datetime.now(), recommendation_id],
        )

        self._log_approval(recommendation_id, actor, "rejected", reason)

        return ApprovalResult(
            recommendation_id=recommendation_id,
            action="cancelled",
            actor=actor,
            reason=reason,
            message=f"Recommendation {recommendation_id} rejected: {reason}" if reason
            else f"Recommendation {recommendation_id} rejected",
        )

    def _execute_order(self, rec: dict) -> str | None:
        """
        Convert an approved recommendation to a broker order.

        Returns the order_id if successful, None otherwise.
        """
        try:
            from hrp.execution.orders import Order, OrderSide, OrderType

            symbol = rec["symbol"]
            action = rec.get("action", "BUY")
            entry_price = Decimal(str(rec.get("entry_price", 0)))
            position_pct = float(rec.get("position_pct", 0.05))

            if entry_price <= 0:
                logger.warning(f"Invalid entry price for {rec['recommendation_id']}")
                return None

            # Compute quantity from portfolio value and position sizing
            portfolio_value = self.api.get_portfolio_value()
            if portfolio_value <= 0:
                # Fallback to env var
                import os
                portfolio_value = Decimal(os.getenv("HRP_PORTFOLIO_VALUE", "100000"))

            target_value = portfolio_value * Decimal(str(position_pct))
            quantity = int(target_value / entry_price)

            if quantity <= 0:
                logger.warning(
                    f"Computed quantity is 0 for {symbol} "
                    f"(portfolio={portfolio_value}, pct={position_pct}, price={entry_price})"
                )
                return None

            side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                hypothesis_id=rec.get("hypothesis_id"),
            )

            # Record the order as a trade (actual broker submission handled by LiveTradingAgent)
            trade_id = self.api.record_trade(
                order=order,
                filled_price=entry_price,
                filled_quantity=quantity,
                hypothesis_id=rec.get("hypothesis_id"),
            )

            logger.info(
                f"Order generated for {symbol}: {side.value} {quantity} shares "
                f"@ ~${entry_price} (trade_id={trade_id})"
            )
            return order.order_id

        except ImportError:
            logger.error("Execution module not available")
            return None
        except Exception as e:
            logger.error(f"Failed to execute order for {rec['recommendation_id']}: {e}")
            return None

    def _log_approval(
        self,
        recommendation_id: str,
        actor: str,
        decision: str,
        reason: str = "",
    ) -> None:
        """Log an approval/rejection event to lineage."""
        from hrp.research.lineage import log_event

        event_type = (
            "recommendation_approved" if decision == "approved"
            else "recommendation_rejected"
        )
        log_event(
            event_type=event_type,
            actor=actor,
            details={
                "recommendation_id": recommendation_id,
                "decision": decision,
                "reason": reason,
            },
        )
