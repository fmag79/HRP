#!/usr/bin/env python
"""
Run a single HRP job and exit.

Replaces the long-lived scheduler daemon with individual job invocations.
Each call opens DB, runs job, logs result, and exits — no persistent process.

Usage:
    python -m hrp.agents.run_job --job prices
    python -m hrp.agents.run_job --job features
    python -m hrp.agents.run_job --job universe
    python -m hrp.agents.run_job --job backup
    python -m hrp.agents.run_job --job fundamentals
    python -m hrp.agents.run_job --job signal-scan --ic-threshold 0.03
    python -m hrp.agents.run_job --job agent-pipeline
    python -m hrp.agents.run_job --job daily-report
    python -m hrp.agents.run_job --job weekly-report
    python -m hrp.agents.run_job --job quality-monitoring
    python -m hrp.agents.run_job --job quality-sentinel
    python -m hrp.agents.run_job --job cio-review
    python -m hrp.agents.run_job --job prices --dry-run
"""

import argparse
import sys
from datetime import date

from loguru import logger


# Configure logging to file + stderr
LOG_DIR = "~/hrp-data/logs"


def _setup_logging(job_name: str) -> None:
    """Configure loguru to write to job-specific log file."""
    import os
    from pathlib import Path

    log_dir = Path(os.path.expanduser(LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{job_name}.log"
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def run_prices(dry_run: bool = False) -> dict:
    """Run daily price ingestion."""
    from hrp.agents.jobs import PriceIngestionJob

    if dry_run:
        logger.info("[DRY RUN] Would run price ingestion")
        return {"status": "dry_run", "job": "prices"}

    job = PriceIngestionJob(symbols=None)
    return job.run()


def run_universe(dry_run: bool = False) -> dict:
    """Run daily universe update."""
    from hrp.agents.jobs import UniverseUpdateJob

    if dry_run:
        logger.info("[DRY RUN] Would run universe update")
        return {"status": "dry_run", "job": "universe"}

    job = UniverseUpdateJob()
    return job.run()


def run_features(dry_run: bool = False) -> dict:
    """Run daily feature computation."""
    from hrp.agents.jobs import FeatureComputationJob

    if dry_run:
        logger.info("[DRY RUN] Would run feature computation")
        return {"status": "dry_run", "job": "features"}

    job = FeatureComputationJob(symbols=None)
    return job.run()


def run_backup(dry_run: bool = False) -> dict:
    """Run weekly backup."""
    from hrp.data.backup import BackupJob

    if dry_run:
        logger.info("[DRY RUN] Would run backup")
        return {"status": "dry_run", "job": "backup"}

    job = BackupJob(include_mlflow=True, keep_days=5)
    return job.run()


def run_fundamentals(dry_run: bool = False, source: str = "simfin") -> dict:
    """Run weekly fundamentals ingestion."""
    from hrp.agents.jobs import FundamentalsIngestionJob

    if dry_run:
        logger.info("[DRY RUN] Would run fundamentals ingestion")
        return {"status": "dry_run", "job": "fundamentals"}

    job = FundamentalsIngestionJob(symbols=None, source=source)
    return job.run()


def run_sec_ingestion(dry_run: bool = False, lookback_days: int = 180) -> dict:
    """Run weekly SEC filings ingestion with sentiment analysis."""
    from hrp.data.ingestion.sec_ingestion import SECIngestionJob

    if dry_run:
        logger.info(f"[DRY RUN] Would run SEC filings ingestion ({lookback_days} days)")
        return {"status": "dry_run", "job": "sec-ingestion", "lookback_days": lookback_days}

    job = SECIngestionJob()
    job.run(lookback_days=lookback_days)
    return {"status": "success", "job": "sec-ingestion"}


def run_fundamentals_backfill(dry_run: bool = False, days: int = 365) -> dict:
    """Run comprehensive fundamentals backfill."""
    from hrp.agents.jobs import ComprehensiveFundamentalsBackfillJob

    if dry_run:
        logger.info(f"[DRY RUN] Would run comprehensive fundamentals backfill ({days} days)")
        return {"status": "dry_run", "job": "fundamentals-backfill", "days": days}

    job = ComprehensiveFundamentalsBackfillJob(symbols=None, lookback_days=days)
    return job.run()


def run_signal_scan(
    dry_run: bool = False, ic_threshold: float = 0.03
) -> dict:
    """Run weekly signal scan."""
    from hrp.agents.research_agents import SignalScientist

    if dry_run:
        logger.info("[DRY RUN] Would run signal scan")
        return {"status": "dry_run", "job": "signal-scan"}

    agent = SignalScientist(
        symbols=None,
        features=None,
        ic_threshold=ic_threshold,
        create_hypotheses=True,
    )
    return agent.run()


def run_agent_pipeline(dry_run: bool = False) -> dict:
    """
    Check lineage for unprocessed events and run downstream agents.

    Replaces LineageEventWatcher: queries lineage table for new events,
    dispatches to appropriate agents, then exits.
    """
    from hrp.agents.scheduler import LineageEventWatcher

    if dry_run:
        logger.info("[DRY RUN] Would poll lineage and run agent pipeline")
        return {"status": "dry_run", "job": "agent-pipeline"}

    # Create watcher with all triggers configured
    from hrp.agents.alpha_researcher import AlphaResearcher
    from hrp.agents.kill_gate_enforcer import KillGateEnforcer
    from hrp.agents.research_agents import (
        MLQualitySentinel,
        MLScientist,
        QuantDeveloper,
        ValidationAnalyst,
    )

    watcher = LineageEventWatcher(poll_interval_seconds=0)

    # Register all triggers (same as setup_research_agent_triggers)
    def on_hypothesis_created(event: dict) -> None:
        details = event.get("details", {})
        hypothesis_id = details.get("hypothesis_id") or event.get("hypothesis_id")
        if hypothesis_id:
            logger.info(f"Triggering Alpha Researcher for hypothesis {hypothesis_id}")
            researcher = AlphaResearcher()
            researcher.run()

    watcher.register_trigger(
        event_type="hypothesis_created",
        callback=on_hypothesis_created,
        actor_filter="agent:signal-scientist",
        name="signal_scientist_to_alpha_researcher",
    )

    def on_alpha_researcher_complete(event: dict) -> None:
        details = event.get("details", {})
        promoted_ids = details.get("reviewed_ids", [])
        for hypothesis_id in promoted_ids:
            logger.info(f"Triggering ML Scientist for hypothesis {hypothesis_id}")
            scientist = MLScientist(hypothesis_ids=[hypothesis_id])
            scientist.run()

    watcher.register_trigger(
        event_type="alpha_researcher_complete",
        callback=on_alpha_researcher_complete,
        actor_filter="agent:alpha-researcher",
        name="alpha_researcher_to_ml_scientist",
    )

    def on_experiment_completed(event: dict) -> None:
        details = event.get("details", {})
        experiment_id = details.get("experiment_id") or event.get("experiment_id")
        hypothesis_id = event.get("hypothesis_id")
        if experiment_id or hypothesis_id:
            logger.info(f"Triggering ML Quality Sentinel for experiment {experiment_id}")
            sentinel = MLQualitySentinel(audit_window_days=1, send_alerts=True)
            sentinel.run()

    watcher.register_trigger(
        event_type="experiment_completed",
        callback=on_experiment_completed,
        actor_filter="agent:ml-scientist",
        name="ml_scientist_to_quality_sentinel",
    )

    def on_quality_audit(event: dict) -> None:
        details = event.get("details", {})
        hypothesis_id = event.get("hypothesis_id")
        overall_passed = details.get("overall_passed", False)
        if overall_passed and hypothesis_id:
            logger.info(f"Triggering Quant Developer for hypothesis {hypothesis_id}")
            developer = QuantDeveloper(hypothesis_ids=[hypothesis_id])
            developer.run()

    watcher.register_trigger(
        event_type="ml_quality_sentinel_audit",
        callback=on_quality_audit,
        actor_filter="agent:ml-quality-sentinel",
        name="ml_quality_sentinel_to_quant_developer",
    )

    def on_quant_developer_complete(event: dict) -> None:
        hypothesis_id = event.get("hypothesis_id")
        if hypothesis_id:
            logger.info(f"Triggering Kill Gate Enforcer for hypothesis {hypothesis_id}")
            enforcer = KillGateEnforcer(hypothesis_ids=[hypothesis_id])
            enforcer.run()

    watcher.register_trigger(
        event_type="quant_developer_backtest_complete",
        callback=on_quant_developer_complete,
        actor_filter="agent:quant-developer",
        name="quant_developer_to_kill_gate_enforcer",
    )

    def on_kill_gate_enforcer_complete(event: dict) -> None:
        details = event.get("details", {})
        hypotheses_processed = details.get("hypotheses_processed", 0)
        hypotheses_killed = details.get("hypotheses_killed", 0)
        if hypotheses_processed > hypotheses_killed:
            logger.info(
                f"Triggering Validation Analyst for "
                f"{hypotheses_processed - hypotheses_killed} hypotheses"
            )
            analyst = ValidationAnalyst(hypothesis_ids=None, send_alerts=True)
            analyst.run()

    watcher.register_trigger(
        event_type="kill_gate_enforcer_complete",
        callback=on_kill_gate_enforcer_complete,
        actor_filter="agent:kill-gate-enforcer",
        name="kill_gate_enforcer_to_validation_analyst",
    )

    def on_validation_analyst_complete(event: dict) -> None:
        details = event.get("details", {})
        passed = details.get("hypotheses_passed", 0)
        if passed > 0:
            logger.info(f"Triggering Risk Manager for {passed} passed hypotheses")
            from hrp.agents.risk_manager import RiskManager
            risk_mgr = RiskManager(hypothesis_ids=None, send_alerts=True)
            risk_mgr.run()

    watcher.register_trigger(
        event_type="validation_analyst_complete",
        callback=on_validation_analyst_complete,
        actor_filter="agent:validation-analyst",
        name="validation_analyst_to_risk_manager",
    )

    def on_risk_manager_assessment(event: dict) -> None:
        details = event.get("details", {})
        passed = details.get("hypotheses_passed", 0)
        if passed > 0:
            logger.info(f"Triggering CIO Agent for {passed} risk-cleared hypotheses")
            from hrp.agents.cio import CIOAgent
            agent = CIOAgent(
                job_id=f"cio-triggered-{date.today().strftime('%Y%m%d')}",
                actor="agent:cio",
            )
            agent.execute()

    watcher.register_trigger(
        event_type="risk_manager_assessment",
        callback=on_risk_manager_assessment,
        actor_filter="agent:risk-manager",
        name="risk_manager_to_cio_agent",
    )

    # Single poll — process all pending events, then exit
    events_processed = watcher.poll()
    logger.info(f"Agent pipeline processed {events_processed} events")
    return {"status": "success", "job": "agent-pipeline", "events_processed": events_processed}


def run_daily_report(dry_run: bool = False) -> dict:
    """Generate daily research report."""
    from hrp.agents.report_generator import ReportGenerator

    if dry_run:
        logger.info("[DRY RUN] Would generate daily report")
        return {"status": "dry_run", "job": "daily-report"}

    generator = ReportGenerator(report_type="daily")
    return generator.run()


def run_weekly_report(dry_run: bool = False) -> dict:
    """Generate weekly research report."""
    from hrp.agents.report_generator import ReportGenerator

    if dry_run:
        logger.info("[DRY RUN] Would generate weekly report")
        return {"status": "dry_run", "job": "weekly-report"}

    generator = ReportGenerator(report_type="weekly")
    return generator.run()


def run_quality_monitoring(dry_run: bool = False) -> dict:
    """Run daily data quality monitoring."""
    from hrp.monitoring.quality_monitor import DataQualityMonitor, MonitoringThresholds

    if dry_run:
        logger.info("[DRY RUN] Would run quality monitoring")
        return {"status": "dry_run", "job": "quality-monitoring"}

    thresholds = MonitoringThresholds(health_score_warning=90.0)
    monitor = DataQualityMonitor(thresholds=thresholds, send_alerts=True)
    result = monitor.run_daily_check()

    logger.info(
        f"Quality monitor: score={result.health_score:.0f}/100, "
        f"trend={result.trend}, alerts={sum(result.alerts_sent.values())}"
    )
    return {"status": "success", "job": "quality-monitoring", "health_score": result.health_score}


def run_quality_sentinel(dry_run: bool = False) -> dict:
    """Run ML Quality Sentinel audit."""
    from hrp.agents.research_agents import MLQualitySentinel

    if dry_run:
        logger.info("[DRY RUN] Would run quality sentinel")
        return {"status": "dry_run", "job": "quality-sentinel"}

    sentinel = MLQualitySentinel(
        audit_window_days=7,
        include_monitoring=True,
        send_alerts=True,
    )
    return sentinel.run()


def run_cio_review(dry_run: bool = False) -> dict:
    """Run weekly CIO Agent review."""
    from hrp.agents.cio import CIOAgent

    if dry_run:
        logger.info("[DRY RUN] Would run CIO review")
        return {"status": "dry_run", "job": "cio-review"}

    agent = CIOAgent(
        job_id=f"cio-weekly-{date.today().strftime('%Y%m%d')}",
        actor="agent:cio",
    )
    result = agent.run()
    logger.info(
        f"CIO review complete: {result.get('decision_count', 0)} decisions, "
        f"report: {result.get('report_path', 'N/A')}"
    )
    return result


def run_predictions(dry_run: bool = False) -> dict:
    """Run daily prediction job for deployed strategies."""
    from hrp.agents.prediction_job import DailyPredictionJob

    if dry_run:
        logger.info("[DRY RUN] Would run daily prediction job")
        return {"status": "dry_run", "job": "predictions"}

    job = DailyPredictionJob()
    return job.run()


def run_live_trader(dry_run: bool = False, trading_dry_run: bool = True) -> dict:
    """Run live trading agent.

    Args:
        dry_run: If True, skip job entirely
        trading_dry_run: If True, generate orders but don't submit to broker
    """
    from hrp.agents.live_trader import LiveTradingAgent, TradingConfig

    if dry_run:
        logger.info("[DRY RUN] Would run live trading agent")
        return {"status": "dry_run", "job": "live-trader"}

    # Override dry_run from argument
    config = TradingConfig.from_env()
    config.dry_run = trading_dry_run

    agent = LiveTradingAgent(trading_config=config)
    return agent.run()


def run_drift_monitor(dry_run: bool = False, auto_rollback: bool = False) -> dict:
    """Run drift monitoring job.

    Args:
        dry_run: If True, skip job entirely
        auto_rollback: If True, automatically rollback drifting models
    """
    from hrp.agents.drift_monitor_job import DriftMonitorJob, DriftConfig

    if dry_run:
        logger.info("[DRY RUN] Would run drift monitor job")
        return {"status": "dry_run", "job": "drift-monitor"}

    config = DriftConfig(auto_rollback=auto_rollback)
    job = DriftMonitorJob(drift_config=config)
    return job.run()


# Job registry
JOBS: dict[str, callable] = {
    "prices": run_prices,
    "universe": run_universe,
    "features": run_features,
    "backup": run_backup,
    "fundamentals": run_fundamentals,
    "fundamentals-backfill": run_fundamentals_backfill,
    "sec-ingestion": run_sec_ingestion,
    "signal-scan": run_signal_scan,
    "agent-pipeline": run_agent_pipeline,
    "daily-report": run_daily_report,
    "weekly-report": run_weekly_report,
    "quality-monitoring": run_quality_monitoring,
    "quality-sentinel": run_quality_sentinel,
    "cio-review": run_cio_review,
    "predictions": run_predictions,
    "live-trader": run_live_trader,
    "drift-monitor": run_drift_monitor,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single HRP job and exit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available jobs:
  prices               Daily price ingestion
  universe             Daily universe update
  features             Daily feature computation
  backup               Daily database backup
  fundamentals         Weekly fundamentals ingestion
  fundamentals-backfill Comprehensive fundamentals backfill (--days N)
  sec-ingestion        Weekly SEC filings ingestion with sentiment analysis (--days N)
  signal-scan          Weekly signal discovery scan
  agent-pipeline       Check lineage events, run downstream agents
  daily-report         Generate daily research report
  weekly-report        Generate weekly research report
  quality-monitoring   Daily data quality checks
  quality-sentinel     ML Quality Sentinel audit
  cio-review           Weekly CIO Agent review
  predictions          Daily predictions for deployed strategies
  live-trader          Execute trades (DISABLED by default)
  drift-monitor        Monitor deployed models for drift
""",
    )

    parser.add_argument(
        "--job",
        required=True,
        choices=list(JOBS.keys()),
        help="Job to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would run without executing",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=0.03,
        help="IC threshold for signal-scan job (default: 0.03)",
    )
    parser.add_argument(
        "--fundamentals-source",
        type=str,
        default="simfin",
        choices=["simfin", "yfinance"],
        help="Data source for fundamentals job (default: simfin)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history for fundamentals-backfill job (default: 365)",
    )
    parser.add_argument(
        "--trading-dry-run",
        action="store_true",
        default=True,
        help="Generate orders without submitting to broker (default: True)",
    )
    parser.add_argument(
        "--execute-trades",
        action="store_true",
        help="Actually submit trades to broker (DANGEROUS - overrides --trading-dry-run)",
    )

    args = parser.parse_args()

    # Setup logging
    _setup_logging(args.job)
    logger.info(f"Starting job: {args.job}")

    try:
        # Build kwargs based on job type
        kwargs: dict = {"dry_run": args.dry_run}
        if args.job == "signal-scan":
            kwargs["ic_threshold"] = args.ic_threshold
        elif args.job == "fundamentals":
            kwargs["source"] = args.fundamentals_source
        elif args.job == "fundamentals-backfill":
            kwargs["days"] = args.days
        elif args.job == "sec-ingestion":
            kwargs["lookback_days"] = args.days
        elif args.job == "live-trader":
            # --execute-trades overrides --trading-dry-run
            kwargs["trading_dry_run"] = not args.execute_trades

        result = JOBS[args.job](**kwargs)
        logger.info(f"Job {args.job} completed: {result}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Job {args.job} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
