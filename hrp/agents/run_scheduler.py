#!/usr/bin/env python
"""
Run the HRP data ingestion scheduler.

This script starts the background scheduler that runs:
- Daily price ingestion at 6:00 PM ET
- Daily universe update at 6:05 PM ET
- Daily feature computation at 6:10 PM ET
- Daily backup at 2:00 AM ET
- Weekly fundamentals ingestion (Saturday 10 AM ET)
- Weekly signal scan (Monday 7 PM ET) [optional]
- Daily ML Quality Sentinel audit (6 AM ET) [optional]
- Daily research report generation (7 AM ET) [optional]
- Weekly research report generation (Sunday 8 PM ET) [optional]
- Event-driven research agent pipeline [optional]

Usage:
    python -m hrp.agents.run_scheduler
    python -m hrp.agents.run_scheduler --with-research-triggers
    python -m hrp.agents.run_scheduler --with-signal-scan --with-research-triggers
    python -m hrp.agents.run_scheduler --with-daily-report --with-weekly-report
    python -m hrp.agents.run_scheduler --with-quality-sentinel --with-research-triggers
"""

import argparse
import signal
import sys
from loguru import logger

from hrp.agents.scheduler import IngestionScheduler


def main():
    parser = argparse.ArgumentParser(
        description="Run HRP Data Ingestion Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--price-time",
        type=str,
        default="18:00",
        help="Time to run price ingestion (HH:MM format, default: 18:00 = 6 PM ET)",
    )
    parser.add_argument(
        "--universe-time",
        type=str,
        default="18:05",
        help="Time to run universe update (HH:MM format, default: 18:05)",
    )
    parser.add_argument(
        "--feature-time",
        type=str,
        default="18:10",
        help="Time to run feature computation (HH:MM format, default: 18:10)",
    )
    parser.add_argument(
        "--backup-time",
        type=str,
        default="02:00",
        help="Time to run daily backup (HH:MM format, default: 02:00 = 2 AM ET)",
    )
    parser.add_argument(
        "--backup-keep-days",
        type=int,
        default=5,
        help="Days of backups to retain (default: 5)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable daily backup job",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to ingest (default: all universe symbols)",
    )
    parser.add_argument(
        "--fundamentals-time",
        type=str,
        default="10:00",
        help="Time to run fundamentals ingestion (HH:MM format, default: 10:00 = 10 AM ET)",
    )
    parser.add_argument(
        "--fundamentals-day",
        type=str,
        default="sat",
        help="Day of week to run fundamentals (mon-sun, default: sat)",
    )
    parser.add_argument(
        "--fundamentals-source",
        type=str,
        default="simfin",
        choices=["simfin", "yfinance"],
        help="Data source for fundamentals (default: simfin)",
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Disable weekly fundamentals ingestion job",
    )

    # Research agent options
    parser.add_argument(
        "--with-research-triggers",
        action="store_true",
        help="Enable event-driven research agent pipeline (Signal Scientist → Alpha Researcher → ML Scientist → ML Quality Sentinel → Report Generator)",
    )
    parser.add_argument(
        "--trigger-poll-interval",
        type=int,
        default=60,
        help="Lineage event poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--with-signal-scan",
        action="store_true",
        help="Enable weekly signal scan (Monday 7 PM ET by default)",
    )
    parser.add_argument(
        "--signal-scan-time",
        type=str,
        default="19:00",
        help="Time to run signal scan (HH:MM format, default: 19:00)",
    )
    parser.add_argument(
        "--signal-scan-day",
        type=str,
        default="mon",
        help="Day of week for signal scan (mon-sun, default: mon)",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=0.03,
        help="Minimum IC to create hypothesis (default: 0.03)",
    )
    parser.add_argument(
        "--with-quality-sentinel",
        action="store_true",
        help="Enable daily ML Quality Sentinel audit (6 AM ET by default)",
    )
    parser.add_argument(
        "--sentinel-time",
        type=str,
        default="06:00",
        help="Time to run ML Quality Sentinel (HH:MM format, default: 06:00)",
    )
    parser.add_argument(
        "--with-daily-report",
        action="store_true",
        help="Enable daily research report generation (7 AM ET by default)",
    )
    parser.add_argument(
        "--daily-report-time",
        type=str,
        default="07:00",
        help="Time to generate daily report (HH:MM format, default: 07:00)",
    )
    parser.add_argument(
        "--with-weekly-report",
        action="store_true",
        help="Enable weekly research report generation (Sunday 8 PM ET by default)",
    )
    parser.add_argument(
        "--weekly-report-time",
        type=str,
        default="20:00",
        help="Time to generate weekly report (HH:MM format, default: 20:00)",
    )

    args = parser.parse_args()

    # Create scheduler
    scheduler = IngestionScheduler()

    # Setup daily ingestion
    logger.info("Setting up daily data ingestion pipeline...")
    scheduler.setup_daily_ingestion(
        symbols=args.symbols,
        price_job_time=args.price_time,
        universe_job_time=args.universe_time,
        feature_job_time=args.feature_time,
    )

    # Setup daily backup
    if not args.no_backup:
        logger.info("Setting up daily backup job...")
        scheduler.setup_daily_backup(
            backup_time=args.backup_time,
            keep_days=args.backup_keep_days,
            include_mlflow=True,
        )

    # Setup weekly fundamentals ingestion
    if not args.no_fundamentals:
        logger.info("Setting up weekly fundamentals ingestion job...")
        scheduler.setup_weekly_fundamentals(
            fundamentals_time=args.fundamentals_time,
            day_of_week=args.fundamentals_day,
            source=args.fundamentals_source,
        )

    # Setup weekly signal scan
    if args.with_signal_scan:
        logger.info("Setting up weekly signal scan job...")
        scheduler.setup_weekly_signal_scan(
            scan_time=args.signal_scan_time,
            day_of_week=args.signal_scan_day,
            ic_threshold=args.ic_threshold,
            create_hypotheses=True,
        )

    # Setup daily ML Quality Sentinel
    if args.with_quality_sentinel:
        logger.info("Setting up daily ML Quality Sentinel audit...")
        scheduler.setup_quality_sentinel(
            audit_time=args.sentinel_time,
            audit_window_days=1,
            send_alerts=True,
        )

    # Setup daily report
    if args.with_daily_report:
        logger.info("Setting up daily research report generation...")
        scheduler.setup_daily_report(report_time=args.daily_report_time)

    # Setup weekly report
    if args.with_weekly_report:
        logger.info("Setting up weekly research report generation...")
        scheduler.setup_weekly_report(report_time=args.weekly_report_time)

    # Setup research agent triggers (event-driven pipeline)
    if args.with_research_triggers:
        logger.info("Setting up event-driven research agent pipeline...")
        scheduler.setup_research_agent_triggers(
            poll_interval_seconds=args.trigger_poll_interval,
        )

    # Start scheduler
    if args.with_research_triggers:
        logger.info("Starting scheduler with research agent triggers...")
        scheduler.start_with_triggers()
    else:
        logger.info("Starting scheduler...")
        scheduler.start()

    # List scheduled jobs
    jobs = scheduler.list_jobs()
    logger.info(f"Scheduler is running with {len(jobs)} jobs:")
    for job in jobs:
        logger.info(f"  - {job['id']}: next run at {job['next_run']}")

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping scheduler...")
        scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep running
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    try:
        signal.pause()  # Wait for signals
    except AttributeError:
        # signal.pause() not available on Windows
        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
