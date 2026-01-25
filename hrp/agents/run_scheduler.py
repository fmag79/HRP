#!/usr/bin/env python
"""
Run the HRP data ingestion scheduler.

This script starts the background scheduler that runs:
- Daily price ingestion at 6:00 PM ET
- Daily universe update at 6:05 PM ET
- Daily feature computation at 6:10 PM ET
- Daily backup at 2:00 AM ET
- Weekly fundamentals ingestion (Saturday 10 AM ET)

Usage:
    python -m hrp.agents.run_scheduler
    python -m hrp.agents.run_scheduler --price-time 18:00 --universe-time 18:05 --feature-time 18:10
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
        default=30,
        help="Days of backups to retain (default: 30)",
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

    # Start scheduler
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
