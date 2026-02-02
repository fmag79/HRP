"""
CLI commands for HRP data ingestion job management.

Allows manual triggering of scheduled jobs for testing and debugging.
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from hrp.agents.jobs import FeatureComputationJob, PriceIngestionJob, UniverseUpdateJob
from hrp.agents.scheduler import IngestionScheduler
from hrp.api.platform import PlatformAPI
from hrp.data.ingestion.prices import TEST_SYMBOLS


def run_job_now(job_name: str, symbols: list[str] | None = None) -> dict[str, Any]:
    """
    Manually trigger a job to run immediately.

    Args:
        job_name: Name of the job to run ('prices', 'features', or 'universe')
        symbols: Optional list of symbols to process

    Returns:
        Dictionary with job execution results
    """
    logger.info(f"Manually triggering job: {job_name}")

    if job_name == "prices":
        # Run price ingestion job
        job = PriceIngestionJob(
            symbols=symbols or TEST_SYMBOLS,
            start=date.today() - timedelta(days=1),
            end=date.today(),
        )
        result = job.run()
        logger.info(f"Price ingestion result: {result}")
        return result

    elif job_name == "features":
        # Run feature computation job
        job = FeatureComputationJob(
            symbols=symbols,  # None = all symbols in database
            start=date.today() - timedelta(days=30),
            end=date.today(),
        )
        result = job.run()
        logger.info(f"Feature computation result: {result}")
        return result

    elif job_name == "universe":
        # Run universe update job
        job = UniverseUpdateJob(
            as_of_date=date.today(),
            actor="user:manual_cli",
        )
        result = job.run()
        logger.info(f"Universe update result: {result}")
        return result

    else:
        raise ValueError(
            f"Unknown job: {job_name}. Must be 'prices', 'features', or 'universe'"
        )


def list_scheduled_jobs() -> list[dict[str, Any]]:
    """
    List all scheduled jobs from the scheduler.

    Returns:
        List of job information dictionaries
    """
    scheduler = IngestionScheduler()

    # Setup jobs to query them (without starting scheduler)
    try:
        scheduler.setup_daily_ingestion()
    except Exception as e:
        logger.warning(f"Could not setup jobs: {e}")

    jobs = scheduler.list_jobs()

    if not jobs:
        logger.info("No scheduled jobs found")
        return []

    logger.info(f"Found {len(jobs)} scheduled jobs:")
    for job in jobs:
        logger.info(f"  - {job['id']}: next run at {job['next_run']}")

    return jobs


def get_job_status(job_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get job execution status from ingestion_log table.

    Args:
        job_id: Optional job ID to filter by (None = all jobs)
        limit: Maximum number of records to return

    Returns:
        List of job execution records
    """
    from hrp.api.platform import PlatformAPI
    api = PlatformAPI()
    logs = api.get_ingestion_logs(job_id=job_id, limit=limit)

    results = [
        (
            log["log_id"],
            log["source_id"],
            log["started_at"],
            log["completed_at"],
            log["status"],
            log["records_fetched"],
            log["records_inserted"],
            log["error_message"],
        )
        for log in logs
    ]

    if not results:
        logger.info(f"No job history found{f' for {job_id}' if job_id else ''}")
        return []

    records = []
    for row in results:
        record = {
            "log_id": row[0],
            "source_id": row[1],
            "started_at": row[2],
            "completed_at": row[3],
            "status": row[4],
            "records_fetched": row[5],
            "records_inserted": row[6],
            "error_message": row[7],
        }
        records.append(record)

        # Log summary
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "running": "🔄",
        }.get(row[4], "❓")

        logger.info(
            f"{status_emoji} {row[1]} - {row[4]} - {row[2]} - "
            f"fetched: {row[5]}, inserted: {row[6]}"
        )

    return records


def clear_job_history(
    job_id: str | None = None,
    before: datetime | None = None,
    status: str | None = None,
) -> int:
    """
    Clear job history from ingestion_log table.

    Args:
        job_id: Optional job ID to filter by (None = all jobs)
        before: Optional datetime to clear records before
        status: Optional status to filter by ('success', 'failed', etc.)

    Returns:
        Number of records deleted
    """
    api = PlatformAPI()
    rows_deleted = api.purge_ingestion_logs(
        job_id=job_id,
        before=before.isoformat() if before else None,
        status=status,
    )
    logger.info(f"Deleted {rows_deleted} records from ingestion_log")
    return rows_deleted


def main():
    """CLI entry point for job management."""
    parser = argparse.ArgumentParser(
        description="HRP Data Ingestion Job Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run price ingestion now
  python -m hrp.agents.cli run-now --job prices

  # Run feature computation now
  python -m hrp.agents.cli run-now --job features

  # Run universe update now
  python -m hrp.agents.cli run-now --job universe

  # Run with specific symbols
  python -m hrp.agents.cli run-now --job prices --symbols AAPL MSFT GOOGL

  # List scheduled jobs
  python -m hrp.agents.cli list-jobs

  # View job status history
  python -m hrp.agents.cli job-status

  # View status for specific job
  python -m hrp.agents.cli job-status --job-id price_ingestion

  # Clear old job history
  python -m hrp.agents.cli clear-history --before 2025-01-01
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # run-now command
    run_parser = subparsers.add_parser(
        "run-now",
        help="Manually trigger a job to run immediately",
    )
    run_parser.add_argument(
        "--job",
        type=str,
        required=True,
        choices=["prices", "features", "universe"],
        help="Job to run",
    )
    run_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to process (default: TEST_SYMBOLS for prices, all for features)",
    )

    # list-jobs command
    subparsers.add_parser(
        "list-jobs",
        help="List all scheduled jobs",
    )

    # job-status command
    status_parser = subparsers.add_parser(
        "job-status",
        help="Get job execution status from history",
    )
    status_parser.add_argument(
        "--job-id",
        type=str,
        help="Filter by specific job ID",
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of records to show (default: 10)",
    )

    # clear-history command
    clear_parser = subparsers.add_parser(
        "clear-history",
        help="Clear job history from ingestion_log",
    )
    clear_parser.add_argument(
        "--job-id",
        type=str,
        help="Clear history for specific job ID",
    )
    clear_parser.add_argument(
        "--before",
        type=str,
        help="Clear records before this date (ISO format: YYYY-MM-DD)",
    )
    clear_parser.add_argument(
        "--status",
        type=str,
        choices=["success", "failed", "running"],
        help="Clear records with specific status",
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion without prompting",
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "run-now":
        result = run_job_now(args.job, args.symbols)
        if result.get("status") == "failed":
            logger.error(f"Job failed: {result.get('error')}")
            sys.exit(1)
        else:
            logger.info(f"Job completed successfully: {result}")
            sys.exit(0)

    elif args.command == "list-jobs":
        jobs = list_scheduled_jobs()
        if jobs:
            print("\nScheduled Jobs:")
            print("-" * 80)
            for job in jobs:
                print(f"ID: {job['id']}")
                print(f"  Name: {job['name']}")
                print(f"  Next Run: {job['next_run']}")
                print(f"  Trigger: {job['trigger']}")
                print()
        else:
            print("No scheduled jobs found")

    elif args.command == "job-status":
        records = get_job_status(args.job_id, args.limit)
        if records:
            print(f"\nJob Status History (last {len(records)} records):")
            print("-" * 120)
            print(
                f"{'ID':<6} {'Job':<20} {'Status':<10} {'Started':<20} "
                f"{'Fetched':<10} {'Inserted':<10}"
            )
            print("-" * 120)
            for record in records:
                print(
                    f"{record['log_id']:<6} {record['source_id']:<20} "
                    f"{record['status']:<10} {str(record['started_at']):<20} "
                    f"{record['records_fetched'] or 0:<10} "
                    f"{record['records_inserted'] or 0:<10}"
                )
                if record['error_message']:
                    print(f"  Error: {record['error_message']}")
            print()
        else:
            print("No job history found")

    elif args.command == "clear-history":
        # Parse before date if provided
        before_dt = None
        if args.before:
            try:
                before_dt = datetime.fromisoformat(args.before)
            except ValueError:
                logger.error(f"Invalid date format: {args.before}. Use YYYY-MM-DD")
                sys.exit(1)

        # Confirm deletion
        if not args.confirm:
            conditions = []
            if args.job_id:
                conditions.append(f"job_id={args.job_id}")
            if args.before:
                conditions.append(f"before {args.before}")
            if args.status:
                conditions.append(f"status={args.status}")

            filter_desc = " AND ".join(conditions) if conditions else "ALL RECORDS"
            response = input(f"Delete {filter_desc} from ingestion_log? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled")
                sys.exit(0)

        count = clear_job_history(args.job_id, before_dt, args.status)
        print(f"Deleted {count} records")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
