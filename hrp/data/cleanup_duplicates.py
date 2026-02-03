"""
Cleanup duplicate hypotheses and lineage events.

This script:
1. Deletes duplicate draft hypotheses (keeping the original validated/backtested ones)
2. Deduplicates lineage events (keeping oldest event per group)

Run with: python -m hrp.data.cleanup_duplicates [--dry-run]
"""

from __future__ import annotations

import argparse
from datetime import datetime

from loguru import logger

from hrp.data.db import get_db


def cleanup_duplicate_hypotheses(db, dry_run: bool = True) -> int:
    """
    Delete duplicate draft hypotheses when a validated/backtested version exists.

    Returns:
        Number of hypotheses deleted
    """
    # Find drafts that duplicate existing hypotheses
    duplicates = db.fetchall(
        """
        SELECT h1.hypothesis_id, h1.title, h1.status, h1.created_at,
               h2.hypothesis_id as original_id, h2.status as original_status
        FROM hypotheses h1
        JOIN hypotheses h2 ON h1.title = h2.title
        WHERE h1.status = 'draft'
          AND h2.status NOT IN ('draft', 'deleted', 'rejected')
          AND h1.hypothesis_id != h2.hypothesis_id
          AND h1.created_at > h2.created_at
        ORDER BY h1.title
        """
    )

    if not duplicates:
        logger.info("No duplicate hypotheses found")
        return 0

    logger.info(f"Found {len(duplicates)} duplicate draft hypotheses:")
    for dup in duplicates:
        logger.info(
            f"  {dup[0]} ({dup[2]}) duplicates {dup[4]} ({dup[5]}): '{dup[1][:50]}...'"
        )

    if dry_run:
        logger.info("DRY RUN: Would delete these hypotheses")
        return 0

    # Delete the duplicates
    for dup in duplicates:
        db.execute(
            "UPDATE hypotheses SET status = 'deleted' WHERE hypothesis_id = ?",
            (dup[0],)
        )
        logger.info(f"Marked {dup[0]} as deleted")

    logger.info(f"Deleted {len(duplicates)} duplicate hypotheses")
    return len(duplicates)


def cleanup_duplicate_lineage(db, dry_run: bool = True) -> int:
    """
    Remove duplicate lineage events, keeping the oldest one per group.

    Groups are defined by: event_type, hypothesis_id, experiment_id, actor,
    and timestamp truncated to the minute.

    Returns:
        Number of lineage events deleted
    """
    # Find duplicates (keeping the one with lowest lineage_id)
    duplicate_count = db.fetchone(
        """
        SELECT COUNT(*) FROM lineage
        WHERE lineage_id NOT IN (
            SELECT MIN(lineage_id)
            FROM lineage
            GROUP BY event_type, hypothesis_id, experiment_id, actor,
                     DATE_TRUNC('minute', timestamp)
        )
        """
    )

    count = duplicate_count[0] if duplicate_count else 0

    if count == 0:
        logger.info("No duplicate lineage events found")
        return 0

    logger.info(f"Found {count} duplicate lineage events")

    if dry_run:
        logger.info("DRY RUN: Would delete these events")
        return 0

    # Delete duplicates (keeping lowest lineage_id per group)
    db.execute(
        """
        DELETE FROM lineage
        WHERE lineage_id NOT IN (
            SELECT MIN(lineage_id)
            FROM lineage
            GROUP BY event_type, hypothesis_id, experiment_id, actor,
                     DATE_TRUNC('minute', timestamp)
        )
        """
    )

    logger.info(f"Deleted {count} duplicate lineage events")
    return count


def run_cleanup(dry_run: bool = True) -> dict:
    """
    Run all cleanup operations.

    Args:
        dry_run: If True, only report what would be deleted

    Returns:
        Summary of cleanup results
    """
    # Use read-only mode for dry-run (allows running even when MCP server has lock)
    db = get_db(read_only=dry_run)

    logger.info(f"Starting cleanup {'(DRY RUN)' if dry_run else ''}")
    logger.info("=" * 50)

    results = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "hypotheses_deleted": 0,
        "lineage_events_deleted": 0,
    }

    # 1. Clean up duplicate hypotheses
    logger.info("\n1. Cleaning up duplicate hypotheses...")
    results["hypotheses_deleted"] = cleanup_duplicate_hypotheses(db, dry_run)

    # 2. Clean up duplicate lineage events
    logger.info("\n2. Cleaning up duplicate lineage events...")
    results["lineage_events_deleted"] = cleanup_duplicate_lineage(db, dry_run)

    logger.info("\n" + "=" * 50)
    logger.info("Cleanup complete!")
    logger.info(f"  Hypotheses deleted: {results['hypotheses_deleted']}")
    logger.info(f"  Lineage events deleted: {results['lineage_events_deleted']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup duplicate data in HRP database")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only show what would be deleted (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the cleanup (overrides --dry-run)"
    )

    args = parser.parse_args()
    dry_run = not args.execute

    run_cleanup(dry_run=dry_run)
