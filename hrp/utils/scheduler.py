"""
Scheduler management utilities for HRP.

Provides functions to detect, control, and manage the HRP scheduler service
via macOS launchd. Used by the dashboard to handle database lock conflicts.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


# Default launchd plist location
LAUNCH_AGENT_PLIST = Path.home() / "Library/LaunchAgents/com.hrp.scheduler.plist"


@dataclass
class SchedulerStatus:
    """
    Status information about the HRP scheduler.

    Attributes:
        is_installed: Whether the scheduler launch agent is installed
        is_running: Whether the scheduler is currently running
        pid: Process ID if running, None otherwise
        command: Full command line if running
        error: Error message if status check failed
    """

    is_installed: bool
    is_running: bool
    pid: Optional[int] = None
    command: Optional[str] = None
    error: Optional[str] = None


def get_scheduler_status() -> SchedulerStatus:
    """
    Get the current status of the HRP scheduler.

    Checks if the scheduler launch agent is installed and whether it's
    currently running via launchctl.

    Returns:
        SchedulerStatus with current state information
    """
    # Check if plist file exists
    if not LAUNCH_AGENT_PLIST.exists():
        logger.debug(f"Scheduler plist not found: {LAUNCH_AGENT_PLIST}")
        return SchedulerStatus(is_installed=False, is_running=False)

    # Try to get scheduler status from launchctl
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return SchedulerStatus(
                is_installed=True,
                is_running=False,
                error=f"launchctl failed: {result.stderr}",
            )

        # Parse launchctl output
        for line in result.stdout.splitlines():
            if "com.hrp.scheduler" in line:
                # Format: "pid  service_label"
                parts = line.split()
                if parts and parts[0].isdigit():
                    pid = int(parts[0])

                    # Get process details
                    try:
                        ps_result = subprocess.run(
                            ["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        command = ps_result.stdout.strip() if ps_result.returncode == 0 else None

                        logger.debug(f"Scheduler running: PID={pid}, command={command}")
                        return SchedulerStatus(
                            is_installed=True,
                            is_running=True,
                            pid=pid,
                            command=command,
                        )
                    except subprocess.TimeoutExpired:
                        logger.warning(f"ps command timed out for PID {pid}")
                        return SchedulerStatus(
                            is_installed=True,
                            is_running=True,
                            pid=pid,
                            command="Unknown (ps timed out)",
                        )

        # Service is installed but not running
        logger.debug("Scheduler installed but not running")
        return SchedulerStatus(is_installed=True, is_running=False)

    except subprocess.TimeoutExpired:
        logger.warning("launchctl list command timed out")
        return SchedulerStatus(
            is_installed=True,
            is_running=False,
            error="launchctl command timed out",
        )
    except Exception as e:
        logger.error(f"Failed to check scheduler status: {e}")
        return SchedulerStatus(
            is_installed=True,
            is_running=False,
            error=str(e),
        )


def stop_scheduler() -> dict[str, any]:
    """
    Stop the HRP scheduler via launchctl.

    Unloads the launch agent, stopping all scheduled jobs and releasing
    the database lock.

    Returns:
        Dictionary with success status and message
    """
    if not LAUNCH_AGENT_PLIST.exists():
        return {
            "success": False,
            "message": f"Scheduler not installed: {LAUNCH_AGENT_PLIST}",
        }

    try:
        result = subprocess.run(
            ["launchctl", "unload", str(LAUNCH_AGENT_PLIST)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            logger.info("Scheduler stopped successfully")
            return {
                "success": True,
                "message": "Scheduler stopped. Database lock released.",
            }
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            logger.error(f"Failed to stop scheduler: {error_msg}")
            return {
                "success": False,
                "message": f"Failed to stop scheduler: {error_msg}",
            }

    except subprocess.TimeoutExpired:
        logger.error("launchctl unload command timed out")
        return {
            "success": False,
            "message": "Command timed out. Scheduler may still be running.",
        }
    except Exception as e:
        logger.error(f"Failed to stop scheduler: {e}")
        return {
            "success": False,
            "message": f"Error: {e}",
        }


def start_scheduler() -> dict[str, any]:
    """
    Start the HRP scheduler via launchctl.

    Loads the launch agent, starting all scheduled jobs.

    Returns:
        Dictionary with success status and message
    """
    if not LAUNCH_AGENT_PLIST.exists():
        return {
            "success": False,
            "message": f"Scheduler not installed: {LAUNCH_AGENT_PLIST}",
        }

    try:
        result = subprocess.run(
            ["launchctl", "load", str(LAUNCH_AGENT_PLIST)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            logger.info("Scheduler started successfully")
            return {
                "success": True,
                "message": "Scheduler started. Jobs will run on schedule.",
            }
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            logger.error(f"Failed to start scheduler: {error_msg}")
            return {
                "success": False,
                "message": f"Failed to start scheduler: {error_msg}",
            }

    except subprocess.TimeoutExpired:
        logger.error("launchctl load command timed out")
        return {
            "success": False,
            "message": "Command timed out. Scheduler may not have started.",
        }
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return {
            "success": False,
            "message": f"Error: {e}",
        }


def is_duckdb_lock_error(error: Exception) -> bool:
    """
    Check if an exception is a DuckDB lock error.

    DuckDB lock errors contain specific patterns in their error messages.

    Args:
        error: The exception to check

    Returns:
        True if this appears to be a DuckDB lock error
    """
    error_str = str(error).lower()
    lock_indicators = [
        "conflicting lock",
        "could not set lock",
        "database is locked",
        "lock file",
    ]
    return any(indicator in error_str for indicator in lock_indicators)


def get_lock_holder_pid(error: Exception) -> Optional[int]:
    """
    Extract the PID of the process holding the database lock from error message.

    DuckDB lock errors include the PID of the conflicting process.

    Args:
        error: The DuckDB lock exception

    Returns:
        Process ID if found in error message, None otherwise
    """
    error_str = str(error)

    # DuckDB format: "Conflicting lock is held by <path> (PID <pid>) by user <user>"
    try:
        if "PID " in error_str:
            # Find "PID " and extract the number after it
            pid_start = error_str.find("PID ") + 4
            pid_end = error_str.find(")", pid_start)
            pid_str = error_str[pid_start:pid_end].strip()
            return int(pid_str)
    except (ValueError, IndexError):
        pass

    return None
