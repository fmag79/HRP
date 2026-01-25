"""Tests for hrp/utils/timing.py."""

import time
from unittest.mock import MagicMock, patch

import pytest

from hrp.utils.timing import (
    Timer,
    TimingMetrics,
    add_timing_to_result,
    timed_section,
)


class TestTimingMetrics:
    """Tests for TimingMetrics dataclass."""

    def test_log_default_level(self):
        """Verify default info logging."""
        metrics = TimingMetrics(name="test_section", elapsed_seconds=1.5)

        with patch("hrp.utils.timing.logger") as mock_logger:
            metrics.log()
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "test_section" in call_args
            assert "1.500s" in call_args

    def test_log_debug_level(self):
        """Verify debug logging."""
        metrics = TimingMetrics(name="test_section", elapsed_seconds=2.0)

        with patch("hrp.utils.timing.logger") as mock_logger:
            metrics.log(level="debug")
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0][0]
            assert "test_section" in call_args

    def test_log_with_sub_timings(self):
        """Verify sub-timing percentages are logged."""
        metrics = TimingMetrics(
            name="test_section",
            elapsed_seconds=10.0,
            sub_timings={"fetch": 3.0, "compute": 7.0},
        )

        with patch("hrp.utils.timing.logger") as mock_logger:
            metrics.log()
            # Should have 3 calls: main + 2 sub-timings
            assert mock_logger.info.call_count == 3

            # Check sub-timing percentages
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("fetch" in c and "30.0%" in c for c in calls)
            assert any("compute" in c and "70.0%" in c for c in calls)

    def test_log_with_zero_elapsed(self):
        """Verify sub-timing percentage is 0 when elapsed is 0."""
        metrics = TimingMetrics(
            name="test_section",
            elapsed_seconds=0.0,
            sub_timings={"fetch": 0.0},
        )

        with patch("hrp.utils.timing.logger") as mock_logger:
            metrics.log()
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("0.0%" in c for c in calls)

    def test_to_dict(self):
        """Verify dict structure."""
        metrics = TimingMetrics(name="test_section", elapsed_seconds=1.5)
        result = metrics.to_dict()

        assert result["name"] == "test_section"
        assert result["total_seconds"] == 1.5

    def test_to_dict_with_sub_timings(self):
        """Verify sub-timing prefixes in dict."""
        metrics = TimingMetrics(
            name="test_section",
            elapsed_seconds=5.0,
            sub_timings={"fetch": 2.0, "compute": 3.0},
        )
        result = metrics.to_dict()

        assert result["name"] == "test_section"
        assert result["total_seconds"] == 5.0
        assert result["sub_fetch"] == 2.0
        assert result["sub_compute"] == 3.0


class TestTimedSection:
    """Tests for timed_section context manager."""

    def test_measures_elapsed_time(self):
        """Verify elapsed_seconds is set correctly."""
        with timed_section("test") as metrics:
            time.sleep(0.05)

        assert metrics.elapsed_seconds >= 0.05
        assert metrics.elapsed_seconds < 0.2  # Should not be much more

    def test_captures_sub_timings(self):
        """Verify sub_timings can be added."""
        with timed_section("test") as metrics:
            metrics.sub_timings["step1"] = 0.1
            metrics.sub_timings["step2"] = 0.2

        assert metrics.sub_timings["step1"] == 0.1
        assert metrics.sub_timings["step2"] == 0.2

    def test_handles_exception(self):
        """Verify timing captured on error."""
        metrics = None
        with pytest.raises(ValueError):
            with timed_section("test") as metrics:
                time.sleep(0.02)
                raise ValueError("test error")

        # Timing should still be captured even on exception
        assert metrics.elapsed_seconds >= 0.02


class TestTimer:
    """Tests for Timer class."""

    def test_auto_start_true(self):
        """Verify timer starts automatically by default."""
        timer = Timer(auto_start=True)
        time.sleep(0.02)
        elapsed = timer.elapsed()

        assert elapsed >= 0.02
        assert timer._start_time is not None

    def test_auto_start_false(self):
        """Verify timer waits for start() when auto_start=False."""
        timer = Timer(auto_start=False)

        assert timer._start_time is None

        with pytest.raises(RuntimeError, match="Timer was never started"):
            timer.elapsed()

    def test_stop_returns_elapsed(self):
        """Verify stop() returns correct time."""
        timer = Timer()
        time.sleep(0.02)
        elapsed = timer.stop()

        assert elapsed >= 0.02
        assert timer._stop_time is not None

    def test_elapsed_without_stop(self):
        """Verify elapsed() works while running."""
        timer = Timer()
        time.sleep(0.02)
        elapsed1 = timer.elapsed()
        time.sleep(0.02)
        elapsed2 = timer.elapsed()

        assert elapsed2 > elapsed1
        assert timer._stop_time is None  # Timer still running

    def test_elapsed_after_stop(self):
        """Verify elapsed() returns stopped time after stop()."""
        timer = Timer()
        time.sleep(0.02)
        stopped = timer.stop()
        time.sleep(0.02)
        elapsed = timer.elapsed()

        # elapsed should equal stopped time (not increase)
        assert elapsed == stopped

    def test_reset_clears_state(self):
        """Verify reset() clears times."""
        timer = Timer()
        time.sleep(0.01)
        timer.stop()

        timer.reset()

        assert timer._start_time is None
        assert timer._stop_time is None

    def test_reset_returns_self(self):
        """Verify reset() returns self for chaining."""
        timer = Timer()
        result = timer.reset()

        assert result is timer

    def test_start_returns_self(self):
        """Verify start() returns self for chaining."""
        timer = Timer(auto_start=False)
        result = timer.start()

        assert result is timer

    def test_stop_without_start_raises(self):
        """Verify RuntimeError on stop() without start()."""
        timer = Timer(auto_start=False)

        with pytest.raises(RuntimeError, match="Timer was never started"):
            timer.stop()

    def test_elapsed_without_start_raises(self):
        """Verify RuntimeError on elapsed() without start()."""
        timer = Timer(auto_start=False)

        with pytest.raises(RuntimeError, match="Timer was never started"):
            timer.elapsed()

    def test_restart_after_stop(self):
        """Verify timer can be restarted after stopping."""
        timer = Timer()
        timer.stop()
        timer.start()  # Restart

        assert timer._stop_time is None  # Stop time cleared
        assert timer._start_time is not None


class TestAddTimingToResult:
    """Tests for add_timing_to_result function."""

    def test_adds_timing_key(self):
        """Verify timing key added to dict."""
        result = {"sharpe": 1.5, "return": 0.15}
        metrics = TimingMetrics(name="backtest", elapsed_seconds=10.0)

        updated = add_timing_to_result(result, metrics)

        assert "timing" in updated
        assert updated["timing"]["name"] == "backtest"
        assert updated["timing"]["total_seconds"] == 10.0

    def test_preserves_existing_keys(self):
        """Verify existing keys are kept."""
        result = {"sharpe": 1.5, "return": 0.15, "trades": 100}
        metrics = TimingMetrics(name="backtest", elapsed_seconds=5.0)

        updated = add_timing_to_result(result, metrics)

        assert updated["sharpe"] == 1.5
        assert updated["return"] == 0.15
        assert updated["trades"] == 100
        assert "timing" in updated

    def test_returns_same_dict(self):
        """Verify the same dict is returned (mutated in place)."""
        result = {"sharpe": 1.5}
        metrics = TimingMetrics(name="test", elapsed_seconds=1.0)

        updated = add_timing_to_result(result, metrics)

        assert updated is result
