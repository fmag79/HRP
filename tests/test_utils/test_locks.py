"""Test job locking utilities."""


class TestJobLock:
    """Tests for JobLock context manager."""

    def test_acquires_lock(self, tmp_path):
        """Should acquire lock when not held."""
        from hrp.utils.locks import JobLock

        lock_file = tmp_path / "test.lock"

        with JobLock(str(lock_file)) as acquired:
            assert acquired is True
            assert lock_file.exists()

    def test_releases_lock_on_exit(self, tmp_path):
        """Should release lock when exiting context."""
        from hrp.utils.locks import JobLock

        lock_file = tmp_path / "test.lock"

        with JobLock(str(lock_file)):
            pass

        # Lock file should be gone or unlocked
        assert not lock_file.exists()

    def test_fails_when_lock_held(self, tmp_path):
        """Should fail to acquire when lock is held."""
        from hrp.utils.locks import JobLock

        lock_file = tmp_path / "test.lock"

        with JobLock(str(lock_file)):
            # Try to acquire same lock
            with JobLock(str(lock_file), timeout=0.1) as acquired:
                assert acquired is False

    def test_cleans_stale_locks(self, tmp_path):
        """Should clean up stale lock files."""
        from hrp.utils.locks import JobLock

        lock_file = tmp_path / "test.lock"

        # Create stale lock with invalid PID
        lock_file.write_text("999999999")  # Very unlikely to exist

        with JobLock(str(lock_file)) as acquired:
            assert acquired is True


class TestAcquireJobLock:
    """Tests for acquire_job_lock decorator."""

    def test_decorator_acquires_lock(self, tmp_path):
        """Decorator should acquire lock before execution."""
        from hrp.utils.locks import acquire_job_lock

        lock_file = tmp_path / "job.lock"
        executed = [False]

        @acquire_job_lock(str(lock_file))
        def my_job():
            executed[0] = True
            assert lock_file.exists()
            return "done"

        result = my_job()
        assert result == "done"
        assert executed[0] is True

    def test_decorator_skips_if_locked(self, tmp_path):
        """Decorator should skip if lock is held."""
        from hrp.utils.locks import JobLock, acquire_job_lock

        lock_file = tmp_path / "job.lock"
        executed = [False]

        @acquire_job_lock(str(lock_file), timeout=0.1)
        def my_job():
            executed[0] = True
            return "done"

        with JobLock(str(lock_file)):
            result = my_job()

        assert result is None
        assert executed[0] is False
