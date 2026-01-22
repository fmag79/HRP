# Manual Dashboard Verification - Concurrent Load Testing

## Objective
Verify that the Streamlit dashboard works correctly with the new connection pooling implementation under concurrent load from multiple browser tabs.

## Prerequisites
- Connection pooling implementation completed (Phase 1 & 2)
- All automated tests passing (subtask-3-1, 3-2, 3-3)
- Streamlit installed (`pip install streamlit>=1.30.0`)

## Test Procedure

### 1. Start the Dashboard
```bash
streamlit run hrp/dashboard/app.py
```

Expected: Dashboard starts on `http://localhost:8501` with no errors

### 2. Open Multiple Browser Tabs
Open 5-10 browser tabs/windows pointing to `http://localhost:8501`

This simulates concurrent users accessing the dashboard simultaneously.

### 3. Concurrent Navigation Test
In each tab, simultaneously perform these actions:
- Navigate between different pages (Home, Data, Research, etc.)
- Refresh pages at random intervals
- Trigger data queries (view hypotheses, experiments, lineage)
- Use any interactive widgets (filters, dropdowns, buttons)

Duration: 2-3 minutes of active concurrent navigation

### 4. Verification Checks

#### ✅ Database Connection Health
- [ ] No "database is locked" errors in any tab
- [ ] No "connection closed" errors in browser console
- [ ] No "unable to acquire connection" errors
- [ ] System Status shows "Database: Connected" consistently

#### ✅ Data Consistency
- [ ] All pages load data correctly in every tab
- [ ] Refresh operations work without errors
- [ ] No data corruption or missing data
- [ ] Queries complete successfully

#### ✅ Performance
- [ ] Pages load within reasonable time (<3 seconds)
- [ ] No indefinite hanging or timeouts
- [ ] Smooth navigation between pages
- [ ] Multiple tabs don't cause degradation

#### ✅ Connection Pool Behavior
Check terminal output for connection pool logs:
- [ ] Connections acquired and released properly
- [ ] Pool size stays within max_connections limit (5)
- [ ] No connection leaks (connections released after use)
- [ ] Debug logs show proper acquire/release cycle

#### ✅ Error Recovery
Test error conditions:
- [ ] Rapidly refresh multiple tabs simultaneously
- [ ] Open 10+ tabs concurrently and navigate
- [ ] Dashboard recovers gracefully from pool saturation
- [ ] No permanent errors or stuck states

### 5. Browser Console Check
Open browser developer tools (F12) in each tab:

**Console Tab:**
- [ ] No JavaScript errors related to data loading
- [ ] No 500 Internal Server Error responses
- [ ] No WebSocket connection failures

**Network Tab:**
- [ ] All API requests return 200 status codes
- [ ] No request timeouts or failures

### 6. Terminal Output Review
Review the Streamlit terminal output:

**Look for:**
- ✅ ConnectionPool debug logs showing acquire/release
- ✅ "Connection acquired" / "Connection released" messages
- ✅ Pool statistics (available connections, in-use connections)

**Should NOT see:**
- ❌ Tracebacks or Python exceptions
- ❌ "Pool exhausted" warnings (unless testing extreme load)
- ❌ Database lock errors
- ❌ Connection timeout errors

## Expected Results

### ✅ Success Criteria
1. **No database errors** in any browser tab
2. **Consistent data loading** across all tabs
3. **Proper connection pooling** visible in logs
4. **Graceful handling** of concurrent requests
5. **No connection leaks** (all connections released)
6. **Dashboard remains responsive** under load

### ❌ Failure Indicators
- Database locked errors
- Connection closed errors
- Hanging pages or timeouts
- Data inconsistencies between tabs
- Connection pool exhaustion errors
- Python tracebacks in terminal

## Connection Pool Implementation Details

The connection pool should:
- Maintain up to 5 concurrent connections (max_connections=5)
- Acquire connections from pool for each query
- Release connections back immediately after use
- Block and wait when pool is exhausted
- Validate connections before returning from pool
- Automatically replace invalid/closed connections

## Test Environment

- **Python Version:** 3.9+
- **DuckDB:** Connection pool with max_size=5
- **Streamlit:** Latest stable version
- **Database:** ~/hrp-data/hrp.duckdb

## Troubleshooting

### Issue: "Database is locked"
**Cause:** Connection not properly released or pool exhausted
**Solution:** Check connection pool logs, verify release() calls

### Issue: "No module named streamlit"
**Cause:** Streamlit not installed
**Solution:** `pip install streamlit>=1.30.0` or `pip install -e .`

### Issue: Dashboard doesn't start
**Cause:** Port 8501 already in use
**Solution:** Use `streamlit run hrp/dashboard/app.py --server.port=8502`

### Issue: Connection pool exhaustion warnings
**Cause:** Concurrent load exceeds pool size temporarily
**Solution:** This is expected under high load; verify connections are released

## Verification Sign-Off

**Tester Name:** _________________
**Date:** _________________
**Test Duration:** _________________
**Number of Tabs Tested:** _________________

**Result:** ✅ PASS / ❌ FAIL

**Notes:**
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

**Issues Found:**
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

## Related Subtasks

- ✅ subtask-3-1: Run existing test suite (53/53 tests pass)
- ✅ subtask-3-2: ThreadPoolExecutor concurrent access tests pass
- ✅ subtask-3-3: PlatformAPI integration tests pass (72/72 tests pass)
- 🔄 subtask-3-4: Manual dashboard verification (this document)

## References

- Implementation Plan: `.auto-claude/specs/007-duckdb-connection-pooling-concurrency/implementation_plan.json`
- Connection Pool Code: `hrp/data/db.py` (ConnectionPool class)
- Test Coverage: `tests/test_data/test_db.py` (TestThreadSafety class)
