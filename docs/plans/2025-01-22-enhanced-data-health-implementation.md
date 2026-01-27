# Enhanced Data Health Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add health score hero, 90-day trend chart, quality checks summary, and expandable anomaly drill-down to the existing data health dashboard page.

**Architecture:** Enhance `hrp/dashboard/pages/data_health.py` by integrating with the existing `QualityReportGenerator` from `hrp/data/quality/report.py`. New sections are inserted at the top of the page (health hero, trend chart) with existing sections preserved below.

**Tech Stack:** Streamlit, Pandas, existing `hrp.data.quality` module

---

## Task 1: Add Quality Report Data Functions

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py:1-14` (imports)
- Modify: `hrp/dashboard/pages/data_health.py:267` (add new functions before render)

**Step 1: Add imports for quality report**

Add after line 12 (`from hrp.data.db import get_db`):

```python
from hrp.data.quality.report import QualityReportGenerator, QualityReport
from hrp.data.quality.checks import IssueSeverity
```

**Step 2: Add get_quality_report function**

Add after `get_ingestion_summary()` function (after line 266):

```python
@st.cache_data(ttl=300)
def get_quality_report(as_of_date: date) -> dict[str, Any]:
    """Generate quality report and return as dict for caching."""
    generator = QualityReportGenerator()
    report = generator.generate_report(as_of_date)
    return report.to_dict()


@st.cache_data(ttl=600)
def get_health_trend(days: int = 90) -> pd.DataFrame:
    """Get historical health scores for trend chart."""
    generator = QualityReportGenerator()
    trend_data = generator.get_health_trend(days=days)
    if not trend_data:
        return pd.DataFrame(columns=["date", "health_score", "critical_issues"])
    return pd.DataFrame(trend_data)
```

**Step 3: Add date import**

Update the datetime import on line 7 to include `date`:

```python
from datetime import date, datetime, timedelta
```

**Step 4: Run to verify no syntax errors**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add quality report data functions for health page"
```

---

## Task 2: Add Health Score Color Helper

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (add helper function)

**Step 1: Add color helper function**

Add after the `get_health_trend()` function:

```python
def get_health_color(score: float) -> str:
    """Return color name based on health score."""
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def get_health_emoji(score: float) -> str:
    """Return emoji based on health score."""
    if score >= 80:
        return "✅"
    elif score >= 50:
        return "⚠️"
    return "🔴"
```

**Step 2: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add health score color helpers"
```

---

## Task 3: Implement Health Score Hero Section

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (add render_health_hero function and call it)

**Step 1: Add render_health_hero function**

Add after the helper functions:

```python
def render_health_hero() -> dict[str, Any] | None:
    """Render the health score hero section. Returns report dict if available."""
    today = date.today()

    col_hero, col_button = st.columns([4, 1])

    with col_button:
        if st.button("Run Check Now", type="secondary"):
            st.cache_data.clear()
            st.rerun()

    try:
        report = get_quality_report(today)
    except Exception as e:
        st.warning(f"Could not generate quality report: {e}")
        return None

    health_score = report.get("health_score", 0)
    color = get_health_color(health_score)
    emoji = get_health_emoji(health_score)

    with col_hero:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;">
                <h1 style="margin: 0; font-size: 3rem;">{emoji} {health_score:.0f}/100</h1>
                <p style="margin: 0.5rem 0 0 0; color: gray;">Data Health Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Show last check time
    generated_at = report.get("generated_at", "")
    if generated_at:
        st.caption(f"Last checked: {generated_at[:19]}")

    # Show critical/warning counts
    critical = report.get("critical_issues", 0)
    warnings = report.get("warning_issues", 0)

    if critical > 0:
        st.error(f"{critical} critical issues require attention")
    elif warnings > 0:
        st.warning(f"{warnings} warnings detected")
    else:
        st.success("All quality checks passed")

    return report
```

**Step 2: Update render() to call health hero first**

Replace the beginning of the `render()` function. Find:

```python
def render() -> None:
    """Render the Data Health page."""
    st.title("Data Health")
    st.markdown("Monitor data completeness, quality, and ingestion status.")

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    st.subheader("Overview")
```

Replace with:

```python
def render() -> None:
    """Render the Data Health page."""
    st.title("Data Health")
    st.markdown("Monitor data completeness, quality, and ingestion status.")

    # -------------------------------------------------------------------------
    # Health Score Hero
    # -------------------------------------------------------------------------
    quality_report = render_health_hero()

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
    st.subheader("Overview")
```

**Step 3: Verify syntax and imports**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add health score hero section"
```

---

## Task 4: Implement Historical Trend Chart

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (add render_trend_chart and call it)

**Step 1: Add render_trend_chart function**

Add after `render_health_hero()`:

```python
def render_trend_chart() -> None:
    """Render the 90-day health score trend chart."""
    st.subheader("Health Score Trend (90 days)")

    trend_df = get_health_trend(days=90)

    if trend_df.empty:
        st.info("No historical data available. Quality reports will be stored after each check.")
        return

    # Ensure date column is datetime for proper plotting
    if "date" in trend_df.columns:
        trend_df["date"] = pd.to_datetime(trend_df["date"])
        trend_df = trend_df.set_index("date")

    # Plot health score trend
    st.line_chart(
        trend_df[["health_score"]],
        use_container_width=True,
        height=200,
    )

    # Show summary stats
    if len(trend_df) > 1:
        avg_score = trend_df["health_score"].mean()
        min_score = trend_df["health_score"].min()
        max_score = trend_df["health_score"].max()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{avg_score:.0f}")
        with col2:
            st.metric("Min", f"{min_score:.0f}")
        with col3:
            st.metric("Max", f"{max_score:.0f}")
```

**Step 2: Call render_trend_chart in render()**

After the health hero section and before Overview, add the trend chart call. Find:

```python
    quality_report = render_health_hero()

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

Replace with:

```python
    quality_report = render_health_hero()

    st.divider()

    # -------------------------------------------------------------------------
    # Historical Trend
    # -------------------------------------------------------------------------
    render_trend_chart()

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

**Step 3: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add 90-day health trend chart"
```

---

## Task 5: Implement Quality Checks Summary Table

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (add render_checks_summary and call it)

**Step 1: Add render_checks_summary function**

Add after `render_trend_chart()`:

```python
def render_checks_summary(report: dict[str, Any] | None) -> str | None:
    """Render quality checks summary table. Returns selected check filter or None."""
    st.subheader("Quality Checks")

    if report is None:
        st.info("Run a quality check to see results.")
        return None

    results = report.get("results", [])
    if not results:
        st.info("No check results available.")
        return None

    # Build summary data
    check_data = []
    for result in results:
        check_name = result.get("check_name", "Unknown")
        passed = result.get("passed", False)
        critical = result.get("critical_count", 0)
        warnings = result.get("warning_count", 0)

        status = "✅ Pass" if passed else "❌ Fail"
        issues_str = f"{critical} critical, {warnings} warnings"

        check_data.append({
            "Check": check_name.replace("_", " ").title(),
            "Status": status,
            "Issues": issues_str,
            "check_key": check_name,  # For filtering
        })

    # Display as table
    display_df = pd.DataFrame(check_data)
    st.dataframe(
        display_df[["Check", "Status", "Issues"]],
        use_container_width=True,
        hide_index=True,
    )

    # Return None for now - filtering will use selectbox in anomalies section
    return None
```

**Step 2: Call render_checks_summary after trend chart**

Find:

```python
    render_trend_chart()

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

Replace with:

```python
    render_trend_chart()

    st.divider()

    # -------------------------------------------------------------------------
    # Quality Checks Summary
    # -------------------------------------------------------------------------
    render_checks_summary(quality_report)

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

**Step 3: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add quality checks summary table"
```

---

## Task 6: Implement Flagged Anomalies with Drill-Down

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (add render_flagged_anomalies and call it)

**Step 1: Add render_flagged_anomalies function**

Add after `render_checks_summary()`:

```python
def render_flagged_anomalies(report: dict[str, Any] | None) -> None:
    """Render flagged anomalies with expandable drill-down."""
    st.subheader("Flagged Anomalies")

    if report is None:
        st.info("Run a quality check to see flagged anomalies.")
        return

    # Collect all issues from all checks
    all_issues = []
    results = report.get("results", [])
    for result in results:
        check_name = result.get("check_name", "Unknown")
        for issue in result.get("issues", []):
            all_issues.append({
                "check_name": check_name,
                **issue
            })

    if not all_issues:
        st.success("No anomalies detected!")
        return

    # Filter controls
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        check_options = ["All"] + list(set(i["check_name"] for i in all_issues))
        check_filter = st.selectbox("Filter by Check", check_options, key="anomaly_check_filter")

    with col_filter2:
        severity_options = ["All", "critical", "warning"]
        severity_filter = st.selectbox("Filter by Severity", severity_options, key="anomaly_severity_filter")

    # Apply filters
    filtered_issues = all_issues
    if check_filter != "All":
        filtered_issues = [i for i in filtered_issues if i["check_name"] == check_filter]
    if severity_filter != "All":
        filtered_issues = [i for i in filtered_issues if i.get("severity") == severity_filter]

    # Sort: critical first, then by date
    def sort_key(issue):
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        return (severity_order.get(issue.get("severity", "info"), 3), issue.get("date", "") or "")

    filtered_issues.sort(key=sort_key)

    # Limit to 50
    filtered_issues = filtered_issues[:50]

    st.caption(f"Showing {len(filtered_issues)} of {len(all_issues)} issues")

    # Render expandable rows
    for issue in filtered_issues:
        severity = issue.get("severity", "info")
        symbol = issue.get("symbol", "N/A")
        issue_date = issue.get("date", "N/A")
        description = issue.get("description", "No description")
        check_name = issue.get("check_name", "Unknown")

        # Severity emoji
        if severity == "critical":
            emoji = "🔴"
        elif severity == "warning":
            emoji = "🟡"
        else:
            emoji = "🔵"

        # Create expander
        with st.expander(f"{emoji} {severity.upper()}  |  {symbol}  |  {issue_date}  |  {description[:50]}..."):
            st.markdown(f"**Check:** {check_name.replace('_', ' ').title()}")
            st.markdown(f"**Symbol:** {symbol}")
            st.markdown(f"**Date:** {issue_date}")
            st.markdown(f"**Description:** {description}")

            # Show details if available
            details = issue.get("details", {})
            if details:
                st.markdown("**Details:**")
                for key, value in details.items():
                    if isinstance(value, float):
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
```

**Step 2: Call render_flagged_anomalies after checks summary**

Find:

```python
    render_checks_summary(quality_report)

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

Replace with:

```python
    render_checks_summary(quality_report)

    st.divider()

    # -------------------------------------------------------------------------
    # Flagged Anomalies
    # -------------------------------------------------------------------------
    render_flagged_anomalies(quality_report)

    st.divider()

    # -------------------------------------------------------------------------
    # Overview Metrics
    # -------------------------------------------------------------------------
```

**Step 3: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): add flagged anomalies with expandable drill-down"
```

---

## Task 7: Remove Redundant Data Quality Section

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (remove old Data Quality section)

The existing "Data Quality" section (lines ~451-502) with "Price Anomalies" and "Date Sequence Gaps" is now redundant since we have the new Flagged Anomalies section with proper drill-down.

**Step 1: Remove the old Data Quality section**

Find and remove this entire block (approximately lines 451-502):

```python
    # -------------------------------------------------------------------------
    # Data Quality Metrics
    # -------------------------------------------------------------------------
    st.subheader("Data Quality")

    col_qual1, col_qual2 = st.columns(2)

    with col_qual1:
        st.markdown("**Price Anomalies**")
        anomalies = get_price_anomalies()
        ...

    with col_qual2:
        st.markdown("**Date Sequence Gaps**")
        gaps = get_missing_dates_summary()
        ...

    st.divider()
```

**Step 2: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "refactor(dashboard): remove redundant data quality section"
```

---

## Task 8: Store Quality Report for Historical Tracking

**Files:**
- Modify: `hrp/dashboard/pages/data_health.py` (store report after generation)

**Step 1: Update render_health_hero to store report**

Find in `render_health_hero()`:

```python
    try:
        report = get_quality_report(today)
    except Exception as e:
        st.warning(f"Could not generate quality report: {e}")
        return None
```

Replace with:

```python
    try:
        report = get_quality_report(today)
        # Store report for historical tracking (only if not cached)
        if st.session_state.get("last_report_date") != str(today):
            try:
                generator = QualityReportGenerator()
                # Regenerate to get actual QualityReport object for storage
                full_report = generator.generate_report(today)
                generator.store_report(full_report)
                st.session_state["last_report_date"] = str(today)
            except Exception:
                pass  # Storage failure is non-critical
    except Exception as e:
        st.warning(f"Could not generate quality report: {e}")
        return None
```

**Step 2: Verify syntax**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -c "from hrp.dashboard.pages import data_health; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/data_health.py
git commit -m "feat(dashboard): store quality reports for historical tracking"
```

---

## Task 9: Manual Testing

**Step 1: Start the dashboard**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -m streamlit run hrp/dashboard/app.py`

**Step 2: Verify each feature**

1. Navigate to "Data Health" page
2. Verify health score hero displays with color coding
3. Verify "Run Check Now" button clears cache and refreshes
4. Verify trend chart shows (or shows "No historical data" message)
5. Verify quality checks summary table shows all 5 checks
6. Verify flagged anomalies section with filters
7. Verify expandable rows show issue details
8. Verify existing sections (Overview, Ingestion Status, Symbol Coverage) still work

**Step 3: Stop the dashboard**

Press Ctrl+C to stop

---

## Task 10: Final Commit and Summary

**Step 1: Verify all tests still pass**

Run: `cd /Users/fer/Documents/GitHub/HRP/.worktrees/enhanced-data-health && /opt/homebrew/bin/python3.11 -m pytest tests/test_data/test_quality.py -v --tb=short`

Expected: All tests pass

**Step 2: Create summary commit if needed**

If any uncommitted changes remain:

```bash
git add -A
git commit -m "chore: finalize enhanced data health page"
```

**Step 3: Summary**

The enhanced data health page now includes:
- Health score hero with color-coded score (0-100)
- "Run Check Now" button for on-demand checks
- 90-day historical trend chart
- Quality checks summary table (5 checks)
- Flagged anomalies with filtering and expandable drill-down
- Automatic storage of reports for historical tracking
