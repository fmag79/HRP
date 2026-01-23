# Enhanced Dashboard Data Health Page - Design

**Date:** 2025-01-22
**Status:** Approved

## Overview

Enhance the existing `hrp/dashboard/pages/data_health.py` page to provide comprehensive data health visibility including health scores, historical trends, and anomaly drill-down.

## User Story

As a researcher, I want to quickly see if my data is complete before running experiments.

## Acceptance Criteria

- [x] Shows ingestion status for each data source (last run, records, status) - existing
- [x] Data completeness percentage by symbol - existing
- [ ] Quality metrics: missing data count, anomaly count - **NEW**
- [ ] Flagged anomalies with drill-down (price spikes, gaps) - **ENHANCED**
- [ ] Historical data quality trend chart - **NEW**

## Page Structure

```
┌─────────────────────────────────────────────────────────┐
│  DATA HEALTH                                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │         HEALTH SCORE: 85/100                    │   │
│  │         ████████████████░░░░  (color-coded)     │   │
│  │         Last checked: 2 hours ago   [Run Now]   │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  [Symbols] [Records] [Date Range] [Freshness]          │
├─────────────────────────────────────────────────────────┤
│  Historical Trend (90 days)                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  📈 Line chart: health_score over time          │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Quality Checks Summary    │   Flagged Anomalies       │
│  (per-check pass/fail)     │   (expandable rows)       │
├─────────────────────────────────────────────────────────┤
│  Ingestion Status (existing)                           │
├─────────────────────────────────────────────────────────┤
│  Symbol Coverage (existing)                            │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Health Score Hero

- Large, centered health score (0-100)
- Progress bar with color coding:
  - Green: 80-100 (healthy)
  - Yellow: 50-79 (warning)
  - Red: 0-49 (critical)
- Shows timestamp of last quality check
- "Run Check Now" button to trigger fresh `QualityReport`

### 2. Historical Trend Chart (90 days)

- Line chart using `st.line_chart` or `st.area_chart`
- X-axis: Date
- Y-axis: Health score (0-100)
- Data source: `QualityReportGenerator.get_health_trend(days=90)`
- Graceful handling when no historical data exists

### 3. Quality Checks Summary Table

| Check | Status | Issues |
|-------|--------|--------|
| Price Anomaly | ✅/❌ | X critical, Y warnings |
| Completeness | ✅/❌ | X critical, Y warnings |
| Gap Detection | ✅/❌ | X critical, Y warnings |
| Stale Data | ✅/❌ | X critical, Y warnings |
| Volume Anomaly | ✅/❌ | X critical, Y warnings |

- Pass = no critical issues
- Clicking a row filters the Flagged Anomalies section

### 4. Flagged Anomalies with Drill-Down

- Filter dropdowns:
  - Check type: All / Price Anomaly / Completeness / Gap Detection / Stale Data / Volume
  - Severity: All / Critical / Warning
- Sorted by severity (critical first), then date (recent first)
- Limited to 50 issues for performance
- Each row shows: severity icon, symbol, date, description
- Expandable rows (`st.expander`) showing full details:
  - Check name
  - All fields from `QualityIssue.details`

## Data Sources

| Component | Source |
|-----------|--------|
| Health Score | `QualityReportGenerator.generate_report(as_of_date)` |
| Trend Chart | `QualityReportGenerator.get_health_trend(days=90)` |
| Check Results | `QualityReport.results` (list of `CheckResult`) |
| Issues | `CheckResult.issues` (list of `QualityIssue`) |

## Implementation Notes

### Caching Strategy

- Health score: Cache for 5 minutes (`@st.cache_data(ttl=300)`)
- Trend data: Cache for 10 minutes (`@st.cache_data(ttl=600)`)
- "Run Check Now" clears cache and generates fresh report

### New Functions Needed

```python
@st.cache_data(ttl=300)
def get_quality_report(as_of_date: date) -> dict:
    """Generate or retrieve cached quality report."""

@st.cache_data(ttl=600)
def get_health_trend(days: int = 90) -> pd.DataFrame:
    """Get historical health scores for trend chart."""

def render_health_hero(report: dict) -> None:
    """Render the health score hero section."""

def render_trend_chart(trend_data: pd.DataFrame) -> None:
    """Render the 90-day trend chart."""

def render_checks_summary(results: list) -> str | None:
    """Render checks table, return selected check filter."""

def render_flagged_anomalies(issues: list, check_filter: str | None) -> None:
    """Render expandable anomaly rows with filtering."""
```

### Color Coding

```python
def get_health_color(score: float) -> str:
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"
```

## Dependencies

No new dependencies. Uses:
- `streamlit` (existing)
- `pandas` (existing)
- `hrp.data.quality.report.QualityReportGenerator` (existing)
- `hrp.data.quality.checks.*` (existing)

## Testing

- Verify health score displays correctly with mock data
- Verify trend chart handles empty data gracefully
- Verify expandable rows show correct details
- Verify filters work correctly
- Manual testing with real database
