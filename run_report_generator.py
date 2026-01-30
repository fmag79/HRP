#!/usr/bin/env python3
"""Run Report Generator and display results."""

from hrp.agents import ReportGenerator
from datetime import date

def main():
    print("=" * 70)
    print("HRP Report Generator")
    print("=" * 70)
    print()

    # Generate daily report
    print("Generating daily research report...")
    daily = ReportGenerator(report_type="daily")
    result = daily.run()

    print()
    print("=" * 70)
    print("Report Generated Successfully!")
    print("=" * 70)
    print()
    print(f"Report Type: {result['report_type']}")
    print(f"Report Path: {result['report_path']}")
    print()
    print("Token Usage:")
    print(f"  Input tokens:     {result['token_usage']['input']}")
    print(f"  Output tokens:    {result['token_usage']['output']}")
    print(f"  Total tokens:     {result['token_usage']['total']}")
    print(f"  Estimated cost:   ${result['token_usage']['estimated_cost_usd']:.4f}")
    print()

    # Show report preview
    print("=" * 70)
    print("Report Preview (first 50 lines)")
    print("=" * 70)
    print()

    try:
        with open(result['report_path'], 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:50], 1):
                print(f"{i:3d}: {line.rstrip()}")
    except Exception as e:
        print(f"Could not preview report: {e}")

    print()
    print("=" * 70)
    print(f"Full report available at: {result['report_path']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
