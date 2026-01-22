"""
Validation report generation.

Creates comprehensive validation reports in markdown format.
"""

from datetime import date
from typing import Any

from loguru import logger


def generate_validation_report(data: dict[str, Any]) -> str:
    """
    Generate comprehensive validation report in markdown.
    
    Args:
        data: Dictionary with validation data including:
            - hypothesis_id
            - metrics
            - significance_test
            - robustness
            - validation_passed
            - confidence_score
            
    Returns:
        Markdown-formatted validation report
    """
    hypothesis_id = data["hypothesis_id"]
    metrics = data["metrics"]
    sig_test = data.get("significance_test", {})
    robustness = data.get("robustness", {})
    passed = data.get("validation_passed", False)
    confidence = data.get("confidence_score", 0.0)
    
    status = "VALIDATED" if passed else "REJECTED"
    
    report = f"""# Validation Report: {hypothesis_id}

## Summary
- **Status:** {status}
- **Confidence Score:** {confidence:.2f}
- **Validated Date:** {date.today().isoformat()}
- **Validated By:** system (auto)

## Performance Metrics (Out-of-Sample)

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} | > 0.5 | {'✅' if metrics.get('sharpe', 0) > 0.5 else '❌'} |
| CAGR | {metrics.get('cagr', 0):.1%} | — | — |
| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} | < 25% | {'✅' if metrics.get('max_drawdown', 1) < 0.25 else '❌'} |
| Win Rate | {metrics.get('win_rate', 0):.1%} | > 40% | {'✅' if metrics.get('win_rate', 0) > 0.40 else '❌'} |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} | > 1.2 | {'✅' if metrics.get('profit_factor', 0) > 1.2 else '❌'} |
| Trade Count | {metrics.get('num_trades', 0)} | ≥ 100 | {'✅' if metrics.get('num_trades', 0) >= 100 else '❌'} |
| OOS Period | {metrics.get('oos_period_days', 0)} days | ≥ 730 days | {'✅' if metrics.get('oos_period_days', 0) >= 730 else '❌'} |

## Statistical Significance

"""
    
    if sig_test:
        report += f"""- Excess return vs benchmark: {sig_test.get('excess_return_annualized', 0):.1%} annualized
- t-statistic: {sig_test.get('t_statistic', 0):.2f}
- p-value: {sig_test.get('p_value', 1):.4f}
- **Significant at α=0.05:** {'✅' if sig_test.get('significant', False) else '❌'}

"""
    else:
        report += "_No significance test performed_\n\n"
    
    report += "## Robustness\n\n| Check | Result |\n|-------|--------|\n"
    
    for check_name, result in robustness.items():
        emoji = "✅" if result == "PASS" else "❌"
        report += f"| {check_name.replace('_', ' ').title()} | {emoji} {result} |\n"
    
    report += "\n## Recommendation\n\n"
    
    if passed:
        report += """Approved for paper trading. Monitor for 30 days before live deployment consideration.

**Next Steps:**
1. Deploy to paper trading account
2. Monitor live performance vs backtest
3. Review after 30 days minimum
4. Consider live deployment if performance holds
"""
    else:
        report += """Strategy did not meet validation criteria. Review failures and consider:

**Options:**
1. Revise strategy and re-test
2. Investigate failed criteria
3. Archive hypothesis as rejected
4. Consider alternative approaches
"""
    
    logger.info(f"Generated validation report for {hypothesis_id}: {status}")
    
    return report


class ValidationReport:
    """Class for managing validation reports."""
    
    def __init__(self, hypothesis_id: str):
        self.hypothesis_id = hypothesis_id
    
    def generate(self, data: dict[str, Any]) -> str:
        """Generate report for this hypothesis."""
        data["hypothesis_id"] = self.hypothesis_id
        return generate_validation_report(data)
    
    def save(self, filepath: str, data: dict[str, Any]):
        """Generate and save report to file."""
        report = self.generate(data)
        
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info(f"Saved validation report to {filepath}")
