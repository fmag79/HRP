# Alpha Researcher Report - 2026-01-30

## Summary
- Hypotheses reviewed: 11
- Promoted to testing: 8
- Deferred: 3
- Token usage: 20152 tokens ($0.1809)

---

## HYP-2026-016

**Status:** ⏸️ Deferred
**Recommendation:** DEFER

### Economic Rationale
Volume ratio likely captures information flow asymmetries and market microstructure effects. When volume deviates significantly from historical norms, it often signals informed trading or institutional repositioning that hasn't been fully reflected in prices. This aligns with the 'volume leads price' principle from technical analysis and is supported by Kyle's (1985) model of informed trading. The positive correlation with forward returns suggests the signal may identify underreactions to information - when volume surges indicate new information arrival, prices may adjust gradually rather than immediately, creating predictable momentum. However, without knowing the specific construction of 'volume_ratio' (vs. moving average? vs. sector average?), the economic interpretation remains incomplete.

### Regime Analysis
Volume-based signals typically exhibit strong regime dependence. In bull markets, high volume often accompanies sustainable breakouts and trend continuation. In bear markets, volume spikes may signal capitulation or dead-cat bounces with different predictive power. During low volatility regimes, volume anomalies may be more meaningful as they stand out against quiet backgrounds. The 20-day forward horizon suggests this captures intermediate-term information incorporation rather than short-term noise, but effectiveness likely varies significantly across volatility regimes and market stress periods.

### Related Hypotheses
HYP-2026-012, HYP-2026-015, HYP-2026-014, HYP-2026-011

### Refined Thesis
Volume_ratio captures information asymmetries in market microstructure, where abnormal volume levels relative to historical baselines signal incomplete price discovery. The positive correlation with 20-day forward returns (IC=0.1473) suggests markets underreact to volume-signaled information flow, creating predictable momentum as prices gradually incorporate new information. This effect likely stems from institutional trading patterns and gradual information diffusion rather than immediate price efficiency.

### Refined Falsification Criteria
Reject if: (1) IC drops below 0.05 in out-of-sample testing across multiple time periods, (2) Signal shows no persistence across different volatility regimes (VIX quintiles), (3) Returns are not statistically significant (p>0.05) after transaction costs and realistic implementation constraints, (4) Signal correlation with existing volume factors exceeds 0.7, indicating redundancy rather than novel information, (5) Performance degrades significantly during the most recent 2-year period, suggesting regime change or alpha decay.

---

## HYP-2026-015

**Status:** ⏸️ Deferred
**Recommendation:** DEFER

### Economic Rationale
Volume ratio signals can exploit the market microstructure inefficiency where informed traders reveal their information through trading patterns. High volume ratios may indicate unusual institutional activity or informed flow, creating predictable price movements. This aligns with Kyle's (1985) model of informed trading and Easley & O'Hara's (2001) PIN model. The 10-day horizon is particularly relevant as it captures the time needed for information to be fully incorporated into prices, consistent with Hou et al. (2018) findings on information diffusion speed. However, without knowing the specific construction of 'volume_ratio' (relative to what baseline?), the economic mechanism remains somewhat unclear.

### Regime Analysis
Volume-based signals typically exhibit strong regime dependence. In high volatility/bear markets, volume ratios may become noisier as panic selling dominates informed flow. During low volatility periods, the signal-to-noise ratio should improve as informed trading becomes more prominent. The bi-weekly horizon may help smooth some intraday volatility noise but could miss rapid regime shifts. Expected performance: stronger in normal/bull markets, weaker during crisis periods when volume spikes are driven by liquidity needs rather than information.

### Related Hypotheses
HYP-2026-012, HYP-2026-016, HYP-2026-011, HYP-2026-014, HYP-2026-010

### Refined Thesis
The volume_ratio feature captures informed trading activity that creates predictable price movements over a 10-day horizon. When trading volume deviates significantly from its baseline (requiring clarification of ratio construction), it signals either informed flow or liquidity imbalances that take approximately two weeks to be fully arbitraged away. This exploits the documented market microstructure inefficiency where information diffusion is gradual rather than instantaneous.

### Refined Falsification Criteria
The hypothesis fails if: (1) IC drops below 0.05 in out-of-sample testing, (2) Signal performance is entirely concentrated in specific sectors/market cap segments suggesting spurious correlation, (3) Returns are not risk-adjusted (no significant alpha after controlling for common factors), (4) Performance disappears when accounting for transaction costs at realistic implementation scale, (5) No statistical significance when tested across multiple market regimes separately.

---

## HYP-2026-014

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis exploits the well-documented behavioral bias of herding and overreaction in technical indicator usage. The economic foundation rests on three pillars: (1) Mechanical oversold conditions (RSI < 30) often reflect temporary sentiment extremes rather than fundamental deterioration, creating mean-reversion opportunities; (2) The divergence between price-based indicators (RSI) and volume-based flow indicators (CMF) suggests institutional smart money may be accumulating while retail sentiment remains pessimistic; (3) Academic literature supports that combining momentum and flow indicators can identify temporary mispricings. The strategy essentially identifies when 'weak hands' have been shaken out (low RSI) but 'strong hands' aren't selling aggressively (CMF > -0.1), creating asymmetric risk-reward setups.

### Regime Analysis
This signal exhibits strong regime dependency. In trending bull markets, oversold bounces tend to be more reliable as the underlying trend supports mean reversion. During bear markets, 'oversold can become more oversold,' making the CMF filter crucial to avoid value traps. In sideways/range-bound markets, this setup should perform optimally as mean reversion forces are strongest. The signal may struggle during momentum crashes or capitulation events when correlations spike and traditional technical relationships break down. Consider adding volatility regime filters (VIX levels) or market breadth conditions to enhance regime awareness.

### Related Hypotheses
HYP-2026-012, HYP-2026-010

### Refined Thesis
Retail-dominated technical selling creates temporary mispricings when RSI-based algorithms trigger oversold signals, but institutional money flow indicators (CMF) reveal that sophisticated investors aren't participating in the selling pressure. This divergence between sentiment indicators and actual money flow creates high-probability mean reversion opportunities. The strategy exploits the predictable behavior of systematic technical trading systems while using volume-weighted flow analysis to confirm that the selling pressure is primarily retail-driven rather than informed institutional selling.

### Refined Falsification Criteria
The hypothesis fails if: (1) Long positions taken when RSI < 30 AND CMF > -0.1 do not outperform random entries over rolling 252-day periods with statistical significance (t-stat < 2.0); (2) The Sharpe ratio of the strategy falls below 0.5 over any 2-year testing period; (3) Maximum drawdown exceeds 15% during non-crisis periods (excluding 2008, 2020 COVID crash); (4) The strategy shows negative returns in more than 40% of calendar months; (5) Performance deteriorates significantly when controlling for standard risk factors (market beta, size, value, momentum).

---

## HYP-2026-013

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis exploits two well-documented market inefficiencies: (1) The 'flight-to-quality' phenomenon where investors systematically overreact during stress, creating temporary mispricings in high-quality stocks that experience short-term selling pressure, and (2) The low volatility anomaly, where boring, stable stocks deliver superior risk-adjusted returns. The economic mechanism is behavioral: loss aversion causes investors to dump even quality stocks during drawdowns, but fundamental analysis suggests these should recover faster due to superior balance sheet strength. Academic support exists from Fama-French quality factors, Asness et al. (2013) on quality investing, and extensive literature on momentum and mean reversion at different time horizons.

### Regime Analysis
This strategy should be strongly regime-dependent. In bull markets, it may underperform as investors chase growth over quality. During market stress and bear markets, it should outperform as the flight-to-quality effect dominates. The 200-day MA filter attempts to avoid value traps during secular bear markets, but this needs careful calibration. In sideways/choppy markets, the 20-day dip buying could generate alpha through mean reversion, but transaction costs become critical.

### Related Hypotheses
HYP-2026-008

### Refined Thesis
During market stress, investors exhibit loss aversion and flee to safety, creating temporary mispricings in quality stocks that experience short-term drawdowns. This strategy exploits the behavioral inefficiency by systematically purchasing fundamentally sound companies (high ROE, low leverage, strong cash flow) when they trade below their 20-day moving average but remain above their 200-day trend - capturing mean reversion in quality names while avoiding secular downtrends and value traps.

### Refined Falsification Criteria
The hypothesis fails if: (1) Quality stocks selected by the composite score underperform the market by more than 200bps annually over a 3-year period, (2) The 20-day dip-buying criterion generates negative alpha compared to buying the same quality universe without timing, (3) The 200-day trend filter fails to avoid major drawdowns, measured as maximum drawdown exceeding 25% during bear markets, or (4) Transaction costs and turnover exceed 150bps annually, negating theoretical alpha.

---

## HYP-2026-012

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis is grounded in solid microstructure theory. It exploits the temporary price impact from informed order flow that exceeds market makers' immediate absorption capacity. The economic mechanism is well-documented: when large informed orders hit the market, dealers must adjust their inventory and quotes, creating predictable price drift as they manage risk. The 'Kyle model' and subsequent research show that private information gets incorporated gradually due to strategic trading and liquidity constraints. Academic support includes Hasbrouck (1991) on permanent vs temporary price impact, and Chordia et al. (2002) on volume-return relationships. The use of money flow (price × volume) as a proxy for informed demand is theoretically sound - it captures both the magnitude and direction of trading pressure.

### Regime Analysis
This signal is likely regime-dependent with stronger performance during: (1) Normal/trending markets where information flow is steady and market makers can predict inventory needs, (2) Lower volatility periods when the signal-to-noise ratio of informed flow is higher. Performance may degrade during: (1) High volatility/crisis periods when all trading appears 'informed' and the 2-sigma threshold loses discriminatory power, (2) Very low volume periods where money flow calculations become unreliable, (3) Market reversals where momentum signals generally underperform. The 5-day vs 60-day comparison should help filter regime effects, but additional volatility or VIX-based filters might improve robustness.

### Related Hypotheses
HYP-2026-014, HYP-2026-010

### Refined Thesis
Informed trading creates temporary order flow imbalances that market makers cannot immediately absorb at constant prices due to inventory risk constraints and adverse selection costs. When money flow (price × volume) significantly exceeds normal levels (>2σ above 60-day average over 5 days), it signals sustained informed demand that requires gradual price adjustment. This creates a predictable short-term momentum effect as market makers slowly accommodate the flow through price discovery, generating alpha during the 1-5 day 'inventory absorption' window before full price adjustment occurs.

### Refined Falsification Criteria
The hypothesis fails if: (1) Securities with high money flow signals (>2σ threshold) do not exhibit statistically significant positive returns over 1-5 day holding periods after controlling for market beta and size factors, (2) The signal shows no incremental predictive power beyond simple price momentum or volume alone, (3) Performance completely breaks down during normal market conditions (excluding extreme crisis periods), (4) The signal works equally well with random price×volume combinations, suggesting it's capturing noise rather than informed flow, (5) Forward returns are not monotonically related to money flow signal strength across quantile rankings.

---

## HYP-2026-011

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
The volume_ratio signal likely captures information flow dynamics and liquidity provision patterns. High volume ratios may indicate: (1) Informed trading activity that precedes price moves, consistent with Kyle (1985) model where informed traders optimize order flow; (2) Institutional accumulation/distribution patterns that take time to be fully reflected in prices due to price impact constraints; (3) Breaking of liquidity equilibrium that signals regime changes in market microstructure. The IC of 0.1473 is economically significant and suggests the market takes time to fully incorporate volume-based information signals.

### Regime Analysis
Volume-based signals typically show regime-dependent performance: (1) Bull markets: High volume ratios may indicate continuation as momentum builds; (2) Bear markets: Volume spikes often precede further declines as forced selling creates cascades; (3) Low volatility regimes: Volume anomalies may be more persistent due to reduced noise; (4) Crisis periods: The signal may break down as correlations spike and volume becomes less informative about individual stock fundamentals. The 20-day horizon suggests this captures medium-term information incorporation rather than short-term noise.

### Related Hypotheses
HYP-2026-012, HYP-2026-015, HYP-2026-010

### Refined Thesis
The volume_ratio feature captures delayed price discovery from informed trading activity. When volume ratios are elevated, it indicates information flow that takes 20 days to be fully incorporated into prices due to market microstructure frictions and gradual institutional positioning. This creates a predictable return pattern exploiting the market's incomplete immediate reaction to volume-based information signals.

### Refined Falsification Criteria
The hypothesis is falsified if: (1) IC drops below 0.05 in out-of-sample testing across multiple market regimes; (2) Signal decay analysis shows no persistence beyond 5 days, suggesting noise rather than information; (3) Performance fails during high-volume market stress periods when the underlying mechanism should be strongest; (4) Cross-sectional analysis shows no variation in signal strength by market cap, liquidity, or analyst coverage (which would be expected if information-based).

---

## HYP-2026-010

**Status:** ⏸️ Deferred
**Recommendation:** DEFER

### Economic Rationale
The volume_ratio signal likely captures information asymmetry and informed trading patterns. When volume deviates from historical norms, it often signals the presence of informed traders or institutional flow that hasn't been fully incorporated into prices. The positive correlation with forward returns suggests volume_ratio identifies stocks experiencing accumulation by informed participants. This aligns with the academic literature on volume-price relationships (Karpoff 1987, Campbell et al. 1993) and the practitioner concept of 'volume leads price.' However, the mechanism is unclear without knowing how volume_ratio is constructed - is it relative to recent history, sector peers, or market cap adjusted?

### Regime Analysis
Volume-based signals typically exhibit regime dependency. In risk-on environments, high volume_ratio may indicate momentum and institutional buying, supporting positive returns. In risk-off periods, elevated volume could signal distressed selling or deleveraging, potentially reversing the relationship. The 10-day forward horizon suggests this captures medium-term price discovery rather than short-term noise. Performance likely deteriorates during volatility regimes when volume spikes are driven by fear rather than information.

### Related Hypotheses
HYP-2026-012, HYP-2026-016, HYP-2026-011, HYP-2026-014, HYP-2026-015

### Refined Thesis
The volume_ratio feature captures informed trading activity that leads price discovery over 10-day periods. When current volume patterns deviate significantly from historical norms (as measured by volume_ratio), it signals the presence of informed participants whose trading has not yet been fully incorporated into prices, creating a predictable return pattern with IC=0.1319.

### Refined Falsification Criteria
The hypothesis fails if: (1) IC drops below 0.05 in out-of-sample testing, (2) signal decay is evident within 6 months suggesting rapid alpha decay, (3) performance is driven entirely by a single sector or market cap segment, (4) returns are not risk-adjusted (fails to beat size/value/momentum benchmarks), or (5) transaction costs exceed expected returns when accounting for typical volume-based implementation challenges.

---

## HYP-2026-009

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
Post-earnings announcement drift (PEAD) is one of the most robust and well-documented market anomalies in academic literature. The economic rationale is grounded in three behavioral and structural factors: (1) Analysts are slow to fully incorporate earnings information into their models, creating gradual revision cycles that sustain price momentum; (2) Institutional investors face capacity constraints and compliance processes that delay their response to earnings signals, particularly for smaller positions; (3) Retail investors exhibit anchoring bias and limited attention, causing gradual rather than immediate price discovery. Bernard & Thomas (1989, 1990) provided seminal work, while more recent studies by Chordia et al. (2009) show the effect persists despite increased market efficiency. The anomaly exploits the market's systematic underreaction to earnings information.

### Regime Analysis
PEAD exhibits regime-dependent characteristics. In bull markets, positive earnings surprises tend to have stronger drift effects due to investor optimism amplifying good news. During bear markets or high volatility periods, the effect may weaken as investors become more skeptical and risk-averse. The signal typically performs best in moderate volatility environments where information processing is neither too rushed (high vol) nor too efficient (extremely low vol). Earnings season timing also matters - Q4 earnings often show stronger drift due to annual guidance updates and tax-loss selling reversals.

### Related Hypotheses
None identified

### Refined Thesis
Markets systematically underreact to earnings surprises due to behavioral biases (anchoring, limited attention) and structural frictions (analyst revision cycles, institutional flow constraints, gradual information diffusion). This creates predictable price continuation patterns lasting 1-3 months post-announcement. The effect is strongest for stocks with clear positive earnings surprises relative to consensus estimates, particularly when accompanied by forward guidance raises.

### Refined Falsification Criteria
The hypothesis fails if: (1) Risk-adjusted returns from positive earnings surprise deciles show no statistical outperformance over 1-3 month horizons; (2) The effect disappears when controlling for size, value, and momentum factors; (3) Transaction costs exceed gross alpha generation; (4) The signal shows no persistence across different market regimes or time periods; (5) The drift reverses within 6 months, indicating temporary mispricings rather than genuine alpha.

---

## HYP-2026-008

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis exploits the intersection of three well-documented market anomalies: the quality factor premium, momentum persistence, and flight-to-quality dynamics. High-ROE, low-debt firms exhibit more predictable earnings streams and lower tail risk, making them 'bond-like' during stress periods. The momentum overlay captures the slow information diffusion about fundamental quality - markets are initially slow to recognize sustained competitive advantages, but once quality trends establish, they persist due to: (1) institutional herding into 'safe' names during uncertainty, (2) retail investor preference for steady growers over volatile value traps, and (3) fundamental momentum in business quality that takes time to fully price in. The 120-day MA filter ensures entry only after trend establishment, avoiding value traps disguised as quality.

### Regime Analysis
This strategy should exhibit strong regime-dependence. During market stress/bear markets, the flight-to-quality component should provide downside protection and potentially positive returns as investors rotate into defensive quality names. In bull markets, it may underperform growth momentum strategies but should still capture quality premium. The biggest risk is in late-cycle environments where quality stocks become overvalued and momentum reverses sharply. Sideways/choppy markets could see whipsaws around the 120-day MA threshold. The strategy's defensive nature suggests lower volatility but potentially muted upside in risk-on environments.

### Related Hypotheses
HYP-2026-012, HYP-2026-013, HYP-2026-014

### Refined Thesis
High-quality companies (top decile ROE, bottom decile D/E) exhibit persistent competitive advantages that markets are slow to fully recognize and price. When these quality stocks establish positive technical momentum (trading above 120-day MA with 12-month positive returns), they capture three behavioral biases: (1) under-reaction to quality improvements, (2) institutional flight-to-quality during stress, and (3) momentum persistence in fundamental business trends. The strategy exploits the time lag between quality recognition and full valuation by entering established uptrends in fundamentally superior companies, avoiding both value traps and momentum crashes in low-quality names.

### Refined Falsification Criteria
The hypothesis fails if: (1) Quality momentum portfolio shows negative Sharpe ratio over 3+ year periods, (2) Strategy underperforms simple quality factor by >200bps annually after transaction costs, (3) No statistically significant outperformance during market stress periods (VIX >25), (4) High turnover (>100% annually) negates returns after realistic transaction costs, (5) Strategy shows no momentum persistence beyond 6-month holding periods, or (6) Quality screens fail to predict lower maximum drawdowns vs broad momentum strategies during bear markets.

---

## HYP-2026-007

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis exploits the well-documented behavioral bias of anchoring combined with institutional liquidity dynamics. Academic research supports several components: (1) Anchoring bias causes investors to overweight recent reference points (52-week highs, moving averages), making them perceive temporary pullbacks as opportunities rather than trend changes; (2) Bollinger Band breaches often represent temporary liquidity shocks or forced selling rather than fundamental deterioration; (3) The momentum literature (Jegadeesh & Titman, 1993) shows trend persistence, while mean reversion studies show short-term reversals within longer trends. The economic mechanism is that informed investors recognize oversold conditions in quality uptrends and provide liquidity to noise traders, causing price recovery toward the trend line.

### Regime Analysis
This signal is highly regime-dependent and likely performs best in bull markets with moderate volatility. In bear markets, Bollinger Band breaches may signal genuine trend breaks rather than temporary oversold conditions, leading to continued declines. During market stress periods, correlations spike and the signal may fail as systematic selling overwhelms mean reversion forces. In low-volatility regimes, the signal frequency may be too low to be actionable. The signal should perform best in trending but not parabolic markets where healthy pullbacks create entry opportunities.

### Related Hypotheses
HYP-2026-014, HYP-2026-005, HYP-2026-013, HYP-2026-008

### Refined Thesis
In established uptrends (50MA > 200MA), high-volatility stocks that breach lower Bollinger Bands represent temporary liquidity-driven oversold conditions rather than trend reversals. Behavioral anchoring causes investors to view these pullbacks as buying opportunities relative to recent highs, while institutional rebalancing and contrarian investors provide liquidity. The combination of trend persistence and short-term mean reversion creates predictable price recovery toward the moving average trend line, with effect size proportional to the degree of oversold condition.

### Refined Falsification Criteria
The hypothesis fails if: (1) Stocks selected by this criteria underperform their sector/market over 1-week, 2-week, and 1-month horizons in >60% of observations; (2) The signal shows no statistically significant alpha (t-stat < 2.0) when controlling for market beta, size, value, and momentum factors; (3) Performance degrades significantly (Sharpe ratio drops >30%) during market stress periods (VIX > 25); (4) The signal frequency is too low (<5% of universe monthly) to be practically implementable with reasonable diversification.

---

## HYP-2026-006

**Status:** ✅ Promoted to testing
**Recommendation:** PROCEED

### Economic Rationale
This hypothesis aligns with established volatility risk premium literature. Higher realized volatility often signals periods of market stress or uncertainty, which can create temporary mispricings as investors demand higher risk premiums. The positive correlation (IC=0.0504) suggests that high volatility periods may be followed by above-average returns, consistent with volatility being mean-reverting and risk premiums being time-varying. This could exploit behavioral biases where investors overreact to recent volatility, creating subsequent return opportunities. However, the mechanism differs from typical volatility risk premium strategies that short implied volatility - this appears to be a direct relationship between realized volatility and subsequent equity returns.

### Regime Analysis
This signal is likely highly regime-dependent. In crisis periods (2008, COVID), high volatility may precede further drawdowns rather than recoveries, making the relationship unstable. In normal markets, volatility spikes often mark temporary bottoms, supporting the positive relationship. Bull markets may show weaker signal strength as volatility compression reduces predictive power. The 60-day lookback may be too long to capture regime shifts effectively - consider shorter windows for regime adaptation.

### Related Hypotheses
HYP-2026-005

### Refined Thesis
60-day realized volatility positively predicts 5-day forward returns (IC=0.0504) by exploiting the volatility risk premium and mean-reversion in risk sentiment. High volatility periods create temporary undervaluation as investors demand excessive risk premiums, leading to subsequent return rebounds. This relationship may be stronger during normal market conditions and weaker during sustained crisis periods.

### Refined Falsification Criteria
The hypothesis fails if: (1) IC falls below 0.02 in out-of-sample testing, (2) positive relationship breaks down in >50% of rolling 12-month periods, (3) signal shows no statistical significance (p>0.05) in regime-adjusted analysis, (4) Sharpe ratio of volatility-based portfolio fails to exceed 0.3 after transaction costs, or (5) signal effectiveness deteriorates significantly during high-stress market periods (VIX >30).

---

