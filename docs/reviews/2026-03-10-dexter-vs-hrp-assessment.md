# Dexter vs HRP — Platform Assessment
**Date:** March 10, 2026  
**Author:** Bob (AI Assistant)  
**Purpose:** Evaluate Dexter (autonomous financial research agent) against HRP and identify gaps to improve daily utility

---

## References
- **Dexter repo:** https://github.com/virattt/dexter
- **HRP repo:** https://github.com/fmag-labs/HRP

---

## At a Glance

| Dimension | Dexter | HRP |
|---|---|---|
| Purpose | Ad-hoc financial Q&A agent | Systematic quant research + trading platform |
| Stack | TypeScript/Bun | Python/DuckDB/MLflow/Streamlit |
| AI role | LLM **IS** the agent (planner + executor) | LLM **assists** specific pipeline stages |
| Interface | Conversational (CLI + WhatsApp) | Streamlit dashboard (13 pages) |
| Depth | Shallow-but-fast | Deep-but-batch |
| Data | Fundamental (income stmt, balance sheet, CF) | 45 technical features + shallow fundamentals + intraday |
| Trading | None | Live (IBKR + Robinhood, VaR-aware sizing) |
| Scope | Any question, any ticker, right now | S&P 500 equities, systematic pipeline |
| News/Web | ✅ Exa / Tavily web search | ❌ SEC EDGAR NLP only |
| Eval Suite | ✅ LangSmith + LLM-as-judge | ❌ None |
| Telegram/WhatsApp | ✅ WhatsApp native | ❌ Email only |

---

## Where Dexter Wins

### 1. Conversational access
You can ask Dexter: *"Compare Apple's gross margin trend against Microsoft over 5 years"* and get a structured answer in 30 seconds. HRP requires you to know the API, open the dashboard, and navigate. Daily research velocity is higher with Dexter's pattern.

### 2. Live news + web context
Dexter uses Exa/Tavily for real-time web search — so an answer about NVDA includes earnings news, analyst upgrades, and macro context. HRP's sentiment layer is limited to SEC EDGAR filings (slow, backward-looking).

### 3. Evaluation rigor for agents
Dexter uses LangSmith + LLM-as-judge to score whether agent answers are correct. HRP has no equivalent — there's no way to know if the CIO Agent's quality is improving or degrading over time.

### 4. Instant deployment, zero friction
Dexter is `bun start`. HRP requires launchd, Streamlit, MLflow, DuckDB, and ~50 env vars. For daily use, Dexter is frictionless.

---

## Where HRP Wins (by a lot)

### 1. Execution pipeline — end to end
HRP goes from hypothesis → signal discovery → ML validation → live trading. Dexter has no execution capability whatsoever. HRP is a real trading system; Dexter is a research Q&A tool.

### 2. Statistical rigor
Walk-forward validation, Sharpe decay monitoring, overfitting detection, kill gates, VaR/CVaR — none of this exists in Dexter. These are institutional-grade guardrails.

### 3. ML depth
Ridge, Lasso, ElasticNet, RandomForest, LightGBM with MLflow tracking, feature SHAP importance, regime detection (HMM), parameter sweep with Sharpe decay — Dexter has no ML.

### 4. Reproducibility + audit trail
HRP's hypothesis lifecycle with full lineage tracking means every signal has a history. Dexter's scratchpad is ephemeral JSONL files per session — no long-term tracking.

### 5. Data breadth
45 technical features + intraday Polygon WebSocket + Brinson-Fachler attribution + Fama-French factor analysis. Dexter is fundamentals only (income statement, balance sheet, cash flow via Financial Datasets API).

### 6. Risk management
VaR/CVaR, independent Risk Manager agent with veto authority, pre-trade checks, circuit breakers, position drift monitoring — Dexter has none.

---

## What HRP Is Missing — Recommended Additions

### 🔴 HIGH IMPACT

#### 1. Natural language query interface
Right now, to ask "What's AAPL's momentum profile vs. sector?" you need to write code or navigate the dashboard. With a conversational layer on top of HRP's feature store and DuckDB, you could ask it in plain English and get a researched answer back. The MCP servers are already built — this is about wiring a Claude-based Q&A agent to the HRP data layer with Dexter's task planning pattern.

**Effort:** Medium | **Value:** High — daily research velocity multiplier

#### 2. Telegram integration for daily workflow
HRP delivers via email only. Should be getting:
- Morning briefing: top signals, open positions, risk flags → Telegram
- Ad-hoc queries via chat: "What's the portfolio VaR today?" → HRP responds
- Trade alerts: "Kill gate triggered on HYP-2026-014" → Telegram push

The Report Generator already builds summaries — it just needs a Telegram delivery layer.

**Effort:** Low | **Value:** High — eliminates email friction for daily ops

#### 3. Live news + macro context in agent prompts
HRP's AlphaResearcher and CIO agents have no access to current news. A signal that looks strong technically might have a negative earnings surprise or macro headwind that kills it. Adding Exa/Brave/Tavily web search into the Alpha Researcher and CIO agent prompts would dramatically improve signal quality.

**Effort:** Medium | **Value:** High — fundamental weakness in current agent reasoning

---

### 🟡 MEDIUM IMPACT

#### 4. LLM-as-judge eval layer for agents
No way to know if the CIO Agent's 4-dimension scoring is getting better or worse over time. A lightweight eval framework (similar to Dexter's LangSmith integration) that tracks: *"Did the CIO reject hypotheses that later would have been profitable? Did it approve ones that failed?"* — would let you tune the agents with data instead of intuition.

**Effort:** Medium | **Value:** Medium — long-term agent quality improvement

#### 5. On-demand research mode
HRP's pipeline is scheduled/batch. Adding a `hrp ask "What are the top 5 momentum signals right now?"` CLI command that routes to the signal store + a Claude reasoner would make HRP dramatically more useful outside market hours for exploration.

**Effort:** Medium | **Value:** Medium — complements existing pipeline

#### 6. Deeper fundamental data pipeline
HRP has pe_ratio/pb_ratio via SimFin but it's shallow. Building a proper fundamentals layer (revenue growth, margin trends, FCF yield) using Financial Datasets API would give the CIO Agent real fundamental context alongside technical signals — and prevent the platform from being purely momentum-driven.

**Effort:** High | **Value:** Medium — better signal quality, less momentum bias

---

### 🟢 NICE TO HAVE

#### 7. Historical eval on past CIO decisions
Use Dexter's LLM-as-judge pattern to retrospectively score HRP's past CIO decisions. *"Given what we knew on date X, was the rejection of HYP-2026-007 correct?"* Feeds directly into `KillGateCalibrator`.

#### 8. Mobile-accessible dashboard
HRP's Streamlit dashboard is localhost only. A Cloudflare tunnel + Tailscale would give mobile access. Lower priority given Telegram integration covers the daily-use case.

---

## Bottom Line

HRP and Dexter aren't competing — they're complementary. **Dexter is a research assistant. HRP is a trading system.** Dexter asks questions well; HRP executes answers well.

**The three things that would make HRP genuinely more useful every single day:**

1. **Telegram integration** — morning briefing + ad-hoc chat queries (fastest win, Report Generator already exists)
2. **Conversational query layer** — natural language on top of HRP's feature/signal store (MCP infra already there)
3. **Live news into agent prompts** — give AlphaResearcher and CIO real-world context, not just statistical signals

---

## Action Items

| Priority | Task | Effort | Owner |
|---|---|---|---|
| 🔴 1 | Telegram delivery for Report Generator | Low | Fernando / Bob |
| 🔴 2 | Natural language query CLI (`hrp ask`) | Medium | Fernando / Forge |
| 🔴 3 | Web search integration in AlphaResearcher + CIO | Medium | Fernando / Forge |
| 🟡 4 | LLM-as-judge eval framework for agents | Medium | Fernando / Forge |
| 🟡 5 | On-demand research mode | Medium | Fernando / Forge |
| 🟡 6 | Financial Datasets API fundamentals pipeline | High | Fernando / Forge |
| 🟡 7 | SEC Form 4 insider trading signals | Medium | Fernando / Forge |

---

## Feature: SEC Form 4 Insider Trading Signals
*Added 2026-03-10 — inspired by [@nolimitgains](https://x.com/nolimitgains/status/2031337305911287967)*

### What It Is
SEC Form 4 filings track when corporate insiders (CEOs, CFOs, board members, 10%+ shareholders) buy or sell their own stock. Insiders must file within 2 business days of a transaction. Historically one of the most reliable alpha sources — especially **cluster buying** (multiple insiders buying simultaneously).

### Why It Matters for HRP
- Insiders know their companies better than anyone — their buying/selling is signal, not noise
- Cluster buys (3+ insiders buying within 30 days) have historically outperformed SPY by 5–8% annually
- Complements HRP's purely technical/momentum signal stack with fundamental conviction signals
- Particularly powerful combined with existing momentum features — insider buy + strong momentum = high-conviction entry

### Data Sources
| Source | Cost | Quality | Notes |
|---|---|---|---|
| **SEC EDGAR** | Free | Raw XML | Already integrated in HRP for NLP sentiment — Form 4 endpoint at `/cgi-bin/browse-edgar?action=getcompany&type=4` |
| **OpenInsider.com** | Free | Clean | Scraped/structured, good for backfill |
| **Quiver Quantitative** | Paid | Excellent | Structured API, real-time, bulk endpoints |
| **sec-api.io** | Paid | Excellent | Real-time Form 4 webhooks, clean JSON |

**Recommendation:** Start with SEC EDGAR (free, already have integration pattern) → upgrade to Quiver Quant if signal proves valuable.

### New Features to Add
| Feature | Description | Window |
|---|---|---|
| `insider_buy_count` | # of distinct insiders buying | 30d, 60d, 90d |
| `insider_sell_count` | # of distinct insiders selling | 30d, 60d, 90d |
| `insider_net_value` | Total $ bought minus $ sold by insiders | 30d, 90d |
| `insider_cluster_signal` | 1 if 3+ insiders bought in 30d, 0 otherwise | 30d |
| `ceo_bought` | 1 if CEO specifically bought in last 60d | 60d |

### Implementation Plan
1. **New ingestion pipeline** — `hrp/data/ingestion/insider_transactions.py`
   - Poll EDGAR Form 4 RSS feed or Quiver API daily after market close
   - Parse: ticker, insider name, title, transaction type (P=purchase, S=sale), shares, price, date
   - Store in new DuckDB table: `insider_transactions`

2. **Feature computation** — `hrp/data/features/definitions.py`
   - Add 5 new insider features (see table above)
   - Roll up per ticker per date for feature store consistency

3. **Agent integration**
   - Alpha Researcher: include insider summary in hypothesis context (*"3 insiders bought $4.2M in last 30 days"*)
   - CIO Agent: add insider signal as 5th scoring dimension (current: 4 dimensions)

4. **Dashboard widget** — new panel on Hypotheses or Trading page showing insider activity for universe

### Backtest Hypothesis to Validate First
> **HYP: Insider cluster buying predicts 60-day outperformance**
> - Universe: S&P 500 stocks with 3+ insider purchases in a 30-day window
> - Signal: `insider_cluster_signal = 1`
> - Target: `returns_60d`
> - Falsification: Sharpe < SPY or p-value > 0.05 over 5-year walk-forward

Run this through the existing ML pipeline before committing to full integration.

---

*Assessment generated by Bob on 2026-03-10. Revisit quarterly or when adding new agents.*
