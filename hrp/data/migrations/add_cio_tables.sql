-- CIO Agent: Paper Portfolio and Decision Tracking Tables
-- Migration: 2026-01-27-cio-agent

-- Paper portfolio current allocations
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR NOT NULL UNIQUE,
    weight DECIMAL(5, 4),  -- Position weight (0-1)
    entry_price DECIMAL(10, 4),
    entry_date DATE,
    current_price DECIMAL(10, 4),
    unrealized_pnl DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Daily portfolio history
CREATE TABLE IF NOT EXISTS paper_portfolio_history (
    id INTEGER PRIMARY KEY,
    as_of_date DATE UNIQUE,
    nav DECIMAL(12, 2),  -- Net asset value
    cash DECIMAL(12, 2),
    total_positions INTEGER,
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 3),
    returns_daily DECIMAL(8, 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simulated trade log
CREATE TABLE IF NOT EXISTS paper_portfolio_trades (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR,
    action VARCHAR,  -- 'ADD', 'REMOVE', 'REBALANCE'
    weight_before DECIMAL(5, 4),
    weight_after DECIMAL(5, 4),
    price DECIMAL(10, 4),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- CIO decisions
CREATE TABLE IF NOT EXISTS cio_decisions (
    id INTEGER PRIMARY KEY,
    decision_id VARCHAR UNIQUE,
    report_date DATE,
    hypothesis_id VARCHAR,
    decision VARCHAR,  -- CONTINUE, CONDITIONAL, KILL, PIVOT
    score_total DECIMAL(4, 2),
    score_statistical DECIMAL(4, 2),
    score_risk DECIMAL(4, 2),
    score_economic DECIMAL(4, 2),
    score_cost DECIMAL(4, 2),
    rationale TEXT,
    approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Model cemetery for killed strategies
CREATE TABLE IF NOT EXISTS model_cemetery (
    id INTEGER PRIMARY KEY,
    hypothesis_id VARCHAR UNIQUE,
    killed_date DATE,
    reason TEXT,
    final_score DECIMAL(4, 2),
    experiment_count INTEGER,
    archived_by VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);

-- Adaptive threshold tracking
CREATE TABLE IF NOT EXISTS cio_threshold_history (
    id INTEGER PRIMARY KEY,
    threshold_name VARCHAR,
    old_value DECIMAL(10, 4),
    new_value DECIMAL(10, 4),
    reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
