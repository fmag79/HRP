# Plan: Add NLP & Bayesian Optimization to Roadmap

## What
Add two new capability sections to `docs/plans/Project-Status.md` under a new **Tier 2.5: Intelligence Extensions** section (between Tier 2 and Tier 3), covering:

1. **Fundamental NLP** — text-based features from earnings calls, SEC filings, news
2. **Bayesian Hyperparameter Optimization** — replace grid/random search with Optuna

## File to Modify
- `docs/plans/Project-Status.md` — insert new section after line 120 (end of Tier 2)

## Changes

Insert a new section `## Tier 2.5: Intelligence Extensions (Not Started)` with two subsections:

### Bayesian Optimization
- Status: Optuna already in dependencies, unused
- Scope: Replace grid/random search in `cross_validated_optimize()` with Optuna TPE sampler
- Files: `hrp/ml/optimization.py`, `hrp/ml/training.py`
- Benefit: Better hyperparameter search with fewer trials, respects existing trial counter limits

### Fundamental NLP
- Phase 1: SEC EDGAR ingestion (10-Q/10-K text) → new data source + ingestion job
- Phase 2: Earnings sentiment features (FinBERT or Claude API) → 6-8 new features
- Phase 3: News sentiment aggregation → rolling sentiment signals
- New files: `hrp/data/sources/sec_edgar_source.py`, `hrp/data/ingestion/nlp_text.py`, `hrp/data/features/nlp_features.py`
- Schema: new `raw_text_data` table, NLP features stored in existing `features` table
- Integration: zero changes to backtest engine, ML pipeline, or risk system — uses existing feature store pattern

Also update the Quick Status table to show the new tier.

## Verification
- Read the modified file and confirm formatting is consistent with existing tiers
- Confirm no existing content was altered
