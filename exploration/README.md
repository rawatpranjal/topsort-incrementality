# Exploration Overview

This directory contains self-contained analyses for ad incrementality research. Each subfolder is numbered and includes its own code, artifacts, and summary README.

Contents (Phase 1 - Foundational EDA):
- 01_eda_auctions: competition structure, bidders per auction, win rate by rank, anomalies.
- 02_eda_impressions: impression probability by rank; identify fold cutoff k.
- 03_eda_clicks: CTR by rank; time-to-click distribution.
- 04_eda_purchases: time-to-purchase from click; unadvertised product sales.
- 05_build_master_funnel: build master funnel table/view in Snowflake and validations.

Conventions
- Snowflake SQL dialect.
- Time windows are parameterized in queries via WHERE clauses; adjust as needed.
- Save plots and extracts into the same subfolder as artifacts.

How to use
- Open each subfolder, read its README.md, run the SQL against Snowflake, export artifacts, and write a brief summary.txt.
- Update this README with links and brief outcomes as analyses complete.
