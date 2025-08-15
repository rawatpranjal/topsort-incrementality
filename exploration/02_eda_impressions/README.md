# EDA: Impression Delivery & The Fold

Goal
- Estimate P(impression | rank) and identify cutoff k (fold) where probability drops to ~0.

Inputs
- Tables: `Auctions`, `Impressions`.

Outputs
- `impression_prob_by_rank.png`
- `summary.txt`

Run
- Execute `analysis.sql` in Snowflake. Export rates and plot step-function; derive k.
