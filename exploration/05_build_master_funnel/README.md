# Build: Master Funnel Table/View

Goal
- Create a single analytical view representing the user journey per auction participant with impression/click/purchase flags and attribution.

Inputs
- Tables: `Auctions`, `Impressions`, `Clicks`, `Purchases`.

Outputs
- `build_view.sql` (Snowflake)
- `schema.txt`
- `validation_queries.sql`
- `summary.txt`

Run
- Execute `build_view.sql` in Snowflake as a privileged role in the target schema (set names below).
