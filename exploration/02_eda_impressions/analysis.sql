-- Analysis: Impression probability by rank (Snowflake SQL)
-- Purpose: Join auction participants to impressions and compute P(impression | rank)
-- Notes:
-- - Impressions include winners that were actually shown; non-winners usually have no impression.
-- - If some winners are paced out, they will also show no impression.

-- Parameters (edit as needed)
-- SET start_ts = '2025-03-14';
-- SET end_ts   = '2025-07-21';

WITH base AS (
  SELECT
    a.AUCTION_ID,
    a.PRODUCT_ID,
    a.USER_ID,
    a.CAMPAIGN_ID,
    a.VENDOR_ID,
    a.RANKING,
    a.IS_WINNER,
    i.INTERACTION_ID AS impression_id
  FROM Auctions a
  LEFT JOIN Impressions i
    ON a.AUCTION_ID = i.AUCTION_ID
   AND a.PRODUCT_ID = i.PRODUCT_ID
   AND a.USER_ID = i.USER_ID
  -- WHERE a.AUCTION_TIMESTAMP BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
)
SELECT
  RANKING,
  COUNT(*) AS participants,
  SUM(CASE WHEN impression_id IS NOT NULL THEN 1 ELSE 0 END) AS impressions,
  CAST(SUM(CASE WHEN impression_id IS NOT NULL THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0) AS impression_prob
FROM base
GROUP BY RANKING
ORDER BY RANKING;

-- Heuristic to propose cutoff k: the largest rank with impression_prob > 0.01 (adjust threshold)
-- SELECT MAX(RANKING) AS k
-- FROM (
--   SELECT RANKING,
--          CAST(SUM(CASE WHEN impression_id IS NOT NULL THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0) AS impression_prob
--   FROM base
--   GROUP BY RANKING
-- )
-- WHERE impression_prob > 0.01;
