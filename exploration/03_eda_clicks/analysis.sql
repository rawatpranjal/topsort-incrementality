-- Analysis: CTR by rank and time-to-click (Snowflake SQL)
-- Purpose: Compute CTR by rank and latency from impression to click.

-- Parameters (edit)
-- SET start_ts = '2025-03-14';
-- SET end_ts   = '2025-07-21';

WITH base AS (
  SELECT
    i.AUCTION_ID,
    i.PRODUCT_ID,
    i.USER_ID,
    i.CAMPAIGN_ID,
    i.VENDOR_ID,
    a.RANKING,
    i.OCCURRED_AT AS impression_time,
    c.INTERACTION_ID AS click_id,
    c.OCCURRED_AT AS click_time
  FROM Impressions i
  JOIN Auctions a
    ON i.AUCTION_ID = a.AUCTION_ID
   AND i.PRODUCT_ID = a.PRODUCT_ID
   AND i.USER_ID = a.USER_ID
  LEFT JOIN Clicks c
    ON i.AUCTION_ID = c.AUCTION_ID
   AND i.PRODUCT_ID = c.PRODUCT_ID
   AND i.USER_ID = c.USER_ID
   AND c.OCCURRED_AT >= i.OCCURRED_AT
  -- WHERE i.OCCURRED_AT BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
)
, ctr AS (
  SELECT
    RANKING,
    COUNT(*) AS impressions,
    COUNT_IF(click_id IS NOT NULL) AS clicks,
    CAST(COUNT_IF(click_id IS NOT NULL) AS DOUBLE) / NULLIF(COUNT(*), 0) AS ctr
  FROM base
  GROUP BY RANKING
)
SELECT * FROM ctr ORDER BY RANKING;

-- Time-to-click distribution (minutes)
SELECT
  DATEDIFF(minute, impression_time, click_time) AS minutes_to_click,
  COUNT(*) AS clicks
FROM base
WHERE click_id IS NOT NULL
GROUP BY 1
ORDER BY 1;
