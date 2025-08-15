-- Build MASTER FUNNEL VIEW (Snowflake SQL)
-- Purpose: Construct a performant analytical view/table for downstream causal analyses.
-- Notes:
-- - Adjust database/schema names.
-- - Attribution window: 60 minutes after click (can be parameterized).

-- USE ROLE ANALYST;  -- adjust
-- USE DATABASE ANALYSIS;  -- adjust
-- USE SCHEMA ANALYSIS;    -- adjust

CREATE OR REPLACE VIEW ANALYSIS.USER_FUNNEL_MASTER AS
WITH base AS (
  SELECT
    a.AUCTION_ID,
    a.USER_ID,
    a.PRODUCT_ID,
    a.CAMPAIGN_ID,
    a.VENDOR_ID,
    a.RANKING,
    a.IS_WINNER,
    a.AUCTION_TIMESTAMP,
    i.INTERACTION_ID AS IMPRESSION_ID,
    i.OCCURRED_AT     AS IMPRESSION_TIME,
    c.INTERACTION_ID  AS CLICK_ID,
    c.OCCURRED_AT     AS CLICK_TIME
  FROM Auctions a
  LEFT JOIN Impressions i
    ON a.AUCTION_ID = i.AUCTION_ID
   AND a.PRODUCT_ID = i.PRODUCT_ID
   AND a.USER_ID = i.USER_ID
  LEFT JOIN Clicks c
    ON a.AUCTION_ID = c.AUCTION_ID
   AND a.PRODUCT_ID = c.PRODUCT_ID
   AND a.USER_ID = c.USER_ID
   AND c.OCCURRED_AT >= i.OCCURRED_AT
), purchases_attrib AS (
  SELECT
    b.*,
    p.PURCHASE_ID,
    p.PURCHASED_AT AS PURCHASE_TIME,
    p.QUANTITY,
    p.UNIT_PRICE
  FROM base b
  LEFT JOIN Purchases p
    ON p.USER_ID = b.USER_ID
   AND p.PRODUCT_ID = b.PRODUCT_ID
   AND p.PURCHASED_AT >= b.CLICK_TIME
   AND p.PURCHASED_AT <  DATEADD(minute, 60, b.CLICK_TIME)
)
SELECT
  *,
  IFF(IMPRESSION_ID IS NOT NULL, 1, 0) AS HAS_IMPRESSION,
  IFF(CLICK_ID IS NOT NULL, 1, 0)      AS HAS_CLICK,
  IFF(PURCHASE_ID IS NOT NULL, 1, 0)   AS HAS_PURCHASE,
  (QUANTITY * UNIT_PRICE)              AS REVENUE
FROM purchases_attrib;
