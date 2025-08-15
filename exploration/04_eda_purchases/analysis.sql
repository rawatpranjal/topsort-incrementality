-- Analysis: Purchases and attribution window (Snowflake SQL)
-- Purpose: time-to-purchase from last click; purchases on unadvertised products.

-- Parameters
-- SET start_ts = '2025-03-14';
-- SET end_ts   = '2025-07-21';
-- SET attribution_minutes = 60;

-- 1) Time to purchase from click (minutes)
WITH clicks_purchases AS (
  SELECT
    c.USER_ID,
    c.PRODUCT_ID,
    c.OCCURRED_AT AS click_time,
    p.PURCHASE_ID,
    p.PURCHASED_AT AS purchase_time,
    DATEDIFF(minute, c.OCCURRED_AT, p.PURCHASED_AT) AS minutes_to_purchase
  FROM Clicks c
  JOIN Purchases p
    ON p.USER_ID = c.USER_ID
   AND p.PRODUCT_ID = c.PRODUCT_ID
   AND p.PURCHASED_AT >= c.OCCURRED_AT
   AND p.PURCHASED_AT <  DATEADD(minute, 120, c.OCCURRED_AT)
  -- WHERE c.OCCURRED_AT BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
)
SELECT minutes_to_purchase, COUNT(*) AS purchases
FROM clicks_purchases
GROUP BY 1
ORDER BY 1;

-- 2) Products never advertised (no presence in Auctions)
SELECT COUNT(DISTINCT p.PRODUCT_ID) AS num_unadvertised_products
FROM Purchases p
LEFT JOIN Auctions a
  ON p.PRODUCT_ID = a.PRODUCT_ID
WHERE a.PRODUCT_ID IS NULL;

-- 2b) Extract sales of unadvertised products (export to CSV)
SELECT 
  p.PRODUCT_ID,
  COUNT(*) AS purchase_events,
  SUM(p.QUANTITY) AS total_quantity,
  SUM(p.QUANTITY * p.UNIT_PRICE) AS total_revenue
FROM Purchases p
LEFT JOIN Auctions a
  ON p.PRODUCT_ID = a.PRODUCT_ID
WHERE a.PRODUCT_ID IS NULL
GROUP BY p.PRODUCT_ID
ORDER BY total_revenue DESC;
