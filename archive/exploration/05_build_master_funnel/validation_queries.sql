-- Validation queries for USER_FUNNEL_MASTER

-- 1) Row count vs Auctions
SELECT (SELECT COUNT(*) FROM ANALYSIS.USER_FUNNEL_MASTER) AS funnel_rows,
       (SELECT COUNT(*) FROM Auctions) AS auction_rows;

-- 2) Non-decreasing flags along funnel
SELECT 
  COUNT_IF(HAS_CLICK=1 AND HAS_IMPRESSION=0) AS clicks_without_impression,
  COUNT_IF(HAS_PURCHASE=1 AND HAS_CLICK=0)  AS purchases_without_click
FROM ANALYSIS.USER_FUNNEL_MASTER;

-- 3) Basic rates
SELECT 
  AVG(HAS_IMPRESSION) AS impression_rate,
  AVG(HAS_CLICK)      AS click_rate,
  AVG(HAS_PURCHASE)   AS purchase_rate
FROM ANALYSIS.USER_FUNNEL_MASTER;

-- 4) Sanity by rank
SELECT RANKING,
       AVG(HAS_IMPRESSION) AS p_impression,
       AVG(HAS_CLICK)      AS p_click,
       AVG(HAS_PURCHASE)   AS p_purchase
FROM ANALYSIS.USER_FUNNEL_MASTER
GROUP BY RANKING
ORDER BY RANKING;
