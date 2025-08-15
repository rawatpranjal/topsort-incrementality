-- Analysis: Auction Dynamics (Snowflake SQL)
-- Purpose: bidders per auction distribution; win rate by rank; anomalies
-- Notes:
-- - Assumes each row in Auctions is a bid/participant; IS_WINNER marks the winner.
-- - Lower RANKING means better position (assumption); verify if needed.
-- - Time window filters are optional; uncomment to scope analysis.

-- Parameters (edit as needed)
-- SET start_ts = '2025-03-14';
-- SET end_ts   = '2025-07-21';

-- Optional window filter template
-- WHERE AUCTION_TIMESTAMP >= TO_TIMESTAMP($start_ts)
--   AND AUCTION_TIMESTAMP <  TO_TIMESTAMP($end_ts)

------------------------------------------------------------
-- 1) Bidders per auction distribution
------------------------------------------------------------
WITH bidders AS (
  SELECT AUCTION_ID, COUNT(*) AS bidders
  FROM Auctions
  -- WHERE AUCTION_TIMESTAMP BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
  GROUP BY AUCTION_ID
)
SELECT bidders, COUNT(*) AS auctions
FROM bidders
GROUP BY bidders
ORDER BY bidders;

------------------------------------------------------------
-- 2) Win rate by rank
------------------------------------------------------------
SELECT
  RANKING,
  COUNT(*) AS participants,
  SUM(CASE WHEN IS_WINNER THEN 1 ELSE 0 END) AS wins,
  CAST(SUM(CASE WHEN IS_WINNER THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0) AS win_rate
FROM Auctions
-- WHERE AUCTION_TIMESTAMP BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
GROUP BY RANKING
ORDER BY RANKING;

------------------------------------------------------------
-- 3) Anomalies: winner not equal to best rank in the auction
------------------------------------------------------------
WITH best_rank AS (
  SELECT AUCTION_ID, MIN(RANKING) AS best_rank
  FROM Auctions
  -- WHERE AUCTION_TIMESTAMP BETWEEN TO_TIMESTAMP($start_ts) AND TO_TIMESTAMP($end_ts)
  GROUP BY AUCTION_ID
), winners AS (
  SELECT AUCTION_ID, RANKING AS winner_rank
  FROM Auctions
  WHERE IS_WINNER = TRUE
)
SELECT
  a.AUCTION_ID,
  b.best_rank,
  w.winner_rank,
  (w.winner_rank <> b.best_rank) AS is_anomaly
FROM best_rank b
JOIN winners w USING (AUCTION_ID)
JOIN Auctions a USING (AUCTION_ID)
WHERE w.winner_rank <> b.best_rank
LIMIT 1000;
