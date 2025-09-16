# Topsort Incrementality Analysis

## Project Overview

This repository analyzes ad incrementality for Topsort, an ad platform serving marketplaces (Etsy/DoorDash style). We're measuring the causal impact of sponsored ads on user purchases using 4 months of data (March 14 - July 21, 2025).

**Scale**: ~100M clicks, 0.5M daily auctions leading to clicks, 0.25M unique daily clickers

## Data Schema

### Core Tables

#### 1. AUCTIONS_USER (Auction-level)
- `AUCTION_ID` (Binary): Unique auction identifier
- `OPAQUE_USER_ID` (Varchar): Anonymized user identifier  
- `CREATED_AT` (Timestamp_NTZ): Auction timestamp

#### 2. AUCTIONS_RESULTS (Bid-level)
- `AUCTION_ID` (Binary): Links to AUCTIONS_USER
- `VENDOR_ID` (Binary): Advertiser/vendor identifier
- `CAMPAIGN_ID` (Binary): Campaign identifier
- `PRODUCT_ID` (Varchar): Product being advertised
- `RANKING` (Number): Bid rank (lower = better)
- `IS_WINNER` (Boolean): Whether bid won the auction
- `CREATED_AT` (Timestamp_NTZ): Bid timestamp

#### 3. IMPRESSIONS
- `INTERACTION_ID` (Varchar): Unique impression identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product shown
- `USER_ID` (Varchar): User who saw the ad
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Impression timestamp

#### 4. CLICKS
- `INTERACTION_ID` (Varchar): Unique click identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product clicked
- `USER_ID` (Varchar): User who clicked
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Click timestamp

#### 5. PURCHASES
- `PURCHASE_ID` (Varchar): Unique purchase identifier
- `PURCHASED_AT` (Timestamp_NTZ): Purchase timestamp
- `PRODUCT_ID` (Varchar): Product purchased
- `QUANTITY` (Number): Quantity purchased
- `UNIT_PRICE` (Number): Price per unit
- `USER_ID` (Varchar): User who purchased
- `PURCHASE_LINE` (Number): Line item in purchase

### Key Relationships
- Auctions → Impressions → Clicks linked via `AUCTION_ID`
- User tracking via `OPAQUE_USER_ID` (auctions) and `USER_ID` (impressions/clicks/purchases)
- Products tracked via `PRODUCT_ID` across all tables

## Business Context

### Auction Mechanics
- **Ranking Score**: `Score = bid × quality_score` (quality_score = predicted CTR)
- **Auction Type**: First-price auctions
- **Winner Selection**: Not always lowest rank due to budget constraints, pacing
- **Slots**: Variable (N = 4, 6, 10 depending on context)

### Autobidding System
Advertisers use automated bidding agents:
- `Bid = α × valuation`
- `Valuation = product_price × P(purchase)`
- `α(i,t) = f(α(t-1), performance)` - adapts over time

### Attribution
- **Window**: 60-minute post-click attribution
- **Type**: Last-click attribution
- **Organic**: Purchases include both organic and sponsored

## Repository Structure

```
topsort-incrementality/
├── exploration/          # SQL-based exploratory analysis
│   ├── 01_eda_auctions/     # Competition, bidders, win rates
│   ├── 02_eda_impressions/  # Impression rates, fold detection
│   ├── 03_eda_clicks/       # CTR by rank, time-to-click
│   ├── 04_eda_purchases/    # Conversion, unadvertised sales
│   ├── 05_build_master_funnel/ # Funnel construction
│   └── utils/               # Snowflake connection utilities
├── data/                # Data preparation
│   └── prepare_funnel.py   # Build master funnel with attribution
├── analysis/            # Causal inference models
│   ├── fixed_effects.py    # User/time fixed effects
│   ├── regression_discontinuity.py  # RDD at rank threshold
│   └── instrumental_variables.py    # IV using auction mechanics
├── for_gemini/          # AI analysis integration
│   └── concatenate_all.py  # Combine code and reports
└── outputs/             # Analysis results (text format)
```

## Workflow

### 1. Exploration Phase
Run SQL queries in `exploration/` folders to understand:
- Auction dynamics and competition
- Impression patterns and fold cutoff
- Click behavior and timing
- Purchase patterns and attribution

### 2. Data Preparation
```bash
python data/prepare_funnel.py
```
Creates master funnel table with proper attribution logic.

### 3. Causal Analysis
Three complementary approaches:

**Fixed Effects**: Control for unobserved heterogeneity
```bash
python analysis/fixed_effects.py
```

**Regression Discontinuity**: Exploit rank threshold
```bash
python analysis/regression_discontinuity.py
```

**Instrumental Variables**: Use auction mechanics
```bash
python analysis/instrumental_variables.py
```

### 4. Generate Reports
Results saved to `outputs/` as formatted text files.

## Key Findings

### User Behavior
- **9,300 persistent users**: Appear every week throughout 4 months
- **Quick decisions**: 90% of purchases happen within 2 clicks
- **Peak hours**: Most activity between 4pm-2am
- **Attribution window**: Most purchases within 60 minutes of click

### Market Dynamics
- **Daily scale**: 0.5M auctions → clicks, 0.25M unique clickers
- **Product diversity**: 50% of vendors promote 8+ products
- **CTR/CVR paradox**: Negative correlation (-0.25) between CTR and CVR

### Causal Identification Opportunities

1. **Persistent User Cohort**: 9,300 users with consistent weekly activity
2. **Unadvertised Products**: Natural control group
3. **Rank Threshold (Fold)**: Sharp cutoff in impression probability
4. **Top-Ranked Non-Winners**: Instrument for exogenous variation
5. **Mid-Campaign Starters**: Products beginning advertising mid-period
6. **Exposure Variation**: Users with 0, 1, 2+ ad exposures

## Important SQL Queries

### Master Funnel with Attribution
```sql
WITH base AS (
  SELECT
    au.AUCTION_ID,
    au.OPAQUE_USER_ID as USER_ID,
    ar.PRODUCT_ID,
    ar.VENDOR_ID,
    ar.RANKING,
    ar.IS_WINNER,
    i.OCCURRED_AT AS IMPRESSION_TIME,
    c.OCCURRED_AT AS CLICK_TIME,
    p.PURCHASE_ID,
    p.PURCHASED_AT
  FROM AUCTIONS_USER au
  JOIN AUCTIONS_RESULTS ar ON au.AUCTION_ID = ar.AUCTION_ID
  LEFT JOIN IMPRESSIONS i 
    ON ar.AUCTION_ID = i.AUCTION_ID 
    AND ar.PRODUCT_ID = i.PRODUCT_ID
  LEFT JOIN CLICKS c 
    ON i.INTERACTION_ID = c.INTERACTION_ID
  LEFT JOIN PURCHASES p 
    ON au.OPAQUE_USER_ID = p.USER_ID 
    AND ar.PRODUCT_ID = p.PRODUCT_ID
    AND p.PURCHASED_AT > c.OCCURRED_AT
    AND DATEDIFF(minute, c.OCCURRED_AT, p.PURCHASED_AT) <= 60
)
SELECT * FROM base;
```

### Persistent Weekly Users
```sql
WITH weekly_activity AS (
  SELECT 
    USER_ID,
    DATE_TRUNC('WEEK', OCCURRED_AT) AS week
  FROM CLICKS
  GROUP BY USER_ID, DATE_TRUNC('WEEK', OCCURRED_AT)
),
week_counts AS (
  SELECT 
    USER_ID,
    COUNT(DISTINCT week) AS weeks_active
  FROM weekly_activity
  GROUP BY USER_ID
)
SELECT USER_ID
FROM week_counts
WHERE weeks_active = (SELECT COUNT(DISTINCT DATE_TRUNC('WEEK', OCCURRED_AT)) FROM CLICKS);
```

### Unadvertised Products
```sql
SELECT COUNT(DISTINCT p.PRODUCT_ID) AS unadvertised_products
FROM PURCHASES p
WHERE NOT EXISTS (
  SELECT 1 FROM AUCTIONS_RESULTS ar 
  WHERE ar.PRODUCT_ID = p.PRODUCT_ID
);
```

### Rank Threshold (Fold) Detection
```sql
WITH impression_ranks AS (
  SELECT 
    ar.RANKING,
    COUNT(*) as impressions
  FROM AUCTIONS_RESULTS ar
  JOIN IMPRESSIONS i ON ar.AUCTION_ID = i.AUCTION_ID AND ar.PRODUCT_ID = i.PRODUCT_ID
  WHERE ar.IS_WINNER = TRUE
  GROUP BY ar.RANKING
)
SELECT 
  RANKING,
  impressions,
  SUM(impressions) OVER (ORDER BY RANKING) / SUM(impressions) OVER () as cumulative_pct
FROM impression_ranks
ORDER BY RANKING;
```

## Configuration

### Environment Setup
1. Copy `.env.example` to `.env`
2. Update with your Snowflake credentials:
```
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=MKOYFLY-RESEARCH
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=INCREMENTALITY
SNOWFLAKE_SCHEMA=INCREMENTALITY_RESEARCH
```

### Dependencies
```bash
pip install -r requirements.txt
```

## For AI Analysis (Gemini)

To prepare repository for AI review:
```bash
python for_gemini/concatenate_all.py
```

This creates:
- `for_gemini/repository_snapshot.txt`: All code in single file
- `for_gemini/reports_combined.txt`: All analysis outputs combined

## Notes

- All timestamps are in UTC
- User IDs persist across sessions
- Auction IDs link the entire funnel
- Some ranks above threshold don't win (budget exhaustion)
- Campaigns are short-lived and product-specific