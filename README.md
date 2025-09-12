# Ad-Platform Incrementality Analysis

This repository contains a comprehensive analysis of advertising effectiveness in an online marketplace, examining the causal impact of sponsored search on vendor revenue and user behavior.

## Overview

We analyze six months of auction data from a large online marketplace to measure the incremental return on ad spend (iROAS). The analysis employs multiple econometric approaches including fixed effects models, mixed effects models, and distributed lag specifications across three complementary panel structures.

## Repository Structure

```
├── latex/              
│   ├── main.tex      
│   ├── vendor_week.tex # Vendor-level analysis
│   ├── user_week.tex   # User-level analysis
│   ├── user_vendor_week.tex # User-vendor interaction analysis
│   └── holdouts.tex   # Holdout experiment results
│
├── panel/             # Panel methods
│   ├── vendor_week.ipynb
│   ├── user_week.ipynb
│   └── user_vendor_week.ipynb
│
├── holdouts/          # Using randomized user holdouts
│   ├── detect_holdouts.ipynb    # Holdout group detection
│   ├── analyse_holdouts.ipynb   # Treatment effect analysis
│   └── various reports (.txt)   # Analysis outputs
│
├── rd/                # Regression discontinuity analysis
│   ├── rd.ipynb
│   └── rd_eda.ipynb
│
├── eda/               # Exploratory data analysis
│   └── eda.ipynb
│
└── time-series/       # Time series econometric models
    ├── ardl_models/   # ARDL master framework (1 file)
    ├── var/           # VAR master framework (1 file)
    ├── vecm/          # VECM master framework (1 file)
    ├── dfm/           # Dynamic factor models (1 file)
    ├── favar_models/  # FAVAR models (1 file)
    └── policy_analysis/ # Policy impact analysis (1 file)
```
## Data

The analysis uses proprietary data from a major online marketplace including:
- Auction-level bid level data (ad ranks)
- Impression and click events (promoted only)
- Purchase transactions (all)

## Requirements

See `requirements.txt` for Python dependencies. Key packages:
- `polars` for data manipulation
- `pyfixest` for econometric models
- `numpy`, `pandas` for analysis
- `matplotlib`, `seaborn` for visualization

## Key Findings

### Time-Series Analysis: Vector Error Correction Models (VECM)

We conducted comprehensive VECM analysis across 20 model specifications to examine cointegration relationships and dynamic adjustments in the marketplace.

#### Key Results
- **Optimal specification**: M6_Supply_Density_Daily (BIC = -9421.6, Log-L = 4970.2)
- Strong cointegration between GMV, clicks, and auction metrics (rank = 5)
- Adjustment speeds indicate rapid correction to equilibrium (α_GMV = -0.36)

#### Model Performance Comparison
| Model | Variables | Coint. Rank | BIC | Description |
|-------|-----------|-------------|-----|-------------|
| M6_Supply_Density | 5 | 5 | -9421.6 | Auction density metrics |
| M8_Efficiency_Revenue | 3 | 3 | -4482.1 | Revenue per click focus |
| M3_Full_Funnel | 4 | 4 | -2455.4 | Complete funnel dynamics |
| M2_Trivariate | 3 | 1 | -1669.7 | Baseline GMV-Clicks-Auctions |

#### Economic Interpretation
1. **Multiple cointegrating vectors** confirm long-run equilibrium relationships between advertising and revenue
2. **Fast adjustment speeds** (α < -0.3) indicate efficient market dynamics
3. **Supply-side metrics** (vendor diversity, bid density) significantly improve model fit
4. **Bidirectional Granger causality** between clicks and GMV supports feedback effects

### Time-Series Analysis: Dynamic Factor Models

We performed comprehensive Dynamic Factor Model (DFM) analysis on 17 standardized marketplace variables to identify common latent factors driving system dynamics.

#### Model Selection
- Tested 1-5 factor specifications with various lag structures
- **Optimal model**: 2-factor DFM (BIC = 3979.49)
- Log-likelihood: -1792.20, AIC: 3736.41

#### Factor Interpretation

**Factor 1: Core Marketplace Activity**
- Explains 70-85% of variance in transaction metrics (GMV, units, purchasers)
- Strong loadings on all purchase-related variables (-2.4 to -2.8)
- Represents fundamental demand and conversion dynamics

**Factor 2: Advertising/Discovery Channel**  
- Explains 40-50% of variance in vendor/campaign diversity metrics
- High loadings on click_vendors (-5.03) and click_campaigns (-5.03)
- Captures advertising engagement and product discovery patterns

#### Variance Decomposition (Top Variables)
| Variable | Factor 1 | Factor 2 | Idiosyncratic |
|----------|----------|----------|---------------|
| Transactions | 79.6% | 20.4% | 0.0% |
| Units Sold | 84.3% | 15.4% | 0.3% |
| GMV | 79.9% | 19.5% | 0.6% |
| Click Vendors | 52.4% | 47.7% | 0.0% |
| Impressions | 70.6% | 14.1% | 15.3% |

#### Rolling Window Stability
- 60-day rolling window analysis shows stable factor loadings
- GMV loading coefficient of variation: 23.3
- Indicates consistent structural relationships over time

### Policy Implications

1. **Two-factor structure** suggests advertising operates through a distinct discovery channel rather than directly driving core marketplace activity
2. **High common factor explanation** (>95% for key metrics) indicates strong system-wide co-movement
3. **Stable parameters** over rolling windows suggest consistent advertising effectiveness

## Paper

The full paper is available in the `latex/` directory. Compile with:
```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```