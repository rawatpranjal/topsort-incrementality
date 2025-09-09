# Ad-Platform Incrementality Analysis

This repository contains a comprehensive analysis of advertising effectiveness in an online marketplace, examining the causal impact of sponsored search on vendor revenue and user behavior.

## Overview

We analyze six months of auction data from a large online marketplace to measure the incremental return on ad spend (iROAS). The analysis employs multiple econometric approaches including fixed effects models, mixed effects models, and distributed lag specifications across three complementary panel structures.

## Key Findings

- **Vendor-level elasticity**: 0.77 average click-to-revenue elasticity
- **iROAS at marketplace CPCs ($0.75)**: 20.77x median return, substantially exceeding industry benchmarks
- **User heterogeneity**: Significant variation in ad responsiveness, with median user showing negative elasticity
- **Temporal dynamics**: Strong immediate effects (425% lift) with modest carryover (19% next-week lift)

## Repository Structure

```
├── latex/              # Academic paper (LaTeX source)
│   ├── main.tex       # Main document
│   ├── vendor_week.tex # Vendor-level analysis
│   ├── user_week.tex   # User-level analysis
│   ├── user_vendor_week.tex # User-vendor interaction analysis
│   └── holdouts.tex   # Holdout experiment results
│
├── panel/             # Panel construction and analysis notebooks
│   ├── vendor_week.ipynb
│   ├── user_week.ipynb
│   └── user_vendor_week.ipynb
│
├── holdouts/          # Natural experiment analysis
│   ├── detect_holdouts.ipynb    # Holdout group detection
│   ├── analyse_holdouts.ipynb   # Treatment effect analysis
│   └── various reports (.txt)   # Analysis outputs
│
├── rd/                # Regression discontinuity analysis
│   └── rd.ipynb
│
└── eda/               # Exploratory data analysis
    ├── eda.ipynb
    └── rd_eda.ipynb
```

## Methods

### Panel Analyses
1. **Vendor-Week Panel**: Fixed effects model measuring supply-side advertising returns
2. **User-Week Panel**: Mixed effects model capturing demand-side heterogeneity
3. **User-Vendor-Week Panel**: Distributed lag logit model for granular interaction effects

### Identification Strategy
- Two-way fixed effects controlling for time-invariant heterogeneity
- Empirical Bayes shrinkage for vendor-specific elasticities
- Natural experiment leveraging platform-implemented holdouts

## Data

The analysis uses proprietary data from a major online marketplace including:
- Auction-level bidding data
- Impression and click events
- Purchase transactions
- User and vendor characteristics

## Requirements

See `requirements.txt` for Python dependencies. Key packages:
- `polars` for data manipulation
- `pyfixest` for econometric models
- `numpy`, `pandas` for analysis
- `matplotlib`, `seaborn` for visualization

## Paper

The full academic paper is available in the `latex/` directory. Compile with:
```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Contact

For questions about this analysis, please contact: pp712@georgetown.edu