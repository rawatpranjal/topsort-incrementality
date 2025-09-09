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
└── eda/               # Exploratory data analysis
    └── eda.ipynb
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

## Paper

The full paper is available in the `latex/` directory. Compile with:
```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```