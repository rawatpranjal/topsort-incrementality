# Time Series Analysis for Ad Platform Incrementality

This directory contains comprehensive time series econometric analyses for measuring advertising incrementality and platform efficiency.

## 📊 Directory Structure

### Model Implementations

- **`ardl_models/`** - Autoregressive Distributed Lag models for short/long-run elasticities
- **`var_models/`** - Vector Autoregression models for endogenous system dynamics
- **`vecm_models/`** - Vector Error Correction models for cointegrated time series
- **`dfm_models/`** - Dynamic Factor Models for dimension reduction and latent factors
- **`favar_models/`** - Factor-Augmented VAR combining factor models with VAR
- **`policy_analysis/`** - Interrupted Time Series and causal inference for policy impacts

### Support

- **`utils/`** - Helper functions for data handling and diagnostics
- **`legacy/`** - Original analysis files (preserved for reference)
- **`results/`** - All analysis outputs and generated reports
- **`literature/`** - Academic papers and references

## 🔬 Key Analyses

### 1. ARDL Models
- Captures both short-run dynamics and long-run equilibrium relationships
- Handles mixed order integration (I(0) and I(1) variables)
- Implements Pesaran-Shin-Smith bounds testing

### 2. VAR/VECM Models
- VAR: Captures dynamic interdependencies between multiple time series
- VECM: Models cointegrated relationships and error correction mechanisms
- Includes Forecast Error Variance Decomposition (FEVD)

### 3. Dynamic Factor Models
- Extracts latent factors driving system dynamics
- Reduces dimensionality while preserving information
- BIC-based optimal factor selection

### 4. FAVAR Analysis
- Combines high-dimensional information via factors
- Separates policy variables from market dynamics
- Provides structural impulse responses

### 5. Policy Impact Analysis
- Interrupted Time Series (ITS) design
- Regression Discontinuity analysis
- Synthetic control methods
- Comprehensive funnel efficiency metrics

## 📈 Key Findings

### Excessive Listing Removal Policy (May 1, 2025)
- **Conversion Rate collapsed 20.3%**
- **Revenue per click down 22%**
- **Funnel efficiency declined 14.8%**
- Structural break highly significant (p < 0.001)

### Model Consensus
- All models confirm significant negative impact on platform efficiency
- Trade-off between quantity (impressions/clicks) and quality (conversions)
- Ad platform contribution to GMV variance: ~19% (FAVAR estimate)

## 🚀 Quick Start

### Run Individual Analyses
```bash
# ARDL analysis
python3 ardl_models/modular_ardl_models.py

# VAR with FEVD
python3 var_models/var_fevd_analysis.py

# Dynamic Factor Model
python3 dfm_models/dfm_comprehensive_analysis.py

# Policy Impact Analysis
python3 policy_analysis/policy_impact_consolidated.py
```

### Data Requirements
All models expect hourly data in parquet format located in `../data/`:
- `hourly_clicks_*.parquet`
- `hourly_purchases_*.parquet`
- `hourly_impressions_*.parquet`
- `hourly_auctions_*.parquet`

## 📝 Output

All results are saved to the `results/` directory:
- Statistical tables in `.txt` format
- Diagnostic plots in `.png` format
- Comprehensive reports with all metrics

## 🔧 Dependencies

- `statsmodels` - Core econometric models
- `pandas` - Data manipulation
- `numpy` - Numerical computation
- `scipy` - Statistical functions
- `matplotlib` - Visualization
- `scikit-learn` - PCA for factor models

## 📚 References

Key methodological papers are available in the `literature/` directory, including:
- Pesaran, Shin & Smith (2001) - Bounds testing approaches
- Bernanke, Boivin & Eliasz (2005) - FAVAR methodology
- Stock & Watson (2002) - Dynamic Factor Models

## 🎯 Main Conclusions

The time series analysis provides robust evidence that the platform experienced a fundamental shift in its economic dynamics following the policy intervention. While engagement metrics increased, the quality of that engagement deteriorated significantly, resulting in a broken conversion funnel and reduced platform efficiency.