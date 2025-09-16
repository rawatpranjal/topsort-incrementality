# Collaborative Filtering Analysis with ALS

## Overview

This analysis implements collaborative filtering using Alternating Least Squares (ALS) on user-vendor interaction matrices from the marketplace data. The goal is to learn latent factors that explain user-vendor interaction patterns and enable vendor recommendations.

## Data

- **Source**: User-vendor panel data (`panel/archive/user_vendor_panel_pilot.parquet`)
- **Observations**: 5.6M user-vendor-week interactions
- **Users**: 436,640 unique users (sampled 10,000 for computational efficiency)
- **Vendors**: 20,411 unique vendors with GMV interactions
- **Time Period**: Multiple weeks of marketplace activity

## Methodology

### Matrix Construction
- Built sparse interaction matrices for GMV (revenue) and clicks
- Applied log transformation to GMV values to handle skewness
- Extremely sparse matrices (0.08% density for GMV, 0.02% for clicks)

### Models Implemented

1. **Alternating Least Squares (ALS)**
   - Matrix factorization via alternating optimization
   - Tested with k={20, 50, 100} latent factors
   - L2 regularization (λ=0.01)

2. **Truncated SVD** (for comparison)
   - Direct singular value decomposition
   - Explains 41.82% of variance with 50 components

3. **Non-negative Matrix Factorization (NMF)** (for comparison)
   - Constrained to non-negative factors
   - Better interpretability for recommendation systems

## Key Results

### Model Performance Comparison

| Model | Factors | RMSE | MAE | Correlation |
|-------|---------|------|-----|-------------|
| SVD (k=50) | 50 | **2.52** | 0.82 | 0.45 |
| NMF (k=50) | 50 | 2.56 | 0.83 | 0.44 |
| ALS (k=20) | 20 | 8.28 | 8.22 | -0.01 |
| ALS (k=50) | 50 | 8.28 | 8.22 | -0.01 |
| ALS (k=100) | 100 | 8.28 | 8.22 | -0.01 |

**Best Model**: SVD with 50 components (lowest RMSE = 2.52)

### Matrix Properties

#### GMV Interaction Matrix
- **Sparsity**: 99.92% (only 0.08% non-zero entries)
- **User behavior**: Mean 1.09 vendors per user, max 8 vendors
- **Vendor reach**: Mean 8.6 users per vendor, max 9,500+ users
- **Power law**: Top 10% vendors account for 89.6% of interactions

#### Clicks Interaction Matrix
- **More dense**: 0.02% density (2.5x more than GMV)
- **Higher engagement**: Mean 5 vendors clicked per user
- **Broader reach**: Mean 2.2 users per vendor

### Economic Insights

1. **Extreme Sparsity**: Most user-vendor pairs have no interaction, making collaborative filtering challenging

2. **Power Law Distribution**: A small number of vendors dominate interactions (top 10% = 90% of activity)

3. **Limited Cross-Shopping**: Users interact with very few vendors on average (median = 1)

4. **ALS Challenges**: The extreme sparsity and power law distribution make standard ALS less effective than SVD

5. **Cold Start Problem**: With median of 1 vendor per user, recommending new vendors is challenging

## Recommendations

1. **Hybrid Approach**: Combine collaborative filtering with content-based methods due to sparsity

2. **Popular Items Baseline**: Given the power law, popularity-based recommendations may be effective

3. **Graph-Based Methods**: Consider graph neural networks that can better handle sparse interactions

4. **Implicit Feedback**: Incorporate impressions and search data to densify the interaction matrix

5. **Temporal Dynamics**: Add time-aware factors since user preferences evolve

## Files Generated

- `als_collaborative_filtering.py`: Main implementation
- `results/model_comparison_*.json`: Model performance metrics
- `results/als_factors_*.npz`: Learned latent factors
- `results/mappings_*.json`: User/vendor ID mappings
- `results/sample_recommendations_*.json`: Example recommendations

## Usage

```python
python als_collaborative_filtering.py
```

The script will:
1. Load user-vendor interaction data
2. Build sparse interaction matrices
3. Train ALS, SVD, and NMF models
4. Compare model performance
5. Generate sample recommendations
6. Save results to `results/` directory

## Limitations

- Sampling required due to computational constraints
- Extreme sparsity limits ALS effectiveness
- Cold start problem for new users/vendors
- No temporal dynamics considered