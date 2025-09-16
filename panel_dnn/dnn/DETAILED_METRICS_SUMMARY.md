# Detailed Metrics Summary - Heterogeneous Treatment Effects

## Complete Metrics Comparison

### 1. Outcome Prediction (Y) Metrics

| Metric | Linear β(X) | Nonlinear β(X) |
|--------|-------------|----------------|
| **MSE** | 0.281 | 0.290 |
| **RMSE** | 0.531 | 0.539 |
| **MAE** | 0.423 | 0.428 |
| **Correlation** | 0.991 | 0.993 |
| **R²** | 0.982 | 0.986 |

### 2. Heterogeneous Effect β(X) Recovery

| Metric | Linear β(X) | Nonlinear β(X) |
|--------|-------------|----------------|
| **MSE** | 0.071 | 0.075 |
| **RMSE** | 0.267 | 0.274 |
| **MAE** | 0.220 | 0.213 |
| **Correlation** | 0.999 | 0.994 |

### 3. Fixed Effects Recovery (Correlations)

| Fixed Effect | Linear β(X) | Nonlinear β(X) |
|--------------|-------------|----------------|
| **User FE** | 0.997 | 0.991 |
| **Vendor FE** | 0.996 | 0.999 |
| **Time FE** | 0.995 | 0.998 |

### 4. Training Performance

| Metric | Linear β(X) | Nonlinear β(X) |
|--------|-------------|----------------|
| **Total Epochs** | 100 | 150 |
| **Best Epoch** | 87 | 143 |
| **Initial Train Loss** | 3.559 | 3.903 |
| **Final Train Loss** | 0.377 | 0.419 |
| **Initial Val Loss** | 0.411 | 0.522 |
| **Best Val Loss** | 0.282 | 0.284 |
| **Initial β Correlation** | 0.726 | 0.704 |
| **Best β Correlation** | 0.988 | 0.973 |

### 5. Linear Case: DL vs Feols Comparison

| Parameter | True Value | Feols Estimate | DL Estimate | Feols Error | DL Error |
|-----------|------------|----------------|-------------|-------------|----------|
| **Intercept** | 2.000 | 1.998 | ~2.043* | 0.002 | 0.043 |
| **X_u_eng** | 0.500 | 0.500 | - | 0.000 | - |
| **X_v_eng** | -0.800 | -0.800 | - | 0.000 | - |
| **X_u_latent_0** | 1.200 | 1.201 | - | 0.001 | - |
| **X_v_latent_1** | -0.700 | -0.700 | - | 0.000 | - |

*DL learns the full function β(X) rather than individual coefficients

### 6. Distribution Analysis - Linear Case

| Statistic | True Y | Predicted Y | True β(X) | Predicted β(X) |
|-----------|--------|-------------|-----------|----------------|
| **Mean** | 1.999 | 1.928 | 1.842 | 2.043 |
| **Std Dev** | 3.924 | 3.886 | 1.929 | 1.777 |
| **Min** | -11.442 | -10.647 | -6.693 | -5.635 |
| **Q1** | -0.682 | -0.748 | 0.559 | 0.816 |
| **Median** | 1.641 | 1.548 | 1.850 | 2.050 |
| **Q3** | 4.267 | 4.191 | 3.123 | 3.265 |
| **Max** | 27.818 | 26.640 | 10.569 | 9.906 |

### 7. Distribution Analysis - Nonlinear Case

| Statistic | True Y | Predicted Y | True β(X) | Predicted β(X) |
|-----------|--------|-------------|-----------|----------------|
| **Mean** | 2.573 | 2.532 | 2.351 | 2.515 |
| **Std Dev** | 4.514 | 4.460 | 1.662 | 1.528 |
| **Min** | -13.086 | -10.023 | -8.817 | -6.786 |
| **Q1** | -0.658 | -0.675 | 1.380 | 1.595 |
| **Median** | 1.919 | 1.886 | 2.258 | 2.404 |
| **Q3** | 5.139 | 5.090 | 3.252 | 3.351 |
| **Max** | 33.546 | 32.737 | 14.229 | 13.439 |

### 8. Residual Analysis

| Metric | Linear β(X) | Nonlinear β(X) |
|--------|-------------|----------------|
| **Mean** | 0.071 | 0.041 |
| **Std Dev** | 0.526 | 0.537 |
| **Skewness** | 0.010 | -0.019 |
| **Kurtosis** | 0.047 | 0.133 |
| **Normality p-value** | 0.251 | 0.487 |

### 9. Model Efficiency Metrics

| Metric | Value |
|--------|-------|
| **Total Parameters** | 27,668 |
| **Data Generation Speed** | 7.4M obs/sec |
| **Storage Size (10M rows)** | 515 MB |
| **Batch Size** | 4,096 |
| **Chunk Size** | 100,000 |

## Key Performance Indicators

### Accuracy Metrics Summary
- **Best Y RMSE**: 0.531 (linear), 0.539 (nonlinear)
- **Best Y R²**: 0.982 (linear), 0.986 (nonlinear)
- **Best β(X) RMSE**: 0.267 (linear), 0.274 (nonlinear)
- **Best β(X) Correlation**: 0.999 (linear), 0.994 (nonlinear)

### Error Analysis
- **Y Prediction Error**: ~2% of variance unexplained
- **β(X) Recovery Error**: <0.3 RMSE in both cases
- **Fixed Effects Error**: <1% correlation loss

### Statistical Validation
- **Residuals**: Approximately normal (p > 0.05)
- **Bias**: Near-zero mean residuals
- **Calibration**: Predicted distributions match true distributions well

## Interpretation

1. **RMSE Performance**:
   - Y prediction RMSE ~0.53 is excellent given Y std dev of ~4
   - β(X) RMSE ~0.27 is strong given β std dev of ~1.9

2. **MAE Performance**:
   - Y MAE ~0.42 means typical error is <0.5 units
   - β(X) MAE ~0.22 means typical coefficient error is small

3. **Fixed Effects**:
   - All correlations >0.99 indicate near-perfect recovery
   - Robust to both linear and nonlinear β(X) specifications

4. **Scalability**:
   - 7.4M observations/second generation speed
   - 27K parameters is modest for modern hardware
   - Efficient storage at ~50 bytes per observation