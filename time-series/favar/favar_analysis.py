#!/usr/bin/env python3
"""
Factor-Augmented VAR (FAVAR) Analysis
Based on Bernanke, Boivin and Eliasz (QJE, 2005)
"""

import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*80)
print("FACTOR-AUGMENTED VAR (FAVAR) ANALYSIS")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# DATA PREPARATION
print("\nDATA LOADING")
print("-"*80)

clicks_df = pd.read_parquet('../../data/hourly_clicks.parquet')
purchases_df = pd.read_parquet('../../data/hourly_purchases.parquet')
impressions_df = pd.read_parquet('../../data/hourly_impressions.parquet')
auctions_df = pd.read_parquet('../../data/hourly_auctions.parquet')

merged = purchases_df.merge(
    clicks_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    impressions_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    auctions_df, on='ACTIVITY_HOUR', how='outer'
)

merged['ACTIVITY_HOUR'] = pd.to_datetime(merged['ACTIVITY_HOUR'])
merged = merged.fillna(0)
merged['date'] = merged['ACTIVITY_HOUR'].dt.date

# Daily aggregation
agg_rules = {
    'HOURLY_GMV': 'sum',
    'HOURLY_TRANSACTION_COUNT': 'sum',
    'HOURLY_UNITS_SOLD': 'sum',
    'HOURLY_PURCHASING_USERS': 'sum',
    'HOURLY_PRODUCTS_PURCHASED': 'max',
    'HOURLY_CLICK_COUNT': 'sum',
    'HOURLY_CLICKING_USERS': 'sum',
    'HOURLY_CLICKED_VENDORS': 'max',
    'HOURLY_CLICKED_CAMPAIGNS': 'max',
    'HOURLY_CLICKED_PRODUCTS': 'max',
    'HOURLY_IMPRESSION_COUNT': 'sum',
    'HOURLY_IMPRESSED_USERS': 'sum',
    'HOURLY_IMPRESSED_VENDORS': 'max',
    'HOURLY_IMPRESSED_CAMPAIGNS': 'max',
    'HOURLY_IMPRESSED_PRODUCTS': 'max',
    'HOURLY_AUCTION_COUNT': 'sum',
    'HOURLY_AUCTION_USERS': 'sum'
}

daily = merged.groupby('date').agg(agg_rules).reset_index()
print(f"Daily observations: {len(daily)}")
print(f"Variables: {len(agg_rules)}")

# Variable mapping for cleaner names
var_mapping = {
    'HOURLY_GMV': 'gmv',
    'HOURLY_TRANSACTION_COUNT': 'transactions',
    'HOURLY_UNITS_SOLD': 'units',
    'HOURLY_PURCHASING_USERS': 'purchasers',
    'HOURLY_PRODUCTS_PURCHASED': 'prod_variety',
    'HOURLY_CLICK_COUNT': 'clicks',
    'HOURLY_CLICKING_USERS': 'clickers',
    'HOURLY_CLICKED_VENDORS': 'click_vendors',
    'HOURLY_CLICKED_CAMPAIGNS': 'click_campaigns',
    'HOURLY_CLICKED_PRODUCTS': 'click_products',
    'HOURLY_IMPRESSION_COUNT': 'impressions',
    'HOURLY_IMPRESSED_USERS': 'viewers',
    'HOURLY_IMPRESSED_VENDORS': 'imp_vendors',
    'HOURLY_IMPRESSED_CAMPAIGNS': 'imp_campaigns',
    'HOURLY_IMPRESSED_PRODUCTS': 'imp_products',
    'HOURLY_AUCTION_COUNT': 'auctions',
    'HOURLY_AUCTION_USERS': 'searchers'
}

# STEP 1: STANDARDIZE DATA
print("\nSTEP 1: DATA STANDARDIZATION")
print("-"*80)

# Convert to log differences and standardize
data_transformed = pd.DataFrame(index=pd.to_datetime(daily['date']))

for orig_col, short_name in var_mapping.items():
    log_series = np.log(daily[orig_col].values + 1)
    diff_series = pd.Series(log_series).diff() * 100
    data_transformed[short_name] = diff_series.values

# Remove first observation (NA from differencing)
data_transformed = data_transformed.iloc[1:]

# Standardize
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data_transformed),
    columns=data_transformed.columns,
    index=data_transformed.index
)

print(f"Observations after differencing: {len(data_scaled)}")

# STEP 2: EXTRACT PRINCIPAL COMPONENTS FROM ALL VARIABLES
print("\nSTEP 2: EXTRACT PRINCIPAL COMPONENTS")
print("-"*80)

n_factors = 3
pca_all = PCA(n_components=n_factors)
factors_all = pca_all.fit_transform(data_scaled)
explained_var = pca_all.explained_variance_ratio_

print(f"Variance explained by {n_factors} factors:")
for i in range(n_factors):
    print(f"  Factor {i+1}: {explained_var[i]*100:.2f}%")
print(f"  Cumulative: {explained_var.cumsum()[-1]*100:.2f}%")

# STEP 3: IDENTIFY SLOW-MOVING VARIABLES
print("\nSTEP 3: IDENTIFY SLOW-MOVING VARIABLES")
print("-"*80)

# In BBE, slow variables are those not directly affected by monetary policy
# For ad platform: slow = demand-side variables not directly controlled by platform
slow_vars = ['gmv', 'transactions', 'units', 'purchasers', 'prod_variety',
             'searchers', 'auctions']  # Demand/outcome variables

fast_vars = ['clicks', 'clickers', 'click_vendors', 'click_campaigns', 'click_products',
             'impressions', 'viewers', 'imp_vendors', 'imp_campaigns', 'imp_products']  # Supply/policy variables

print(f"Slow variables (demand-side): {len(slow_vars)}")
print(f"Fast variables (supply-side): {len(fast_vars)}")

# Extract factors from slow variables only
pca_slow = PCA(n_components=n_factors)
factors_slow = pca_slow.fit_transform(data_scaled[slow_vars])

# STEP 4: CLEAN FACTORS FROM OBSERVED POLICY VARIABLES
print("\nSTEP 4: CLEAN FACTORS FROM POLICY VARIABLES")
print("-"*80)

# Select key observed policy variables (Y in FAVAR notation)
# These are the platform's main control variables
observed_vars = ['impressions', 'clicks', 'gmv']

# Regress all factors on slow factors + observed variables
# This removes the effect of observed variables from factors
factors_df = pd.DataFrame(factors_all, columns=[f'F{i+1}' for i in range(n_factors)])
slow_factors_df = pd.DataFrame(factors_slow, columns=[f'F_slow{i+1}' for i in range(n_factors)])
observed_df = data_scaled[observed_vars].reset_index(drop=True)

# Clean each factor
factors_cleaned = np.zeros_like(factors_all)
for i in range(n_factors):
    # Regress factor i on slow factors + observed vars
    X = pd.concat([slow_factors_df, observed_df], axis=1)
    X = sm.add_constant(X)
    y = factors_df[f'F{i+1}']
    
    model = sm.OLS(y, X).fit()
    
    # Remove effect of observed variables
    obs_effect = np.zeros(len(y))
    for j, var in enumerate(observed_vars):
        obs_effect += model.params[var] * observed_df[var].values
    
    factors_cleaned[:, i] = y - obs_effect

print("Factors cleaned of observed variable effects")

# STEP 5: ESTIMATE FAVAR
print("\nSTEP 5: ESTIMATE FAVAR MODEL")
print("-"*80)

# Combine cleaned factors with observed variables
favar_data = pd.DataFrame(factors_cleaned, columns=[f'F{i+1}' for i in range(n_factors)])
for var in observed_vars:
    favar_data[var] = data_scaled[var].values

# Estimate VAR
lag_order = 4  # Reduced from 13 for stability with daily data
var_model = sm.tsa.VAR(favar_data)
var_results = var_model.fit(lag_order)

print(f"VAR({lag_order}) estimated")
print(f"Log-likelihood: {var_results.llf:.2f}")
print(f"AIC: {var_results.aic:.2f}")
print(f"BIC: {var_results.bic:.2f}")

# STEP 6: COMPUTE LOADINGS
print("\nSTEP 6: COMPUTE FACTOR LOADINGS")
print("-"*80)

# Regress each variable on factors + observed vars
loadings = {}
r_squared = {}

for var in data_scaled.columns:
    X = pd.concat([
        pd.DataFrame(factors_cleaned, columns=[f'F{i+1}' for i in range(n_factors)]),
        observed_df
    ], axis=1)
    X = sm.add_constant(X)
    y = data_scaled[var].reset_index(drop=True)
    
    model = sm.OLS(y, X).fit()
    loadings[var] = model.params.to_dict()
    r_squared[var] = model.rsquared

# Display loadings for key variables
print("\nFactor Loadings (Top Variables):")
loading_df = pd.DataFrame(loadings).T
for i in range(n_factors):
    print(f"\nFactor {i+1} - Top 5 loadings:")
    top5 = loading_df[f'F{i+1}'].abs().nlargest(5)
    for var, load in top5.items():
        actual_load = loading_df.loc[var, f'F{i+1}']
        print(f"  {var:20s}: {actual_load:7.4f}")

# STEP 7: IMPULSE RESPONSE FUNCTIONS
print("\nSTEP 7: IMPULSE RESPONSE FUNCTIONS")
print("-"*80)

# Compute IRFs for key shocks
horizon = 30
irf_results = var_results.irf(horizon)

# Focus on impression shock (platform's main policy tool)
imp_shock_idx = list(favar_data.columns).index('impressions')
gmv_response_idx = list(favar_data.columns).index('gmv')
click_response_idx = list(favar_data.columns).index('clicks')

# Get IRFs
gmv_response_to_imp = irf_results.irfs[:, gmv_response_idx, imp_shock_idx]
click_response_to_imp = irf_results.irfs[:, click_response_idx, imp_shock_idx]

print(f"\nGMV response to impression shock:")
print(f"  Impact (t=0):     {gmv_response_to_imp[0]:.4f}")
print(f"  After 7 days:     {gmv_response_to_imp[7]:.4f}")
print(f"  After 30 days:    {gmv_response_to_imp[-1]:.4f}")
print(f"  Cumulative (30d): {gmv_response_to_imp.sum():.4f}")

# STEP 8: FORECAST ERROR VARIANCE DECOMPOSITION
print("\nSTEP 8: VARIANCE DECOMPOSITION")
print("-"*80)

fevd_results = var_results.fevd(horizon)

# Extract FEVD for GMV at different horizons
# Use min to avoid index out of bounds
h7 = min(7, fevd_results.decomp.shape[0] - 1)
h30 = min(30, fevd_results.decomp.shape[0] - 1)
gmv_fevd_7 = fevd_results.decomp[h7, gmv_response_idx, :]
gmv_fevd_30 = fevd_results.decomp[h30, gmv_response_idx, :]

print("\nGMV Variance Decomposition:")
print(f"\n{h7}-day horizon:")
for i, col in enumerate(favar_data.columns):
    print(f"  {col:15s}: {gmv_fevd_7[i]*100:6.2f}%")

print(f"\n{h30}-day horizon:")
for i, col in enumerate(favar_data.columns):
    print(f"  {col:15s}: {gmv_fevd_30[i]*100:6.2f}%")

# Calculate ad-platform contribution
ad_contribution_7 = (gmv_fevd_7[imp_shock_idx] + gmv_fevd_7[click_response_idx]) * 100
ad_contribution_30 = (gmv_fevd_30[imp_shock_idx] + gmv_fevd_30[click_response_idx]) * 100

print(f"\nAd Platform Contribution to GMV Variance:")
print(f"  7-day:  {ad_contribution_7:.2f}%")
print(f"  30-day: {ad_contribution_30:.2f}%")

# STEP 9: BOOTSTRAPPED CONFIDENCE INTERVALS
print("\nSTEP 9: BOOTSTRAP CONFIDENCE INTERVALS")
print("-"*80)

n_bootstrap = 100
print(f"Running {n_bootstrap} bootstrap simulations...")

# Store bootstrap results
boot_irfs = np.zeros((n_bootstrap, horizon + 1))

for b in range(n_bootstrap):
    # Resample residuals
    residuals = var_results.resid
    boot_resid = residuals.sample(n=len(residuals), replace=True)
    
    # Generate bootstrap data
    boot_data = var_results.fittedvalues + boot_resid
    
    # Re-estimate VAR
    try:
        boot_var = sm.tsa.VAR(boot_data)
        boot_results = boot_var.fit(lag_order)
        boot_irf = boot_results.irf(horizon)
        
        # Store GMV response to impression shock
        boot_irfs[b, :] = boot_irf.irfs[:, gmv_response_idx, imp_shock_idx]
    except:
        boot_irfs[b, :] = np.nan

# Calculate confidence intervals
lower_ci = np.nanpercentile(boot_irfs, 5, axis=0)
upper_ci = np.nanpercentile(boot_irfs, 95, axis=0)

print(f"Bootstrap completed")
print(f"\nGMV Response to Impressions (90% CI):")
print(f"  Impact: {gmv_response_to_imp[0]:.4f} [{lower_ci[0]:.4f}, {upper_ci[0]:.4f}]")
print(f"  7-day:  {gmv_response_to_imp[7]:.4f} [{lower_ci[7]:.4f}, {upper_ci[7]:.4f}]")
print(f"  30-day: {gmv_response_to_imp[-1]:.4f} [{lower_ci[-1]:.4f}, {upper_ci[-1]:.4f}]")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: IRF - GMV response to impression shock
ax = axes[0, 0]
ax.plot(gmv_response_to_imp, 'b-', linewidth=2, label='Point estimate')
ax.fill_between(range(horizon + 1), lower_ci, upper_ci, alpha=0.3, color='blue')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_title('GMV Response to Impression Shock')
ax.set_xlabel('Days')
ax.set_ylabel('Response')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Click response to impression shock  
ax = axes[0, 1]
ax.plot(click_response_to_imp, 'orange', linewidth=2)
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_title('Click Response to Impression Shock')
ax.set_xlabel('Days')
ax.set_ylabel('Response')
ax.grid(True, alpha=0.3)

# Plot 3: Variance decomposition over time
ax = axes[1, 0]
fevd_steps = min(horizon + 1, fevd_results.decomp.shape[0])
fevd_gmv = np.zeros((fevd_steps, len(favar_data.columns)))
for h in range(fevd_steps):
    fevd_gmv[h, :] = fevd_results.decomp[h, gmv_response_idx, :]

# Stack plot
bottom = np.zeros(fevd_steps)
colors = plt.cm.Set3(np.linspace(0, 1, len(favar_data.columns)))
for i, col in enumerate(favar_data.columns):
    ax.fill_between(range(fevd_steps), bottom, bottom + fevd_gmv[:, i], 
                    label=col, color=colors[i], alpha=0.7)
    bottom += fevd_gmv[:, i]

ax.set_title('GMV Variance Decomposition')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Proportion')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)

# Plot 4: Factor loadings heatmap
ax = axes[1, 1]
loading_matrix = loading_df[[f'F{i+1}' for i in range(n_factors)]].values
im = ax.imshow(loading_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_yticks(range(n_factors))
ax.set_yticklabels([f'Factor {i+1}' for i in range(n_factors)])
ax.set_xticks(range(len(loading_df)))
ax.set_xticklabels(loading_df.index, rotation=45, ha='right')
ax.set_title('Factor Loadings')
plt.colorbar(im, ax=ax)

plt.suptitle('FAVAR Analysis Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/favar_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved to: results/favar_analysis.png")

# SAVE DETAILED RESULTS
with open('results/favar_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FACTOR-AUGMENTED VAR (FAVAR) ANALYSIS\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    f.write("MODEL SPECIFICATION:\n")
    f.write(f"  Number of factors: {n_factors}\n")
    f.write(f"  Observed variables: {', '.join(observed_vars)}\n")
    f.write(f"  VAR lag order: {lag_order}\n")
    f.write(f"  Bootstrap iterations: {n_bootstrap}\n\n")
    
    f.write("KEY RESULTS:\n")
    f.write("-"*60 + "\n")
    f.write(f"1. Ad Platform Contribution to GMV Variance:\n")
    f.write(f"   7-day horizon:  {ad_contribution_7:.2f}%\n")
    f.write(f"   30-day horizon: {ad_contribution_30:.2f}%\n\n")
    
    f.write(f"2. GMV Elasticity to Impressions:\n")
    f.write(f"   Impact effect: {gmv_response_to_imp[0]:.4f}\n")
    f.write(f"   Cumulative 30-day: {gmv_response_to_imp.sum():.4f}\n\n")
    
    f.write(f"3. Model Fit:\n")
    f.write(f"   Log-likelihood: {var_results.llf:.2f}\n")
    f.write(f"   AIC: {var_results.aic:.2f}\n")
    f.write(f"   BIC: {var_results.bic:.2f}\n")

print("\nResults saved to: results/favar_results.txt")
print("="*80)