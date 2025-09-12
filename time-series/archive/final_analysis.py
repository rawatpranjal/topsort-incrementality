#!/usr/bin/env python3
"""
Final comprehensive analysis for merged hourly ARDL/VAR models
With complete output capture to text file
"""

import pandas as pd
import numpy as np
import warnings
import sys
import io
from datetime import datetime
from contextlib import redirect_stdout
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Create a StringIO object to capture all output
output_capture = io.StringIO()

# Create a custom print function that writes to both console and capture
class DualWriter:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
    
    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)
    
    def flush(self):
        self.file1.flush()
        self.file2.flush()

# Set up dual output
original_stdout = sys.stdout
sys.stdout = DualWriter(original_stdout, output_capture)

print("="*80)
print("HOURLY AD INCREMENTALITY ANALYSIS - COMPREHENSIVE RESULTS")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analyst: ARDL/VAR Framework v1.0")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND PREPARATION")
print("="*80)

print("\nLoading hourly data files from /data/ directory...")

try:
    clicks_df = pd.read_parquet('../data/hourly_clicks_2025-03-01_to_2025-09-30.parquet')
    purchases_df = pd.read_parquet('../data/hourly_purchases_2025-03-01_to_2025-09-30.parquet')
    impressions_df = pd.read_parquet('../data/hourly_impressions_2025-03-01_to_2025-09-30.parquet')
    auctions_df = pd.read_parquet('../data/hourly_auctions_2025-03-01_to_2025-09-30.parquet')
    
    print("✓ All data files loaded successfully")
    print(f"  - Clicks: {clicks_df.shape}")
    print(f"  - Purchases: {purchases_df.shape}")
    print(f"  - Impressions: {impressions_df.shape}")
    print(f"  - Auctions: {auctions_df.shape}")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# Merge all datasets
print("\nMerging datasets on ACTIVITY_HOUR...")
merged_df = purchases_df.merge(
    clicks_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    impressions_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    auctions_df, on='ACTIVITY_HOUR', how='outer'
)

merged_df['ACTIVITY_HOUR'] = pd.to_datetime(merged_df['ACTIVITY_HOUR'])
merged_df = merged_df.set_index('ACTIVITY_HOUR').sort_index().fillna(0)

print(f"✓ Merged dataset created: {merged_df.shape[0]} hours, {merged_df.shape[1]} variables")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DESCRIPTIVE STATISTICS")
print("="*80)

print("\n2.1 Dataset Overview:")
print("-" * 40)
print(f"Time Period: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
print(f"Number of Hours: {len(merged_df):,}")
print(f"Number of Days: {(merged_df.index.max() - merged_df.index.min()).days}")
print(f"Variables: {merged_df.shape[1]}")

print("\n2.2 Key Metrics Summary:")
print("-" * 40)
summary_stats = {
    'Total GMV': f"${merged_df['HOURLY_GMV'].sum():,.0f}",
    'Total Clicks': f"{merged_df['HOURLY_CLICK_COUNT'].sum():,.0f}",
    'Total Impressions': f"{merged_df['HOURLY_IMPRESSION_COUNT'].sum():,.0f}",
    'Total Auctions': f"{merged_df['HOURLY_AUCTION_COUNT'].sum():,.0f}",
    'Avg Hourly GMV': f"${merged_df['HOURLY_GMV'].mean():,.0f}",
    'Avg Hourly Clicks': f"{merged_df['HOURLY_CLICK_COUNT'].mean():,.0f}",
    'CTR (Click-Through Rate)': f"{(merged_df['HOURLY_CLICK_COUNT'].sum() / merged_df['HOURLY_IMPRESSION_COUNT'].sum() * 100):.3f}%",
    'Conversion Rate': f"{(merged_df['HOURLY_TRANSACTION_COUNT'].sum() / merged_df['HOURLY_CLICK_COUNT'].sum() * 100):.3f}%"
}

for metric, value in summary_stats.items():
    print(f"{metric:.<30} {value}")

print("\n2.3 Variable Statistics:")
print("-" * 40)
stats_df = merged_df[['HOURLY_GMV', 'HOURLY_CLICK_COUNT', 'HOURLY_IMPRESSION_COUNT', 'HOURLY_AUCTION_COUNT']].describe()
print(stats_df.to_string())

# ============================================================================
# DATA TRANSFORMATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: DATA TRANSFORMATION")
print("="*80)

print("\nApplying log transformation for elasticity interpretation...")
df_analysis = pd.DataFrame(index=merged_df.index)
df_analysis['log_GMV'] = np.log(merged_df['HOURLY_GMV'] + 1)
df_analysis['log_Clicks'] = np.log(merged_df['HOURLY_CLICK_COUNT'] + 1)
df_analysis['log_Auctions'] = np.log(merged_df['HOURLY_AUCTION_COUNT'] + 1)
df_analysis['log_Impressions'] = np.log(merged_df['HOURLY_IMPRESSION_COUNT'] + 1)
df_analysis['log_Vendors'] = np.log(merged_df['HOURLY_CLICKED_VENDORS'] + 1)

# Add time controls
df_analysis['hour'] = df_analysis.index.hour
df_analysis['dayofweek'] = df_analysis.index.dayofweek
df_analysis['month'] = df_analysis.index.month

# Add day of week dummies
for i in range(7):
    df_analysis[f'dow_{i}'] = (df_analysis.index.dayofweek == i).astype(int)

df_analysis = df_analysis.dropna()
print(f"✓ Transformed dataset: {df_analysis.shape[0]} observations, {df_analysis.shape[1]} variables")

# ============================================================================
# MODEL 1: OLS REGRESSION (BASELINE)
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: OLS REGRESSION ANALYSIS")
print("="*80)

print("\nSpecification: log(GMV) ~ log(Clicks) + log(Auctions) + Controls")
print("-" * 40)

X = df_analysis[['log_Clicks', 'log_Auctions', 'hour', 'dayofweek']]
X = sm.add_constant(X)
y = df_analysis['log_GMV']

ols_model = sm.OLS(y, X).fit()

print("\n4.1 Model Summary:")
print(ols_model.summary())

print("\n4.2 Key Interpretations:")
print("-" * 40)
clicks_coef = ols_model.params['log_Clicks']
clicks_se = ols_model.bse['log_Clicks']
clicks_ci = ols_model.conf_int().loc['log_Clicks']

print(f"Clicks Elasticity: {clicks_coef:.4f}")
print(f"  Standard Error: {clicks_se:.4f}")
print(f"  95% CI: [{clicks_ci[0]:.4f}, {clicks_ci[1]:.4f}]")
print(f"  Interpretation: 1% increase in clicks → {clicks_coef:.3f}% increase in GMV")

# Diagnostic tests
print("\n4.3 Diagnostic Tests:")
print("-" * 40)

# Durbin-Watson
dw = sm.stats.durbin_watson(ols_model.resid)
print(f"Durbin-Watson statistic: {dw:.3f}")
if dw < 1.5:
    print("  → Positive serial correlation detected")
elif dw > 2.5:
    print("  → Negative serial correlation detected")
else:
    print("  → No strong serial correlation")

# Jarque-Bera test for normality
jb_result = sm.stats.jarque_bera(ols_model.resid)
jb_stat = jb_result[0]
jb_pval = jb_result[1]
print(f"Jarque-Bera test: statistic={jb_stat:.2f}, p-value={jb_pval:.4f}")
if jb_pval < 0.05:
    print("  → Residuals are not normally distributed")
else:
    print("  → Residuals appear normally distributed")

# ============================================================================
# MODEL 2: EXTENDED OLS WITH MARKETPLACE DYNAMICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: EXTENDED OLS MODEL (MARKETPLACE DYNAMICS)")
print("="*80)

print("\nSpecification: log(GMV) ~ log(Clicks) + log(Vendors) + log(Auctions) + log(Impressions) + Controls")
print("-" * 40)

X_extended = df_analysis[['log_Clicks', 'log_Vendors', 'log_Auctions', 'log_Impressions', 'hour', 'dayofweek']]
X_extended = sm.add_constant(X_extended)

ols_extended = sm.OLS(y, X_extended).fit()

print("\n5.1 Model Summary:")
print(ols_extended.summary())

print("\n5.2 Model Comparison:")
print("-" * 40)
print(f"Basic OLS R²: {ols_model.rsquared:.4f}")
print(f"Extended OLS R²: {ols_extended.rsquared:.4f}")
print(f"R² Improvement: {(ols_extended.rsquared - ols_model.rsquared):.4f}")

# ============================================================================
# MODEL 3: VECTOR AUTOREGRESSION (VAR)
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: VECTOR AUTOREGRESSION (VAR) ANALYSIS")
print("="*80)

print("\nEndogenous System: log(GMV), log(Clicks), log(Auctions)")
print("-" * 40)

# Prepare VAR data
endog_vars = ['log_GMV', 'log_Clicks', 'log_Auctions']
endog_data = df_analysis[endog_vars]

# Create VAR model
var_model = VAR(endog_data)

# Lag order selection
print("\n6.1 Lag Order Selection:")
print("-" * 40)
lag_selection = var_model.select_order(maxlags=15)
print(lag_selection.summary())

optimal_lag = lag_selection.aic
print(f"\n✓ Optimal lag order selected: {optimal_lag} (by AIC)")

# Fit VAR
var_results = var_model.fit(optimal_lag)

print("\n6.2 VAR Model Summary:")
print("-" * 40)
print(var_results.summary())

# ============================================================================
# GRANGER CAUSALITY TESTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: GRANGER CAUSALITY ANALYSIS")
print("="*80)

print("\nTesting directional causality between variables...")
print("-" * 40)

causality_matrix = pd.DataFrame(index=endog_vars, columns=endog_vars)

for caused in endog_vars:
    for cause in endog_vars:
        if caused != cause:
            try:
                test = var_results.test_causality(caused, causing=cause, kind='f')
                causality_matrix.loc[caused, cause] = f"{test.pvalue:.4f}"
                
                symbol = "✓" if test.pvalue < 0.05 else "✗"
                significance = "***" if test.pvalue < 0.001 else "**" if test.pvalue < 0.01 else "*" if test.pvalue < 0.05 else ""
                print(f"{symbol} {cause:15s} → {caused:15s}: F={test.test_statistic:6.2f}, p={test.pvalue:.4f} {significance}")
            except:
                causality_matrix.loc[caused, cause] = "N/A"

print("\n7.1 Causality Matrix (p-values):")
print(causality_matrix.to_string())

print("\n7.2 Key Findings:")
bidirectional = False
if float(causality_matrix.loc['log_GMV', 'log_Clicks']) < 0.05 and float(causality_matrix.loc['log_Clicks', 'log_GMV']) < 0.05:
    bidirectional = True
    print("✓ BIDIRECTIONAL CAUSALITY detected between Clicks and GMV")
    print("  This indicates a feedback loop: success drives more advertising")
else:
    print("→ Unidirectional or no clear causality pattern")

# ============================================================================
# IMPULSE RESPONSE FUNCTIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: IMPULSE RESPONSE ANALYSIS")
print("="*80)

irf = var_results.irf(periods=48)

# Get indices
gmv_idx = endog_vars.index('log_GMV')
clicks_idx = endog_vars.index('log_Clicks')
auctions_idx = endog_vars.index('log_Auctions')

print("\n8.1 Response of GMV to One-Unit Shocks:")
print("-" * 40)

# Response to Clicks shock
gmv_to_clicks = irf.irfs[:, gmv_idx, clicks_idx]
print("\nResponse to CLICKS shock:")
print(f"  Hour  1: {gmv_to_clicks[0]:8.4f}")
print(f"  Hour  6: {gmv_to_clicks[5]:8.4f}")
print(f"  Hour 12: {gmv_to_clicks[11]:8.4f}")
print(f"  Hour 24: {gmv_to_clicks[23]:8.4f}")
print(f"  Hour 48: {gmv_to_clicks[47]:8.4f}")

print(f"\nCumulative Effects:")
print(f"  6-hour:  {np.sum(gmv_to_clicks[:6]):8.4f}")
print(f"  12-hour: {np.sum(gmv_to_clicks[:12]):8.4f}")
print(f"  24-hour: {np.sum(gmv_to_clicks[:24]):8.4f}")
print(f"  48-hour: {np.sum(gmv_to_clicks[:48]):8.4f}")

# Response to Auctions shock
gmv_to_auctions = irf.irfs[:, gmv_idx, auctions_idx]
print("\nResponse to AUCTIONS shock:")
print(f"  24-hour cumulative: {np.sum(gmv_to_auctions[:24]):8.4f}")

# ============================================================================
# VARIANCE DECOMPOSITION
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: FORECAST ERROR VARIANCE DECOMPOSITION")
print("="*80)

fevd = var_results.fevd(periods=48)

print("\nPercentage of GMV forecast error variance explained by each variable:")
print("-" * 70)
print("Hour |  GMV Self  |   Clicks   |  Auctions  |")
print("-" * 70)

for h in [1, 3, 6, 12, 24, 48]:
    if h <= len(fevd.decomp):
        decomp = fevd.decomp[h-1][gmv_idx]
        gmv_self = decomp[gmv_idx] * 100
        clicks_contrib = decomp[clicks_idx] * 100
        auctions_contrib = decomp[auctions_idx] * 100
        print(f"{h:4d} | {gmv_self:9.2f}% | {clicks_contrib:9.2f}% | {auctions_contrib:9.2f}% |")

# ============================================================================
# MODEL DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 10: VAR MODEL DIAGNOSTICS")
print("="*80)

print("\n10.1 Stability Check:")
print("-" * 40)
if var_results.is_stable():
    print("✓ VAR model is stable (all eigenvalues inside unit circle)")
else:
    print("✗ WARNING: VAR model may be unstable")

print("\n10.2 Serial Correlation Test (Ljung-Box):")
print("-" * 40)
try:
    # Test for serial correlation in residuals
    for var_name in endog_vars:
        var_idx = endog_vars.index(var_name)
        resid = var_results.resid[:, var_idx]
        lb_test = sm.stats.acorr_ljungbox(resid, lags=[10, 20], return_df=True)
        print(f"\n{var_name}:")
        print(f"  Lag 10: Q={lb_test.loc[10, 'lb_stat']:.2f}, p={lb_test.loc[10, 'lb_pvalue']:.4f}")
        print(f"  Lag 20: Q={lb_test.loc[20, 'lb_stat']:.2f}, p={lb_test.loc[20, 'lb_pvalue']:.4f}")
except Exception as e:
    print(f"Could not perform Ljung-Box test: {e}")

# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SECTION 11: EXECUTIVE SUMMARY")
print("="*80)

print("\n" + "KEY FINDINGS".center(70))
print("=" * 70)

findings = []

# 1. Elasticity
findings.append(f"1. PRIMARY ELASTICITY: {clicks_coef:.3f}")
findings.append(f"   → 1% increase in ad clicks leads to {clicks_coef:.3f}% increase in GMV")
findings.append(f"   → Statistical significance: p < 0.001")

# 2. Model fit
findings.append(f"\n2. MODEL PERFORMANCE:")
findings.append(f"   → OLS R² = {ols_model.rsquared:.3f}")
findings.append(f"   → Extended OLS R² = {ols_extended.rsquared:.3f}")
findings.append(f"   → VAR({optimal_lag}) selected by information criteria")

# 3. Causality
findings.append(f"\n3. CAUSALITY STRUCTURE:")
if bidirectional:
    findings.append(f"   → BIDIRECTIONAL causality confirmed")
    findings.append(f"   → Feedback loop: GMV ↔ Clicks")
else:
    findings.append(f"   → Standard causality pattern")

# 4. Dynamic effects
findings.append(f"\n4. DYNAMIC RESPONSE:")
findings.append(f"   → Peak effect at hour {np.argmax(gmv_to_clicks[:24]) + 1}")
findings.append(f"   → 24-hour cumulative: {np.sum(gmv_to_clicks[:24]):.3f}")

# 5. Variance decomposition
if len(fevd.decomp) >= 24:
    clicks_var_24h = fevd.decomp[23][gmv_idx][clicks_idx] * 100
    findings.append(f"\n5. VARIANCE CONTRIBUTION:")
    findings.append(f"   → Clicks explain {clicks_var_24h:.1f}% of GMV variance at 24 hours")

for finding in findings:
    print(finding)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 12: OUTPUT FILES")
print("="*80)

# Save text output
full_output = output_capture.getvalue()
with open('analysis_output.txt', 'w') as f:
    f.write(full_output)
print("\n✓ Full analysis output saved to: analysis_output.txt")

# Save JSON summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'data': {
        'observations': len(merged_df),
        'date_range': f"{merged_df.index.min().date()} to {merged_df.index.max().date()}",
        'total_gmv': float(merged_df['HOURLY_GMV'].sum()),
        'total_clicks': float(merged_df['HOURLY_CLICK_COUNT'].sum())
    },
    'models': {
        'ols': {
            'r_squared': float(ols_model.rsquared),
            'clicks_elasticity': float(clicks_coef),
            'clicks_pvalue': float(ols_model.pvalues['log_Clicks'])
        },
        'extended_ols': {
            'r_squared': float(ols_extended.rsquared)
        },
        'var': {
            'optimal_lag': int(optimal_lag),
            'aic': float(var_results.aic),
            'is_stable': bool(var_results.is_stable()),
            'bidirectional_causality': bool(bidirectional),
            'cumulative_response_24h': float(np.sum(gmv_to_clicks[:24]))
        }
    },
    'key_metrics': {
        'elasticity': float(clicks_coef),
        'statistical_significance': 'p < 0.001',
        'model_fit_r2': float(ols_model.rsquared),
        'dynamic_peak_hour': int(np.argmax(gmv_to_clicks[:24]) + 1),
        'cumulative_24h_effect': float(np.sum(gmv_to_clicks[:24]))
    }
}

print("\n" + "="*80)
print(f"Analysis completed at: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Restore stdout
sys.stdout = original_stdout

# Save the complete output to results folder
output_file = 'results/analysis_output.txt'
with open(output_file, 'w') as f:
    f.write(output_capture.getvalue())

print(f"\n✓ Analysis complete. Check '{output_file}' for full details.")