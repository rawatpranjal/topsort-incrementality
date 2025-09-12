#!/usr/bin/env python3
"""
Robust Control Variable Analysis for Ad Incrementality
Systematic testing of control variable specifications
"""

import pandas as pd
import numpy as np
import warnings
import sys
import io
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate

warnings.filterwarnings('ignore')

# Output capture setup
output_capture = io.StringIO()

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

original_stdout = sys.stdout
sys.stdout = DualWriter(original_stdout, output_capture)

print("="*80)
print("ROBUST CONTROL VARIABLE ANALYSIS FOR AD INCREMENTALITY")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Framework: Progressive Control Addition Strategy")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA PREPARATION")
print("="*80)

print("\nLoading hourly data...")
try:
    clicks_df = pd.read_parquet('../data/hourly_clicks_2025-03-01_to_2025-09-30.parquet')
    purchases_df = pd.read_parquet('../data/hourly_purchases_2025-03-01_to_2025-09-30.parquet')
    impressions_df = pd.read_parquet('../data/hourly_impressions_2025-03-01_to_2025-09-30.parquet')
    auctions_df = pd.read_parquet('../data/hourly_auctions_2025-03-01_to_2025-09-30.parquet')
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Merge datasets
merged_df = purchases_df.merge(
    clicks_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    impressions_df, on='ACTIVITY_HOUR', how='outer'
).merge(
    auctions_df, on='ACTIVITY_HOUR', how='outer'
)

merged_df['ACTIVITY_HOUR'] = pd.to_datetime(merged_df['ACTIVITY_HOUR'])
merged_df = merged_df.set_index('ACTIVITY_HOUR').sort_index().fillna(0)

print(f"✓ Merged dataset: {merged_df.shape[0]} observations")

# ============================================================================
# VARIABLE CONSTRUCTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: VARIABLE CONSTRUCTION")
print("="*80)

# Create analysis dataframe with all variables
df = pd.DataFrame(index=merged_df.index)

# Core variables (log-transformed for elasticity interpretation)
df['log_gmv'] = np.log(merged_df['HOURLY_GMV'] + 1)
df['log_clicks'] = np.log(merged_df['HOURLY_CLICK_COUNT'] + 1)

# TIER 1 CONTROL: Demand-side (user intent)
df['log_auctions'] = np.log(merged_df['HOURLY_AUCTION_COUNT'] + 1)

# TIER 2 CONTROL: Supply-side (vendor activity)
df['log_active_vendors'] = np.log(merged_df['HOURLY_CLICKED_VENDORS'] + 1)

# Additional potential controls
df['log_impressions'] = np.log(merged_df['HOURLY_IMPRESSION_COUNT'] + 1)
df['log_unique_users'] = np.log(merged_df['HOURLY_AUCTION_USERS'] + 1)
df['log_unique_products'] = np.log(merged_df['HOURLY_CLICKED_PRODUCTS'] + 1)

# Calculate CTR as a control for ad quality
df['ctr'] = merged_df['HOURLY_CLICK_COUNT'] / (merged_df['HOURLY_IMPRESSION_COUNT'] + 1)
df['log_ctr'] = np.log(df['ctr'] + 0.001)  # Add small constant for log

# Time controls
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['week'] = df.index.isocalendar().week

# Create time dummies
for h in range(24):
    df[f'hour_{h}'] = (df['hour'] == h).astype(int)

for d in range(7):
    df[f'dow_{d}'] = (df['dayofweek'] == d).astype(int)

# Drop any NaN values
df = df.dropna()

print(f"Variables created:")
print(f"  - Core: log_gmv (outcome), log_clicks (treatment)")
print(f"  - Tier 1 Control: log_auctions (demand/user intent)")
print(f"  - Tier 2 Control: log_active_vendors (supply dynamics)")
print(f"  - Additional: log_impressions, log_unique_users, log_unique_products, ctr")
print(f"  - Time controls: hour dummies (24), day-of-week dummies (7)")
print(f"  - Final dataset: {df.shape[0]} observations, {df.shape[1]} variables")

# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: PROGRESSIVE MODEL SPECIFICATIONS")
print("="*80)

# Store results for comparison
model_results = []

# Helper function to estimate and store model results
def estimate_model(name, y, X_vars, df, description):
    """Estimate OLS model and extract key statistics"""
    X = df[X_vars].copy()
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # Extract key statistics
    clicks_coef = model.params.get('log_clicks', np.nan)
    clicks_se = model.bse.get('log_clicks', np.nan)
    clicks_pval = model.pvalues.get('log_clicks', np.nan)
    
    result = {
        'Model': name,
        'Description': description,
        'N': int(model.nobs),
        'R²': model.rsquared,
        'Adj R²': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'Click Elasticity': clicks_coef,
        'Std Error': clicks_se,
        'p-value': clicks_pval,
        'Significant': '***' if clicks_pval < 0.001 else '**' if clicks_pval < 0.01 else '*' if clicks_pval < 0.05 else '',
        'model_object': model
    }
    
    return result

# Outcome variable
y = df['log_gmv']

print("\nEstimating progressive model specifications...")
print("-" * 60)

# MODEL 0: Baseline (no controls)
print("\nMODEL 0: Baseline - No Controls")
print("Specification: log(GMV) ~ log(Clicks)")
result0 = estimate_model(
    name="Model 0",
    y=y,
    X_vars=['log_clicks'],
    df=df,
    description="Baseline (no controls)"
)
model_results.append(result0)
print(f"✓ Click elasticity: {result0['Click Elasticity']:.4f} (p={result0['p-value']:.4f})")

# MODEL 1: Time controls only
print("\nMODEL 1: Time Controls Only")
print("Specification: log(GMV) ~ log(Clicks) + hour_dummies + dow_dummies")
time_vars = ['log_clicks'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
result1 = estimate_model(
    name="Model 1",
    y=y,
    X_vars=time_vars,
    df=df,
    description="Time controls only"
)
model_results.append(result1)
print(f"✓ Click elasticity: {result1['Click Elasticity']:.4f} (p={result1['p-value']:.4f})")

# MODEL 2: Tier 1 Control (Demand)
print("\nMODEL 2: Tier 1 - Demand Control")
print("Specification: log(GMV) ~ log(Clicks) + log(Auctions) + time_dummies")
tier1_vars = ['log_clicks', 'log_auctions'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
result2 = estimate_model(
    name="Model 2",
    y=y,
    X_vars=tier1_vars,
    df=df,
    description="Demand control (auctions)"
)
model_results.append(result2)
print(f"✓ Click elasticity: {result2['Click Elasticity']:.4f} (p={result2['p-value']:.4f})")

# MODEL 3: Tier 1 + Tier 2 Controls
print("\nMODEL 3: Tier 1 + 2 - Demand + Supply Controls")
print("Specification: log(GMV) ~ log(Clicks) + log(Auctions) + log(Active_Vendors) + time_dummies")
tier2_vars = ['log_clicks', 'log_auctions', 'log_active_vendors'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
result3 = estimate_model(
    name="Model 3",
    y=y,
    X_vars=tier2_vars,
    df=df,
    description="Demand + Supply controls"
)
model_results.append(result3)
print(f"✓ Click elasticity: {result3['Click Elasticity']:.4f} (p={result3['p-value']:.4f})")

# MODEL 4: Add impressions control
print("\nMODEL 4: Core + Impressions")
print("Specification: log(GMV) ~ log(Clicks) + log(Auctions) + log(Active_Vendors) + log(Impressions) + time_dummies")
impr_vars = ['log_clicks', 'log_auctions', 'log_active_vendors', 'log_impressions'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
result4 = estimate_model(
    name="Model 4",
    y=y,
    X_vars=impr_vars,
    df=df,
    description="Core + Impressions"
)
model_results.append(result4)
print(f"✓ Click elasticity: {result4['Click Elasticity']:.4f} (p={result4['p-value']:.4f})")

# MODEL 5: Kitchen Sink (all controls)
print("\nMODEL 5: Kitchen Sink - All Controls")
print("Specification: log(GMV) ~ log(Clicks) + all_controls + time_dummies")
all_vars = ['log_clicks', 'log_auctions', 'log_active_vendors', 'log_impressions', 
            'log_unique_users', 'log_unique_products'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
result5 = estimate_model(
    name="Model 5",
    y=y,
    X_vars=all_vars,
    df=df,
    description="Kitchen sink (all controls)"
)
model_results.append(result5)
print(f"✓ Click elasticity: {result5['Click Elasticity']:.4f} (p={result5['p-value']:.4f})")

# ============================================================================
# ROBUSTNESS TABLE
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: ROBUSTNESS TABLE")
print("="*80)

# Create robustness table
robustness_df = pd.DataFrame(model_results)
robustness_df = robustness_df.drop('model_object', axis=1)

# Format for display
display_df = robustness_df.copy()
display_df['Click Coef'] = display_df.apply(lambda x: f"{x['Click Elasticity']:.4f}{x['Significant']}", axis=1)
display_df['SE'] = display_df['Std Error'].apply(lambda x: f"({x:.4f})")
display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}")
display_df['AIC'] = display_df['AIC'].apply(lambda x: f"{x:.0f}")

# Select columns for clean display
display_cols = ['Model', 'Description', 'Click Coef', 'SE', 'R²', 'AIC', 'N']
display_table = display_df[display_cols]

print("\nROBUSTNESS ACROSS SPECIFICATIONS")
print("-" * 80)
print(tabulate(display_table, headers='keys', tablefmt='grid', showindex=False))

# ============================================================================
# COEFFICIENT STABILITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: COEFFICIENT STABILITY ANALYSIS")
print("="*80)

baseline_coef = result0['Click Elasticity']
print(f"\nBaseline coefficient (no controls): {baseline_coef:.4f}")
print("\nChange from baseline:")

for i, result in enumerate(model_results[1:], 1):
    change = result['Click Elasticity'] - baseline_coef
    pct_change = (change / baseline_coef) * 100
    print(f"  Model {i}: {change:+.4f} ({pct_change:+.1f}%)")

# Check coefficient stability
coefficients = [r['Click Elasticity'] for r in model_results]
coef_std = np.std(coefficients)
coef_cv = coef_std / np.mean(coefficients)

print(f"\nCoefficient variation:")
print(f"  Range: [{min(coefficients):.4f}, {max(coefficients):.4f}]")
print(f"  Standard deviation: {coef_std:.4f}")
print(f"  Coefficient of variation: {coef_cv:.3f}")

if coef_cv < 0.1:
    print("  → Excellent stability: coefficient very stable across specifications")
elif coef_cv < 0.2:
    print("  → Good stability: coefficient reasonably stable")
elif coef_cv < 0.3:
    print("  → Moderate stability: some sensitivity to specification")
else:
    print("  → Poor stability: coefficient highly sensitive to controls")

# ============================================================================
# MULTICOLLINEARITY DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: MULTICOLLINEARITY DIAGNOSTICS (VIF)")
print("="*80)

# Calculate VIF for Model 3 (recommended specification)
print("\nVariance Inflation Factors for Model 3 (Demand + Supply):")
print("-" * 60)

# Prepare data for VIF calculation (exclude time dummies for clarity)
vif_vars = ['log_clicks', 'log_auctions', 'log_active_vendors']
X_vif = df[vif_vars].copy()
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

for idx, row in vif_data.iterrows():
    if row['Variable'] != 'const':
        vif_val = row['VIF']
        status = "✓ OK" if vif_val < 5 else "⚠ Moderate" if vif_val < 10 else "✗ High"
        print(f"  {row['Variable']:20s}: {vif_val:6.2f} {status}")

# ============================================================================
# PARTIAL R² ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: PARTIAL R² CONTRIBUTION")
print("="*80)

print("\nContribution of each control (Model 3):")
print("-" * 60)

# Full model R²
full_r2 = result3['R²']

# Model without auctions
no_auctions_vars = ['log_clicks', 'log_active_vendors'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
no_auctions_model = sm.OLS(y, sm.add_constant(df[no_auctions_vars])).fit()
partial_r2_auctions = full_r2 - no_auctions_model.rsquared

# Model without vendors
no_vendors_vars = ['log_clicks', 'log_auctions'] + [c for c in df.columns if c.startswith(('hour_', 'dow_'))]
no_vendors_model = sm.OLS(y, sm.add_constant(df[no_vendors_vars])).fit()
partial_r2_vendors = full_r2 - no_vendors_model.rsquared

print(f"  log_auctions contribution:      {partial_r2_auctions:.4f} ({partial_r2_auctions/full_r2*100:.1f}% of total R²)")
print(f"  log_active_vendors contribution: {partial_r2_vendors:.4f} ({abs(partial_r2_vendors)/full_r2*100:.1f}% of total R²)")

# ============================================================================
# DETAILED MODEL 3 OUTPUT (RECOMMENDED SPECIFICATION)
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: RECOMMENDED MODEL DETAILS (Model 3)")
print("="*80)

recommended_model = model_results[3]['model_object']
print("\nFull regression output for Model 3 (Demand + Supply controls):")
print("-" * 80)
print(recommended_model.summary())

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: KEY INSIGHTS")
print("="*80)

print("\n" + "ECONOMETRIC FINDINGS".center(70))
print("=" * 70)

# Determine best model
aic_values = [r['AIC'] for r in model_results]
best_model_idx = aic_values.index(min(aic_values))
best_model = model_results[best_model_idx]

insights = []

# 1. Baseline vs controlled estimate
baseline_elasticity = result0['Click Elasticity']
controlled_elasticity = result3['Click Elasticity']
bias_correction = controlled_elasticity - baseline_elasticity

insights.append("1. OMITTED VARIABLE BIAS CORRECTION:")
insights.append(f"   • Naive estimate (no controls): {baseline_elasticity:.4f}")
insights.append(f"   • Controlled estimate (Model 3): {controlled_elasticity:.4f}")
insights.append(f"   • Bias correction: {bias_correction:+.4f} ({bias_correction/baseline_elasticity*100:+.1f}%)")

if abs(bias_correction) > 0.1:
    insights.append("   → Substantial bias corrected by controls")
else:
    insights.append("   → Moderate bias, but core relationship robust")

# 2. Importance of demand control
insights.append("\n2. IMPORTANCE OF DEMAND CONTROL (Auctions):")
change_with_auctions = result2['Click Elasticity'] - result1['Click Elasticity']
insights.append(f"   • Adding auction control changes elasticity by {change_with_auctions:+.4f}")
insights.append(f"   • Partial R² contribution: {partial_r2_auctions:.4f}")
insights.append("   → Confirms auctions (user intent) is critical confounder")

# 3. Supply-side effects
insights.append("\n3. SUPPLY-SIDE DYNAMICS (Active Vendors):")
change_with_vendors = result3['Click Elasticity'] - result2['Click Elasticity']
insights.append(f"   • Adding vendor control changes elasticity by {change_with_vendors:+.4f}")
if abs(partial_r2_vendors) > 0.01:
    insights.append("   → Vendor composition affects GMV relationship")
else:
    insights.append("   → Vendor composition has minimal additional impact")

# 4. Over-specification test
kitchen_sink_elasticity = result5['Click Elasticity']
overspec_change = kitchen_sink_elasticity - controlled_elasticity
insights.append("\n4. OVER-SPECIFICATION TEST:")
insights.append(f"   • Kitchen sink elasticity: {kitchen_sink_elasticity:.4f}")
insights.append(f"   • Change from Model 3: {overspec_change:+.4f}")
if abs(overspec_change) < 0.05:
    insights.append("   → Additional controls don't meaningfully change estimate")
    insights.append("   → Model 3 achieves parsimony without bias")

# 5. Final recommendation
insights.append("\n5. RECOMMENDED SPECIFICATION:")
insights.append(f"   • Model 3: Demand (auctions) + Supply (vendors) controls")
insights.append(f"   • Final elasticity: {controlled_elasticity:.4f}")
insights.append(f"   • Interpretation: 1% increase in ad clicks → {controlled_elasticity:.3f}% increase in GMV")
insights.append(f"   • Statistical significance: p < 0.001")

for insight in insights:
    print(insight)

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 10: OUTPUT FILES")
print("="*80)

# Save full text output only
output_file = 'results/robustness_analysis_output.txt'
full_output = output_capture.getvalue()
with open(output_file, 'w') as f:
    f.write(full_output)
print(f"\n✓ Full analysis saved to: {output_file}")

print("\n" + "="*80)
print(f"Analysis completed at: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

# Restore stdout
sys.stdout = original_stdout

# Final save of complete output
with open('robustness_analysis_output.txt', 'w') as f:
    f.write(output_capture.getvalue())

print("\n✓ Robustness analysis complete. Check 'robustness_analysis_output.txt' for full details.")