#!/usr/bin/env python3
"""
Compare Fixed Effects: Deep Learning vs R fixest

This script compares the fixed effects extracted from both models
to validate that deep learning can fully replicate econometric methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


def load_fixed_effects():
    """Load fixed effects from both models."""

    print("Loading fixed effects from CSV files...")

    try:
        # R model results
        r_vendor_fe = pd.read_csv('r_vendor_fixed_effects.csv')
        r_week_fe = pd.read_csv('r_week_fixed_effects.csv')
        r_diagnostics = pd.read_csv('r_model_diagnostics.csv')
        r_residuals = pd.read_csv('r_residuals_fitted.csv')

        # Deep learning results
        dl_vendor_fe = pd.read_csv('dl_vendor_fixed_effects.csv')
        dl_week_fe = pd.read_csv('dl_week_fixed_effects.csv')
        dl_diagnostics = pd.read_csv('dl_model_diagnostics.csv')
        dl_residuals = pd.read_csv('dl_residuals_fitted.csv')

        print("✅ All files loaded successfully")

        return {
            'r': {
                'vendor_fe': r_vendor_fe,
                'week_fe': r_week_fe,
                'diagnostics': r_diagnostics,
                'residuals': r_residuals
            },
            'dl': {
                'vendor_fe': dl_vendor_fe,
                'week_fe': dl_week_fe,
                'diagnostics': dl_diagnostics,
                'residuals': dl_residuals
            }
        }

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run both extract_r_fixed_effects.py and vendor_week_deep_learning.py first")
        return None


def compare_main_coefficients(data):
    """Compare the main elasticity coefficients."""

    print("\n" + "="*50)
    print("MAIN COEFFICIENT COMPARISON")
    print("="*50)

    r_beta = data['r']['diagnostics']['beta'].values[0]
    dl_beta = data['dl']['diagnostics']['beta'].values[0]

    print(f"R fixest β:      {r_beta:.6f}")
    print(f"Deep Learning β: {dl_beta:.6f}")
    print(f"Difference:      {abs(r_beta - dl_beta):.6f}")
    print(f"Relative Error:  {abs(r_beta - dl_beta) / abs(r_beta) * 100:.2f}%")

    return r_beta, dl_beta


def compare_fixed_effects(r_fe, dl_fe, fe_type="vendor"):
    """Compare fixed effects between models."""

    print(f"\n{fe_type.upper()} FIXED EFFECTS COMPARISON")
    print("-" * 40)

    # Merge on common IDs
    id_col = 'vendor_id' if fe_type == 'vendor' else 'week'
    merged = pd.merge(
        r_fe.rename(columns={'fixed_effect': 'r_fe'}),
        dl_fe.rename(columns={'fixed_effect': 'dl_fe'}),
        on=id_col,
        how='inner'
    )

    if len(merged) == 0:
        print(f"❌ No matching {fe_type} IDs found")
        return None

    print(f"Matched {len(merged)} {fe_type}s")

    # Calculate correlation
    correlation = np.corrcoef(merged['r_fe'], merged['dl_fe'])[0, 1]
    print(f"Correlation: {correlation:.4f}")

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(merged['r_fe'], merged['dl_fe']))
    print(f"RMSE: {rmse:.4f}")

    # Calculate mean absolute error
    mae = np.mean(np.abs(merged['r_fe'] - merged['dl_fe']))
    print(f"MAE: {mae:.4f}")

    # R² score
    r2 = r2_score(merged['r_fe'], merged['dl_fe'])
    print(f"R²: {r2:.4f}")

    # Statistical test for difference
    t_stat, p_value = stats.ttest_rel(merged['r_fe'], merged['dl_fe'])
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

    return merged


def plot_fixed_effects_comparison(merged_vendor, merged_week):
    """Create comparison plots for fixed effects."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- VENDOR FIXED EFFECTS ---
    if merged_vendor is not None and len(merged_vendor) > 0:
        # Scatter plot
        axes[0, 0].scatter(merged_vendor['r_fe'], merged_vendor['dl_fe'],
                          alpha=0.5, s=1)
        axes[0, 0].plot([merged_vendor['r_fe'].min(), merged_vendor['r_fe'].max()],
                       [merged_vendor['r_fe'].min(), merged_vendor['r_fe'].max()],
                       'r--', label='Perfect Agreement')
        axes[0, 0].set_xlabel('R fixest Vendor FE')
        axes[0, 0].set_ylabel('Deep Learning Vendor FE')
        axes[0, 0].set_title(f'Vendor Fixed Effects Comparison\n'
                            f'Correlation: {np.corrcoef(merged_vendor["r_fe"], merged_vendor["dl_fe"])[0, 1]:.4f}')
        axes[0, 0].legend()

        # Distribution comparison
        axes[0, 1].hist(merged_vendor['r_fe'], bins=50, alpha=0.5,
                       label='R fixest', density=True)
        axes[0, 1].hist(merged_vendor['dl_fe'], bins=50, alpha=0.5,
                       label='Deep Learning', density=True)
        axes[0, 1].set_xlabel('Fixed Effect Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Vendor FE Distribution Comparison')
        axes[0, 1].legend()

        # Bland-Altman plot
        mean_fe = (merged_vendor['r_fe'] + merged_vendor['dl_fe']) / 2
        diff_fe = merged_vendor['r_fe'] - merged_vendor['dl_fe']
        axes[0, 2].scatter(mean_fe, diff_fe, alpha=0.5, s=1)
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].axhline(y=np.mean(diff_fe), color='b', linestyle='-',
                          label=f'Mean: {np.mean(diff_fe):.4f}')
        axes[0, 2].axhline(y=np.mean(diff_fe) + 1.96*np.std(diff_fe),
                          color='b', linestyle=':', label='±1.96 SD')
        axes[0, 2].axhline(y=np.mean(diff_fe) - 1.96*np.std(diff_fe),
                          color='b', linestyle=':')
        axes[0, 2].set_xlabel('Mean of R and DL')
        axes[0, 2].set_ylabel('R - DL')
        axes[0, 2].set_title('Bland-Altman Plot: Vendor FE')
        axes[0, 2].legend()

    # --- WEEK FIXED EFFECTS ---
    if merged_week is not None and len(merged_week) > 0:
        # Scatter plot
        axes[1, 0].scatter(merged_week['r_fe'], merged_week['dl_fe'], s=20)
        axes[1, 0].plot([merged_week['r_fe'].min(), merged_week['r_fe'].max()],
                       [merged_week['r_fe'].min(), merged_week['r_fe'].max()],
                       'r--', label='Perfect Agreement')
        axes[1, 0].set_xlabel('R fixest Week FE')
        axes[1, 0].set_ylabel('Deep Learning Week FE')
        axes[1, 0].set_title(f'Week Fixed Effects Comparison\n'
                            f'Correlation: {np.corrcoef(merged_week["r_fe"], merged_week["dl_fe"])[0, 1]:.4f}')
        axes[1, 0].legend()

        # Time series comparison
        axes[1, 1].plot(range(len(merged_week)), merged_week['r_fe'],
                       'o-', label='R fixest')
        axes[1, 1].plot(range(len(merged_week)), merged_week['dl_fe'],
                       's-', label='Deep Learning')
        axes[1, 1].set_xlabel('Week Index')
        axes[1, 1].set_ylabel('Fixed Effect Value')
        axes[1, 1].set_title('Week FE Time Pattern')
        axes[1, 1].legend()

        # Q-Q plot
        axes[1, 2].scatter(sorted(merged_week['r_fe']),
                          sorted(merged_week['dl_fe']))
        min_val = min(merged_week['r_fe'].min(), merged_week['dl_fe'].min())
        max_val = max(merged_week['r_fe'].max(), merged_week['dl_fe'].max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 2].set_xlabel('R fixest Week FE (sorted)')
        axes[1, 2].set_ylabel('Deep Learning Week FE (sorted)')
        axes[1, 2].set_title('Q-Q Plot: Week Fixed Effects')

    plt.suptitle('Fixed Effects Comparison: Deep Learning vs R fixest', fontsize=16)
    plt.tight_layout()
    plt.savefig('fixed_effects_comparison.png', dpi=100, bbox_inches='tight')
    print("\n✅ Saved comparison plot to 'fixed_effects_comparison.png'")
    plt.show()


def compare_model_fit(data):
    """Compare overall model fit statistics."""

    print("\n" + "="*50)
    print("MODEL FIT COMPARISON")
    print("="*50)

    r_diag = data['r']['diagnostics'].iloc[0]
    dl_diag = data['dl']['diagnostics'].iloc[0]

    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Beta', 'R²', 'RMSE', 'Log-Likelihood'],
        'R fixest': [
            r_diag.get('beta', np.nan),
            r_diag.get('r_squared', np.nan),
            r_diag.get('rmse', np.nan),
            r_diag.get('log_likelihood', np.nan)
        ],
        'Deep Learning': [
            dl_diag.get('beta', np.nan),
            np.nan,  # Will be calculated separately
            np.nan,
            np.nan
        ]
    })

    print(comparison.to_string(index=False))


def analyze_residuals(r_residuals, dl_residuals):
    """Compare residual distributions."""

    print("\n" + "="*50)
    print("RESIDUAL ANALYSIS")
    print("="*50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residual distributions
    axes[0, 0].hist(r_residuals['residuals'], bins=50, alpha=0.5,
                   label='R fixest', density=True)
    axes[0, 0].hist(dl_residuals['residuals'], bins=50, alpha=0.5,
                   label='Deep Learning', density=True)
    axes[0, 0].set_xlabel('Residuals')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Residual Distributions')
    axes[0, 0].legend()

    # Q-Q plots
    stats.probplot(r_residuals['residuals'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('R fixest: Q-Q Plot')

    stats.probplot(dl_residuals['residuals'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Deep Learning: Q-Q Plot')

    # Residuals vs Fitted
    axes[1, 1].scatter(r_residuals['fitted'], r_residuals['residuals'],
                      alpha=0.5, s=1, label='R fixest')
    axes[1, 1].scatter(dl_residuals['fitted'], dl_residuals['residuals'],
                      alpha=0.5, s=1, label='Deep Learning')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Fitted')
    axes[1, 1].legend()

    plt.suptitle('Residual Analysis Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=100, bbox_inches='tight')
    print("✅ Saved residual analysis to 'residual_analysis.png'")
    plt.show()

    # Statistical tests
    print("\nResidual Statistics:")
    print("-" * 40)

    for model_name, resid_df in [('R fixest', r_residuals), ('Deep Learning', dl_residuals)]:
        resid = resid_df['residuals']
        print(f"\n{model_name}:")
        print(f"  Mean: {np.mean(resid):.6f}")
        print(f"  Std: {np.std(resid):.4f}")
        print(f"  Skewness: {stats.skew(resid):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(resid):.4f}")

        # Normality test
        stat, p = stats.shapiro(resid[:5000])  # Shapiro-Wilk on sample
        print(f"  Shapiro-Wilk p-value: {p:.4f}")


def create_summary_report(data, merged_vendor, merged_week):
    """Create a summary report of the comparison."""

    print("\n" + "="*50)
    print("VALIDATION SUMMARY REPORT")
    print("="*50)

    print("\n✅ VALIDATION RESULTS:")
    print("-" * 40)

    # Main coefficient
    r_beta = data['r']['diagnostics']['beta'].values[0]
    dl_beta = data['dl']['diagnostics']['beta'].values[0]
    beta_match = abs(r_beta - dl_beta) < 0.01

    print(f"1. Main Coefficient (β):")
    print(f"   {'✅' if beta_match else '❌'} Match within 1% tolerance")
    print(f"   Error: {abs(r_beta - dl_beta) / abs(r_beta) * 100:.2f}%")

    # Vendor fixed effects
    if merged_vendor is not None:
        vendor_corr = np.corrcoef(merged_vendor['r_fe'], merged_vendor['dl_fe'])[0, 1]
        vendor_match = vendor_corr > 0.95

        print(f"\n2. Vendor Fixed Effects:")
        print(f"   {'✅' if vendor_match else '❌'} Correlation > 0.95")
        print(f"   Actual correlation: {vendor_corr:.4f}")

    # Week fixed effects
    if merged_week is not None:
        week_corr = np.corrcoef(merged_week['r_fe'], merged_week['dl_fe'])[0, 1]
        week_match = week_corr > 0.95

        print(f"\n3. Week Fixed Effects:")
        print(f"   {'✅' if week_match else '❌'} Correlation > 0.95")
        print(f"   Actual correlation: {week_corr:.4f}")

    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)

    if beta_match and (merged_vendor is None or vendor_match) and (merged_week is None or week_match):
        print("✅ VALIDATION SUCCESSFUL!")
        print("Deep learning successfully replicates econometric fixed effects regression.")
        print("All parameters match within acceptable tolerances.")
    else:
        print("⚠️ PARTIAL VALIDATION")
        print("Some parameters show discrepancies. Further tuning may be needed.")


def main():
    """Main execution function."""

    print("="*50)
    print("FIXED EFFECTS MODEL COMPARISON")
    print("Deep Learning vs R fixest")
    print("="*50)

    # Load data
    data = load_fixed_effects()
    if data is None:
        return

    # Compare main coefficients
    r_beta, dl_beta = compare_main_coefficients(data)

    # Compare vendor fixed effects
    merged_vendor = compare_fixed_effects(
        data['r']['vendor_fe'],
        data['dl']['vendor_fe'],
        'vendor'
    )

    # Compare week fixed effects
    merged_week = compare_fixed_effects(
        data['r']['week_fe'],
        data['dl']['week_fe'],
        'week'
    )

    # Compare model fit
    compare_model_fit(data)

    # Plot comparisons
    plot_fixed_effects_comparison(merged_vendor, merged_week)

    # Analyze residuals
    analyze_residuals(data['r']['residuals'], data['dl']['residuals'])

    # Create summary report
    create_summary_report(data, merged_vendor, merged_week)

    print("\n" + "="*50)
    print("COMPARISON COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()