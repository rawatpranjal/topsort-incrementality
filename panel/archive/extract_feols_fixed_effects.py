#!/usr/bin/env python3
"""
Extract Fixed Effects from R feols Model

This script runs the EXACT same vendor-week panel regression as in vendor_week.ipynb
using feols (Fixed Effects OLS) and extracts the fixed effects for comparison.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # Import R packages
    base = importr('base')
    fixest = importr('fixest')
    print("✅ R environment and fixest loaded successfully")

except ImportError as e:
    print(f"❌ Error loading R environment: {e}")
    print("Please ensure R and rpy2 are installed")
    sys.exit(1)


def load_and_prepare_data(filepath='vendor_panel_full_history_clicks_only.parquet'):
    """Load and prepare the vendor-week panel data exactly as in the notebook."""

    print("\nLoading vendor-week panel data...")
    df = pd.read_parquet(filepath)

    # Convert types exactly as in notebook
    df['revenue_dollars'] = df['revenue_dollars'].astype(float)
    df['clicks'] = df['clicks'].astype(int)
    df['purchases'] = df['purchases'].astype(int)
    df['week'] = pd.to_datetime(df['week'])

    # Create log-transformed variables
    df['log_revenue_plus_1'] = np.log1p(df['revenue_dollars'])
    df['log_clicks_plus_1'] = np.log1p(df['clicks'])

    print(f"Data shape: {df.shape}")
    print(f"Unique vendors: {df['vendor_id'].nunique()}")
    print(f"Unique weeks: {df['week'].nunique()}")
    print(f"Date range: {df['week'].min()} to {df['week'].max()}")

    # Important: The notebook uses ALL data, including zero-click observations
    # This is different from the deep learning model which filters clicks > 0
    print(f"Total observations: {len(df):,}")
    print(f"Zero-click observations: {(df['clicks'] == 0).sum():,}")

    return df


def run_feols_and_extract_fixed_effects(df):
    """Run the exact feols model from the notebook and extract fixed effects."""

    print("\n" + "="*50)
    print("Running R feols Model (Exact Notebook Specification)")
    print("="*50)

    # Convert to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['df_panel_full'] = df

    # Run the EXACT model from the notebook
    print("\nModel: log_revenue_plus_1 ~ log_clicks_plus_1 | vendor_id + week")
    print("Standard Errors: Clustered by vendor_id")

    ro.r("""
    library(fixest)

    # Convert to factors for fixed effects
    df_panel_full$week <- as.factor(df_panel_full$week)
    df_panel_full$vendor_id <- as.factor(df_panel_full$vendor_id)

    # Run the exact model from vendor_week.ipynb
    model_main <- feols(
        log_revenue_plus_1 ~ log_clicks_plus_1 | vendor_id + week,
        data = df_panel_full,
        vcov = ~vendor_id
    )

    # Print model summary (matches notebook output)
    print(etable(model_main, digits = 4))

    # Extract main coefficient
    beta_main <- coef(model_main)["log_clicks_plus_1"]

    # Extract fixed effects using fixef()
    fe_all <- fixef(model_main)
    vendor_fe <- fe_all$vendor_id
    week_fe <- fe_all$week

    # Get model statistics
    n_obs <- nobs(model_main)
    r2_stats <- r2(model_main)
    r_squared <- r2_stats["r2"]
    within_r2 <- r2_stats["wr2"]

    # Calculate RMSE
    residuals_vec <- residuals(model_main)
    rmse <- sqrt(mean(residuals_vec^2))

    # Get fitted values
    fitted_vec <- fitted(model_main)

    # Calculate log-likelihood
    n <- length(residuals_vec)
    sigma2 <- sum(residuals_vec^2) / n
    log_lik <- -n/2 * log(2*pi) - n/2 * log(sigma2) - 1/(2*sigma2) * sum(residuals_vec^2)

    # Create dataframes for export
    vendor_fe_df <- data.frame(
        vendor_id = names(vendor_fe),
        fixed_effect = as.numeric(vendor_fe),
        stringsAsFactors = FALSE
    )

    week_fe_df <- data.frame(
        week = names(week_fe),
        fixed_effect = as.numeric(week_fe),
        stringsAsFactors = FALSE
    )

    # Print diagnostics
    cat("\n")
    cat("="*50, "\n")
    cat("MODEL DIAGNOSTICS\n")
    cat("="*50, "\n")
    cat(sprintf("Beta coefficient: %.6f\n", beta_main))
    cat(sprintf("R-squared: %.6f\n", r_squared))
    cat(sprintf("Within R-squared: %.6f\n", within_r2))
    cat(sprintf("RMSE: %.6f\n", rmse))
    cat(sprintf("Log-likelihood: %.2f\n", log_lik))
    cat(sprintf("Number of observations: %d\n", n_obs))
    cat(sprintf("Number of vendor FEs: %d\n", length(vendor_fe)))
    cat(sprintf("Number of week FEs: %d\n", length(week_fe)))
    """)

    # Extract results back to Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        vendor_fe_df = ro.r('vendor_fe_df')
        week_fe_df = ro.r('week_fe_df')

        # Get diagnostics
        diagnostics = {
            'beta': float(ro.r('beta_main')[0]),
            'r_squared': float(ro.r('r_squared')[0]),
            'within_r2': float(ro.r('within_r2')[0]),
            'rmse': float(ro.r('rmse')[0]),
            'log_likelihood': float(ro.r('log_lik')[0]),
            'n_obs': int(ro.r('n_obs')[0]),
            'residuals': np.array(ro.r('residuals_vec')),
            'fitted': np.array(ro.r('fitted_vec'))
        }

    return vendor_fe_df, week_fe_df, diagnostics


def save_results(vendor_fe_df, week_fe_df, diagnostics):
    """Save fixed effects and diagnostics to CSV files."""

    # Save vendor fixed effects
    vendor_fe_df.to_csv('feols_vendor_fixed_effects.csv', index=False)
    print(f"\n✅ Saved feols_vendor_fixed_effects.csv ({len(vendor_fe_df)} vendor FEs)")

    # Save week fixed effects
    week_fe_df.to_csv('feols_week_fixed_effects.csv', index=False)
    print(f"✅ Saved feols_week_fixed_effects.csv ({len(week_fe_df)} week FEs)")

    # Save model diagnostics
    diagnostics_df = pd.DataFrame([{
        'model': 'R_feols',
        'beta': diagnostics['beta'],
        'r_squared': diagnostics['r_squared'],
        'within_r2': diagnostics['within_r2'],
        'rmse': diagnostics['rmse'],
        'log_likelihood': diagnostics['log_likelihood'],
        'n_obs': diagnostics['n_obs']
    }])
    diagnostics_df.to_csv('feols_model_diagnostics.csv', index=False)
    print("✅ Saved feols_model_diagnostics.csv")

    # Save residuals and fitted values
    residuals_df = pd.DataFrame({
        'residuals': diagnostics['residuals'],
        'fitted': diagnostics['fitted']
    })
    residuals_df.to_csv('feols_residuals_fitted.csv', index=False)
    print(f"✅ Saved feols_residuals_fitted.csv ({len(residuals_df)} observations)")


def print_fixed_effects_summary(vendor_fe_df, week_fe_df):
    """Print summary statistics of fixed effects for comparison."""

    print("\n" + "="*50)
    print("FIXED EFFECTS SUMMARY STATISTICS")
    print("="*50)

    print("\nVendor Fixed Effects:")
    vendor_fe_vals = vendor_fe_df['fixed_effect'].values
    print(f"  Count: {len(vendor_fe_vals)}")
    print(f"  Mean: {np.mean(vendor_fe_vals):.6f}")
    print(f"  Std: {np.std(vendor_fe_vals):.6f}")
    print(f"  Min: {np.min(vendor_fe_vals):.6f}")
    print(f"  25%: {np.percentile(vendor_fe_vals, 25):.6f}")
    print(f"  50%: {np.median(vendor_fe_vals):.6f}")
    print(f"  75%: {np.percentile(vendor_fe_vals, 75):.6f}")
    print(f"  Max: {np.max(vendor_fe_vals):.6f}")

    print("\nWeek Fixed Effects:")
    week_fe_vals = week_fe_df['fixed_effect'].values
    print(f"  Count: {len(week_fe_vals)}")
    print(f"  Mean: {np.mean(week_fe_vals):.6f}")
    print(f"  Std: {np.std(week_fe_vals):.6f}")
    print(f"  Min: {np.min(week_fe_vals):.6f}")
    print(f"  25%: {np.percentile(week_fe_vals, 25):.6f}")
    print(f"  50%: {np.median(week_fe_vals):.6f}")
    print(f"  75%: {np.percentile(week_fe_vals, 75):.6f}")
    print(f"  Max: {np.max(week_fe_vals):.6f}")


def main():
    """Main execution function."""

    print("="*50)
    print("EXTRACTING FIXED EFFECTS FROM R FEOLS MODEL")
    print("="*50)

    # Load data
    df_full = load_and_prepare_data()

    # Run model and extract effects
    vendor_fe_df, week_fe_df, diagnostics = run_feols_and_extract_fixed_effects(df_full)

    # Save results
    save_results(vendor_fe_df, week_fe_df, diagnostics)

    # Print summary
    print_fixed_effects_summary(vendor_fe_df, week_fe_df)

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print("\nFixed effects saved. Ready for comparison with deep learning model.")
    print("\nNote: This model includes ALL observations (including zero clicks)")
    print("while the deep learning model filters to clicks > 0 only.")

    return vendor_fe_df, week_fe_df, diagnostics


if __name__ == "__main__":
    vendor_fe_df, week_fe_df, diagnostics = main()