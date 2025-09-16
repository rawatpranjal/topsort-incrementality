#!/usr/bin/env python3
"""
Extract Fixed Effects from R fixest Model

This script runs the same vendor-week panel regression as in the notebooks
but extracts and saves the fixed effects for comparison with the deep learning model.
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
    """Load and prepare the vendor-week panel data."""

    print("\nLoading vendor-week panel data...")
    df = pd.read_parquet(filepath)

    # Convert types
    df['revenue_dollars'] = df['revenue_dollars'].astype(float)
    df['clicks'] = df['clicks'].astype(int)
    df['week'] = pd.to_datetime(df['week'])

    # Create log-transformed variables
    df['log_revenue_plus_1'] = np.log1p(df['revenue_dollars'])
    df['log_clicks_plus_1'] = np.log1p(df['clicks'])

    # Remove zero-click observations for consistency with DL model
    df_clean = df[df['clicks'] > 0].copy()

    print(f"Data shape: {df_clean.shape}")
    print(f"Unique vendors: {df_clean['vendor_id'].nunique()}")
    print(f"Unique weeks: {df_clean['week'].nunique()}")

    return df_clean


def run_fixest_model_and_extract_effects(df):
    """Run fixest model and extract all fixed effects and diagnostics."""

    print("\n" + "="*50)
    print("Running R fixest Model")
    print("="*50)

    # Convert to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['df_panel'] = df

    # Run the model and extract everything we need
    ro.r("""
    library(fixest)

    # Prepare data
    df_panel$week <- as.factor(df_panel$week)
    df_panel$vendor_id <- as.factor(df_panel$vendor_id)

    # Run the main fixed effects model (same as in notebooks)
    model <- feols(
        log_revenue_plus_1 ~ log_clicks_plus_1 | vendor_id + week,
        data = df_panel,
        vcov = ~vendor_id
    )

    # Print summary
    print(summary(model))

    # Extract main coefficient
    beta_main <- coef(model)["log_clicks_plus_1"]

    # Extract fixed effects
    fe_all <- fixef(model)
    vendor_fe <- fe_all$vendor_id
    week_fe <- fe_all$week

    # Get model diagnostics
    n_obs <- nobs(model)
    r_squared <- r2(model)["r2"]
    within_r2 <- r2(model)["wr2"]
    rmse <- sqrt(mean(residuals(model)^2))

    # Create dataframes for export
    vendor_fe_df <- data.frame(
        vendor_id = names(vendor_fe),
        fixed_effect = as.numeric(vendor_fe)
    )

    week_fe_df <- data.frame(
        week = names(week_fe),
        fixed_effect = as.numeric(week_fe)
    )

    # Get residuals and fitted values for additional diagnostics
    residuals_vec <- residuals(model)
    fitted_vec <- fitted(model)

    # Calculate log-likelihood (approximate)
    n <- length(residuals_vec)
    sigma2 <- sum(residuals_vec^2) / n
    log_lik <- -n/2 * log(2*pi) - n/2 * log(sigma2) - 1/(2*sigma2) * sum(residuals_vec^2)

    print(paste("Beta coefficient:", round(beta_main, 4)))
    print(paste("R-squared:", round(r_squared, 4)))
    print(paste("Within R-squared:", round(within_r2, 4)))
    print(paste("RMSE:", round(rmse, 4)))
    print(paste("Log-likelihood:", round(log_lik, 2)))
    print(paste("Number of vendor FEs:", length(vendor_fe)))
    print(paste("Number of week FEs:", length(week_fe)))
    """)

    # Extract results back to Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        vendor_fe_df = ro.r('vendor_fe_df')
        week_fe_df = ro.r('week_fe_df')

        # Get diagnostics
        diagnostics = {
            'beta': ro.r('beta_main')[0],
            'r_squared': ro.r('r_squared')[0],
            'within_r2': ro.r('within_r2')[0],
            'rmse': ro.r('rmse')[0],
            'log_likelihood': ro.r('log_lik')[0],
            'n_obs': ro.r('n_obs')[0],
            'residuals': np.array(ro.r('residuals_vec')),
            'fitted': np.array(ro.r('fitted_vec'))
        }

    return vendor_fe_df, week_fe_df, diagnostics


def save_results(vendor_fe_df, week_fe_df, diagnostics):
    """Save fixed effects and diagnostics to files."""

    # Save fixed effects
    vendor_fe_df.to_csv('r_vendor_fixed_effects.csv', index=False)
    week_fe_df.to_csv('r_week_fixed_effects.csv', index=False)

    # Save diagnostics
    diagnostics_df = pd.DataFrame([{
        'model': 'R_fixest',
        'beta': diagnostics['beta'],
        'r_squared': diagnostics['r_squared'],
        'within_r2': diagnostics['within_r2'],
        'rmse': diagnostics['rmse'],
        'log_likelihood': diagnostics['log_likelihood'],
        'n_obs': diagnostics['n_obs']
    }])
    diagnostics_df.to_csv('r_model_diagnostics.csv', index=False)

    # Save residuals and fitted values
    residuals_df = pd.DataFrame({
        'residuals': diagnostics['residuals'],
        'fitted': diagnostics['fitted']
    })
    residuals_df.to_csv('r_residuals_fitted.csv', index=False)

    print("\n" + "="*50)
    print("Results Saved:")
    print("="*50)
    print("✅ r_vendor_fixed_effects.csv")
    print("✅ r_week_fixed_effects.csv")
    print("✅ r_model_diagnostics.csv")
    print("✅ r_residuals_fitted.csv")


def print_summary_statistics(vendor_fe_df, week_fe_df, diagnostics):
    """Print summary statistics for comparison."""

    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)

    print(f"\nMain Effect:")
    print(f"  β (log_clicks): {diagnostics['beta']:.4f}")

    print(f"\nModel Fit:")
    print(f"  R²: {diagnostics['r_squared']:.4f}")
    print(f"  Within R²: {diagnostics['within_r2']:.4f}")
    print(f"  RMSE: {diagnostics['rmse']:.4f}")
    print(f"  Log-likelihood: {diagnostics['log_likelihood']:.2f}")

    print(f"\nVendor Fixed Effects:")
    vendor_fe_vals = vendor_fe_df['fixed_effect'].values
    print(f"  Count: {len(vendor_fe_vals)}")
    print(f"  Mean: {np.mean(vendor_fe_vals):.4f}")
    print(f"  Std: {np.std(vendor_fe_vals):.4f}")
    print(f"  Min: {np.min(vendor_fe_vals):.4f}")
    print(f"  25%: {np.percentile(vendor_fe_vals, 25):.4f}")
    print(f"  50%: {np.median(vendor_fe_vals):.4f}")
    print(f"  75%: {np.percentile(vendor_fe_vals, 75):.4f}")
    print(f"  Max: {np.max(vendor_fe_vals):.4f}")

    print(f"\nWeek Fixed Effects:")
    week_fe_vals = week_fe_df['fixed_effect'].values
    print(f"  Count: {len(week_fe_vals)}")
    print(f"  Mean: {np.mean(week_fe_vals):.4f}")
    print(f"  Std: {np.std(week_fe_vals):.4f}")
    print(f"  Min: {np.min(week_fe_vals):.4f}")
    print(f"  Max: {np.max(week_fe_vals):.4f}")


def main():
    """Main execution function."""

    print("="*50)
    print("EXTRACTING FIXED EFFECTS FROM R FIXEST MODEL")
    print("="*50)

    # Load data
    df_clean = load_and_prepare_data()

    # Run model and extract effects
    vendor_fe_df, week_fe_df, diagnostics = run_fixest_model_and_extract_effects(df_clean)

    # Save results
    save_results(vendor_fe_df, week_fe_df, diagnostics)

    # Print summary
    print_summary_statistics(vendor_fe_df, week_fe_df, diagnostics)

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print("\nFixed effects and diagnostics saved for comparison with deep learning model.")

    return vendor_fe_df, week_fe_df, diagnostics


if __name__ == "__main__":
    vendor_fe_df, week_fe_df, diagnostics = main()