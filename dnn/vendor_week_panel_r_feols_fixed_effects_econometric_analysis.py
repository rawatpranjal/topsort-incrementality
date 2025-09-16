#!/usr/bin/env python3
"""
Vendor-Week Panel Analysis using R feols
Outputs all results to vendor_week_panel_feols_results.txt
"""

import pandas as pd
import numpy as np
import sys
import os

# Redirect all output to file
output_file = open('results/vendor_week_panel_feols_results.txt', 'w')
original_stdout = sys.stdout
sys.stdout = output_file

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    base = importr('base')
    fixest = importr('fixest')
    print("R ENVIRONMENT LOADED")
    print("="*80)

except ImportError as e:
    print(f"ERROR LOADING R ENVIRONMENT: {e}")
    sys.exit(1)

# Load data
print("LOADING DATA: vendor_panel_full_history_clicks_only.parquet")
df = pd.read_parquet('data/vendor_panel_full_history_clicks_only.parquet')

print(f"DATA SHAPE: {df.shape}")
print(f"COLUMNS: {list(df.columns)}")
print(f"MEMORY USAGE: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("="*80)

# Data type conversion
df['revenue_dollars'] = df['revenue_dollars'].astype(float)
df['clicks'] = df['clicks'].astype(int)
df['purchases'] = df['purchases'].astype(int)
df['week'] = pd.to_datetime(df['week'])

# Create log variables
df['log_revenue_plus_1'] = np.log1p(df['revenue_dollars'])
df['log_clicks_plus_1'] = np.log1p(df['clicks'])

print("DATA SUMMARY STATISTICS")
print("="*80)
print(df.describe())
print("="*80)

print("\nDATE RANGE")
print(f"MIN DATE: {df['week'].min()}")
print(f"MAX DATE: {df['week'].max()}")
print(f"UNIQUE WEEKS: {df['week'].nunique()}")
print(f"UNIQUE VENDORS: {df['vendor_id'].nunique()}")
print("="*80)

print("\nZERO COUNTS")
print(f"ZERO CLICKS: {(df['clicks'] == 0).sum()} ({(df['clicks'] == 0).mean()*100:.2f}%)")
print(f"ZERO REVENUE: {(df['revenue_dollars'] == 0).sum()} ({(df['revenue_dollars'] == 0).mean()*100:.2f}%)")
print(f"ZERO PURCHASES: {(df['purchases'] == 0).sum()} ({(df['purchases'] == 0).mean()*100:.2f}%)")
print("="*80)

# Convert to R
with localconverter(ro.default_converter + pandas2ri.converter):
    ro.globalenv['df_panel'] = df

# Run feols model
print("\nRUNNING FEOLS MODEL")
print("="*80)

ro.r("""
library(fixest)

df_panel$week <- as.factor(df_panel$week)
df_panel$vendor_id <- as.factor(df_panel$vendor_id)

cat("MODEL SPECIFICATION: log_revenue_plus_1 ~ log_clicks_plus_1 | vendor_id + week\n")
cat("STANDARD ERRORS: Clustered by vendor_id\n")
cat("="*80, "\n")

model <- feols(
    log_revenue_plus_1 ~ log_clicks_plus_1 | vendor_id + week,
    data = df_panel,
    vcov = ~vendor_id
)

cat("\nMODEL SUMMARY\n")
cat("="*80, "\n")
print(summary(model))

cat("\nMODEL TABLE\n")
cat("="*80, "\n")
print(etable(model, digits = 6))

beta_main <- coef(model)["log_clicks_plus_1"]
cat("\nMAIN COEFFICIENT (BETA)\n")
cat("="*80, "\n")
cat(sprintf("log_clicks_plus_1 coefficient: %.10f\n", beta_main))

fe_all <- fixef(model)
vendor_fe <- fe_all$vendor_id
week_fe <- fe_all$week

cat("\nFIXED EFFECTS COUNTS\n")
cat("="*80, "\n")
cat(sprintf("Number of vendor fixed effects: %d\n", length(vendor_fe)))
cat(sprintf("Number of week fixed effects: %d\n", length(week_fe)))

cat("\nVENDOR FIXED EFFECTS SUMMARY\n")
cat("="*80, "\n")
cat(sprintf("Mean: %.10f\n", mean(vendor_fe)))
cat(sprintf("Std Dev: %.10f\n", sd(vendor_fe)))
cat(sprintf("Min: %.10f\n", min(vendor_fe)))
cat(sprintf("Q1: %.10f\n", quantile(vendor_fe, 0.25)))
cat(sprintf("Median: %.10f\n", median(vendor_fe)))
cat(sprintf("Q3: %.10f\n", quantile(vendor_fe, 0.75)))
cat(sprintf("Max: %.10f\n", max(vendor_fe)))

cat("\nWEEK FIXED EFFECTS SUMMARY\n")
cat("="*80, "\n")
cat(sprintf("Mean: %.10f\n", mean(week_fe)))
cat(sprintf("Std Dev: %.10f\n", sd(week_fe)))
cat(sprintf("Min: %.10f\n", min(week_fe)))
cat(sprintf("Q1: %.10f\n", quantile(week_fe, 0.25)))
cat(sprintf("Median: %.10f\n", median(week_fe)))
cat(sprintf("Q3: %.10f\n", quantile(week_fe, 0.75)))
cat(sprintf("Max: %.10f\n", max(week_fe)))

cat("\nFIRST 20 VENDOR FIXED EFFECTS\n")
cat("="*80, "\n")
for(i in 1:min(20, length(vendor_fe))) {
    cat(sprintf("%s: %.10f\n", names(vendor_fe)[i], vendor_fe[i]))
}

cat("\nALL WEEK FIXED EFFECTS\n")
cat("="*80, "\n")
for(i in 1:length(week_fe)) {
    cat(sprintf("%s: %.10f\n", names(week_fe)[i], week_fe[i]))
}

n_obs <- nobs(model)
r2_stats <- r2(model)
r_squared <- r2_stats["r2"]
within_r2 <- r2_stats["wr2"]
adjr2 <- r2_stats["ar2"]
residuals_vec <- residuals(model)
rmse <- sqrt(mean(residuals_vec^2))
fitted_vec <- fitted(model)

cat("\nMODEL DIAGNOSTICS\n")
cat("="*80, "\n")
cat(sprintf("Observations: %d\n", n_obs))
cat(sprintf("R-squared: %.10f\n", r_squared))
cat(sprintf("Within R-squared: %.10f\n", within_r2))
cat(sprintf("Adjusted R-squared: %.10f\n", adjr2))
cat(sprintf("RMSE: %.10f\n", rmse))

cat("\nRESIDUAL STATISTICS\n")
cat("="*80, "\n")
cat(sprintf("Mean: %.10f\n", mean(residuals_vec)))
cat(sprintf("Std Dev: %.10f\n", sd(residuals_vec)))
cat(sprintf("Min: %.10f\n", min(residuals_vec)))
cat(sprintf("Q1: %.10f\n", quantile(residuals_vec, 0.25)))
cat(sprintf("Median: %.10f\n", median(residuals_vec)))
cat(sprintf("Q3: %.10f\n", quantile(residuals_vec, 0.75)))
cat(sprintf("Max: %.10f\n", max(residuals_vec)))
cat(sprintf("Skewness: %.10f\n", moments::skewness(residuals_vec)))
cat(sprintf("Kurtosis: %.10f\n", moments::kurtosis(residuals_vec)))

cat("\nFITTED VALUES STATISTICS\n")
cat("="*80, "\n")
cat(sprintf("Mean: %.10f\n", mean(fitted_vec)))
cat(sprintf("Std Dev: %.10f\n", sd(fitted_vec)))
cat(sprintf("Min: %.10f\n", min(fitted_vec)))
cat(sprintf("Q1: %.10f\n", quantile(fitted_vec, 0.25)))
cat(sprintf("Median: %.10f\n", median(fitted_vec)))
cat(sprintf("Q3: %.10f\n", quantile(fitted_vec, 0.75)))
cat(sprintf("Max: %.10f\n", max(fitted_vec)))

n <- length(residuals_vec)
sigma2 <- sum(residuals_vec^2) / n
log_lik <- -n/2 * log(2*pi) - n/2 * log(sigma2) - 1/(2*sigma2) * sum(residuals_vec^2)
aic <- -2*log_lik + 2*(length(vendor_fe) + length(week_fe) + 1)
bic <- -2*log_lik + log(n)*(length(vendor_fe) + length(week_fe) + 1)

cat("\nINFORMATION CRITERIA\n")
cat("="*80, "\n")
cat(sprintf("Log-likelihood: %.10f\n", log_lik))
cat(sprintf("AIC: %.10f\n", aic))
cat(sprintf("BIC: %.10f\n", bic))

actual <- df_panel$log_revenue_plus_1
predicted <- fitted_vec
ss_tot <- sum((actual - mean(actual))^2)
ss_res <- sum((actual - predicted)^2)
r2_manual <- 1 - ss_res/ss_tot

cat("\nMANUAL R-SQUARED VERIFICATION\n")
cat("="*80, "\n")
cat(sprintf("SS Total: %.10f\n", ss_tot))
cat(sprintf("SS Residual: %.10f\n", ss_res))
cat(sprintf("R-squared (manual): %.10f\n", r2_manual))

cat("\nEND OF ANALYSIS\n")
cat("="*80, "\n")
""")

print("\nPROCESSING COMPLETE")
print("="*80)

# Close output file
sys.stdout = original_stdout
output_file.close()

print(f"Results saved to results/vendor_week_panel_feols_results.txt")