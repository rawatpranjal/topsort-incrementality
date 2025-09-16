#!/usr/bin/env python3
import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

# Set R environment variables to fix BLAS library issue
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
os.environ['DYLD_LIBRARY_PATH'] = f"{os.environ['R_HOME']}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Use the new conversion context instead of deprecated activate()
def run_weighted_fe_logistic_regression():
    # Import R packages
    base = importr('base')
    fixest = importr('fixest')
    arrow_r = importr('arrow')
    
    # Use the conversion context for pandas <-> R dataframe conversions
    with pandas2ri.localconverter(robjects.default_converter + pandas2ri.converter):
        
        # R code to read the Parquet file and prepare variables
        r_script = """
        library(arrow)
        library(fixest)
        
        # Read Parquet data
        df_analysis <- read_parquet('downsampled_power_panel.parquet')
        
        # Data preparation
        df_analysis$ihs_clicks <- asinh(df_analysis$total_clicks_promoted)
        df_analysis$ihs_revenue <- asinh(df_analysis$total_revenue_vendor_product)
        df_analysis$user_id <- as.factor(df_analysis$user_id)
        df_analysis$vendor_id <- as.factor(df_analysis$vendor_id)
        df_analysis$revenue_binary <- as.integer(df_analysis$ihs_revenue > 0)
        
        # Check if weight column exists
        has_weights <- "weight" %in% names(df_analysis)
        
        # Run the weighted fixed effects logistic regression
        if (has_weights) {
            model <- feglm(
                revenue_binary ~ ihs_clicks | user_id + vendor_id,
                data = df_analysis,
                family = binomial(),
                weights = df_analysis$weight
            )
        } else {
            model <- feglm(
                revenue_binary ~ ihs_clicks | user_id + vendor_id,
                data = df_analysis,
                family = binomial()
            )
        }
        
        # Get clustered standard errors
        res <- summary(model, cluster = ~ user_id)
        print(res)
        
        # Save results to text file
        sink("fe_logistic_results.txt")
        print("Fixed Effects Logistic Regression Results")
        print("==========================================")
        print(res)
        sink()
        
        # Return the model for further analysis if needed
        model
        """
        
        # Execute the R script
        print("Running fixed effects logistic regression...")
        model = r(r_script)
        
        print("\nResults have been saved to 'fe_logistic_results.txt'")
        
        return model

if __name__ == "__main__":
    try:
        model = run_weighted_fe_logistic_regression()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure R is installed: https://cran.r-project.org/")
        print("2. Install required R packages in R console:")
        print("   install.packages(c('arrow', 'fixest'))")
        print("3. On macOS, try reinstalling R for your architecture (arm64 for M1/M2/M3)")