#!/usr/bin/env python3
"""
Version-compatible R integration for Jupyter notebooks
This works with different versions of rpy2
"""

import os
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.conversion import localconverter

# Set R environment variables to fix BLAS library issue
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
os.environ['DYLD_LIBRARY_PATH'] = f"{os.environ['R_HOME']}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Check and install required R packages
utils = importr('utils')
required_packages = ['arrow', 'fixest']

print("Checking R packages...")
for package in required_packages:
    if not isinstalled(package):
        print(f"Installing {package}...")
        utils.install_packages(package, repos='https://cran.rstudio.com/')
    else:
        print(f"✅ {package} is installed")

# Import R packages
base = importr('base')
fixest = importr('fixest')
arrow_r = importr('arrow')

print("\n✅ All R packages loaded successfully")

# Use the correct conversion context based on rpy2 version
try:
    # Try the newer approach first
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # R code execution
        r("""
        library(arrow)
        library(fixest)
        
        # Check if the parquet file exists
        file_path <- 'downsampled_power_panel.parquet'
        
        if (!file.exists(file_path)) {
            stop(paste("File not found:", file_path))
        }
        
        # Read Parquet data
        df_analysis <- read_parquet(file_path)
        
        print(paste("Data loaded:", nrow(df_analysis), "rows"))
        
        # Data preparation
        df_analysis$ihs_clicks <- asinh(df_analysis$total_clicks_promoted)
        df_analysis$ihs_revenue <- asinh(df_analysis$total_revenue_vendor_product)
        df_analysis$user_id <- as.factor(df_analysis$user_id)
        df_analysis$vendor_id <- as.factor(df_analysis$vendor_id)
        df_analysis$revenue_binary <- as.integer(df_analysis$ihs_revenue > 0)
        
        # Run the fixed effects logistic regression
        model <- feglm(
            revenue_binary ~ ihs_clicks | user_id + vendor_id,
            data = df_analysis,
            family = binomial()
        )
        
        # Get results with clustered standard errors
        res <- summary(model, cluster = ~ user_id)
        print(res)
        """)
        
except (AttributeError, ImportError):
    print("Using alternative approach for older rpy2 version...")
    
    # Alternative: Just run R code directly without conversion context
    r("""
    library(arrow)
    library(fixest)
    
    # Check if the parquet file exists
    file_path <- 'downsampled_power_panel.parquet'
    
    if (!file.exists(file_path)) {
        stop(paste("File not found:", file_path))
    }
    
    # Read Parquet data
    df_analysis <- read_parquet(file_path)
    
    print(paste("Data loaded:", nrow(df_analysis), "rows"))
    
    # Data preparation
    df_analysis$ihs_clicks <- asinh(df_analysis$total_clicks_promoted)
    df_analysis$ihs_revenue <- asinh(df_analysis$total_revenue_vendor_product)
    df_analysis$user_id <- as.factor(df_analysis$user_id)
    df_analysis$vendor_id <- as.factor(df_analysis$vendor_id)
    df_analysis$revenue_binary <- as.integer(df_analysis$ihs_revenue > 0)
    
    # Run the fixed effects logistic regression
    model <- feglm(
        revenue_binary ~ ihs_clicks | user_id + vendor_id,
        data = df_analysis,
        family = binomial()
    )
    
    # Get results with clustered standard errors
    res <- summary(model, cluster = ~ user_id)
    print(res)
    """)

print("\n✅ Analysis completed successfully!")