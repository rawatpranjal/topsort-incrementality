#!/usr/bin/env python3
"""
Modern rpy2 setup that avoids deprecated pandas2ri.activate()
Uses the conversion context manager approach recommended for newer rpy2 versions.
"""

import warnings
import sys
import pandas as pd
import numpy as np

def setup_rpy2():
    """
    Setup rpy2 with proper conversion context handling.
    Uses the modern context manager approach instead of deprecated activate().
    
    Returns:
        tuple: (success_flag, robjects, pandas2ri, localconverter)
    """
    try:
        import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
        
        # DO NOT use pandas2ri.activate() - it's deprecated!
        # We'll use the conversion context when needed instead
        
        # Load R libraries
        try:
            fixest = importr('fixest')
            print("✅ R package 'fixest' loaded successfully")
        except Exception as e:
            print(f"❌ Could not load R package 'fixest': {e}")
            print("   Please install it in R with: install.packages('fixest')")
            return False, None, None, None
        
        print("✅ rpy2 setup complete (using modern context manager approach)")
        return True, ro, pandas2ri, localconverter
        
    except ImportError as e:
        print(f"❌ Could not import rpy2: {e}")
        print("   Please install it with: pip install rpy2")
        return False, None, None, None

def run_fixed_effects_model(data_df, ro, pandas2ri, localconverter):
    """
    Run R fixed effects model using the modern conversion context.
    
    Args:
        data_df: Pandas DataFrame with the data
        ro: rpy2.robjects module
        pandas2ri: pandas to R conversion module
        localconverter: conversion context manager
    
    Returns:
        R model results
    """
    # Use the conversion context for pandas<->R conversion
    with localconverter(ro.default_converter + pandas2ri.converter):
        # Convert pandas DataFrame to R dataframe
        r_dataframe = ro.conversion.py2rpy(data_df)
        
        # Assign to R global environment
        ro.globalenv['model_data'] = r_dataframe
        
        # Run the R model
        result = ro.r('''
            library(fixest)
            
            # Prepare the data
            model_data$week <- as.factor(model_data$week)
            model_data$user_id <- as.factor(model_data$user_id)
            
            # Run fixed effects regression
            model <- feols(
                revenue_dollars ~ clicks | user_id + week,
                data = model_data,
                cluster = ~user_id
            )
            
            # Return summary
            summary(model)
        ''')
        
        return result

def run_logistic_fixed_effects(data_df, ro, pandas2ri, localconverter):
    """
    Run R fixed effects logistic regression using the modern conversion context.
    
    Args:
        data_df: Pandas DataFrame with the data
        ro: rpy2.robjects module
        pandas2ri: pandas to R conversion module
        localconverter: conversion context manager
    
    Returns:
        R model results
    """
    # Use the conversion context for pandas<->R conversion
    with localconverter(ro.default_converter + pandas2ri.converter):
        # Convert pandas DataFrame to R dataframe
        r_dataframe = ro.conversion.py2rpy(data_df)
        
        # Assign to R global environment
        ro.globalenv['model_data'] = r_dataframe
        
        # Run the logistic R model
        result = ro.r('''
            library(fixest)
            
            # Prepare the data
            model_data$revenue_binary <- as.integer(model_data$revenue_dollars > 0)
            model_data$user_id <- as.factor(model_data$user_id)
            model_data$week <- as.factor(model_data$week)
            
            # Run fixed effects logistic regression
            model <- feglm(
                revenue_binary ~ clicks | user_id + week,
                data = model_data,
                family = binomial()
            )
            
            # Get results with clustered standard errors
            res <- summary(model, cluster = ~ user_id)
            res
        ''')
        
        return result

def main():
    """
    Example usage of the modern rpy2 setup.
    """
    # Setup rpy2
    rpy2_ready, ro, pandas2ri, localconverter = setup_rpy2()
    
    if not rpy2_ready:
        print("❌ Failed to setup rpy2. Exiting.")
        sys.exit(1)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2'] * 2,
        'week': ['2025-W01', '2025-W02'] * 4,
        'clicks': np.random.poisson(5, 8),
        'revenue_dollars': np.random.gamma(2, 10, 8)
    })
    
    print("\n📊 Sample data:")
    print(sample_data.head())
    
    try:
        # Run the fixed effects model
        print("\n🔧 Running fixed effects model...")
        result = run_fixed_effects_model(sample_data, ro, pandas2ri, localconverter)
        print("\n📈 Model results:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Error running model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()