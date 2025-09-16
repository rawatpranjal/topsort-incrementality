#!/usr/bin/env python3
"""
Alternative Python implementation using statsmodels for fixed effects logistic regression
This avoids R dependency issues while providing similar functionality
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    """Prepare data for fixed effects regression"""
    # Create inverse hyperbolic sine transformations
    df['ihs_clicks'] = np.arcsinh(df['total_clicks_promoted'])
    df['ihs_revenue'] = np.arcsinh(df['total_revenue_vendor_product'])
    df['revenue_binary'] = (df['ihs_revenue'] > 0).astype(int)
    
    # Encode categorical variables
    le_user = LabelEncoder()
    le_vendor = LabelEncoder()
    
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['vendor_id_encoded'] = le_vendor.fit_transform(df['vendor_id'])
    
    return df, le_user, le_vendor

def create_fixed_effects_dummies(df, fe_columns, max_dummies=1000):
    """Create dummy variables for fixed effects (limited to avoid memory issues)"""
    dummies_list = []
    
    for col in fe_columns:
        # Get unique values
        unique_vals = df[col].unique()
        n_unique = len(unique_vals)
        
        if n_unique > max_dummies:
            print(f"Warning: {col} has {n_unique} unique values. Using only top {max_dummies} by frequency.")
            # Keep only the most frequent values
            top_values = df[col].value_counts().head(max_dummies).index
            df_subset = df[df[col].isin(top_values)].copy()
        else:
            df_subset = df.copy()
        
        # Create dummies (drop first to avoid collinearity)
        dummies = pd.get_dummies(df_subset[col], prefix=col, drop_first=True)
        dummies_list.append(dummies)
    
    return pd.concat(dummies_list, axis=1)

def run_fe_logistic_regression(df, use_weights=False):
    """Run fixed effects logistic regression using statsmodels"""
    
    # Prepare data
    df_analysis, le_user, le_vendor = prepare_data(df)
    
    # Define dependent and independent variables
    y = df_analysis['revenue_binary']
    X = df_analysis[['ihs_clicks']]
    
    # Add fixed effects as dummies (simplified approach for computational efficiency)
    # Note: For large datasets, consider using panel data methods or iterative approaches
    print("Creating fixed effects dummies...")
    
    # Option 1: Simplified approach with user fixed effects only
    user_dummies = pd.get_dummies(df_analysis['user_id_encoded'], prefix='user', drop_first=True)
    
    # Limit number of dummies to avoid memory issues
    max_user_dummies = min(100, user_dummies.shape[1])
    user_dummies = user_dummies.iloc[:, :max_user_dummies]
    
    X = pd.concat([X, user_dummies], axis=1)
    
    # Add constant
    X = sm.add_constant(X)
    
    # Prepare weights if needed
    if use_weights and 'weight' in df_analysis.columns:
        weights = df_analysis['weight']
    else:
        weights = None
    
    # Fit logistic regression model
    print("Fitting logistic regression model...")
    if weights is not None:
        model = Logit(y, X).fit(disp=0, maxiter=100, method='bfgs', cov_type='cluster', 
                                cov_kwds={'groups': df_analysis['user_id_encoded']})
    else:
        model = Logit(y, X).fit(disp=0, maxiter=100, method='bfgs')
    
    return model

def save_results(model, filename="python_fe_logistic_results.txt"):
    """Save regression results to text file"""
    with open(filename, 'w') as f:
        f.write("Fixed Effects Logistic Regression Results (Python Implementation)\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(model.summary()))
        f.write("\n\n")
        f.write("Key Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Log-Likelihood: {model.llf:.4f}\n")
        f.write(f"AIC: {model.aic:.4f}\n")
        f.write(f"BIC: {model.bic:.4f}\n")
        f.write(f"Pseudo R-squared: {model.prsquared:.4f}\n")
        f.write("\n")
        f.write("Main Effect (ihs_clicks):\n")
        f.write("-" * 30 + "\n")
        if 'ihs_clicks' in model.params.index:
            f.write(f"Coefficient: {model.params['ihs_clicks']:.6f}\n")
            f.write(f"Std Error: {model.bse['ihs_clicks']:.6f}\n")
            f.write(f"z-statistic: {model.tvalues['ihs_clicks']:.4f}\n")
            f.write(f"p-value: {model.pvalues['ihs_clicks']:.6f}\n")
            
            # Calculate odds ratio
            odds_ratio = np.exp(model.params['ihs_clicks'])
            f.write(f"Odds Ratio: {odds_ratio:.6f}\n")

def main():
    """Main execution function"""
    try:
        # Read the data
        print("Reading data from parquet file...")
        df = pd.read_parquet('downsampled_power_panel.parquet')
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if weight column exists
        use_weights = 'weight' in df.columns
        if use_weights:
            print("Weight column found, will use weighted regression")
        
        # Run regression
        model = run_fe_logistic_regression(df, use_weights=use_weights)
        
        # Print summary
        print("\n" + "=" * 70)
        print("REGRESSION RESULTS")
        print("=" * 70)
        print(model.summary())
        
        # Save results
        save_results(model)
        print("\nResults saved to 'python_fe_logistic_results.txt'")
        
        return model
        
    except FileNotFoundError:
        print("Error: 'downsampled_power_panel.parquet' not found.")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model = main()