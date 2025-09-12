#!/usr/bin/env python3
"""
Unified Dynamic Factor Model Analysis
Complete structural analysis with all variables and model specifications
"""

import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from datetime import datetime
from tabulate import tabulate
from scipy import stats
import sys
import io
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')

class DFMAnalysis:
    def __init__(self, data_path='../../data/'):
        self.data_path = data_path
        self.endog_data = None
        self.daily_data = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        print("="*80)
        print("DATA LOADING AND PREPARATION")
        print("="*80)
        print(f"Timestamp: {datetime.now()}")
        
        # Load all data files
        clicks_df = pd.read_parquet(f'{self.data_path}hourly_clicks_2025-03-01_to_2025-09-30.parquet')
        purchases_df = pd.read_parquet(f'{self.data_path}hourly_purchases_2025-03-01_to_2025-09-30.parquet')
        impressions_df = pd.read_parquet(f'{self.data_path}hourly_impressions_2025-03-01_to_2025-09-30.parquet')
        auctions_df = pd.read_parquet(f'{self.data_path}hourly_auctions_2025-03-01_to_2025-09-30.parquet')
        
        print(f"\nRaw data shapes:")
        print(f"Clicks: {clicks_df.shape}")
        print(f"Purchases: {purchases_df.shape}")
        print(f"Impressions: {impressions_df.shape}")
        print(f"Auctions: {auctions_df.shape}")
        
        print(f"\nColumn names:")
        print(f"Clicks columns: {list(clicks_df.columns)}")
        print(f"Purchases columns: {list(purchases_df.columns)}")
        print(f"Impressions columns: {list(impressions_df.columns)}")
        print(f"Auctions columns: {list(auctions_df.columns)}")
        
        # Merge all data
        merged = purchases_df.merge(
            clicks_df, on='ACTIVITY_HOUR', how='outer'
        ).merge(
            impressions_df, on='ACTIVITY_HOUR', how='outer'
        ).merge(
            auctions_df, on='ACTIVITY_HOUR', how='outer'
        )
        
        merged['ACTIVITY_HOUR'] = pd.to_datetime(merged['ACTIVITY_HOUR'])
        merged = merged.fillna(0)
        
        print(f"\nMerged data shape: {merged.shape}")
        print(f"Date range: {merged['ACTIVITY_HOUR'].min()} to {merged['ACTIVITY_HOUR'].max()}")
        
        # Daily aggregation
        merged['date'] = merged['ACTIVITY_HOUR'].dt.date
        
        # Define all aggregation rules
        agg_rules = {
            'HOURLY_GMV': 'sum',
            'HOURLY_TRANSACTION_COUNT': 'sum',
            'HOURLY_UNITS_SOLD': 'sum',
            'HOURLY_PURCHASING_USERS': 'sum',
            'HOURLY_PRODUCTS_PURCHASED': 'max',
            'HOURLY_CLICK_COUNT': 'sum',
            'HOURLY_CLICKING_USERS': 'sum',
            'HOURLY_CLICKED_VENDORS': 'max',
            'HOURLY_CLICKED_CAMPAIGNS': 'max',
            'HOURLY_CLICKED_PRODUCTS': 'max',
            'HOURLY_IMPRESSION_COUNT': 'sum',
            'HOURLY_IMPRESSED_USERS': 'sum',
            'HOURLY_IMPRESSED_VENDORS': 'max',
            'HOURLY_IMPRESSED_CAMPAIGNS': 'max',
            'HOURLY_IMPRESSED_PRODUCTS': 'max',
            'HOURLY_AUCTION_COUNT': 'sum',
            'HOURLY_AUCTION_USERS': 'sum'
        }
        
        # Check which columns exist
        available_cols = {}
        for col, agg_func in agg_rules.items():
            if col in merged.columns:
                available_cols[col] = agg_func
                
        print(f"\nAvailable columns for aggregation: {len(available_cols)}")
        print(f"Columns: {list(available_cols.keys())}")
        
        self.daily_data = merged.groupby('date').agg(available_cols).reset_index()
        print(f"\nDaily data shape: {self.daily_data.shape}")
        print(f"Daily observations: {len(self.daily_data)}")
        
        # Show first few rows of daily data
        print("\nFirst 5 rows of daily aggregated data:")
        print(self.daily_data.head())
        
        # Show summary statistics
        print("\nSummary statistics of daily data:")
        print(self.daily_data.describe())
        
        return self.daily_data
    
    def create_standardized_variables(self):
        print("\n" + "="*80)
        print("VARIABLE STANDARDIZATION")
        print("="*80)
        
        # Variable mapping
        var_mapping = {
            'HOURLY_GMV': 'gmv',
            'HOURLY_TRANSACTION_COUNT': 'transactions',
            'HOURLY_UNITS_SOLD': 'units',
            'HOURLY_PURCHASING_USERS': 'purchasers',
            'HOURLY_PRODUCTS_PURCHASED': 'prod_variety',
            'HOURLY_CLICK_COUNT': 'clicks',
            'HOURLY_CLICKING_USERS': 'clickers',
            'HOURLY_CLICKED_VENDORS': 'click_vendors',
            'HOURLY_CLICKED_CAMPAIGNS': 'click_campaigns',
            'HOURLY_CLICKED_PRODUCTS': 'click_products',
            'HOURLY_IMPRESSION_COUNT': 'impressions',
            'HOURLY_IMPRESSED_USERS': 'viewers',
            'HOURLY_IMPRESSED_VENDORS': 'imp_vendors',
            'HOURLY_IMPRESSED_CAMPAIGNS': 'imp_campaigns',
            'HOURLY_IMPRESSED_PRODUCTS': 'imp_products',
            'HOURLY_AUCTION_COUNT': 'auctions',
            'HOURLY_AUCTION_USERS': 'searchers'
        }
        
        # Create standardized dataframe
        self.endog_data = pd.DataFrame(index=pd.to_datetime(self.daily_data['date']))
        
        print("\nStandardization process:")
        for orig_col, short_name in var_mapping.items():
            if orig_col in self.daily_data.columns:
                # Log transform
                log_series = np.log(self.daily_data[orig_col].values + 1)
                print(f"\n{orig_col} -> {short_name}:")
                print(f"  Original mean: {self.daily_data[orig_col].mean():.4f}")
                print(f"  Original std: {self.daily_data[orig_col].std():.4f}")
                print(f"  Log mean: {np.mean(log_series):.4f}")
                
                # First difference
                diff_series = pd.Series(log_series).diff() * 100
                print(f"  Diff mean: {diff_series.mean():.4f}")
                print(f"  Diff std: {diff_series.std():.4f}")
                
                # Standardize
                standardized = (diff_series - diff_series.mean()) / diff_series.std()
                self.endog_data[f'std_{short_name}'] = standardized.values
                print(f"  Standardized mean: {standardized.mean():.6f}")
                print(f"  Standardized std: {standardized.std():.6f}")
        
        # Remove first row (NaN from differencing)
        self.endog_data = self.endog_data.iloc[1:]
        
        print(f"\nFinal standardized data shape: {self.endog_data.shape}")
        print(f"Variables: {list(self.endog_data.columns)}")
        print(f"Observations: {len(self.endog_data)}")
        
        # Correlation matrix
        print("\nCorrelation Matrix:")
        corr_matrix = self.endog_data.corr()
        print(tabulate(corr_matrix, headers=corr_matrix.columns, 
                      tablefmt='simple', floatfmt='.3f', showindex=True))
        
        # Find highly correlated pairs
        print("\nHighly correlated pairs (|r| > 0.7):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        return self.endog_data
    
    def fit_single_model(self, k_factors):
        print(f"\nFitting {k_factors}-factor model...")
        print("-"*60)
        
        # Model specification
        if k_factors <= 2:
            factor_order = 2
            error_order = 1
        elif k_factors <= 3:
            factor_order = 1
            error_order = 1
        else:
            factor_order = 1
            error_order = 0
            
        print(f"Model specification: k_factors={k_factors}, factor_order={factor_order}, error_order={error_order}")
        
        mod = sm.tsa.DynamicFactor(
            self.endog_data,
            k_factors=k_factors,
            factor_order=factor_order,
            error_order=error_order
        )
        
        try:
            # Different optimization strategies based on complexity
            if k_factors <= 2:
                print("Using two-step optimization (powell then bfgs)...")
                initial = mod.fit(method='powell', maxiter=200, disp=False)
                print(f"Initial optimization: Log-L = {initial.llf:.2f}")
                res = mod.fit(initial.params, maxiter=300, disp=False)
            else:
                print("Using direct optimization...")
                res = mod.fit(maxiter=300, disp=False)
            
            print(f"Optimization successful")
            print(f"Log-likelihood: {res.llf:.2f}")
            print(f"AIC: {res.aic:.2f}")
            print(f"BIC: {res.bic:.2f}")
            print(f"Number of parameters: {len(res.params)}")
            
            # Store results
            self.models[k_factors] = res
            
            # Extract and display parameters
            print(f"\nParameter estimates ({k_factors} factors):")
            for param_name, param_value in res.params.items():
                if param_name in res.pvalues:
                    print(f"  {param_name}: {param_value:.6f} (p-value: {res.pvalues[param_name]:.4f})")
                else:
                    print(f"  {param_name}: {param_value:.6f}")
            
            return res
            
        except Exception as e:
            print(f"Model failed to converge: {str(e)}")
            return None
    
    def fit_all_models(self, max_factors=5):
        print("\n" + "="*80)
        print("MODEL ESTIMATION: 1 TO {} FACTORS".format(max_factors))
        print("="*80)
        
        for k in range(1, max_factors + 1):
            self.fit_single_model(k)
        
        # Model comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison_data = []
        for k, model in self.models.items():
            if model is not None:
                comparison_data.append([
                    k,
                    model.llf,
                    model.aic,
                    model.bic,
                    len(model.params),
                    model.nobs
                ])
        
        headers = ['k_factors', 'Log-L', 'AIC', 'BIC', 'n_params', 'n_obs']
        print(tabulate(comparison_data, headers=headers, tablefmt='grid', floatfmt='.2f'))
        
        # Likelihood ratio tests
        print("\nLikelihood Ratio Tests:")
        for i in range(len(comparison_data) - 1):
            k1 = comparison_data[i][0]
            k2 = comparison_data[i + 1][0]
            if k1 in self.models and k2 in self.models:
                lr_stat = 2 * (self.models[k2].llf - self.models[k1].llf)
                df = len(self.models[k2].params) - len(self.models[k1].params)
                p_value = 1 - stats.chi2.cdf(lr_stat, df)
                print(f"  {k1} vs {k2} factors: LR={lr_stat:.2f}, df={df}, p-value={p_value:.4f}")
        
        # Find optimal by BIC
        bic_values = {k: model.bic for k, model in self.models.items() if model is not None}
        if bic_values:
            optimal_k = min(bic_values, key=bic_values.get)
            print(f"\nOptimal number of factors (by BIC): {optimal_k}")
            print(f"BIC value: {bic_values[optimal_k]:.2f}")
        
        return self.models
    
    def extract_loadings(self, model, k_factors):
        print(f"\nExtracting loadings for {k_factors}-factor model...")
        
        loadings = {}
        for i in range(1, k_factors + 1):
            loadings[f'Factor {i}'] = {}
            
        for col in self.endog_data.columns:
            var_name = col.replace('std_', '')
            for i in range(1, k_factors + 1):
                param = f'loading.f{i}.{col}'
                if param in model.params.index:
                    loadings[f'Factor {i}'][var_name] = model.params[param]
        
        # Display loading matrix
        print(f"\nLoading Matrix ({k_factors} factors):")
        loading_matrix = []
        for var in loadings['Factor 1'].keys():
            row = [var]
            for i in range(1, k_factors + 1):
                row.append(loadings[f'Factor {i}'].get(var, 0))
            loading_matrix.append(row)
        
        headers = ['Variable'] + [f'Factor {i}' for i in range(1, k_factors + 1)]
        print(tabulate(loading_matrix, headers=headers, tablefmt='simple', floatfmt='.4f'))
        
        return loadings
    
    def variance_decomposition(self, model, k_factors):
        print(f"\nVariance Decomposition ({k_factors} factors):")
        print("-"*60)
        
        fevd_results = []
        for col in self.endog_data.columns:
            var_name = col.replace('std_', '')
            
            # Get loadings for all factors
            var_components = []
            for i in range(1, k_factors + 1):
                param = f'loading.f{i}.{col}'
                if param in model.params.index:
                    load = model.params[param]
                    var_components.append(load**2)
                else:
                    var_components.append(0)
            
            # Idiosyncratic variance
            error_var_param = f'sigma2.{col}'
            if error_var_param in model.params.index:
                var_idio = model.params[error_var_param]
            else:
                var_idio = 1.0  # Default normalized
            
            total_var = sum(var_components) + var_idio
            
            # Calculate percentages
            row = [var_name]
            for comp in var_components:
                row.append((comp/total_var)*100)
            row.append((var_idio/total_var)*100)
            row.append(total_var)
            
            fevd_results.append(row)
        
        # Sort by total factor contribution
        fevd_results.sort(key=lambda x: sum(x[1:-2]), reverse=True)
        
        headers = ['Variable'] + [f'F{i} (%)' for i in range(1, k_factors + 1)] + ['Idio (%)', 'Total Var']
        print(tabulate(fevd_results, headers=headers, tablefmt='simple', floatfmt='.2f'))
        
        return fevd_results
    
    def impulse_responses(self, model, k_factors, steps=30):
        print(f"\nImpulse Response Functions ({k_factors} factors, {steps} periods):")
        print("-"*60)
        
        try:
            irf = model.impulse_responses(steps=steps)
            
            # Display IRF for each variable at key horizons
            horizons = [0, 1, 2, 3, 5, 10, 15, 20, 30]
            irf_table = []
            
            for h in horizons:
                if h <= steps and h < len(irf):
                    row = [f"t={h}"]
                    for col in self.endog_data.columns:
                        if col in irf.columns:
                            row.append(irf[col].iloc[h])
                        else:
                            row.append(0)
                    irf_table.append(row)
            
            headers = ['Horizon'] + [col.replace('std_', '') for col in self.endog_data.columns]
            print(tabulate(irf_table, headers=headers, tablefmt='simple', floatfmt='.4f'))
            
            # Cumulative IRF
            print(f"\nCumulative IRF (sum over {steps} periods):")
            cum_irf = []
            for col in self.endog_data.columns:
                if col in irf.columns:
                    cum_response = irf[col].sum()
                    cum_irf.append([col.replace('std_', ''), cum_response])
            
            print(tabulate(cum_irf, headers=['Variable', 'Cumulative'], tablefmt='simple', floatfmt='.4f'))
            
            return irf
            
        except Exception as e:
            print(f"IRF calculation failed: {str(e)}")
            return None
    
    def factor_analysis(self, model, k_factors):
        print(f"\nFactor Analysis ({k_factors} factors):")
        print("-"*60)
        
        try:
            # Extract factors
            factors = pd.DataFrame()
            for i in range(k_factors):
                factors[f'Factor {i+1}'] = model.factors.filtered[i]
            
            # Factor statistics
            print("\nFactor Statistics:")
            print(factors.describe())
            
            # Factor correlations
            print("\nFactor Correlations:")
            factor_corr = factors.corr()
            print(tabulate(factor_corr, headers=factor_corr.columns, 
                          tablefmt='simple', floatfmt='.3f', showindex=True))
            
            # Identify extreme periods for each factor
            for i in range(1, k_factors + 1):
                factor_name = f'Factor {i}'
                factor_series = factors[factor_name]
                
                print(f"\n{factor_name} Extreme Values:")
                print(f"  Top 5 values:")
                top_5 = factor_series.nlargest(5)
                for date_idx, value in top_5.items():
                    if date_idx < len(self.endog_data.index):
                        print(f"    {self.endog_data.index[date_idx]}: {value:.4f}")
                
                print(f"  Bottom 5 values:")
                bottom_5 = factor_series.nsmallest(5)
                for date_idx, value in bottom_5.items():
                    if date_idx < len(self.endog_data.index):
                        print(f"    {self.endog_data.index[date_idx]}: {value:.4f}")
            
            return factors
            
        except Exception as e:
            print(f"Factor analysis failed: {str(e)}")
            return None
    
    def rolling_window_analysis(self, window_size=60):
        print("\n" + "="*80)
        print(f"ROLLING WINDOW ANALYSIS (window={window_size} days)")
        print("="*80)
        
        n_windows = len(self.endog_data) - window_size + 1
        
        if n_windows <= 0:
            print(f"Insufficient data for rolling window analysis (need at least {window_size} observations)")
            return
        
        print(f"Number of possible windows: {n_windows}")
        print(f"Analyzing every 10th window for computational efficiency...")
        
        rolling_results = []
        window_dates = []
        
        for i in range(0, n_windows, 10):
            window_data = self.endog_data.iloc[i:i+window_size]
            window_date = window_data.index[window_size//2]
            
            print(f"\nWindow {i//10 + 1} (centered at {window_date}):")
            
            try:
                # Fit simple 1-factor model
                mod_window = sm.tsa.DynamicFactor(
                    window_data,
                    k_factors=1,
                    factor_order=1,
                    error_order=0
                )
                res_window = mod_window.fit(maxiter=200, disp=False)
                
                print(f"  Log-L: {res_window.llf:.2f}, AIC: {res_window.aic:.2f}")
                
                # Extract key parameters
                window_params = {}
                for param in ['loading.f1.std_gmv', 'loading.f1.std_clicks', 
                             'loading.f1.std_impressions', 'loading.f1.std_auctions']:
                    if param in res_window.params.index:
                        window_params[param] = res_window.params[param]
                        print(f"  {param}: {res_window.params[param]:.4f}")
                
                window_params['date'] = window_date
                window_params['llf'] = res_window.llf
                rolling_results.append(window_params)
                window_dates.append(window_date)
                
            except Exception as e:
                print(f"  Failed: {str(e)[:50]}")
                continue
        
        if rolling_results:
            # Calculate stability metrics
            print("\n" + "-"*60)
            print("Loading Stability Analysis:")
            
            params_to_analyze = ['loading.f1.std_gmv', 'loading.f1.std_clicks', 
                                'loading.f1.std_impressions', 'loading.f1.std_auctions']
            
            stability_table = []
            for param in params_to_analyze:
                values = [r.get(param, np.nan) for r in rolling_results]
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    stability_table.append([
                        param.replace('loading.f1.std_', ''),
                        mean_val,
                        std_val,
                        cv,
                        min_val,
                        max_val
                    ])
            
            headers = ['Variable', 'Mean', 'Std Dev', 'CV', 'Min', 'Max']
            print(tabulate(stability_table, headers=headers, tablefmt='simple', floatfmt='.4f'))
    
    def comprehensive_analysis(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE DFM ANALYSIS - COMPLETE OUTPUT")
        print("="*80)
        
        # Select models for detailed analysis
        detailed_k = [1, 2, 3] if 3 in self.models else [1, 2] if 2 in self.models else [1]
        
        for k in detailed_k:
            if k not in self.models or self.models[k] is None:
                continue
                
            print("\n" + "="*80)
            print(f"{k}-FACTOR MODEL DETAILED ANALYSIS")
            print("="*80)
            
            model = self.models[k]
            
            # Loadings
            loadings = self.extract_loadings(model, k)
            
            # Variance decomposition
            fevd = self.variance_decomposition(model, k)
            
            # Impulse responses
            irf = self.impulse_responses(model, k)
            
            # Factor analysis
            factors = self.factor_analysis(model, k)
            
            # Model diagnostics
            print(f"\nModel Diagnostics ({k} factors):")
            print(f"  Observations: {model.nobs}")
            print(f"  Log-likelihood: {model.llf:.4f}")
            print(f"  AIC: {model.aic:.4f}")
            print(f"  BIC: {model.bic:.4f}")
            print(f"  HQIC: {model.hqic:.4f}")
            
            # Parameter count by type
            param_counts = {}
            for param in model.params.index:
                param_type = param.split('.')[0]
                param_counts[param_type] = param_counts.get(param_type, 0) + 1
            
            print(f"\nParameter counts by type:")
            for ptype, count in sorted(param_counts.items()):
                print(f"  {ptype}: {count}")
            
            # Residual analysis
            print(f"\nResidual Analysis:")
            try:
                resid = model.resid
                print(f"  Residual shape: {resid.shape}")
                print(f"  Mean residuals by variable:")
                for i, col in enumerate(self.endog_data.columns):
                    if i < resid.shape[1]:
                        print(f"    {col}: {np.mean(resid[:, i]):.6f}")
            except:
                print("  Residual analysis not available")
    
    def run_full_analysis(self):
        print("="*80)
        print("UNIFIED DYNAMIC FACTOR MODEL ANALYSIS")
        print("="*80)
        print(f"Analysis started: {datetime.now()}")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Create standardized variables
        self.create_standardized_variables()
        
        # Fit all models
        self.fit_all_models(max_factors=5)
        
        # Comprehensive analysis
        self.comprehensive_analysis()
        
        # Rolling window analysis
        self.rolling_window_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Analysis ended: {datetime.now()}")
        print("="*80)

def main():
    # Create output buffer to capture all stdout
    output_buffer = io.StringIO()
    
    # Run analysis with output capture
    with redirect_stdout(output_buffer):
        analysis = DFMAnalysis()
        analysis.run_full_analysis()
    
    # Get the complete output
    complete_output = output_buffer.getvalue()
    
    # Print to console
    print(complete_output)
    
    # Save to file
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/dfm_complete_results.txt', 'w') as f:
        f.write(complete_output)
    
    print("\n" + "="*80)
    print("Results saved to: results/dfm_complete_results.txt")
    print("="*80)

if __name__ == "__main__":
    main()