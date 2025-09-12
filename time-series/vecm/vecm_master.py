#!/usr/bin/env python3
"""
VECM Master Framework - Consolidated Analysis System
Combines all VECM models into single intelligent framework
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
import warnings
import json
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tabulate import tabulate
import itertools

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for VECM model"""
    name: str
    variables: List[str]
    k_ar_diff: int = 2
    coint_rank: Optional[int] = None
    deterministic: str = 'ci'
    freq: str = 'daily'
    description: str = ""
    exog_vars: Optional[List[str]] = None

class DualWriter:
    """Dual output to stdout and StringIO for capture"""
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
    
    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)
    
    def flush(self):
        self.file1.flush()
        self.file2.flush()

class DataLoader:
    """Handles all data loading and merging operations"""
    
    def __init__(self, base_path: str = '../../data'):
        self.base_path = Path(base_path)
        self.raw_data = {}
        self.merged_data = {}
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw parquet files"""
        files = {
            'hourly_clicks': 'hourly_clicks.parquet',
            'hourly_purchases': 'hourly_purchases.parquet', 
            'hourly_impressions': 'hourly_impressions.parquet',
            'hourly_auctions': 'hourly_auctions.parquet'
        }
        
        print("\nLOADING RAW DATA FILES")
        print("-"*80)
        
        for key, filename in files.items():
            path = self.base_path / filename
            print(f"Loading {path}...")
            self.raw_data[key] = pd.read_parquet(path)
            print(f"  Shape: {self.raw_data[key].shape}")
            print(f"  Columns: {', '.join(self.raw_data[key].columns)}")
            print(f"  Date range: {self.raw_data[key]['ACTIVITY_HOUR'].min()} to {self.raw_data[key]['ACTIVITY_HOUR'].max()}")
            
        return self.raw_data
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge all datasets on common time column"""
        print("\nMERGING DATASETS")
        print("-"*80)
        
        clicks_df = self.raw_data['hourly_clicks'].copy()
        clicks_df['occurred_at_hour'] = clicks_df['ACTIVITY_HOUR']
        clicks_df['clicks_total_clicks'] = clicks_df['HOURLY_CLICK_COUNT']
        clicks_df['clicks_unique_users'] = clicks_df['HOURLY_CLICKING_USERS']
        clicks_df['clicks_unique_vendors'] = clicks_df['HOURLY_CLICKED_VENDORS']
        clicks_df = clicks_df[['occurred_at_hour', 'clicks_total_clicks', 'clicks_unique_users', 'clicks_unique_vendors']]
        
        purchases_df = self.raw_data['hourly_purchases'].copy()
        purchases_df['occurred_at_hour'] = purchases_df['ACTIVITY_HOUR']
        purchases_df['purchases_gmv'] = purchases_df['HOURLY_GMV']
        purchases_df['purchases_num_purchases'] = purchases_df['HOURLY_TRANSACTION_COUNT']
        purchases_df['purchases_unique_users'] = purchases_df['HOURLY_PURCHASING_USERS']
        purchases_df = purchases_df[['occurred_at_hour', 'purchases_gmv', 'purchases_num_purchases', 'purchases_unique_users']]
        
        impressions_df = self.raw_data['hourly_impressions'].copy()
        impressions_df['occurred_at_hour'] = impressions_df['ACTIVITY_HOUR']
        impressions_df['impressions_total_impressions'] = impressions_df['HOURLY_IMPRESSION_COUNT']
        impressions_df['impressions_unique_users'] = impressions_df['HOURLY_IMPRESSED_USERS']
        impressions_df['impressions_unique_vendors'] = impressions_df['HOURLY_IMPRESSED_VENDORS']
        impressions_df['impressions_unique_products'] = impressions_df['HOURLY_IMPRESSED_PRODUCTS']
        impressions_df = impressions_df[['occurred_at_hour', 'impressions_total_impressions', 'impressions_unique_users', 
                                         'impressions_unique_vendors', 'impressions_unique_products']]
        
        auctions_df = self.raw_data['hourly_auctions'].copy()
        auctions_df['occurred_at_hour'] = auctions_df['ACTIVITY_HOUR']
        auctions_df['auctions_total_auctions'] = auctions_df['HOURLY_AUCTION_COUNT']
        auctions_df['auctions_total_bids'] = auctions_df['HOURLY_AUCTION_COUNT'] * 2
        auctions_df = auctions_df[['occurred_at_hour', 'auctions_total_auctions', 'auctions_total_bids']]
        
        print("Merging clicks and purchases...")
        merged = pd.merge(clicks_df, purchases_df, on='occurred_at_hour', how='outer')
        print(f"  After merge: {merged.shape}")
        
        print("Adding impressions...")
        merged = pd.merge(merged, impressions_df, on='occurred_at_hour', how='outer')
        print(f"  After merge: {merged.shape}")
        
        print("Adding auctions...")
        merged = pd.merge(merged, auctions_df, on='occurred_at_hour', how='outer')
        print(f"  After merge: {merged.shape}")
        
        merged = merged.sort_values('occurred_at_hour')
        merged = merged.fillna(0)
        
        print(f"\nFinal merged shape: {merged.shape}")
        print(f"Columns: {', '.join(merged.columns)}")
        
        self.merged_data['hourly'] = merged
        return merged
    
    def aggregate_to_frequency(self, freq: str = 'daily') -> pd.DataFrame:
        """Aggregate data to specified frequency"""
        if 'hourly' not in self.merged_data:
            self.merge_datasets()
            
        print(f"\nAGGREGATING TO {freq.upper()} FREQUENCY")
        print("-"*80)
        
        df = self.merged_data['hourly'].copy()
        if 'occurred_at_hour' in df.columns:
            df['occurred_at_hour'] = pd.to_datetime(df['occurred_at_hour'])
            df.set_index('occurred_at_hour', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
        
        print(f"Original hourly data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        if freq == 'hourly':
            result = df
        elif freq == 'daily':
            result = df.resample('D').sum()
            print(f"Aggregated to daily: {result.shape}")
        elif freq == 'weekly':
            result = df.resample('W').sum()
            print(f"Aggregated to weekly: {result.shape}")
        else:
            raise ValueError(f"Unknown frequency: {freq}")
            
        result = result.fillna(0)
        
        print(f"Final shape: {result.shape}")
        print(f"Date range: {result.index.min()} to {result.index.max()}")
        
        self.merged_data[freq] = result
        return result

class VariableBuilder:
    """Builds features and transformations"""
    
    @staticmethod
    def add_log_transforms(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add log transformations for specified columns"""
        print("\nADDING LOG TRANSFORMATIONS")
        print("-"*80)
        
        for col in columns:
            if col in df.columns:
                df[f'log_{col}'] = np.log(df[col] + 1)
                print(f"Created log_{col}")
                
                # Print statistics
                print(f"  Original {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
                print(f"  Log {col}: mean={df[f'log_{col}'].mean():.4f}, std={df[f'log_{col}'].std():.4f}")
                
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Add time-based features"""
        print("\nADDING TIME FEATURES")
        print("-"*80)
        
        if freq == 'hourly':
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            print("Added: hour, day_of_week, is_weekend")
        elif freq in ['daily', 'weekly']:
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['trend'] = np.arange(len(df))
            print("Added: day_of_week, month, quarter, is_weekend, trend")
            
        return df
    
    @staticmethod
    def add_supply_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-side metrics"""
        print("\nADDING SUPPLY METRICS")
        print("-"*80)
        
        if 'auctions_total_auctions' in df.columns and 'auctions_total_bids' in df.columns:
            df['bid_density'] = df['auctions_total_bids'] / (df['auctions_total_auctions'] + 1)
            df['log_bid_density'] = np.log(df['bid_density'] + 1)
            print(f"Created bid_density: mean={df['bid_density'].mean():.2f}")
            
        if 'impressions_total_impressions' in df.columns and 'auctions_total_auctions' in df.columns:
            df['auction_fill_rate'] = df['impressions_total_impressions'] / (df['auctions_total_auctions'] + 1)
            df['log_auction_fill_rate'] = np.log(df['auction_fill_rate'] + 1)
            print(f"Created auction_fill_rate: mean={df['auction_fill_rate'].mean():.2f}")
            
        if 'impressions_unique_products' in df.columns:
            df['log_unique_products'] = np.log(df['impressions_unique_products'] + 1)
            print(f"Created log_unique_products")
            
        if 'impressions_unique_vendors' in df.columns and 'log_unique_vendors' not in df.columns:
            df['log_unique_vendors'] = np.log(df['impressions_unique_vendors'] + 1)
            print(f"Created log_unique_vendors")
            
        return df
    
    @staticmethod
    def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Add funnel efficiency metrics"""
        print("\nADDING EFFICIENCY METRICS")
        print("-"*80)
        
        if 'clicks_total_clicks' in df.columns and 'impressions_total_impressions' in df.columns:
            df['ctr'] = df['clicks_total_clicks'] / (df['impressions_total_impressions'] + 1)
            df['log_ctr'] = np.log(df['ctr'] + 0.001)
            print(f"Created CTR: mean={df['ctr'].mean():.4f}")
            
        if 'purchases_gmv' in df.columns and 'clicks_total_clicks' in df.columns:
            df['revenue_per_click'] = df['purchases_gmv'] / (df['clicks_total_clicks'] + 1)
            df['log_revenue_per_click'] = np.log(df['revenue_per_click'] + 1)
            print(f"Created revenue_per_click: mean={df['revenue_per_click'].mean():.2f}")
            
        if 'purchases_num_purchases' in df.columns and 'clicks_total_clicks' in df.columns:
            df['cvr'] = df['purchases_num_purchases'] / (df['clicks_total_clicks'] + 1)
            df['log_cvr'] = np.log(df['cvr'] + 0.001)
            print(f"Created CVR: mean={df['cvr'].mean():.4f}")
            
        if 'purchases_gmv' in df.columns and 'purchases_num_purchases' in df.columns:
            df['avg_order_value'] = df['purchases_gmv'] / (df['purchases_num_purchases'] + 1)
            df['log_avg_order_value'] = np.log(df['avg_order_value'] + 1)
            print(f"Created avg_order_value: mean={df['avg_order_value'].mean():.2f}")
            
        return df

class VECMEstimator:
    """Handles VECM model estimation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.results = {}
        
    def run_unit_root_tests(self, variables: List[str]) -> Dict[str, Dict]:
        """Run ADF and KPSS tests"""
        print("\nUNIT ROOT TESTS")
        print("-"*80)
        
        results = {}
        
        # Print header
        print(f"{'Variable':<25} {'ADF Stat':<12} {'ADF p-val':<12} {'KPSS Stat':<12} {'KPSS p-val':<12}")
        print("-"*80)
        
        for var in variables:
            if var in self.df.columns:
                series = self.df[var].dropna()
                
                # ADF test
                adf_result = adfuller(series, autolag='AIC')
                adf_stat, adf_pval = adf_result[0], adf_result[1]
                
                # KPSS test
                try:
                    kpss_result = kpss(series, regression='c', nlags='auto')
                    kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]
                except:
                    kpss_stat, kpss_pval = np.nan, np.nan
                
                results[var] = {
                    'adf_stat': adf_stat,
                    'adf_pval': adf_pval,
                    'kpss_stat': kpss_stat,
                    'kpss_pval': kpss_pval
                }
                
                print(f"{var:<25} {adf_stat:<12.4f} {adf_pval:<12.4f} {kpss_stat:<12.4f} {kpss_pval:<12.4f}")
                
        return results
    
    def run_johansen_test(self, endog: pd.DataFrame, k_ar_diff: int = 2, det_order: int = 0) -> Dict:
        """Run Johansen cointegration test"""
        print("\nJOHANSEN COINTEGRATION TEST")
        print("-"*80)
        
        print(f"Variables: {', '.join(endog.columns)}")
        print(f"Lag order (k_ar_diff): {k_ar_diff}")
        print(f"Deterministic order: {det_order}")
        
        johansen_result = coint_johansen(endog, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Trace statistics
        print("\nTRACE STATISTICS")
        print("-"*40)
        trace_table = []
        for i in range(len(johansen_result.lr1)):
            trace_table.append([
                f"r <= {i}",
                f"{johansen_result.lr1[i]:.2f}",
                f"{johansen_result.cvt[i, 0]:.2f}",
                f"{johansen_result.cvt[i, 1]:.2f}",
                f"{johansen_result.cvt[i, 2]:.2f}",
                "*" if johansen_result.lr1[i] > johansen_result.cvt[i, 1] else ""
            ])
        print(tabulate(trace_table, 
                      headers=['H0', 'Statistic', '90% CV', '95% CV', '99% CV', 'Sig'],
                      tablefmt='simple'))
        
        # Max eigenvalue statistics
        print("\nMAX EIGENVALUE STATISTICS")
        print("-"*40)
        max_table = []
        for i in range(len(johansen_result.lr2)):
            max_table.append([
                f"r = {i}",
                f"{johansen_result.lr2[i]:.2f}",
                f"{johansen_result.cvm[i, 0]:.2f}",
                f"{johansen_result.cvm[i, 1]:.2f}",
                f"{johansen_result.cvm[i, 2]:.2f}",
                "*" if johansen_result.lr2[i] > johansen_result.cvm[i, 1] else ""
            ])
        print(tabulate(max_table,
                      headers=['H0', 'Statistic', '90% CV', '95% CV', '99% CV', 'Sig'],
                      tablefmt='simple'))
        
        # Eigenvalues
        print("\nEIGENVALUES")
        print("-"*40)
        for i, eig in enumerate(johansen_result.eig):
            print(f"λ{i+1} = {eig:.6f}")
        
        return {
            'trace_stats': johansen_result.lr1.tolist(),
            'max_eig_stats': johansen_result.lr2.tolist(),
            'trace_cv': johansen_result.cvt.tolist(),
            'max_eig_cv': johansen_result.cvm.tolist(),
            'eigenvalues': johansen_result.eig.tolist()
        }
    
    def estimate_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Estimate single VECM model"""
        
        # Select variables
        endog = self.df[config.variables].dropna()
        
        if len(endog) < 20:
            return {
                'config': asdict(config),
                'error': f"Insufficient observations: {len(endog)}"
            }
        
        try:
            # Unit root tests
            unit_root_results = self.run_unit_root_tests(config.variables)
            
            # Johansen test
            johansen_results = self.run_johansen_test(endog, config.k_ar_diff)
            
            # Select cointegration rank
            print("\nRANK SELECTION")
            print("-"*80)
            
            if config.coint_rank is None:
                rank_test = select_coint_rank(endog, det_order=0, k_ar_diff=config.k_ar_diff, method='trace')
                coint_rank = rank_test.rank if rank_test.rank > 0 else 1
                print(f"Automatically selected rank (trace method): {coint_rank}")
            else:
                coint_rank = config.coint_rank
                print(f"Using specified rank: {coint_rank}")
            
            # Fit VECM
            print("\nVECM ESTIMATION")
            print("-"*80)
            
            vecm = VECM(endog, k_ar_diff=config.k_ar_diff, coint_rank=coint_rank, 
                       deterministic=config.deterministic)
            vecm_fit = vecm.fit()
            
            print(f"Model: VECM(k_ar_diff={config.k_ar_diff}, rank={coint_rank}, det={config.deterministic})")
            print(f"Sample size: {vecm_fit.nobs}")
            print(f"Log-likelihood: {vecm_fit.llf:.2f}")
            
            # Calculate AIC and BIC manually since VECMResults doesn't have them
            k_params = coint_rank * len(config.variables) * 2 + config.k_ar_diff * len(config.variables)**2
            aic = -2 * vecm_fit.llf + 2 * k_params
            bic = -2 * vecm_fit.llf + np.log(vecm_fit.nobs) * k_params
            hqic = -2 * vecm_fit.llf + 2 * np.log(np.log(vecm_fit.nobs)) * k_params
            
            print(f"AIC: {aic:.2f}")
            print(f"BIC: {bic:.2f}")
            print(f"HQIC: {hqic:.2f}")
            
            # Extract coefficients
            alpha = vecm_fit.alpha
            beta = vecm_fit.beta
            
            # ALPHA (Loading coefficients)
            print("\nALPHA (LOADING COEFFICIENTS)")
            print("-"*80)
            alpha_table = []
            for i, var in enumerate(config.variables):
                row = [var]
                for j in range(coint_rank):
                    if j < alpha.shape[1]:
                        coef = alpha[i, j]
                        # Calculate t-stat if available
                        row.append(f"{coef:.4f}")
                alpha_table.append(row)
            
            headers = ['Variable'] + [f'EC{i+1}' for i in range(coint_rank)]
            print(tabulate(alpha_table, headers=headers, tablefmt='simple'))
            
            # BETA (Cointegrating vectors)
            print("\nBETA (COINTEGRATING VECTORS)")
            print("-"*80)
            beta_table = []
            for i, var in enumerate(config.variables):
                row = [var]
                for j in range(coint_rank):
                    if j < beta.shape[1]:
                        row.append(f"{beta[i, j]:.4f}")
                beta_table.append(row)
            
            headers = ['Variable'] + [f'Coint{i+1}' for i in range(coint_rank)]
            print(tabulate(beta_table, headers=headers, tablefmt='simple'))
            
            # Normalized beta (first variable = 1)
            print("\nNORMALIZED BETA (First variable = 1)")
            print("-"*80)
            for j in range(coint_rank):
                if j < beta.shape[1]:
                    normalized = beta[:, j] / beta[0, j] if abs(beta[0, j]) > 1e-10 else beta[:, j]
                    print(f"\nCointegrating vector {j+1}:")
                    for i, var in enumerate(config.variables):
                        print(f"  {var}: {normalized[i]:.4f}")
            
            # GAMMA (Short-run dynamics)
            if config.k_ar_diff > 0 and hasattr(vecm_fit, 'gamma'):
                try:
                    gamma = vecm_fit.gamma
                    print("\nGAMMA (SHORT-RUN DYNAMICS)")
                    print("-"*80)
                    
                    if gamma.ndim == 3:
                        for eq_idx, var in enumerate(config.variables):
                            print(f"\nEquation for Δ{var}:")
                            gamma_table = []
                            for lag in range(gamma.shape[2]):
                                row = [f"Lag {lag+1}"]
                                for j in range(gamma.shape[1]):
                                    row.append(f"{gamma[eq_idx, j, lag]:.4f}")
                                gamma_table.append(row)
                            
                            headers = [''] + [f'Δ{v}(t-{lag+1})' for v in config.variables]
                            print(tabulate(gamma_table, headers=headers, tablefmt='simple'))
                    elif gamma.ndim == 2:
                        gamma_table = []
                        for i, var in enumerate(config.variables):
                            row = [f"Δ{var}"]
                            for j in range(gamma.shape[1]):
                                row.append(f"{gamma[i, j]:.4f}")
                            gamma_table.append(row)
                        
                        headers = ['Equation'] + [f'Δ{v}(t-1)' for v in config.variables]
                        print(tabulate(gamma_table, headers=headers, tablefmt='simple'))
                except Exception as e:
                    print(f"Could not display gamma coefficients: {e}")
            
            # Diagnostic tests
            diagnostics = self.run_diagnostics(vecm_fit, config.variables)
            
            # Granger causality
            granger_results = self.run_granger_causality(vecm_fit, config.variables)
            
            # Forecast error variance decomposition
            try:
                print("\nFORECAST ERROR VARIANCE DECOMPOSITION (10 periods)")
                print("-"*80)
                fevd = vecm_fit.fevd(10)
                fevd_summary = fevd.summary()
                print(fevd_summary)
            except:
                print("FEVD could not be computed")
                fevd = None
            
            # Impulse response
            try:
                print("\nIMPULSE RESPONSE FUNCTIONS")
                print("-"*80)
                irf = vecm_fit.irf(10)
                for i, var in enumerate(config.variables[:2]):  # Show first 2 variables
                    print(f"\nResponse of {var} to unit shock in:")
                    for j, shock_var in enumerate(config.variables[:2]):
                        responses = irf.irfs[:5, i, j]  # First 5 periods
                        print(f"  {shock_var}: {' '.join([f'{r:.4f}' for r in responses])}")
            except:
                print("IRF could not be computed")
            
            result = {
                'config': asdict(config),
                'nobs': int(vecm_fit.nobs),
                'llf': float(vecm_fit.llf),
                'aic': float(aic),
                'bic': float(bic),
                'hqic': float(hqic),
                'coint_rank': int(coint_rank),
                'alpha': alpha.tolist(),
                'beta': beta.tolist(),
                'unit_root_tests': unit_root_results,
                'johansen_test': johansen_results,
                'diagnostics': diagnostics,
                'granger_causality': granger_results
            }
            
            self.models[config.name] = vecm_fit
            self.results[config.name] = result
            return result
            
        except Exception as e:
            import traceback
            print(f"\nERROR in model {config.name}: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            
            return {
                'config': asdict(config),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_diagnostics(self, model, variables: List[str]) -> Dict[str, Any]:
        """Run diagnostic tests"""
        print("\nDIAGNOSTIC TESTS")
        print("-"*80)
        
        diagnostics = {}
        
        # Ljung-Box test for autocorrelation
        try:
            print("\nLjung-Box Test for Residual Autocorrelation:")
            for lag in [5, 10, 15]:
                whiteness = model.test_whiteness(nlags=lag)
                print(f"  Lag {lag}: stat={whiteness.test_statistic:.2f}, p-value={whiteness.pvalue:.4f}")
                diagnostics[f'ljung_box_lag{lag}'] = {
                    'statistic': float(whiteness.test_statistic),
                    'pvalue': float(whiteness.pvalue),
                    'df': int(whiteness.df)
                }
        except:
            print("  Ljung-Box test could not be performed")
        
        # Normality test
        try:
            print("\nJarque-Bera Test for Normality:")
            normality = model.test_normality()
            print(f"  Combined: stat={normality.test_statistic:.2f}, p-value={normality.pvalue:.4f}")
            diagnostics['jarque_bera'] = {
                'statistic': float(normality.test_statistic),
                'pvalue': float(normality.pvalue)
            }
            
            # Individual equation normality
            residuals = model.resid
            for i, var in enumerate(variables):
                jb_stat, jb_pval = jarque_bera(residuals[:, i])[:2]
                print(f"  {var}: stat={jb_stat:.2f}, p-value={jb_pval:.4f}")
                diagnostics[f'jb_{var}'] = {
                    'statistic': float(jb_stat),
                    'pvalue': float(jb_pval)
                }
        except:
            print("  Normality test could not be performed")
        
        # Stability test
        try:
            print("\nStability Test:")
            stability = model.test_stability()
            print(f"  All eigenvalues inside unit circle: {stability}")
            diagnostics['stable'] = bool(stability)
        except:
            print("  Stability test could not be performed")
        
        # Residual statistics
        try:
            print("\nResidual Statistics:")
            residuals = model.resid
            resid_table = []
            for i, var in enumerate(variables):
                resid_table.append([
                    var,
                    f"{residuals[:, i].mean():.6f}",
                    f"{residuals[:, i].std():.4f}",
                    f"{residuals[:, i].min():.4f}",
                    f"{residuals[:, i].max():.4f}",
                    f"{stats.skew(residuals[:, i]):.4f}",
                    f"{stats.kurtosis(residuals[:, i]):.4f}"
                ])
            
            print(tabulate(resid_table, 
                          headers=['Variable', 'Mean', 'Std', 'Min', 'Max', 'Skew', 'Kurtosis'],
                          tablefmt='simple'))
            
            for i, var in enumerate(variables):
                diagnostics[f'resid_{var}'] = {
                    'mean': float(residuals[:, i].mean()),
                    'std': float(residuals[:, i].std()),
                    'min': float(residuals[:, i].min()),
                    'max': float(residuals[:, i].max()),
                    'skew': float(stats.skew(residuals[:, i])),
                    'kurtosis': float(stats.kurtosis(residuals[:, i]))
                }
        except:
            print("  Residual statistics could not be computed")
        
        return diagnostics
    
    def run_granger_causality(self, model, variables: List[str]) -> Dict[str, Any]:
        """Run Granger causality tests"""
        print("\nGRANGER CAUSALITY TESTS")
        print("-"*80)
        
        results = {}
        causality_table = []
        
        for caused in variables[:3]:  # Test main variables
            for causing in variables[:3]:
                if caused != causing:
                    try:
                        granger = model.test_granger_causality(caused=caused, causing=causing)
                        causality_table.append([
                            f"{causing} → {caused}",
                            f"{granger.test_statistic:.2f}",
                            f"{granger.pvalue:.4f}",
                            "*" if granger.pvalue < 0.05 else ""
                        ])
                        results[f"{causing}_to_{caused}"] = {
                            'statistic': float(granger.test_statistic),
                            'pvalue': float(granger.pvalue)
                        }
                    except:
                        pass
        
        if causality_table:
            print(tabulate(causality_table, 
                          headers=['Direction', 'Statistic', 'p-value', 'Sig'],
                          tablefmt='simple'))
        
        return results

class ResultsManager:
    """Manages output and results storage"""
    
    def __init__(self, output_dir: str = '../results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = self.output_dir / f"vecm_master_results_{timestamp}.txt"
        self.json_file = self.output_dir / f"vecm_master_results_{timestamp}.json"
        self.output_capture = io.StringIO()
        
    def setup_dual_output(self):
        """Setup dual output to stdout and StringIO"""
        self.original_stdout = sys.stdout
        sys.stdout = DualWriter(self.original_stdout, self.output_capture)
        
    def restore_output(self):
        """Restore original stdout"""
        sys.stdout = self.original_stdout
        
    def write_header(self):
        """Write analysis header"""
        print("="*80)
        print("VECM MASTER FRAMEWORK - COMPREHENSIVE RESULTS")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    def write_model_results(self, name: str, result: Dict[str, Any]):
        """Write individual model results"""
        print("\n" + "="*80)
        print(f"MODEL: {name}")
        print("="*80)
        
        config = result.get('config', {})
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Frequency: {config.get('freq', 'N/A')}")
        print(f"Variables: {', '.join(config.get('variables', []))}")
        print(f"Lag order (k_ar_diff): {config.get('k_ar_diff', 'N/A')}")
        print(f"Deterministic: {config.get('deterministic', 'N/A')}")
        
        if 'error' not in result:
            print(f"\nMODEL FIT STATISTICS:")
            print(f"Sample size: {result.get('nobs', 'N/A')}")
            print(f"Log-likelihood: {result.get('llf', 'N/A'):.2f}")
            print(f"AIC: {result.get('aic', 'N/A'):.2f}")
            print(f"BIC: {result.get('bic', 'N/A'):.2f}")
            print(f"HQIC: {result.get('hqic', 'N/A'):.2f}")
            print(f"Cointegration rank: {result.get('coint_rank', 'N/A')}")
        
    def write_comparison_table(self, results: Dict[str, Dict]):
        """Write comparison table across models"""
        print("\n" + "="*80)
        print("CROSS-MODEL COMPARISON")
        print("="*80)
        
        # Model fit comparison
        print("\nMODEL FIT COMPARISON")
        print("-"*80)
        
        rows = []
        for name, result in results.items():
            if 'error' not in result:
                config = result.get('config', {})
                
                rows.append({
                    'Model': name,
                    'Freq': config.get('freq', 'N/A'),
                    'Vars': len(config.get('variables', [])),
                    'Rank': result.get('coint_rank', 0),
                    'N': result.get('nobs', 0),
                    'Log-L': f"{result.get('llf', 0):.1f}",
                    'AIC': f"{result.get('aic', 0):.1f}",
                    'BIC': f"{result.get('bic', 0):.1f}"
                })
        
        if rows:
            df = pd.DataFrame(rows)
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Diagnostics comparison
        print("\nDIAGNOSTICS COMPARISON")
        print("-"*80)
        
        diag_rows = []
        for name, result in results.items():
            if 'error' not in result and 'diagnostics' in result:
                diag = result['diagnostics']
                
                jb_pval = diag.get('jarque_bera', {}).get('pvalue', np.nan)
                lb10_pval = diag.get('ljung_box_lag10', {}).get('pvalue', np.nan)
                stable = diag.get('stable', None)
                
                diag_rows.append({
                    'Model': name,
                    'JB p-val': f"{jb_pval:.4f}" if not np.isnan(jb_pval) else "—",
                    'LB(10) p-val': f"{lb10_pval:.4f}" if not np.isnan(lb10_pval) else "—",
                    'Stable': "Yes" if stable else ("No" if stable is False else "—")
                })
        
        if diag_rows:
            df = pd.DataFrame(diag_rows)
            print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))
        
        # Adjustment speeds comparison
        print("\nADJUSTMENT SPEEDS (ALPHA) FOR KEY VARIABLES")
        print("-"*80)
        
        alpha_rows = []
        for name, result in results.items():
            if 'error' not in result and 'alpha' in result:
                config = result.get('config', {})
                variables = config.get('variables', [])
                alpha = np.array(result['alpha'])
                
                row = {'Model': name}
                
                # Get first EC term coefficients for main variables
                if 'log_gmv' in variables and alpha.shape[1] > 0:
                    idx = variables.index('log_gmv')
                    row['α(GMV)'] = f"{alpha[idx, 0]:.4f}"
                
                if 'log_clicks' in variables and alpha.shape[1] > 0:
                    idx = variables.index('log_clicks')
                    row['α(Clicks)'] = f"{alpha[idx, 0]:.4f}"
                
                if 'log_impressions' in variables and alpha.shape[1] > 0:
                    idx = variables.index('log_impressions')
                    row['α(Impressions)'] = f"{alpha[idx, 0]:.4f}"
                
                alpha_rows.append(row)
        
        if alpha_rows:
            df = pd.DataFrame(alpha_rows)
            print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))
    
    def save_results(self, results: Dict):
        """Save results to files"""
        # Save text output
        with open(self.results_file, 'w') as f:
            f.write(self.output_capture.getvalue())
        
        # Save JSON
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        clean_results = json.loads(json.dumps(results, default=convert_types))
        
        with open(self.json_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        return self.results_file, self.json_file

def get_model_registry() -> List[ModelConfig]:
    """Define all model configurations"""
    return [
        # BASELINE MODELS
        ModelConfig(
            name="M1_Bivariate_Daily",
            variables=["log_gmv", "log_clicks"],
            freq="daily",
            description="Baseline bivariate: GMV-Clicks"
        ),
        ModelConfig(
            name="M2_Trivariate_Daily",
            variables=["log_gmv", "log_clicks", "log_auctions"],
            freq="daily",
            description="Trivariate: GMV-Clicks-Auctions"
        ),
        ModelConfig(
            name="M3_Full_Funnel_Daily",
            variables=["log_gmv", "log_clicks", "log_impressions", "log_auctions"],
            freq="daily",
            description="Full funnel with impressions"
        ),
        
        # SUPPLY-SIDE MODELS
        ModelConfig(
            name="M4_Supply_Vendors_Daily",
            variables=["log_gmv", "log_clicks", "log_auctions", "log_unique_vendors"],
            freq="daily",
            description="With vendor diversity"
        ),
        ModelConfig(
            name="M5_Supply_Products_Daily",
            variables=["log_gmv", "log_clicks", "log_auctions", "log_unique_products"],
            freq="daily",
            description="With product variety"
        ),
        ModelConfig(
            name="M6_Supply_Density_Daily",
            variables=["log_gmv", "log_clicks", "log_auctions", "log_bid_density", "log_auction_fill_rate"],
            freq="daily",
            description="With auction metrics"
        ),
        
        # EFFICIENCY MODELS
        ModelConfig(
            name="M7_Efficiency_CTR_Daily",
            variables=["log_gmv", "log_clicks", "log_ctr"],
            freq="daily",
            description="With click-through rate"
        ),
        ModelConfig(
            name="M8_Efficiency_Revenue_Daily",
            variables=["log_gmv", "log_clicks", "log_revenue_per_click"],
            freq="daily",
            description="With revenue efficiency"
        ),
        
        # REVERSE CAUSALITY
        ModelConfig(
            name="M9_Reverse_Clicks_Daily",
            variables=["log_clicks", "log_gmv", "log_auctions"],
            freq="daily",
            description="Reverse: Clicks as outcome"
        ),
        ModelConfig(
            name="M10_Reverse_Impressions_Daily",
            variables=["log_impressions", "log_clicks", "log_gmv"],
            freq="daily",
            description="Reverse: Impressions as outcome"
        ),
        
        # WEEKLY AGGREGATION
        ModelConfig(
            name="M11_Baseline_Weekly",
            variables=["log_gmv", "log_clicks", "log_auctions"],
            freq="weekly",
            description="Weekly baseline for stability"
        ),
        ModelConfig(
            name="M12_Full_Weekly",
            variables=["log_gmv", "log_clicks", "log_impressions", "log_auctions", "log_unique_vendors"],
            freq="weekly",
            description="Weekly full model"
        ),
        
        # HOURLY MODELS
        ModelConfig(
            name="M13_Baseline_Hourly",
            variables=["log_gmv", "log_clicks"],
            freq="hourly",
            k_ar_diff=24,
            description="Hourly bivariate"
        ),
        ModelConfig(
            name="M14_Hourly_Auctions",
            variables=["log_gmv", "log_clicks", "log_auctions"],
            freq="hourly",
            k_ar_diff=24,
            description="Hourly with auctions"
        ),
        
        # ALTERNATIVE DETERMINISTIC SPECS
        ModelConfig(
            name="M15_Constant_Only",
            variables=["log_gmv", "log_clicks", "log_auctions"],
            freq="daily",
            deterministic="co",
            description="Constant only (no trend)"
        ),
        ModelConfig(
            name="M16_Linear_Trend",
            variables=["log_gmv", "log_clicks", "log_auctions"],
            freq="daily",
            deterministic="colo",
            description="Constant outside, linear trend"
        ),
        
        # COMPREHENSIVE MODELS
        ModelConfig(
            name="M17_Kitchen_Sink_Daily",
            variables=["log_gmv", "log_clicks", "log_impressions", "log_auctions", 
                      "log_unique_vendors", "log_unique_products", "log_bid_density"],
            freq="daily",
            description="All variables daily"
        ),
        ModelConfig(
            name="M18_User_Focused_Daily",
            variables=["log_gmv", "log_clicks", "log_clicks_unique_users", "log_purchases_unique_users"],
            freq="daily",
            description="User-centric metrics"
        ),
        ModelConfig(
            name="M19_Transaction_Focused_Daily",
            variables=["log_gmv", "log_purchases", "log_avg_order_value", "log_clicks"],
            freq="daily",
            description="Transaction metrics"
        ),
        ModelConfig(
            name="M20_Funnel_Efficiency_Daily",
            variables=["log_gmv", "log_ctr", "log_cvr", "log_revenue_per_click"],
            freq="daily",
            description="Pure efficiency metrics"
        )
    ]

def main():
    """Main execution function"""
    
    # Setup results manager
    results_manager = ResultsManager()
    results_manager.setup_dual_output()
    results_manager.write_header()
    
    print("\nINITIALIZING VECM MASTER FRAMEWORK")
    print("="*80)
    
    # Load data
    loader = DataLoader()
    loader.load_raw_data()
    
    # Get model registry
    model_registry = get_model_registry()
    print(f"\nMODEL REGISTRY: {len(model_registry)} models configured")
    
    # Store all results
    all_results = {}
    
    # Process each model
    for i, config in enumerate(model_registry, 1):
        print("\n" + "="*80)
        print(f"PROCESSING MODEL {i}/{len(model_registry)}: {config.name}")
        print("="*80)
        
        # Aggregate to required frequency
        df = loader.aggregate_to_frequency(config.freq)
        
        # Build variables
        variable_builder = VariableBuilder()
        
        # Add all transformations
        df = variable_builder.add_log_transforms(df, [
            'clicks_total_clicks', 'purchases_gmv', 'impressions_total_impressions',
            'auctions_total_auctions', 'auctions_total_bids', 'impressions_unique_products',
            'impressions_unique_vendors', 'purchases_num_purchases', 'clicks_unique_users',
            'purchases_unique_users', 'clicks_unique_vendors'
        ])
        
        df = variable_builder.add_time_features(df, config.freq)
        df = variable_builder.add_supply_metrics(df)
        df = variable_builder.add_efficiency_metrics(df)
        
        # Rename columns for consistency
        df.columns = [col.replace('clicks_total_clicks', 'clicks')
                     .replace('purchases_gmv', 'gmv')
                     .replace('impressions_total_impressions', 'impressions')
                     .replace('auctions_total_auctions', 'auctions')
                     .replace('purchases_num_purchases', 'purchases')
                     .replace('clicks_unique_users', 'clicks_unique_users')
                     .replace('purchases_unique_users', 'purchases_unique_users')
                     .replace('impressions_unique_vendors', 'unique_vendors')
                     .replace('impressions_unique_products', 'unique_products')
                     .replace('clicks_unique_vendors', 'clicked_vendors')
                     for col in df.columns]
        
        # Estimate model
        estimator = VECMEstimator(df)
        result = estimator.estimate_model(config)
        
        # Store result
        all_results[config.name] = result
        
        # Print summary
        if 'error' not in result:
            print(f"\n✓ Model completed successfully")
            print(f"  Cointegration rank: {result.get('coint_rank', 'N/A')}")
            print(f"  Log-likelihood: {result.get('llf', 'N/A'):.2f}")
            print(f"  BIC: {result.get('bic', 'N/A'):.2f}")
        else:
            print(f"\n✗ Model failed: {result['error'][:100]}...")
    
    # Write comparison tables
    results_manager.write_comparison_table(all_results)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    successful = sum(1 for r in all_results.values() if 'error' not in r)
    failed = len(all_results) - successful
    
    print(f"\nModels processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Save results
    text_file, json_file = results_manager.save_results(all_results)
    
    # Restore output
    results_manager.restore_output()
    
    print(f"\nResults saved to:")
    print(f"  Text: {text_file}")
    print(f"  JSON: {json_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()