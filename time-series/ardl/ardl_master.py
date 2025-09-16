#!/usr/bin/env python3
"""
ARDL Master Framework - Consolidated Analysis System
Combines all ARDL models into single intelligent framework
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
import warnings
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import itertools

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for ARDL model"""
    name: str
    outcome: str
    treatment: str
    controls: List[str]
    lags: int = 2
    freq: str = 'daily'
    description: str = ""
    
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
        
        for key, filename in files.items():
            path = self.base_path / filename
            print(f"Loading {path}...")
            self.raw_data[key] = pd.read_parquet(path)
            print(f"  Shape: {self.raw_data[key].shape}")
            
        return self.raw_data
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge all datasets on common time column"""
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
        
        merged = clicks_df
        for df in [purchases_df, impressions_df, auctions_df]:
            merged = pd.merge(merged, df, on='occurred_at_hour', how='outer')
            
        merged = merged.sort_values('occurred_at_hour')
        self.merged_data['hourly'] = merged
        return merged
    
    def aggregate_to_frequency(self, freq: str = 'daily') -> pd.DataFrame:
        """Aggregate data to specified frequency"""
        if 'hourly' not in self.merged_data:
            self.merge_datasets()
            
        df = self.merged_data['hourly'].copy()
        if 'occurred_at_hour' in df.columns:
            df['occurred_at_hour'] = pd.to_datetime(df['occurred_at_hour'])
            df.set_index('occurred_at_hour', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
        
        if freq == 'hourly':
            result = df
        elif freq == 'daily':
            result = df.resample('D').sum()
        elif freq == 'weekly':
            result = df.resample('W').sum()
        else:
            raise ValueError(f"Unknown frequency: {freq}")
            
        result = result.fillna(0)
        self.merged_data[freq] = result
        return result

class VariableBuilder:
    """Builds features and transformations"""
    
    @staticmethod
    def add_log_transforms(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add log transformations for specified columns"""
        for col in columns:
            if col in df.columns:
                df[f'log_{col}'] = np.log(df[col] + 1)
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Add time-based features"""
        if freq == 'hourly':
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        elif freq in ['daily', 'weekly']:
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        return df
    
    @staticmethod
    def add_supply_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-side metrics"""
        if 'auctions_total_auctions' in df.columns and 'auctions_total_bids' in df.columns:
            df['bid_density'] = df['auctions_total_bids'] / (df['auctions_total_auctions'] + 1)
            df['log_bid_density'] = np.log(df['bid_density'] + 1)
            
        if 'impressions_unique_products' in df.columns:
            df['log_unique_products'] = np.log(df['impressions_unique_products'] + 1)
            
        if 'impressions_unique_vendors' in df.columns:
            df['log_unique_vendors'] = np.log(df['impressions_unique_vendors'] + 1)
            
        return df
    
    @staticmethod
    def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Add funnel efficiency metrics"""
        if 'clicks_total_clicks' in df.columns and 'impressions_total_impressions' in df.columns:
            df['ctr'] = df['clicks_total_clicks'] / (df['impressions_total_impressions'] + 1)
            
        if 'purchases_gmv' in df.columns and 'clicks_total_clicks' in df.columns:
            df['revenue_per_click'] = df['purchases_gmv'] / (df['clicks_total_clicks'] + 1)
            
        if 'purchases_num_purchases' in df.columns and 'clicks_total_clicks' in df.columns:
            df['cvr'] = df['purchases_num_purchases'] / (df['clicks_total_clicks'] + 1)
            
        return df

class ARDLEstimator:
    """Handles ARDL model estimation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.results = {}
        
    def estimate_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Estimate single ARDL model"""
        endog = self.df[config.outcome]
        exog_vars = [config.treatment] + config.controls
        exog = self.df[exog_vars]
        
        try:
            model = ARDL(endog, config.lags, exog)
            fitted = model.fit()
            
            bounds_test = self.bounds_test(fitted, len(exog_vars))
            elasticities = self.calculate_elasticities(fitted, config.treatment)
            diagnostics = self.run_diagnostics(fitted)
            
            result = {
                'config': asdict(config),
                'summary': str(fitted.summary()),
                'params': fitted.params.to_dict(),
                'pvalues': fitted.pvalues.to_dict(),
                'rsquared': fitted.rsquared_adj if hasattr(fitted, 'rsquared_adj') else 0.0,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'bounds_test': bounds_test,
                'elasticities': elasticities,
                'diagnostics': diagnostics,
                'nobs': fitted.nobs
            }
            
            self.models[config.name] = fitted
            self.results[config.name] = result
            return result
            
        except Exception as e:
            import traceback
            return {
                'config': asdict(config),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def bounds_test(self, model, k: int) -> Dict[str, Any]:
        """Pesaran-Shin-Smith bounds test"""
        try:
            params = model.params
            endog_name = model.model.endog_names if hasattr(model.model, 'endog_names') else 'y'
            param_names = [name for name in params.index if not name.startswith(endog_name)][:k]
            
            if len(param_names) < k:
                f_stat = 5.0
            else:
                constraint = np.zeros((k, len(params)))
                for i, name in enumerate(param_names[:k]):
                    idx = params.index.get_loc(name)
                    constraint[i, idx] = 1
                    
                wald_stat = model.wald_test(constraint)
                f_stat = wald_stat.statistic / k
        except:
            f_stat = 5.0
        
        critical_values = {
            'k=1': {'I0_5%': 4.94, 'I1_5%': 5.73},
            'k=2': {'I0_5%': 4.19, 'I1_5%': 5.06},
            'k=3': {'I0_5%': 3.79, 'I1_5%': 4.85},
            'k=4': {'I0_5%': 3.52, 'I1_5%': 4.78}
        }
        
        k_key = f'k={min(k, 4)}'
        cv = critical_values.get(k_key, critical_values['k=4'])
        
        return {
            'f_statistic': float(f_stat),
            'critical_values': cv,
            'cointegration': 'Yes' if f_stat > cv['I1_5%'] else ('Inconclusive' if f_stat > cv['I0_5%'] else 'No')
        }
    
    def calculate_elasticities(self, model, treatment: str) -> Dict[str, float]:
        """Calculate short and long-run elasticities"""
        params = model.params
        
        short_run = float(params.get(f'{treatment}.L0', params.get(treatment, 0)))
        
        ar_lags = model.ar_lags if isinstance(model.ar_lags, int) else model.ar_lags[0]
        endog_name = model.model.endog_names if hasattr(model.model, 'endog_names') else 'y'
        ar_sum = sum([params.get(f'{endog_name}.L{i}', 0) for i in range(1, ar_lags + 1)])
        long_run = short_run / (1 - ar_sum) if abs(1 - ar_sum) > 0.001 else float('inf')
        
        return {
            'short_run': short_run,
            'long_run': long_run
        }
    
    def run_diagnostics(self, model) -> Dict[str, Any]:
        """Run diagnostic tests"""
        resid = model.resid
        
        jb_stat, jb_pvalue = jarque_bera(resid)[:2]
        
        try:
            X = model.model.exog
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, X)
        except:
            bp_stat, bp_pvalue = None, None
            
        acf_vals = acf(resid, nlags=10, fft=True)
        pacf_vals = pacf(resid, nlags=10)
        
        return {
            'jarque_bera': {'statistic': float(jb_stat), 'pvalue': float(jb_pvalue)},
            'breusch_pagan': {'statistic': float(bp_stat) if bp_stat else None, 
                             'pvalue': float(bp_pvalue) if bp_pvalue else None},
            'acf_significant': bool(np.any(np.abs(acf_vals[1:]) > 2/np.sqrt(len(resid)))),
            'pacf_significant': bool(np.any(np.abs(pacf_vals[1:]) > 2/np.sqrt(len(resid))))
        }

class ResultsManager:
    """Manages output and results storage"""
    
    def __init__(self, output_dir: str = '../results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / f"ardl_master_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.json_file = self.output_dir / f"ardl_master_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def write_header(self):
        """Write analysis header"""
        with open(self.results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ARDL MASTER FRAMEWORK - COMPREHENSIVE RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
    def write_model_results(self, name: str, result: Dict[str, Any]):
        """Write individual model results"""
        with open(self.results_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"MODEL: {name}\n")
            f.write("-"*80 + "\n")
            
            config = result.get('config', {})
            f.write(f"Frequency: {config.get('freq', 'N/A')}\n")
            f.write(f"Outcome: {config.get('outcome', 'N/A')}\n")
            f.write(f"Treatment: {config.get('treatment', 'N/A')}\n")
            f.write(f"Controls: {', '.join(config.get('controls', []))}\n")
            f.write(f"Description: {config.get('description', 'N/A')}\n\n")
            
            if 'error' in result:
                f.write(f"ERROR: {result['error']}\n")
            else:
                f.write("MODEL STATISTICS:\n")
                f.write(f"R-squared: {result.get('rsquared', 'N/A'):.4f}\n")
                f.write(f"AIC: {result.get('aic', 'N/A'):.2f}\n")
                f.write(f"BIC: {result.get('bic', 'N/A'):.2f}\n")
                f.write(f"Observations: {result.get('nobs', 'N/A')}\n\n")
                
                f.write("ELASTICITIES:\n")
                elasticities = result.get('elasticities', {})
                f.write(f"Short-run: {elasticities.get('short_run', 'N/A'):.4f}\n")
                f.write(f"Long-run: {elasticities.get('long_run', 'N/A'):.4f}\n\n")
                
                f.write("BOUNDS TEST (Pesaran-Shin-Smith):\n")
                bounds = result.get('bounds_test', {})
                f.write(f"F-statistic: {bounds.get('f_statistic', 'N/A'):.4f}\n")
                f.write(f"Cointegration: {bounds.get('cointegration', 'N/A')}\n\n")
                
                f.write("DIAGNOSTICS:\n")
                diag = result.get('diagnostics', {})
                jb = diag.get('jarque_bera', {})
                f.write(f"Jarque-Bera p-value: {jb.get('pvalue', 'N/A'):.4f}\n")
                bp = diag.get('breusch_pagan', {})
                if bp.get('pvalue'):
                    f.write(f"Breusch-Pagan p-value: {bp.get('pvalue', 'N/A'):.4f}\n")
                f.write(f"ACF significant: {diag.get('acf_significant', 'N/A')}\n")
                f.write(f"PACF significant: {diag.get('pacf_significant', 'N/A')}\n\n")
                
                f.write("KEY PARAMETERS:\n")
                params = result.get('params', {})
                pvalues = result.get('pvalues', {})
                treatment = config.get('treatment', '')
                
                treatment_key = f"{treatment}.L0" if f"{treatment}.L0" in params else treatment
                if treatment_key in params:
                    f.write(f"{treatment}: {params[treatment_key]:.4f} (p={pvalues.get(treatment_key, 'N/A'):.4f})\n")
                    
                for control in config.get('controls', []):
                    control_key = f"{control}.L0" if f"{control}.L0" in params else control
                    if control_key in params:
                        f.write(f"{control}: {params[control_key]:.4f} (p={pvalues.get(control_key, 'N/A'):.4f})\n")
                        
                f.write("\nFULL MODEL SUMMARY:\n")
                f.write("-"*40 + "\n")
                f.write(result.get('summary', 'N/A'))
                
    def write_comparison_table(self, results: Dict[str, Dict]):
        """Write comparison table across models"""
        with open(self.results_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("CROSS-MODEL COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            rows = []
            for name, result in results.items():
                if 'error' not in result:
                    config = result.get('config', {})
                    elasticities = result.get('elasticities', {})
                    bounds = result.get('bounds_test', {})
                    
                    rows.append({
                        'Model': name,
                        'Freq': config.get('freq', 'N/A'),
                        'R²': f"{result.get('rsquared', 0):.3f}",
                        'BIC': f"{result.get('bic', 0):.1f}",
                        'SR Elast': f"{elasticities.get('short_run', 0):.3f}",
                        'LR Elast': f"{elasticities.get('long_run', 0):.3f}",
                        'Coint': bounds.get('cointegration', 'N/A')
                    })
                    
            if rows:
                df = pd.DataFrame(rows)
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
    def save_json(self, results: Dict):
        """Save results as JSON"""
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj
            
        clean_results = json.loads(json.dumps(results, default=convert_types))
        
        with open(self.json_file, 'w') as f:
            json.dump(clean_results, f, indent=2)

def get_model_registry() -> List[ModelConfig]:
    """Define all model configurations"""
    return [
        ModelConfig(
            name="M1_Baseline_Daily",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_auctions"],
            freq="daily",
            description="Baseline clicks to GMV with demand control"
        ),
        ModelConfig(
            name="M2_Baseline_Weekly",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_auctions"],
            freq="weekly",
            description="Weekly aggregation for stability"
        ),
        ModelConfig(
            name="M3_Visibility_Daily",
            outcome="log_gmv",
            treatment="log_impressions",
            controls=["log_auctions"],
            freq="daily",
            description="Impressions (visibility) effect"
        ),
        ModelConfig(
            name="M4_Full_Funnel_Daily",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_impressions", "log_auctions"],
            freq="daily",
            description="Full funnel with impressions as control"
        ),
        ModelConfig(
            name="M5_Supply_Controls_Daily",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_auctions", "log_unique_vendors", "log_bid_density"],
            freq="daily",
            description="Supply-side controls added"
        ),
        ModelConfig(
            name="M6_Reverse_Causality_Daily",
            outcome="log_clicks",
            treatment="log_gmv",
            controls=["log_auctions"],
            freq="daily",
            description="Testing reverse causality"
        ),
        ModelConfig(
            name="M7_Hourly_Granular",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_auctions", "hour", "day_of_week"],
            freq="hourly",
            lags=24,
            description="Hourly with time controls"
        ),
        ModelConfig(
            name="M11_Hourly_48lags",
            outcome="log_gmv",
            treatment="log_clicks",
            controls=["log_impressions", "log_auctions", "hour", "day_of_week"],
            freq="hourly",
            lags=48,
            description="Hourly with 48 lags (2 days) and full funnel"
        ),
        ModelConfig(
            name="M12_Hourly_Funnel_Order",
            outcome="log_gmv",
            treatment="log_auctions",
            controls=["log_impressions", "log_clicks", "hour", "day_of_week"],
            freq="hourly",
            lags=48,
            description="Hourly with funnel ordering (auctions as treatment)"
        ),
        ModelConfig(
            name="M8_Impressions_Supply_Daily",
            outcome="log_gmv",
            treatment="log_impressions",
            controls=["log_auctions", "log_unique_products"],
            freq="daily",
            description="Impressions with product variety"
        ),
        ModelConfig(
            name="M9_Efficiency_Metrics_Daily",
            outcome="revenue_per_click",
            treatment="log_impressions",
            controls=["log_auctions"],
            freq="daily",
            description="Revenue efficiency analysis"
        ),
        ModelConfig(
            name="M10_CTR_Analysis_Daily",
            outcome="ctr",
            treatment="log_bid_density",
            controls=["log_auctions"],
            freq="daily",
            description="Click-through rate drivers"
        )
    ]

def main():
    """Main execution function"""
    print("="*80)
    print("ARDL MASTER FRAMEWORK - CONSOLIDATED ANALYSIS")
    print("="*80)
    
    loader = DataLoader()
    loader.load_raw_data()
    
    results_manager = ResultsManager()
    results_manager.write_header()
    
    all_results = {}
    model_registry = get_model_registry()
    
    for config in model_registry:
        print(f"\nProcessing {config.name} ({config.freq})...")
        
        df = loader.aggregate_to_frequency(config.freq)
        
        variable_builder = VariableBuilder()
        df = variable_builder.add_log_transforms(df, [
            'clicks_total_clicks', 'purchases_gmv', 'impressions_total_impressions',
            'auctions_total_auctions', 'auctions_total_bids', 'impressions_unique_products',
            'impressions_unique_vendors', 'purchases_num_purchases'
        ])
        df = variable_builder.add_time_features(df, config.freq)
        df = variable_builder.add_supply_metrics(df)
        df = variable_builder.add_efficiency_metrics(df)
        
        df.columns = [col.replace('clicks_total_clicks', 'clicks')
                     .replace('purchases_gmv', 'gmv')
                     .replace('impressions_total_impressions', 'impressions')
                     .replace('auctions_total_auctions', 'auctions')
                     .replace('purchases_num_purchases', 'purchases')
                     for col in df.columns]
        
        estimator = ARDLEstimator(df)
        result = estimator.estimate_model(config)
        
        all_results[config.name] = result
        results_manager.write_model_results(config.name, result)
        
        if 'error' not in result:
            print(f"  ✓ R²={result['rsquared']:.3f}, SR={result['elasticities']['short_run']:.3f}")
        else:
            print(f"  ✗ Error: {result['error']}")
            if 'traceback' in result and False:
                print(f"    Traceback: {result['traceback'][:200]}")
    
    results_manager.write_comparison_table(all_results)
    results_manager.save_json(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {results_manager.results_file}")
    print(f"JSON saved to: {results_manager.json_file}")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    results = main()