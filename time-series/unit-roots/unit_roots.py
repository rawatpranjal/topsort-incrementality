#!/usr/bin/env python3
"""
Comprehensive Unit Root Testing for Time Series Variables
Tests for I(0), I(1), and I(2) integration orders
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
# from arch.unitroot import PhillipsPerron, DFGLS, ZivotAndrews  # Optional: install arch package for these tests
from datetime import datetime
from tabulate import tabulate
import sys
import io
from contextlib import redirect_stdout
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class UnitRootTester:
    """
    Comprehensive unit root testing for all marketplace variables
    """
    
    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.raw_data = {}
        self.hourly_data = None
        self.daily_data = None
        self.weekly_data = None
        self.test_results = {}
        self.integration_orders = {}
        
    def load_data(self):
        """Load all hourly data files"""
        print("="*80)
        print("DATA LOADING")
        print("="*80)
        print(f"Timestamp: {datetime.now()}")
        
        # Load all data files
        self.raw_data['clicks'] = pd.read_parquet(f'{self.data_path}hourly_clicks_2025-03-01_to_2025-09-30.parquet')
        self.raw_data['purchases'] = pd.read_parquet(f'{self.data_path}hourly_purchases_2025-03-01_to_2025-09-30.parquet')
        self.raw_data['impressions'] = pd.read_parquet(f'{self.data_path}hourly_impressions_2025-03-01_to_2025-09-30.parquet')
        self.raw_data['auctions'] = pd.read_parquet(f'{self.data_path}hourly_auctions_2025-03-01_to_2025-09-30.parquet')
        
        print("\nRaw data loaded:")
        for name, df in self.raw_data.items():
            print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"    Columns: {list(df.columns)}")
        
        return self.raw_data
    
    def prepare_daily_data(self):
        """Merge and aggregate to daily frequency"""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Merge all dataframes
        merged = self.raw_data['purchases'].merge(
            self.raw_data['clicks'], on='ACTIVITY_HOUR', how='outer'
        ).merge(
            self.raw_data['impressions'], on='ACTIVITY_HOUR', how='outer'
        ).merge(
            self.raw_data['auctions'], on='ACTIVITY_HOUR', how='outer'
        )
        
        merged['ACTIVITY_HOUR'] = pd.to_datetime(merged['ACTIVITY_HOUR'])
        merged = merged.fillna(0)
        merged['date'] = merged['ACTIVITY_HOUR'].dt.date
        
        # Aggregation rules
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
        
        # Filter available columns
        available_agg = {col: func for col, func in agg_rules.items() if col in merged.columns}
        
        # Aggregate to daily
        self.daily_data = merged.groupby('date').agg(available_agg).reset_index()
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        self.daily_data.set_index('date', inplace=True)
        
        print(f"Daily data shape: {self.daily_data.shape}")
        print(f"Date range: {self.daily_data.index.min()} to {self.daily_data.index.max()}")
        print(f"Variables: {len(self.daily_data.columns)}")
        
        return self.daily_data
    
    def prepare_hourly_data(self):
        """Prepare hourly frequency data"""
        print("\n" + "="*80)
        print("HOURLY DATA PREPARATION")
        print("="*80)
        
        # Merge all dataframes
        merged = self.raw_data['purchases'].merge(
            self.raw_data['clicks'], on='ACTIVITY_HOUR', how='outer'
        ).merge(
            self.raw_data['impressions'], on='ACTIVITY_HOUR', how='outer'
        ).merge(
            self.raw_data['auctions'], on='ACTIVITY_HOUR', how='outer'
        )
        
        merged['ACTIVITY_HOUR'] = pd.to_datetime(merged['ACTIVITY_HOUR'])
        merged = merged.fillna(0)
        
        # Set index to datetime for time series analysis
        self.hourly_data = merged.set_index('ACTIVITY_HOUR')
        
        # Remove non-numeric columns if any
        numeric_cols = self.hourly_data.select_dtypes(include=[np.number]).columns
        self.hourly_data = self.hourly_data[numeric_cols]
        
        print(f"Hourly data shape: {self.hourly_data.shape}")
        print(f"Date range: {self.hourly_data.index.min()} to {self.hourly_data.index.max()}")
        print(f"Variables: {len(self.hourly_data.columns)}")
        print(f"Observations: {len(self.hourly_data)}")
        
        return self.hourly_data
    
    def prepare_weekly_data(self):
        """Aggregate to weekly frequency"""
        print("\n" + "="*80)
        print("WEEKLY DATA PREPARATION")
        print("="*80)
        
        if self.daily_data is None:
            self.prepare_daily_data()
        
        # Create week identifier
        daily_with_week = self.daily_data.copy()
        daily_with_week['week'] = pd.to_datetime(daily_with_week.index).to_period('W')
        
        # Aggregation rules for weekly
        agg_rules = {}
        for col in daily_with_week.columns:
            if col != 'week':
                if 'VENDORS' in col or 'CAMPAIGNS' in col or 'PRODUCTS' in col:
                    agg_rules[col] = 'max'  # Max for diversity metrics
                else:
                    agg_rules[col] = 'sum'  # Sum for counts and values
        
        # Aggregate to weekly
        self.weekly_data = daily_with_week.groupby('week').agg(agg_rules)
        self.weekly_data.index = self.weekly_data.index.to_timestamp()
        
        print(f"Weekly data shape: {self.weekly_data.shape}")
        print(f"Date range: {self.weekly_data.index.min()} to {self.weekly_data.index.max()}")
        print(f"Variables: {len(self.weekly_data.columns)}")
        print(f"Observations: {len(self.weekly_data)}")
        
        if len(self.weekly_data) < 30:
            print(f"⚠ WARNING: Only {len(self.weekly_data)} weekly observations - results may be unreliable")
        
        return self.weekly_data
    
    def adf_test(self, series, regression='c', autolag='AIC'):
        """
        Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        series : array-like
            Time series to test
        regression : str
            'c' for constant, 'ct' for constant and trend, 'n' for none
        autolag : str
            Method for lag selection
            
        Returns:
        --------
        dict : Test results
        """
        try:
            result = adfuller(series, regression=regression, autolag=autolag)
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'usedlag': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'icbest': result[5] if len(result) > 5 else None
            }
        except Exception as e:
            print(f"ADF test failed: {str(e)}")
            return None
    
    def kpss_test(self, series, regression='c', nlags='auto'):
        """
        KPSS stationarity test
        
        Parameters:
        -----------
        series : array-like
            Time series to test
        regression : str
            'c' for constant, 'ct' for constant and trend
        nlags : str or int
            Number of lags or 'auto'
            
        Returns:
        --------
        dict : Test results
        """
        try:
            result = kpss(series, regression=regression, nlags=nlags)
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'usedlag': result[2],
                'critical_values': result[3]
            }
        except Exception as e:
            print(f"KPSS test failed: {str(e)}")
            return None
    
    def pp_test(self, series, trend='c'):
        """
        Phillips-Perron test (requires arch package)
        Currently disabled - install arch package to enable
        """
        return None  # Placeholder - install arch package for this test
    
    def dfgls_test(self, series, trend='c'):
        """
        DF-GLS test (Elliott-Rothenberg-Stock)
        Currently disabled - install arch package to enable
        """
        return None  # Placeholder - install arch package for this test
    
    def zivot_andrews_test(self, series, trend='c'):
        """
        Zivot-Andrews test for unit root with structural break
        Currently disabled - install arch package to enable
        """
        return None  # Placeholder - install arch package for this test
    
    def test_single_series(self, series, name, max_diff=2):
        """
        Run all unit root tests on a single series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        name : str
            Variable name
        max_diff : int
            Maximum differencing order to test
            
        Returns:
        --------
        dict : Complete test results
        """
        print(f"\nTesting: {name}")
        print("-"*60)
        
        results = {'name': name, 'tests': {}}
        
        # Clean series
        series_clean = series.dropna()
        if len(series_clean) < 30:
            print(f"  Insufficient observations ({len(series_clean)})")
            return results
        
        # Test at different differencing orders
        for diff_order in range(max_diff + 1):
            if diff_order == 0:
                test_series = series_clean
                label = "Level"
            else:
                test_series = series_clean.diff(diff_order).dropna()
                label = f"Diff({diff_order})"
            
            print(f"\n  {label}:")
            results['tests'][f'diff_{diff_order}'] = {}
            
            # ADF Test
            adf_c = self.adf_test(test_series, regression='c')
            adf_ct = self.adf_test(test_series, regression='ct')
            
            if adf_c:
                print(f"    ADF (c):  stat={adf_c['statistic']:.4f}, p={adf_c['pvalue']:.4f}")
                results['tests'][f'diff_{diff_order}']['adf_c'] = adf_c
            
            if adf_ct:
                print(f"    ADF (ct): stat={adf_ct['statistic']:.4f}, p={adf_ct['pvalue']:.4f}")
                results['tests'][f'diff_{diff_order}']['adf_ct'] = adf_ct
            
            # KPSS Test
            kpss_c = self.kpss_test(test_series, regression='c')
            
            if kpss_c:
                print(f"    KPSS (c): stat={kpss_c['statistic']:.4f}, p={kpss_c['pvalue']:.4f}")
                results['tests'][f'diff_{diff_order}']['kpss_c'] = kpss_c
            
            # PP Test
            pp_c = self.pp_test(test_series, trend='c')
            
            if pp_c:
                print(f"    PP (c):   stat={pp_c['statistic']:.4f}, p={pp_c['pvalue']:.4f}")
                results['tests'][f'diff_{diff_order}']['pp_c'] = pp_c
            
            # DF-GLS Test
            dfgls_c = self.dfgls_test(test_series, trend='c')
            
            if dfgls_c:
                print(f"    DF-GLS:   stat={dfgls_c['statistic']:.4f}, p={dfgls_c['pvalue']:.4f}")
                results['tests'][f'diff_{diff_order}']['dfgls_c'] = dfgls_c
        
        return results
    
    def determine_integration_order(self, test_results, significance=0.05):
        """
        Determine integration order based on test results
        
        Parameters:
        -----------
        test_results : dict
            Results from test_single_series
        significance : float
            Significance level
            
        Returns:
        --------
        tuple : (order, conclusion, details)
        """
        name = test_results['name']
        tests = test_results['tests']
        
        # Check each differencing level
        for diff_order in range(3):
            diff_key = f'diff_{diff_order}'
            if diff_key not in tests:
                continue
            
            diff_tests = tests[diff_key]
            
            # ADF results (null: unit root)
            adf_stationary = False
            if 'adf_c' in diff_tests and diff_tests['adf_c']:
                if diff_tests['adf_c']['pvalue'] < significance:
                    adf_stationary = True
            
            # KPSS results (null: stationary)
            kpss_stationary = True
            if 'kpss_c' in diff_tests and diff_tests['kpss_c']:
                if diff_tests['kpss_c']['pvalue'] < significance:
                    kpss_stationary = False
            
            # PP results (null: unit root)
            pp_stationary = False
            if 'pp_c' in diff_tests and diff_tests['pp_c']:
                if diff_tests['pp_c']['pvalue'] < significance:
                    pp_stationary = True
            
            # Decision logic
            if adf_stationary and kpss_stationary:
                return (diff_order, f"I({diff_order})", "Stationary (ADF rejects, KPSS accepts)")
            elif adf_stationary and not kpss_stationary:
                return (diff_order, f"I({diff_order})*", "Likely stationary (conflicting tests)")
            elif pp_stationary:
                return (diff_order, f"I({diff_order})", "Stationary (PP rejects unit root)")
        
        return (None, "I(2+)", "Non-stationary even after second differencing")
    
    def test_all_variables(self, frequency='daily'):
        """Test all variables at specified frequency"""
        print("\n" + "="*80)
        print(f"UNIT ROOT TESTING - {frequency.upper()} FREQUENCY")
        print("="*80)
        
        # Select appropriate dataset
        if frequency == 'hourly':
            data = self.hourly_data
        elif frequency == 'daily':
            data = self.daily_data
        elif frequency == 'weekly':
            data = self.weekly_data
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        if data is None:
            print(f"No {frequency} data available")
            return
        
        # Initialize results for this frequency
        freq_test_results = {}
        freq_integration_orders = {}
        
        for col in data.columns:
            # Test raw series
            series = data[col]
            results = self.test_single_series(series, f"{col}_{frequency}")
            freq_test_results[col] = results
            
            # Determine integration order
            order, conclusion, details = self.determine_integration_order(results)
            freq_integration_orders[col] = {
                'order': order,
                'conclusion': conclusion,
                'details': details
            }
            
            # Also test log-transformed series for positive variables
            if (series > 0).all():
                log_series = np.log(series)
                log_results = self.test_single_series(log_series, f"log({col})_{frequency}")
                freq_test_results[f"log_{col}"] = log_results
                
                order, conclusion, details = self.determine_integration_order(log_results)
                freq_integration_orders[f"log_{col}"] = {
                    'order': order,
                    'conclusion': conclusion,
                    'details': details
                }
        
        # Store results by frequency
        self.test_results[frequency] = freq_test_results
        self.integration_orders[frequency] = freq_integration_orders
        
        return freq_integration_orders
    
    def test_all_frequencies(self):
        """Test all variables at all available frequencies"""
        print("\n" + "="*80)
        print("MULTI-FREQUENCY UNIT ROOT TESTING")
        print("="*80)
        
        all_freq_results = {}
        
        # Test at each frequency
        for freq in ['hourly', 'daily', 'weekly']:
            print(f"\nProcessing {freq} frequency...")
            if freq == 'hourly' and self.hourly_data is not None:
                all_freq_results[freq] = self.test_all_variables('hourly')
            elif freq == 'daily' and self.daily_data is not None:
                all_freq_results[freq] = self.test_all_variables('daily')
            elif freq == 'weekly' and self.weekly_data is not None:
                all_freq_results[freq] = self.test_all_variables('weekly')
        
        return all_freq_results
    
    def generate_summary_table(self):
        """Generate summary table of integration orders across frequencies"""
        print("\n" + "="*80)
        print("SUMMARY: INTEGRATION ORDERS BY FREQUENCY")
        print("="*80)
        
        # Check which frequencies we have
        available_freqs = [freq for freq in ['hourly', 'daily', 'weekly'] 
                          if freq in self.integration_orders]
        
        if not available_freqs:
            print("No test results available")
            return
        
        # If only one frequency (backward compatibility)
        if len(available_freqs) == 1 and available_freqs[0] == 'daily':
            # Old single-frequency format
            freq_orders = self.integration_orders['daily']
            freq_results = self.test_results['daily']
            
            summary_data = []
            for var_name, order_info in freq_orders.items():
                if var_name in freq_results:
                    level_tests = freq_results[var_name]['tests'].get('diff_0', {})
                    
                    adf_stat = level_tests.get('adf_c', {}).get('statistic', None)
                    adf_pval = level_tests.get('adf_c', {}).get('pvalue', None)
                    kpss_stat = level_tests.get('kpss_c', {}).get('statistic', None)
                    kpss_pval = level_tests.get('kpss_c', {}).get('pvalue', None)
                    
                    summary_data.append([
                        var_name[:30],
                        f"{adf_stat:.3f}" if adf_stat else "—",
                        f"{adf_pval:.3f}" if adf_pval else "—",
                        f"{kpss_stat:.3f}" if kpss_stat else "—",
                        f"{kpss_pval:.3f}" if kpss_pval else "—",
                        order_info['conclusion'],
                        order_info['details'][:40]
                    ])
            
            headers = ['Variable', 'ADF Stat', 'ADF p', 'KPSS Stat', 'KPSS p', 'Order', 'Conclusion']
            print(tabulate(summary_data, headers=headers, tablefmt='grid'))
            
        else:
            # Multi-frequency comparison table
            print("\nCOMPARISON ACROSS FREQUENCIES")
            print("-"*80)
            
            # Get unique variable names (without log prefix)
            all_vars = set()
            for freq in available_freqs:
                for var in self.integration_orders[freq].keys():
                    if not var.startswith('log_'):
                        all_vars.add(var)
            
            comparison_data = []
            for var in sorted(all_vars):
                row = [var[:25]]  # Variable name
                
                # Add integration order for each frequency
                for freq in ['hourly', 'daily', 'weekly']:
                    if freq in available_freqs and var in self.integration_orders[freq]:
                        order = self.integration_orders[freq][var]['conclusion']
                        row.append(order)
                    else:
                        row.append("—")
                
                # Add consensus/notes
                orders = [self.integration_orders[freq][var]['conclusion'] 
                         for freq in available_freqs 
                         if freq in self.integration_orders and var in self.integration_orders[freq]]
                
                if len(set(orders)) == 1:
                    consensus = orders[0]
                else:
                    consensus = "Mixed"
                row.append(consensus)
                
                comparison_data.append(row)
            
            headers = ['Variable', 'Hourly', 'Daily', 'Weekly', 'Consensus']
            print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
            
            # Print frequency-specific summaries
            for freq in available_freqs:
                print(f"\n{freq.upper()} FREQUENCY SUMMARY:")
                print("-"*40)
                
                order_counts = {}
                for var, info in self.integration_orders[freq].items():
                    if not var.startswith('log_'):
                        order = info['conclusion'].split('*')[0]
                        order_counts[order] = order_counts.get(order, 0) + 1
                
                for order, count in sorted(order_counts.items()):
                    print(f"  {order}: {count} variables")
        
        return comparison_data if len(available_freqs) > 1 else summary_data
    
    def test_cointegration_requirements(self):
        """Check if variables meet cointegration requirements"""
        print("\n" + "="*80)
        print("COINTEGRATION REQUIREMENTS CHECK")
        print("="*80)
        
        # Check if we have the new structure (by frequency) or old structure
        if isinstance(self.integration_orders, dict):
            # Check if it's frequency-based or variable-based
            first_key = list(self.integration_orders.keys())[0] if self.integration_orders else None
            
            if first_key in ['hourly', 'daily', 'weekly']:
                # New structure - pick daily if available, otherwise first frequency
                if 'daily' in self.integration_orders:
                    orders_to_check = self.integration_orders['daily']
                else:
                    freq = list(self.integration_orders.keys())[0]
                    orders_to_check = self.integration_orders[freq]
                    print(f"Using {freq} frequency for cointegration assessment")
            else:
                # Old structure - direct variable mapping
                orders_to_check = self.integration_orders
        else:
            print("No integration order results available")
            return
        
        # Separate by integration order
        i0_vars = []
        i1_vars = []
        i2_vars = []
        
        for var, info in orders_to_check.items():
            if not var.startswith('log_'):  # Skip log-transformed duplicates
                order = info['conclusion']
                if 'I(0)' in order:
                    i0_vars.append(var)
                elif 'I(1)' in order:
                    i1_vars.append(var)
                elif 'I(2)' in order:
                    i2_vars.append(var)
        
        print(f"\nI(0) Variables (Stationary): {len(i0_vars)}")
        for var in i0_vars[:10]:  # Show first 10
            print(f"  - {var}")
        
        print(f"\nI(1) Variables (First-difference stationary): {len(i1_vars)}")
        for var in i1_vars[:10]:
            print(f"  - {var}")
        
        print(f"\nI(2) Variables (Second-difference stationary): {len(i2_vars)}")
        for var in i2_vars[:10]:
            print(f"  - {var}")
        
        # Cointegration assessment
        print("\nCointegration Assessment:")
        if len(i1_vars) >= 2:
            print(f"  ✓ {len(i1_vars)} I(1) variables available for cointegration testing")
            print("  → VECM models are appropriate for these variables")
        else:
            print("  ✗ Insufficient I(1) variables for cointegration")
        
        if len(i0_vars) > 0:
            print(f"  ✓ {len(i0_vars)} stationary variables can enter VAR models directly")
        
        if len(i2_vars) > 0:
            print(f"  ⚠ {len(i2_vars)} I(2) variables require second differencing")
    
    def save_results(self, filename='unit_roots_results.txt'):
        """Save all results to file"""
        print(f"\nSaving results to {filename}...")
        
        # Check structure
        if not self.integration_orders:
            print("No results to save")
            return
            
        # Determine if we have multi-frequency results
        first_key = list(self.integration_orders.keys())[0] if self.integration_orders else None
        is_multifreq = first_key in ['hourly', 'daily', 'weekly']
        
        with open(filename, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE UNIT ROOT TESTING RESULTS\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            # Handle different structures
            if is_multifreq:
                # Multi-frequency results
                f.write("\nMULTI-FREQUENCY ANALYSIS\n")
                f.write("-"*80 + "\n")
                
                for freq in ['hourly', 'daily', 'weekly']:
                    if freq in self.integration_orders:
                        f.write(f"\n{freq.upper()} FREQUENCY\n")
                        f.write("-"*40 + "\n")
                        
                        summary_data = []
                        for var_name, order_info in self.integration_orders[freq].items():
                            if not var_name.startswith('log_'):
                                summary_data.append([
                                    var_name,
                                    order_info['conclusion'],
                                    order_info['details'][:50]
                                ])
                        
                        headers = ['Variable', 'Order', 'Details']
                        f.write(tabulate(summary_data, headers=headers, tablefmt='simple') + "\n")
            else:
                # Single frequency results (backward compatible)
                f.write("SUMMARY TABLE\n")
                f.write("-"*80 + "\n")
                summary_data = []
                for var_name, order_info in self.integration_orders.items():
                    if var_name in self.test_results:
                        level_tests = self.test_results[var_name]['tests'].get('diff_0', {})
                        
                        adf_stat = level_tests.get('adf_c', {}).get('statistic', None)
                        adf_pval = level_tests.get('adf_c', {}).get('pvalue', None)
                        kpss_stat = level_tests.get('kpss_c', {}).get('statistic', None)
                        kpss_pval = level_tests.get('kpss_c', {}).get('pvalue', None)
                        
                        summary_data.append([
                            var_name,
                            f"{adf_stat:.4f}" if adf_stat else "—",
                            f"{adf_pval:.4f}" if adf_pval else "—",
                            f"{kpss_stat:.4f}" if kpss_stat else "—",
                            f"{kpss_pval:.4f}" if kpss_pval else "—",
                            order_info['conclusion'],
                            order_info['details']
                        ])
                
                headers = ['Variable', 'ADF Stat', 'ADF p-val', 'KPSS Stat', 'KPSS p-val', 'Order', 'Conclusion']
                f.write(tabulate(summary_data, headers=headers, tablefmt='simple') + "\n\n")
        
        print(f"Results saved to {filename}")
    
    def run_complete_analysis(self, test_frequencies=['daily']):
        """
        Run the complete unit root analysis pipeline
        
        Parameters:
        -----------
        test_frequencies : list
            List of frequencies to test: ['hourly', 'daily', 'weekly']
            Default is ['daily'] for backward compatibility
        """
        print("="*80)
        print("COMPREHENSIVE UNIT ROOT ANALYSIS")
        if len(test_frequencies) > 1:
            print(f"Testing at frequencies: {', '.join(test_frequencies)}")
        print("="*80)
        print(f"Started: {datetime.now()}")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Prepare data at different frequencies
        if 'hourly' in test_frequencies:
            self.prepare_hourly_data()
        if 'daily' in test_frequencies:
            self.prepare_daily_data()
        if 'weekly' in test_frequencies:
            self.prepare_weekly_data()
        
        # Test at specified frequencies
        if len(test_frequencies) == 1:
            # Single frequency - use old method for compatibility
            self.test_all_variables(test_frequencies[0])
        else:
            # Multiple frequencies - use new method
            self.test_all_frequencies()
        
        # Generate summary
        self.generate_summary_table()
        
        # Check cointegration requirements
        self.test_cointegration_requirements()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Ended: {datetime.now()}")
        print("="*80)
        
        return self.integration_orders


def main(test_all_frequencies=True):
    """
    Main execution function
    
    Parameters:
    -----------
    test_all_frequencies : bool
        If True, test at hourly, daily, and weekly frequencies
        If False, test only at daily frequency (backward compatible)
    """
    # Create output buffer for complete capture
    output_buffer = io.StringIO()
    
    # Run analysis with output capture
    with redirect_stdout(output_buffer):
        tester = UnitRootTester()
        
        if test_all_frequencies:
            # Test at all three frequencies
            integration_orders = tester.run_complete_analysis(['hourly', 'daily', 'weekly'])
        else:
            # Test only at daily frequency (backward compatible)
            integration_orders = tester.run_complete_analysis(['daily'])
    
    # Get complete output
    complete_output = output_buffer.getvalue()
    
    # Print to console
    print(complete_output)
    
    # Save comprehensive output
    filename = 'unit_roots_multifreq_output.txt' if test_all_frequencies else 'unit_roots_comprehensive_output.txt'
    with open(filename, 'w') as f:
        f.write(complete_output)
    
    print("\n" + "="*80)
    print(f"Output saved to: {filename}")
    print("Results saved to: unit_roots_results.txt")
    print("="*80)
    
    return integration_orders


if __name__ == "__main__":
    main()