#!/usr/bin/env python3
"""
Main entry point for ARDL analysis
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_handler import load_data, prepare_data, test_stationarity, test_all_stationarity
from ardl_models import run_complete_ardl_analysis
from analysis import (
    calculate_elasticity, 
    calculate_half_life,
    test_long_run_elasticity_equals_one
)
from diagnostics import run_all_diagnostics
from utils import OutputCapture, save_results, format_results_table, print_banner


def analyze_single_model(df, dependent_var, independent_var, dep_name, indep_name, max_lags=7):
    """Run complete analysis for a single model"""
    
    print_banner(f"MODEL: {dep_name} vs {indep_name}")
    
    # Prepare data for analysis
    analysis_df = df[[dependent_var, independent_var]].dropna()
    
    # Run ARDL analysis
    results = run_complete_ardl_analysis(
        endog=analysis_df[dependent_var],
        exog=analysis_df[[independent_var]],
        endog_name=dep_name,
        exog_name=indep_name,
        max_lags=max_lags,
        verbose=True
    )
    
    # If successful, add advanced analyses
    if results.get('uecm_results') is not None:
        # Calculate elasticity
        print("\n5. Calculating Long-Run Elasticity...")
        elasticity = calculate_elasticity(results['uecm_results'])
        results['elasticity'] = elasticity
        if 'error' not in elasticity:
            print(f"   Long-run elasticity: {elasticity['long_run_elasticity']:.4f}")
            
            # Calculate half-life
            print("\n6. Calculating Half-Life...")
            half_life = calculate_half_life(elasticity['speed_of_adjustment'])
            results['half_life'] = half_life
            if 'error' not in half_life:
                print(f"   Half-life: {half_life['half_life_days']:.1f} days")
        
        # Test unit elasticity
        print("\n7. Testing Unit Elasticity...")
        unit_test = test_long_run_elasticity_equals_one(results['uecm_results'])
        results['unit_elasticity_test'] = unit_test
        if 'error' not in unit_test:
            print(f"   {unit_test['conclusion']}")
        
        # Run diagnostics
        print("\n8. Running Diagnostics...")
        diagnostics = run_all_diagnostics(
            results['uecm_results'],
            verbose=False
        )
        results['diagnostics'] = diagnostics
        print(f"   Model quality: {diagnostics['overall_assessment']['model_quality']}")
        if diagnostics['overall_assessment']['issues_detected']:
            print(f"   Issues: {', '.join(diagnostics['overall_assessment']['issues_detected'])}")
    
    return results


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='ARDL Analysis for Click-to-Purchase Funnel')
    parser.add_argument('--data', type=str, default='daily_click_to_purchase_funnel_summary.csv',
                       help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='modular_ardl_results.txt',
                       help='Output file for results')
    parser.add_argument('--max-lags', type=int, default=7,
                       help='Maximum lags to consider')
    parser.add_argument('--models', nargs='+', default=['purchases', 'aov', 'revenue'],
                       choices=['purchases', 'aov', 'revenue'],
                       help='Which models to run')
    
    args = parser.parse_args()
    
    # Start output capture
    with OutputCapture(args.output):
        print_banner("ARDL ANALYSIS OF CLICK-TO-PURCHASE FUNNEL DATA", char="=")
        print()
        
        # Load data
        print("Loading data...")
        df = load_data(args.data)
        print(f"Loaded {len(df)} observations")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print()
        
        # Prepare data
        print("Preparing data (log transformation)...")
        df = prepare_data(df)
        print("Data preparation complete")
        print()
        
        # Test stationarity
        print_banner("STATIONARITY TESTS")
        variables = ['log_clicks', 'log_purchases', 'log_aov', 'log_revenue']
        variables = [v for v in variables if v in df.columns]
        
        stationarity_results = test_all_stationarity(df, variables)
        print("\nStationarity Test Summary:")
        print(stationarity_results[['variable', 'type', 'adf_stat', 'pvalue', 'stationary']])
        print()
        
        # Store all results
        all_results = {
            'data_info': {
                'n_observations': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'variables': variables
            },
            'stationarity': stationarity_results.to_dict(),
            'models': {}
        }
        
        # Run models
        if 'purchases' in args.models and 'log_purchases' in df.columns:
            print("\n" + "="*80)
            model1 = analyze_single_model(
                df, 'log_purchases', 'log_clicks',
                'Log Purchases', 'Log Clicks',
                args.max_lags
            )
            all_results['models']['purchases_vs_clicks'] = model1
        
        if 'aov' in args.models and 'log_aov' in df.columns:
            print("\n" + "="*80)
            model2 = analyze_single_model(
                df, 'log_aov', 'log_clicks',
                'Log AOV', 'Log Clicks',
                args.max_lags
            )
            all_results['models']['aov_vs_clicks'] = model2
        
        if 'revenue' in args.models and 'log_revenue' in df.columns:
            print("\n" + "="*80)
            model3 = analyze_single_model(
                df, 'log_revenue', 'log_clicks',
                'Log Revenue', 'Log Clicks',
                args.max_lags
            )
            all_results['models']['revenue_vs_clicks'] = model3
        
        # Print summary
        print("\n" + "="*80)
        print_banner("FINAL SUMMARY")
        
        for model_name, model_results in all_results['models'].items():
            print(f"\n{model_name.upper()}:")
            
            if 'bounds_test' in model_results:
                print(f"  Cointegration: {model_results['bounds_test'].get('conclusion', 'N/A')}")
            
            if 'elasticity' in model_results:
                elasticity = model_results['elasticity'].get('long_run_elasticity')
                if elasticity:
                    print(f"  Long-run elasticity: {elasticity:.4f}")
            
            if 'half_life' in model_results:
                hl = model_results['half_life'].get('half_life_days')
                if hl:
                    print(f"  Half-life: {hl:.1f} days")
            
            if 'diagnostics' in model_results:
                quality = model_results['diagnostics']['overall_assessment']['model_quality']
                print(f"  Model quality: {quality}")
        
        print("\n" + "="*80)
        print(f"\nResults saved to: {args.output}")
        
        # Also save as JSON for programmatic access
        json_file = args.output.replace('.txt', '.json')
        save_results(all_results, json_file, format='json')
        print(f"JSON results saved to: {json_file}")


if __name__ == "__main__":
    main()