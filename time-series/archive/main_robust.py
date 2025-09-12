#!/usr/bin/env python3
"""
Robust ARDL Analysis Pipeline with Enhanced Statistical Methods
Automatically resolves serial correlation and other statistical issues
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from datetime import datetime
import sys
from io import StringIO

# Import our enhanced modules
from data_handler import (
    load_data, prepare_data, prepare_data_with_controls, test_all_stationarity,
    check_data_quality, test_integration_order, get_control_variable_names
)
from ardl_models import (
    run_robust_ardl_analysis, resolve_serial_correlation,
    perform_bounds_test, select_optimal_lags
)
from analysis import (
    calculate_elasticity, calculate_half_life, test_long_run_elasticity_equals_one,
    perform_granger_causality_test, perform_structural_break_test,
    calculate_forecast_metrics, perform_robustness_checks
)
from diagnostics import (
    run_all_diagnostics, test_residual_autocorrelation,
    test_specification_errors, enhanced_diagnostic_summary
)
from utils import print_banner as print_header, save_results

warnings.filterwarnings('ignore')


def run_robust_analysis(filepath: str = 'daily_click_to_purchase_funnel_summary.csv',
                        models_to_run: list = None,
                        max_lags: int = 14,
                        add_controls: bool = True,
                        auto_fix_serial: bool = True,
                        verbose: bool = True,
                        capture_output: bool = True) -> dict:
    """
    Run robust ARDL analysis with automatic issue resolution
    
    Parameters:
    -----------
    filepath : str
        Path to data file
    models_to_run : list
        Models to analyze (default: all)
    max_lags : int
        Maximum lags to consider
    add_controls : bool
        Whether to add control variables
    auto_fix_serial : bool
        Automatically resolve serial correlation
    verbose : bool
        Print detailed output
    
    Returns:
    --------
    dict
        Complete analysis results
    """
    
    results = {}
    
    # Set up output capture if requested
    captured_output = StringIO()
    original_stdout = sys.stdout
    
    if capture_output:
        # Create a custom stdout that writes to both console and capture
        class TeeOutput:
            def __init__(self, *streams):
                self.streams = streams
            
            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush() if hasattr(stream, 'flush') else None
            
            def flush(self):
                for stream in self.streams:
                    if hasattr(stream, 'flush'):
                        stream.flush()
        
        sys.stdout = TeeOutput(original_stdout, captured_output)
    
    # 1. LOAD AND PREPARE DATA
    print_header("ROBUST ARDL ANALYSIS WITH ENHANCED METHODS")
    print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if verbose:
        print("\n1. Loading and preparing data...")
    
    df = load_data(filepath)
    
    # Check data quality
    quality = check_data_quality(df)
    if verbose:
        print(f"   Loaded {quality['n_observations']} observations")
        print(f"   Date range: {quality['date_range']}")
    
    # Prepare data with controls if requested
    if add_controls:
        if verbose:
            print("   Adding control variables (seasonality, trends)...")
        df = prepare_data_with_controls(
            df,
            add_seasonality=True,
            add_trends=True,
            add_events=False
        )
        control_vars = get_control_variable_names(df)
        if verbose:
            print(f"   Added {len(control_vars['all'])} control variables")
    else:
        df = prepare_data(df)
        control_vars = {'all': []}
    
    # 2. TEST INTEGRATION ORDER
    if verbose:
        print("\n2. Testing integration order of variables...")
    
    test_vars = ['log_clicks', 'log_purchases', 'log_aov', 'log_revenue']
    integration_results = {}
    
    for var in test_vars:
        if var in df.columns:
            int_result = test_integration_order(df[var], var, max_order=2)
            integration_results[var] = int_result
            if verbose:
                print(f"   {var}: {int_result['conclusion']}")
    
    # Check if any variables are I(2)
    i2_vars = [var for var, res in integration_results.items() 
               if res.get('integration_order') == 2]
    if i2_vars and verbose:
        print(f"   WARNING: I(2) variables detected: {i2_vars}")
        print("   Consider first-differencing these variables")
    
    # 3. STATIONARITY TESTS
    if verbose:
        print("\n3. Running stationarity tests...")
    
    stationarity_results = test_all_stationarity(df, test_vars)
    
    # 4. RUN MODELS
    if models_to_run is None:
        models_to_run = ['purchases', 'aov', 'revenue']
    
    for model_name in models_to_run:
        print_header(f"MODEL: {model_name.upper()} vs CLICKS")
        
        # Set up variables
        if model_name == 'purchases':
            dependent = 'log_purchases'
            independent = 'log_clicks'
        elif model_name == 'aov':
            dependent = 'log_aov'
            independent = 'log_clicks'
        elif model_name == 'revenue':
            dependent = 'log_revenue'
            independent = 'log_clicks'
        else:
            continue
        
        # Check variables exist
        if dependent not in df.columns or independent not in df.columns:
            print(f"Skipping {model_name}: variables not found")
            continue
        
        # Run robust ARDL analysis
        if verbose:
            print(f"\nRunning robust ARDL analysis for {model_name}...")
        
        # Prepare fixed regressors if needed
        fixed_df = None
        if add_controls and control_vars['all']:
            fixed_df = df[control_vars['all']]
        
        model_results = run_robust_ardl_analysis(
            endog=df[dependent],
            exog=df[[independent]],
            fixed=fixed_df,
            endog_name=dependent,
            exog_name=independent,
            resolve_serial=auto_fix_serial,
            verbose=verbose
        )
        
        # Store results
        results[model_name] = model_results
        
        # Additional advanced tests
        if model_results.get('final_model'):
            ardl_model = model_results['final_model']
            
            # Granger causality test
            if verbose:
                print("\n5. Testing Granger causality...")
            granger_result = perform_granger_causality_test(
                df, independent, dependent, max_lags=7
            )
            model_results['granger_causality'] = granger_result
            if verbose and 'error' not in granger_result:
                print(f"   {granger_result['interpretation']}")
            
            # Structural break test
            if verbose:
                print("\n6. Testing for structural breaks...")
            break_result = perform_structural_break_test(
                df[dependent], test_type='cusum'
            )
            model_results['structural_break'] = break_result
            if verbose and 'error' not in break_result:
                print(f"   {break_result['interpretation']}")
            
            # Forecast metrics
            if verbose:
                print("\n7. Calculating forecast metrics...")
            forecast_result = calculate_forecast_metrics(
                ardl_model, df, forecast_periods=7
            )
            model_results['forecast_metrics'] = forecast_result
            if verbose and 'error' not in forecast_result:
                print(f"   RMSE: {forecast_result.get('rmse', 'N/A'):.4f}")
                print(f"   MAE: {forecast_result.get('mae', 'N/A'):.4f}")
            
            # Robustness checks
            if verbose:
                print("\n8. Performing robustness checks...")
            robustness = perform_robustness_checks(
                ardl_model, df, dependent, independent
            )
            model_results['robustness'] = robustness
            if verbose:
                if 'parameter_stability' in robustness:
                    print(f"   {robustness['parameter_stability'].get('interpretation', 'N/A')}")
                if 'outlier_analysis' in robustness:
                    print(f"   Outliers: {robustness['outlier_analysis'].get('n_outliers', 'N/A')}")
        
        # Enhanced diagnostic summary
        if model_results.get('diagnostics'):
            enhanced_summary = enhanced_diagnostic_summary(model_results['diagnostics'])
            if verbose:
                print(enhanced_summary)
    
    # 5. FINAL SUMMARY
    print_header("FINAL ROBUST ANALYSIS SUMMARY")
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        
        # Cointegration
        if 'bounds_test' in model_results:
            print(f"  Cointegration: {model_results['bounds_test'].get('conclusion', 'N/A')}")
        
        # Elasticity
        if 'elasticity' in model_results:
            elasticity = model_results['elasticity'].get('long_run_elasticity', 'N/A')
            if isinstance(elasticity, float):
                print(f"  Long-run elasticity: {elasticity:.4f}")
        
        # Half-life
        if 'half_life' in model_results:
            half_life = model_results['half_life'].get('half_life_days', 'N/A')
            if isinstance(half_life, float):
                print(f"  Half-life: {half_life:.1f} days")
        
        # Serial correlation status
        if 'serial_correlation_resolved' in model_results:
            if model_results['serial_correlation_resolved']:
                print(f"  Serial correlation: RESOLVED (lags: {model_results.get('final_lags', 'N/A')})")
            else:
                print(f"  Serial correlation: UNRESOLVED")
        
        # Model quality
        if 'diagnostics' in model_results:
            quality = model_results['diagnostics']['overall_assessment']['model_quality']
            print(f"  Model quality: {quality}")
        
        # Granger causality
        if 'granger_causality' in model_results:
            if model_results['granger_causality'].get('granger_causes'):
                print(f"  Granger causality: Confirmed")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'robust_results_{timestamp}.txt'
    full_output_file = f'robust_results_{timestamp}_full.txt'
    
    # Save formatted results
    with open(output_file, 'w') as f:
        f.write("ROBUST ARDL ANALYSIS RESULTS - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {filepath}\n")
        f.write(f"Controls added: {add_controls}\n")
        f.write(f"Auto-fix serial correlation: {auto_fix_serial}\n")
        f.write(f"Maximum lags tested: {max_lags}\n\n")
        
        # Write detailed results
        for model_name, model_results in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            # Write all available metrics
            if 'bounds_test' in model_results:
                f.write(f"Cointegration: {model_results['bounds_test'].get('conclusion', 'N/A')}\n")
                f.write(f"F-statistic: {model_results['bounds_test'].get('f_statistic', 'N/A'):.4f}\n")
            
            if 'elasticity' in model_results:
                elasticity = model_results['elasticity'].get('long_run_elasticity', 'N/A')
                if isinstance(elasticity, float):
                    f.write(f"Long-run elasticity: {elasticity:.4f}\n")
                speed = model_results['elasticity'].get('speed_of_adjustment', 'N/A')
                if isinstance(speed, float):
                    f.write(f"Speed of adjustment: {speed:.4f}\n")
            
            if 'half_life' in model_results:
                half_life = model_results['half_life'].get('half_life_days', 'N/A')
                if isinstance(half_life, float):
                    f.write(f"Half-life: {half_life:.1f} days\n")
            
            if 'unit_elasticity_test' in model_results:
                f.write(f"Unit elasticity test: {model_results['unit_elasticity_test'].get('conclusion', 'N/A')}\n")
            
            if 'granger_causality' in model_results:
                f.write(f"Granger causality: {model_results['granger_causality'].get('interpretation', 'N/A')}\n")
                f.write(f"  Best lag: {model_results['granger_causality'].get('best_lag', 'N/A')}\n")
                f.write(f"  F-statistic: {model_results['granger_causality'].get('f_statistic', 'N/A'):.4f}\n")
                f.write(f"  P-value: {model_results['granger_causality'].get('f_pvalue', 'N/A'):.4f}\n")
            
            if 'structural_break' in model_results:
                f.write(f"Structural break: {model_results['structural_break'].get('interpretation', 'N/A')}\n")
            
            if 'forecast_metrics' in model_results:
                metrics = model_results['forecast_metrics']
                if 'rmse' in metrics:
                    f.write(f"Forecast RMSE: {metrics['rmse']:.4f}\n")
                if 'mae' in metrics:
                    f.write(f"Forecast MAE: {metrics['mae']:.4f}\n")
            
            if 'serial_correlation_resolved' in model_results:
                f.write(f"Serial correlation resolved: {model_results['serial_correlation_resolved']}\n")
                if 'final_lags' in model_results:
                    f.write(f"Final lag structure: {model_results['final_lags']}\n")
            
            if 'diagnostics' in model_results:
                summary = enhanced_diagnostic_summary(model_results['diagnostics'])
                f.write("\nDiagnostics:\n")
                f.write(summary)
                f.write("\n")
    
    # Save full stdout output if captured
    if capture_output:
        # Restore original stdout
        sys.stdout = original_stdout
        
        # Save complete output with source code
        with open(full_output_file, 'w') as f:
            f.write("ROBUST ARDL ANALYSIS - COMPLETE OUTPUT WITH SOURCE CODE\n")
            f.write("="*80 + "\n\n")
            
            # First write the captured stdout
            f.write("="*80 + "\n")
            f.write("                           ANALYSIS OUTPUT                           \n")
            f.write("="*80 + "\n\n")
            f.write(captured_output.getvalue())
            
            # Now include the full source code of key modules
            f.write("\n\n" + "="*80 + "\n")
            f.write("                           MODULE SOURCE CODE                           \n")
            f.write("="*80 + "\n\n")
            
            # Read and include analysis.py
            try:
                f.write("="*80 + "\n")
                f.write("                              ANALYSIS.PY                              \n")
                f.write("="*80 + "\n\n")
                with open('analysis.py', 'r') as src:
                    f.write(src.read())
            except Exception as e:
                f.write(f"Error reading analysis.py: {e}\n")
            
            # Read and include ardl_models.py
            try:
                f.write("\n\n" + "="*80 + "\n")
                f.write("                            ARDL_MODELS.PY                            \n")
                f.write("="*80 + "\n\n")
                with open('ardl_models.py', 'r') as src:
                    f.write(src.read())
            except Exception as e:
                f.write(f"Error reading ardl_models.py: {e}\n")
            
            # Read and include diagnostics.py
            try:
                f.write("\n\n" + "="*80 + "\n")
                f.write("                            DIAGNOSTICS.PY                            \n")
                f.write("="*80 + "\n\n")
                with open('diagnostics.py', 'r') as src:
                    f.write(src.read())
            except Exception as e:
                f.write(f"Error reading diagnostics.py: {e}\n")
            
            # Read and include data_handler.py
            try:
                f.write("\n\n" + "="*80 + "\n")
                f.write("                           DATA_HANDLER.PY                           \n")
                f.write("="*80 + "\n\n")
                with open('data_handler.py', 'r') as src:
                    f.write(src.read())
            except Exception as e:
                f.write(f"Error reading data_handler.py: {e}\n")
            
            # Read and include utils.py
            try:
                f.write("\n\n" + "="*80 + "\n")
                f.write("                               UTILS.PY                               \n")
                f.write("="*80 + "\n\n")
                with open('utils.py', 'r') as src:
                    f.write(src.read())
            except Exception as e:
                f.write(f"Error reading utils.py: {e}\n")
            
            # Include all numerical results in structured format
            f.write("\n\n" + "="*80 + "\n")
            f.write("                        COMPLETE NUMERICAL RESULTS                        \n")
            f.write("="*80 + "\n\n")
            
            import json
            
            # Convert results to a more complete format
            for model_name, model_results in results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write(f"{'='*60}\n\n")
                
                # Write ALL results in detailed format
                for key, value in model_results.items():
                    f.write(f"\n### {key.upper()} ###\n")
                    if isinstance(value, dict):
                        f.write(json.dumps(value, indent=2, default=str))
                    elif hasattr(value, '__dict__'):
                        # For model objects, try to extract attributes
                        f.write(f"Object type: {type(value).__name__}\n")
                        if hasattr(value, 'summary'):
                            try:
                                f.write(str(value.summary()))
                            except:
                                pass
                        if hasattr(value, 'params'):
                            f.write(f"Parameters:\n{value.params}\n")
                        if hasattr(value, 'pvalues'):
                            f.write(f"P-values:\n{value.pvalues}\n")
                    else:
                        f.write(str(value))
                    f.write("\n")
        
        print(f"\n\nResults saved to:")
        print(f"  Summary: {output_file}")
        print(f"  Full output with source code: {full_output_file}")
    else:
        print(f"\n\nResults saved to: {output_file}")
    
    return results


def main():
    """Main entry point for robust analysis"""
    parser = argparse.ArgumentParser(description='Robust ARDL Analysis with Enhanced Methods')
    parser.add_argument('--data', default='daily_click_to_purchase_funnel_summary.csv',
                       help='Path to data file')
    parser.add_argument('--models', nargs='+', default=['purchases', 'aov', 'revenue'],
                       help='Models to run')
    parser.add_argument('--max-lags', type=int, default=14,
                       help='Maximum lags to test')
    parser.add_argument('--no-controls', action='store_true',
                       help='Do not add control variables')
    parser.add_argument('--no-auto-fix', action='store_true',
                       help='Do not automatically fix serial correlation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_robust_analysis(
        filepath=args.data,
        models_to_run=args.models,
        max_lags=args.max_lags,
        add_controls=not args.no_controls,
        auto_fix_serial=not args.no_auto_fix,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()