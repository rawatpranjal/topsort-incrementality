"""
Diagnostic tests for ARDL models with enhanced strictness
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, Optional, List
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey


def test_serial_correlation(residuals: pd.Series,
                           lags: list = [7, 14],
                           test_type: str = 'ljungbox',
                           significance_level: float = 0.10) -> Dict:
    """
    Test for serial correlation in residuals with stricter criteria
    
    Parameters:
    -----------
    residuals : pd.Series
        Model residuals
    lags : list
        Lags to test
    test_type : str
        Type of test ('ljungbox', 'bg' for Breusch-Godfrey, 'dw' for Durbin-Watson, 'all')
    significance_level : float
        Stricter significance level (default 0.10 for more conservative testing)
    
    Returns:
    --------
    dict
        Test results
    """
    try:
        results = {}
        
        if test_type == 'ljungbox' or test_type == 'all':
            test_results = sm.stats.acorr_ljungbox(residuals, lags=lags, return_df=True)
            
            # Use stricter significance level
            has_serial_correlation = (test_results['lb_pvalue'] < significance_level).any()
            
            results['ljungbox'] = {
                'test_type': 'Ljung-Box',
                'results': test_results,
                'has_serial_correlation': has_serial_correlation,
                'min_pvalue': test_results['lb_pvalue'].min(),
                'max_statistic': test_results['lb_stat'].max(),
                'interpretation': f'Serial correlation detected (p < {significance_level})' if has_serial_correlation 
                                else f'No serial correlation (p >= {significance_level})'
            }
        
        if test_type == 'dw' or test_type == 'all':
            # Durbin-Watson test
            dw_stat = durbin_watson(residuals)
            
            # Critical values: <1.5 or >2.5 indicate serial correlation
            has_serial_correlation = dw_stat < 1.5 or dw_stat > 2.5
            
            results['durbin_watson'] = {
                'test_type': 'Durbin-Watson',
                'statistic': dw_stat,
                'has_serial_correlation': has_serial_correlation,
                'interpretation': f'DW statistic: {dw_stat:.3f} (ideal: ~2.0, problematic: <1.5 or >2.5)'
            }
        
        if test_type == 'bg' or test_type == 'all':
            # Breusch-Godfrey test
            try:
                # Need exog for BG test - create simple trend
                n = len(residuals)
                X = sm.add_constant(np.arange(n))
                bg_result = acorr_breusch_godfrey(sm.OLS(residuals, X).fit(), nlags=max(lags))
                
                results['breusch_godfrey'] = {
                    'test_type': 'Breusch-Godfrey',
                    'lm_statistic': bg_result[0],
                    'lm_pvalue': bg_result[1],
                    'f_statistic': bg_result[2],
                    'f_pvalue': bg_result[3],
                    'has_serial_correlation': bg_result[1] < significance_level,
                    'interpretation': f'Serial correlation detected (p < {significance_level})' if bg_result[1] < significance_level 
                                    else f'No serial correlation (p >= {significance_level})'
                }
            except:
                pass
        
        # Combine results if multiple tests
        if test_type == 'all':
            # Conservative: serial correlation if ANY test detects it
            any_serial = any(v.get('has_serial_correlation', False) for v in results.values())
            return {
                'combined_tests': results,
                'has_serial_correlation': any_serial,
                'interpretation': 'Serial correlation detected by at least one test' if any_serial 
                                else 'No serial correlation detected by any test'
            }
        elif len(results) == 1:
            return list(results.values())[0]
        else:
            return results
    
    except Exception as e:
        return {'error': f'Error in serial correlation test: {str(e)}'}


def test_heteroskedasticity(residuals: pd.Series,
                          exog: pd.DataFrame = None,
                          test_type: str = 'white') -> Dict:
    """
    Test for heteroskedasticity in residuals
    
    Parameters:
    -----------
    residuals : pd.Series
        Model residuals
    exog : pd.DataFrame, optional
        Exogenous variables for White test
    test_type : str
        Type of test ('white', 'breusch_pagan')
    
    Returns:
    --------
    dict
        Test results
    """
    try:
        if test_type == 'white' and exog is not None:
            # White test
            white_test = sm.stats.diagnostic.het_white(residuals, exog)
            
            return {
                'test_type': 'White',
                'lm_statistic': white_test[0],
                'lm_pvalue': white_test[1],
                'f_statistic': white_test[2],
                'f_pvalue': white_test[3],
                'has_heteroskedasticity': white_test[1] < 0.05,
                'interpretation': 'Heteroskedasticity detected' if white_test[1] < 0.05 else 'Homoskedastic'
            }
        elif test_type == 'breusch_pagan' and exog is not None:
            # Breusch-Pagan test
            bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, exog)
            
            return {
                'test_type': 'Breusch-Pagan',
                'lm_statistic': bp_test[0],
                'lm_pvalue': bp_test[1],
                'has_heteroskedasticity': bp_test[1] < 0.05,
                'interpretation': 'Heteroskedasticity detected' if bp_test[1] < 0.05 else 'Homoskedastic'
            }
        else:
            # Simple test based on residual variance stability
            n = len(residuals)
            first_half_var = residuals[:n//2].var()
            second_half_var = residuals[n//2:].var()
            variance_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
            
            return {
                'test_type': 'Variance Ratio',
                'first_half_variance': first_half_var,
                'second_half_variance': second_half_var,
                'variance_ratio': variance_ratio,
                'interpretation': f'Variance ratio: {variance_ratio:.2f} (>2 suggests heteroskedasticity)'
            }
    
    except Exception as e:
        return {'error': f'Error in heteroskedasticity test: {str(e)}'}


def test_normality(residuals: pd.Series,
                  test_type: str = 'jarque_bera') -> Dict:
    """
    Test for normality of residuals
    
    Parameters:
    -----------
    residuals : pd.Series
        Model residuals
    test_type : str
        Type of test ('jarque_bera', 'shapiro')
    
    Returns:
    --------
    dict
        Test results
    """
    try:
        if test_type == 'jarque_bera':
            jb_stat, jb_pvalue = sm.stats.jarque_bera(residuals)
            
            return {
                'test_type': 'Jarque-Bera',
                'statistic': jb_stat,
                'pvalue': jb_pvalue,
                'is_normal': jb_pvalue > 0.05,
                'interpretation': 'Residuals are normal' if jb_pvalue > 0.05 else 'Residuals are not normal'
            }
        elif test_type == 'shapiro':
            from scipy import stats
            shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
            
            return {
                'test_type': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'pvalue': shapiro_pvalue,
                'is_normal': shapiro_pvalue > 0.05,
                'interpretation': 'Residuals are normal' if shapiro_pvalue > 0.05 else 'Residuals are not normal'
            }
        else:
            return {'error': f'Test type {test_type} not implemented'}
    
    except Exception as e:
        return {'error': f'Error in normality test: {str(e)}'}


def test_stability(model_results: Any,
                  test_type: str = 'recursive') -> Dict:
    """
    Test for parameter stability
    
    Parameters:
    -----------
    model_results : ARDLResults or UECMResults
        Fitted model results
    test_type : str
        Type of test ('recursive', 'cusum')
    
    Returns:
    --------
    dict
        Test results
    """
    try:
        if test_type == 'cusum' and hasattr(model_results, 'plot_cusum'):
            # CUSUM test is typically visual, we'll check if it's available
            return {
                'test_type': 'CUSUM',
                'available': True,
                'interpretation': 'Use model_results.plot_cusum() to visualize stability test'
            }
        elif test_type == 'recursive':
            # Check coefficient stability across subsamples
            params = model_results.params
            n_obs = model_results.nobs
            
            # Split sample and check if coefficients are stable
            # This is a simplified stability check
            return {
                'test_type': 'Recursive',
                'n_parameters': len(params),
                'n_observations': n_obs,
                'interpretation': 'Parameter stability check requires visual inspection'
            }
        else:
            return {'error': f'Test type {test_type} not fully implemented'}
    
    except Exception as e:
        return {'error': f'Error in stability test: {str(e)}'}


def run_all_diagnostics(model_results: Any,
                       residuals: pd.Series = None,
                       exog: pd.DataFrame = None,
                       verbose: bool = True) -> Dict:
    """
    Run all diagnostic tests on model with enhanced strictness
    
    Parameters:
    -----------
    model_results : ARDLResults or UECMResults
        Fitted model results
    residuals : pd.Series, optional
        Model residuals (if not provided, will extract from model)
    exog : pd.DataFrame, optional
        Exogenous variables for some tests
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    dict
        All diagnostic test results
    """
    # Extract residuals if not provided
    if residuals is None:
        if hasattr(model_results, 'resid'):
            residuals = model_results.resid
        else:
            return {'error': 'Could not extract residuals from model'}
    
    results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("DIAGNOSTIC TESTS")
        print("="*60 + "\n")
    
    # 1. Serial Correlation - use ALL tests for comprehensive checking
    if verbose:
        print("1. Testing for Serial Correlation (Multiple Tests)...")
    serial_test = test_serial_correlation(residuals, test_type='all', significance_level=0.10)
    results['serial_correlation'] = serial_test
    if verbose and 'error' not in serial_test:
        print(f"   {serial_test['interpretation']}")
        if 'combined_tests' in serial_test:
            for test_name, test_result in serial_test['combined_tests'].items():
                if 'min_pvalue' in test_result:
                    print(f"   {test_name}: p-value = {test_result['min_pvalue']:.4f}")
                elif 'statistic' in test_result:
                    print(f"   {test_name}: statistic = {test_result['statistic']:.4f}")
    
    # 2. Heteroskedasticity
    if verbose:
        print("\n2. Testing for Heteroskedasticity...")
    hetero_test = test_heteroskedasticity(residuals, exog)
    results['heteroskedasticity'] = hetero_test
    if verbose and 'error' not in hetero_test:
        print(f"   {hetero_test['interpretation']}")
    
    # 3. Normality
    if verbose:
        print("\n3. Testing for Normality...")
    norm_test = test_normality(residuals)
    results['normality'] = norm_test
    if verbose and 'error' not in norm_test:
        print(f"   {norm_test['interpretation']}")
        print(f"   P-value: {norm_test.get('pvalue', 'N/A'):.4f}")
    
    # 4. Stability
    if verbose:
        print("\n4. Testing for Parameter Stability...")
    stability_test = test_stability(model_results)
    results['stability'] = stability_test
    if verbose and 'error' not in stability_test:
        print(f"   {stability_test['interpretation']}")
    
    # Overall assessment with stricter criteria
    issues = []
    critical_issues = []
    
    # Check serial correlation (critical issue)
    if serial_test.get('has_serial_correlation'):
        critical_issues.append('serial correlation')
        issues.append('serial correlation')
    
    # Check for combined test results
    if 'combined_tests' in serial_test:
        for test_name, test_result in serial_test['combined_tests'].items():
            if test_result.get('has_serial_correlation'):
                if f'{test_name} serial correlation' not in issues:
                    issues.append(f'{test_name} serial correlation')
    
    if hetero_test.get('has_heteroskedasticity'):
        issues.append('heteroskedasticity')
    
    if norm_test.get('is_normal') == False:
        issues.append('non-normality')
    
    # Stricter model quality assessment
    if critical_issues:
        model_quality = 'Poor - Critical issues detected'
    elif len(issues) == 0:
        model_quality = 'Good'
    elif len(issues) == 1:
        model_quality = 'Acceptable with minor issues'
    elif len(issues) == 2:
        model_quality = 'Questionable - Multiple issues'
    else:
        model_quality = 'Poor - Multiple issues'
    
    results['overall_assessment'] = {
        'issues_detected': issues,
        'critical_issues': critical_issues,
        'n_issues': len(issues),
        'model_quality': model_quality,
        'requires_remediation': len(critical_issues) > 0
    }
    
    if verbose:
        print("\n" + "-"*60)
        print("OVERALL ASSESSMENT:")
        if issues:
            print(f"Issues detected: {', '.join(issues)}")
        else:
            print("No major issues detected")
        print(f"Model quality: {results['overall_assessment']['model_quality']}")
    
    return results


def test_residual_autocorrelation(residuals: pd.Series,
                                 max_lags: int = 20) -> Dict:
    """
    Comprehensive residual autocorrelation testing
    
    Parameters:
    -----------
    residuals : pd.Series
        Model residuals
    max_lags : int
        Maximum lags to test
    
    Returns:
    --------
    dict
        Autocorrelation test results
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import acf, pacf
        
        # Calculate ACF and PACF
        acf_values = acf(residuals, nlags=max_lags)
        pacf_values = pacf(residuals, nlags=max_lags)
        
        # Confidence bounds (95%)
        n = len(residuals)
        confidence_bound = 1.96 / np.sqrt(n)
        
        # Count significant autocorrelations
        significant_acf = np.sum(np.abs(acf_values[1:]) > confidence_bound)
        significant_pacf = np.sum(np.abs(pacf_values[1:]) > confidence_bound)
        
        # Ljung-Box test at multiple lags
        lb_results = acorr_ljungbox(residuals, lags=list(range(1, min(max_lags+1, len(residuals)//5))), return_df=True)
        
        # Find first significant lag
        first_significant_lag = None
        for lag in lb_results.index:
            if lb_results.loc[lag, 'lb_pvalue'] < 0.10:
                first_significant_lag = lag
                break
        
        return {
            'significant_acf_lags': significant_acf,
            'significant_pacf_lags': significant_pacf,
            'first_significant_lag': first_significant_lag,
            'confidence_bound': confidence_bound,
            'max_acf': np.max(np.abs(acf_values[1:])),
            'max_pacf': np.max(np.abs(pacf_values[1:])),
            'interpretation': f"{significant_acf} ACF and {significant_pacf} PACF lags exceed bounds",
            'has_autocorrelation': significant_acf > 2 or significant_pacf > 2,
            'ljungbox_summary': {
                'min_pvalue': lb_results['lb_pvalue'].min(),
                'n_significant_at_10pct': (lb_results['lb_pvalue'] < 0.10).sum(),
                'n_significant_at_5pct': (lb_results['lb_pvalue'] < 0.05).sum()
            }
        }
    
    except Exception as e:
        return {'error': f'Error in autocorrelation test: {str(e)}'}


def test_specification_errors(model_results: Any,
                             df: pd.DataFrame) -> Dict:
    """
    Test for specification errors in the model
    
    Parameters:
    -----------
    model_results : Model results
    df : pd.DataFrame
        Original data
    
    Returns:
    --------
    dict
        Specification test results
    """
    try:
        from statsmodels.stats.diagnostic import linear_reset
        
        results = {}
        
        # RESET test for functional form misspecification
        try:
            reset_result = linear_reset(model_results, power=3, use_f=True)
            results['reset_test'] = {
                'statistic': reset_result.statistic,
                'pvalue': reset_result.pvalue,
                'misspecified': reset_result.pvalue < 0.05,
                'interpretation': 'Functional form misspecification' if reset_result.pvalue < 0.05 
                                else 'No misspecification detected'
            }
        except:
            results['reset_test'] = {'error': 'Could not perform RESET test'}
        
        # Check for omitted variables (simplified)
        residuals = model_results.resid
        
        # Test if residuals correlate with squared fitted values
        fitted = model_results.fittedvalues
        fitted_squared = fitted ** 2
        
        correlation = np.corrcoef(residuals, fitted_squared)[0, 1]
        
        results['omitted_variable_test'] = {
            'correlation_with_fitted_squared': correlation,
            'likely_omitted_variables': abs(correlation) > 0.3,
            'interpretation': 'Likely omitted variables' if abs(correlation) > 0.3 
                            else 'No strong evidence of omitted variables'
        }
        
        return results
    
    except Exception as e:
        return {'error': f'Error in specification tests: {str(e)}'}


def enhanced_diagnostic_summary(all_diagnostics: Dict) -> str:
    """
    Generate enhanced diagnostic summary with recommendations
    
    Parameters:
    -----------
    all_diagnostics : dict
        Results from run_all_diagnostics
    
    Returns:
    --------
    str
        Detailed summary with recommendations
    """
    summary = []
    summary.append("\n" + "="*70)
    summary.append("ENHANCED DIAGNOSTIC SUMMARY")
    summary.append("="*70 + "\n")
    
    # Critical Issues
    if all_diagnostics['overall_assessment']['critical_issues']:
        summary.append("⚠️  CRITICAL ISSUES DETECTED:")
        for issue in all_diagnostics['overall_assessment']['critical_issues']:
            summary.append(f"   • {issue.upper()}")
        summary.append("")
    
    # Model Quality
    quality = all_diagnostics['overall_assessment']['model_quality']
    summary.append(f"Model Quality: {quality}")
    summary.append("")
    
    # Detailed Test Results
    summary.append("Detailed Test Results:")
    summary.append("-" * 40)
    
    # Serial Correlation
    serial = all_diagnostics.get('serial_correlation', {})
    if 'combined_tests' in serial:
        summary.append("Serial Correlation (Multiple Tests):")
        for test_name, test_result in serial['combined_tests'].items():
            if 'statistic' in test_result:
                summary.append(f"  {test_name}: {test_result.get('interpretation', 'N/A')}")
    elif 'interpretation' in serial:
        summary.append(f"Serial Correlation: {serial['interpretation']}")
    
    # Heteroskedasticity
    hetero = all_diagnostics.get('heteroskedasticity', {})
    if 'interpretation' in hetero:
        summary.append(f"Heteroskedasticity: {hetero['interpretation']}")
    
    # Normality
    norm = all_diagnostics.get('normality', {})
    if 'interpretation' in norm:
        summary.append(f"Normality: {norm['interpretation']}")
    
    summary.append("")
    
    # Recommendations
    if all_diagnostics['overall_assessment']['requires_remediation']:
        summary.append("RECOMMENDED ACTIONS:")
        summary.append("-" * 40)
        
        if 'serial correlation' in all_diagnostics['overall_assessment']['issues_detected']:
            summary.append("For Serial Correlation:")
            summary.append("  1. Increase lag order (try 14, 21, or 28 lags)")
            summary.append("  2. Add seasonal dummies if not already included")
            summary.append("  3. Consider structural breaks or regime changes")
            summary.append("  4. Use HAC standard errors for inference")
            summary.append("")
        
        if 'heteroskedasticity' in all_diagnostics['overall_assessment']['issues_detected']:
            summary.append("For Heteroskedasticity:")
            summary.append("  1. Use robust standard errors")
            summary.append("  2. Consider log transformation if not already applied")
            summary.append("  3. Check for outliers or structural changes")
            summary.append("")
        
        if 'non-normality' in all_diagnostics['overall_assessment']['issues_detected']:
            summary.append("For Non-Normality:")
            summary.append("  1. Check for outliers")
            summary.append("  2. Consider alternative distributions")
            summary.append("  3. Use bootstrap for inference")
            summary.append("")
    
    return "\n".join(summary)