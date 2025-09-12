"""
Data handling module for ARDL analysis
Includes loading, preprocessing, and stationarity testing
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from typing import Tuple, Dict, Optional, List


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with datetime index
    """
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime if it exists
    date_columns = ['activity_date', 'date', 'Date', 'DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break
    
    # Clean numeric columns (remove commas if present)
    numeric_columns = ['total_clicks', 'total_purchases', 'aov', 'total_revenue']
    for col in numeric_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
            else:
                df[col] = df[col].astype(float)
    
    return df


def prepare_data(df: pd.DataFrame, 
                 variables: list = None,
                 handle_zeros: bool = True) -> pd.DataFrame:
    """
    Prepare data for ARDL analysis including log transformation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    variables : list
        List of variables to transform. If None, transforms all numeric columns
    handle_zeros : bool
        Whether to replace zeros with 1 before log transformation
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with log-transformed variables
    """
    df = df.copy()
    
    if variables is None:
        variables = ['total_clicks', 'total_purchases', 'aov', 'total_revenue']
        variables = [v for v in variables if v in df.columns]
    
    # Handle zeros before log transformation
    if handle_zeros:
        for var in variables:
            if var in df.columns:
                df[var] = df[var].replace(0, 1)
    
    # Log transform
    for var in variables:
        if var in df.columns:
            log_var = f'log_{var.replace("total_", "")}'
            df[log_var] = np.log(df[var])
    
    return df


def test_stationarity(series: pd.Series, 
                     name: str = "Series",
                     significance_level: float = 0.05,
                     verbose: bool = True) -> Dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Name of the series for reporting
    significance_level : float
        Significance level for the test
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        if verbose:
            print(f"Warning: {name} has insufficient observations for ADF test")
        return {'stationary': None, 'pvalue': None}
    
    result = adfuller(series_clean)
    
    output = {
        'adf_stat': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'stationary': result[1] < significance_level,
        'name': name
    }
    
    if verbose:
        print(f'ADF Test for {name}:')
        print(f'  ADF Statistic: {output["adf_stat"]:.4f}')
        print(f'  p-value: {output["pvalue"]:.4f}')
        if output['stationary']:
            print(f'  Result: The series is likely stationary (I(0)).')
        else:
            print(f'  Result: The series is likely non-stationary (I(1)).')
    
    return output


def test_all_stationarity(df: pd.DataFrame, 
                         variables: list = None,
                         test_differences: bool = True) -> pd.DataFrame:
    """
    Test stationarity for multiple variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the variables
    variables : list
        List of variables to test
    test_differences : bool
        Whether to also test first differences
    
    Returns:
    --------
    pd.DataFrame
        Summary of stationarity test results
    """
    if variables is None:
        # Default to log-transformed variables
        variables = [col for col in df.columns if col.startswith('log_')]
    
    results = []
    
    for var in variables:
        if var in df.columns:
            # Test levels
            level_result = test_stationarity(df[var], f'{var} (Level)', verbose=False)
            level_result['variable'] = var
            level_result['type'] = 'Level'
            results.append(level_result)
            
            # Test first differences
            if test_differences:
                diff_result = test_stationarity(df[var].diff(), f'{var} (Diff)', verbose=False)
                diff_result['variable'] = var
                diff_result['type'] = 'First Difference'
                results.append(diff_result)
    
    results_df = pd.DataFrame(results)
    return results_df


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Check data quality and provide summary statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary with data quality metrics
    """
    quality = {
        'n_observations': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "No datetime index",
        'missing_values': df.isnull().sum().to_dict(),
        'zero_values': (df == 0).sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    # Check for gaps in time series
    if hasattr(df.index, 'to_series'):
        date_diff = df.index.to_series().diff()
        if len(date_diff) > 1:
            quality['max_gap_days'] = date_diff.max().days if hasattr(date_diff.max(), 'days') else None
            quality['has_gaps'] = quality['max_gap_days'] > 1 if quality['max_gap_days'] else False
    
    return quality


def test_integration_order(series: pd.Series, 
                          name: str = "Series",
                          max_order: int = 2,
                          significance_level: float = 0.05) -> Dict:
    """
    Determine integration order of a series (I(0), I(1), or I(2))
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Name of the series
    max_order : int
        Maximum integration order to test
    significance_level : float
        Significance level for tests
    
    Returns:
    --------
    dict
        Integration order and test results
    """
    current_series = series.dropna()
    results = {'name': name, 'tests': []}
    
    for i in range(max_order + 1):
        # ADF test
        adf_result = adfuller(current_series, autolag='AIC')
        adf_pvalue = adf_result[1]
        
        # KPSS test (null hypothesis is stationarity)
        try:
            kpss_result = kpss(current_series, regression='c', nlags='auto')
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_crit = kpss_result[3]
        except:
            kpss_stat, kpss_pvalue = None, None
            kpss_crit = {}
        
        test_info = {
            'difference_order': i,
            'adf_stat': adf_result[0],
            'adf_pvalue': adf_pvalue,
            'adf_stationary': adf_pvalue < significance_level,
            'kpss_stat': kpss_stat,
            'kpss_pvalue': kpss_pvalue,
            'kpss_stationary': kpss_pvalue > significance_level if kpss_pvalue else None
        }
        results['tests'].append(test_info)
        
        # Decision logic: both tests should agree
        if adf_pvalue < significance_level:
            if kpss_pvalue is None or kpss_pvalue > significance_level:
                results['integration_order'] = i
                results['conclusion'] = f"Series is I({i})"
                return results
        
        # Take next difference
        if i < max_order:
            current_series = current_series.diff().dropna()
    
    # If we get here, series might be I(>max_order)
    results['integration_order'] = None
    results['conclusion'] = f"Series may be I({max_order+1}) or higher"
    return results


def add_seasonal_dummies(df: pd.DataFrame, 
                        freq: str = 'day_of_week',
                        drop_first: bool = True) -> pd.DataFrame:
    """
    Add seasonal dummy variables to dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    freq : str
        Type of seasonality ('day_of_week', 'month', 'quarter')
    drop_first : bool
        Whether to drop first dummy to avoid multicollinearity
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with seasonal dummies added
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if freq == 'day_of_week':
        # Create day of week dummies (0=Monday, 6=Sunday)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(day_names):
            df[f'dow_{day}'] = (df.index.dayofweek == i).astype(int)
        
        if drop_first:
            df = df.drop('dow_Mon', axis=1)  # Monday as baseline
            
    elif freq == 'month':
        # Create month dummies
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month in enumerate(month_names, 1):
            df[f'month_{month}'] = (df.index.month == i).astype(int)
        
        if drop_first:
            df = df.drop('month_Jan', axis=1)  # January as baseline
            
    elif freq == 'quarter':
        # Create quarter dummies
        for q in range(1, 5):
            df[f'quarter_Q{q}'] = (df.index.quarter == q).astype(int)
        
        if drop_first:
            df = df.drop('quarter_Q1', axis=1)  # Q1 as baseline
    
    return df


def add_trend_variables(df: pd.DataFrame, 
                       polynomial_order: int = 2) -> pd.DataFrame:
    """
    Add trend and polynomial trend variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    polynomial_order : int
        Maximum polynomial order for trend
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with trend variables added
    """
    df = df.copy()
    n = len(df)
    
    # Linear trend
    df['trend'] = range(1, n + 1)
    
    # Polynomial trends
    for order in range(2, polynomial_order + 1):
        df[f'trend_{order}'] = df['trend'] ** order
    
    # Normalized versions (to avoid numerical issues)
    df['trend_norm'] = df['trend'] / n
    for order in range(2, polynomial_order + 1):
        df[f'trend_{order}_norm'] = (df['trend'] / n) ** order
    
    return df


def add_event_dummies(df: pd.DataFrame, 
                     event_dates: Dict[str, str] = None) -> pd.DataFrame:
    """
    Add dummy variables for specific events or dates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    event_dates : dict
        Dictionary mapping dates to event names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with event dummies added
    """
    df = df.copy()
    
    if event_dates is None:
        # Default major US holidays and shopping events
        event_dates = {
            '2025-07-04': 'independence_day',
            '2025-09-01': 'labor_day',
            '2025-05-26': 'memorial_day',
            # Add Black Friday, Cyber Monday based on year
        }
    
    for date_str, event_name in event_dates.items():
        event_date = pd.to_datetime(date_str)
        if event_date in df.index:
            df[f'event_{event_name}'] = (df.index == event_date).astype(int)
            
            # Also add pre- and post-event effects
            if event_date - pd.Timedelta(days=1) in df.index:
                df[f'event_{event_name}_pre'] = (df.index == event_date - pd.Timedelta(days=1)).astype(int)
            if event_date + pd.Timedelta(days=1) in df.index:
                df[f'event_{event_name}_post'] = (df.index == event_date + pd.Timedelta(days=1)).astype(int)
    
    return df


def prepare_data_with_controls(df: pd.DataFrame,
                              add_seasonality: bool = True,
                              add_trends: bool = True,
                              add_events: bool = False,
                              event_dates: Dict = None) -> pd.DataFrame:
    """
    Prepare data with all control variables for robust ARDL analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    add_seasonality : bool
        Whether to add seasonal dummies
    add_trends : bool
        Whether to add trend variables
    add_events : bool
        Whether to add event dummies
    event_dates : dict
        Dictionary of event dates
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all controls added
    """
    # First do standard preparation (log transforms)
    df = prepare_data(df)
    
    # Add seasonal dummies
    if add_seasonality:
        df = add_seasonal_dummies(df, freq='day_of_week')
    
    # Add trends
    if add_trends:
        df = add_trend_variables(df, polynomial_order=2)
    
    # Add event dummies
    if add_events:
        df = add_event_dummies(df, event_dates)
    
    return df


def get_control_variable_names(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get names of control variables by type
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with control variables
    
    Returns:
    --------
    dict
        Dictionary mapping control type to variable names
    """
    controls = {
        'seasonal': [],
        'trend': [],
        'event': [],
        'all': []
    }
    
    for col in df.columns:
        if col.startswith('dow_') or col.startswith('month_') or col.startswith('quarter_'):
            controls['seasonal'].append(col)
        elif 'trend' in col:
            controls['trend'].append(col)
        elif col.startswith('event_'):
            controls['event'].append(col)
    
    controls['all'] = controls['seasonal'] + controls['trend'] + controls['event']
    
    return controls