"""
Utility functions for ARDL analysis
"""

import sys
import logging
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class OutputCapture:
    """Context manager to capture stdout to file"""
    def __init__(self, filename: str):
        self.filename = filename
        self.original_stdout = None
        self.file = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.file = open(self.filename, 'w')
        sys.stdout = Tee(sys.stdout, self.file)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()


class Tee:
    """Write to multiple file handles simultaneously"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            if hasattr(f, 'flush'):
                f.flush()
    
    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()


def setup_logger(name: str = 'ardl_analysis',
                level: str = 'INFO',
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger for ARDL analysis
    
    Parameters:
    -----------
    name : str
        Logger name
    level : str
        Logging level
    log_file : str, optional
        File to save logs
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def save_results(results: Dict,
                filename: str,
                format: str = 'txt') -> None:
    """
    Save analysis results to file
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filename : str
        Output filename
    format : str
        Output format ('txt', 'json', 'csv')
    """
    if format == 'txt':
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ARDL ANALYSIS RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Write results recursively
            _write_dict_to_file(results, f)
    
    elif format == 'json':
        import json
        
        # Convert non-serializable objects to strings
        clean_results = _clean_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
    
    elif format == 'csv':
        # Extract key metrics to CSV
        metrics = _extract_key_metrics(results)
        df = pd.DataFrame([metrics])
        df.to_csv(filename, index=False)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def _write_dict_to_file(d: Dict, f, indent: int = 0):
    """Helper to write dictionary to file recursively"""
    for key, value in d.items():
        if isinstance(value, dict):
            f.write(" " * indent + f"{key}:\n")
            _write_dict_to_file(value, f, indent + 2)
        elif isinstance(value, (list, tuple)):
            f.write(" " * indent + f"{key}: {value}\n")
        elif isinstance(value, pd.DataFrame):
            f.write(" " * indent + f"{key}:\n")
            f.write(value.to_string())
            f.write("\n")
        elif value is not None:
            f.write(" " * indent + f"{key}: {value}\n")


def _clean_for_json(obj: Any) -> Any:
    """Clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def _extract_key_metrics(results: Dict) -> Dict:
    """Extract key metrics from results for CSV output"""
    metrics = {}
    
    # Try to extract common metrics
    if 'bounds_test' in results:
        metrics['f_statistic'] = results['bounds_test'].get('f_statistic')
        metrics['cointegration'] = results['bounds_test'].get('cointegration')
    
    if 'elasticity' in results:
        metrics['long_run_elasticity'] = results['elasticity'].get('long_run_elasticity')
        metrics['speed_of_adjustment'] = results['elasticity'].get('speed_of_adjustment')
    
    if 'half_life' in results:
        metrics['half_life_days'] = results['half_life'].get('half_life_days')
    
    if 'diagnostics' in results:
        diag = results['diagnostics']
        if 'serial_correlation' in diag:
            metrics['has_serial_correlation'] = diag['serial_correlation'].get('has_serial_correlation')
        if 'heteroskedasticity' in diag:
            metrics['has_heteroskedasticity'] = diag['heteroskedasticity'].get('has_heteroskedasticity')
        if 'normality' in diag:
            metrics['is_normal'] = diag['normality'].get('is_normal')
    
    return metrics


def format_results_table(results: Dict) -> str:
    """
    Format results as a nice table string
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    
    Returns:
    --------
    str
        Formatted table
    """
    lines = []
    lines.append("="*60)
    lines.append("SUMMARY OF KEY RESULTS")
    lines.append("="*60)
    
    # Cointegration
    if 'bounds_test' in results:
        bt = results['bounds_test']
        lines.append(f"\nCointegration Test:")
        lines.append(f"  F-statistic: {bt.get('f_statistic', 'N/A'):.4f}")
        lines.append(f"  Result: {bt.get('conclusion', 'N/A')}")
    
    # Elasticity
    if 'elasticity' in results:
        el = results['elasticity']
        lines.append(f"\nLong-Run Elasticity:")
        lines.append(f"  Value: {el.get('long_run_elasticity', 'N/A'):.4f}")
        lines.append(f"  Speed of Adjustment: {el.get('speed_of_adjustment', 'N/A'):.4f}")
    
    # Half-life
    if 'half_life' in results:
        hl = results['half_life']
        lines.append(f"\nHalf-Life of Shocks:")
        lines.append(f"  Days: {hl.get('half_life_days', 'N/A'):.1f}")
    
    # Diagnostics
    if 'diagnostics' in results:
        diag = results['diagnostics']
        lines.append(f"\nModel Diagnostics:")
        if 'overall_assessment' in diag:
            assessment = diag['overall_assessment']
            lines.append(f"  Issues: {', '.join(assessment['issues_detected']) if assessment['issues_detected'] else 'None'}")
            lines.append(f"  Quality: {assessment['model_quality']}")
    
    return "\n".join(lines)


def print_banner(text: str, width: int = 80, char: str = "="):
    """Print a formatted banner"""
    print(char * width)
    print(text.center(width))
    print(char * width)