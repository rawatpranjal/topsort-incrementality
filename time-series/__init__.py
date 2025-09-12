"""
ARDL Analysis Package for Click-to-Purchase Funnel Data
========================================================

A modular framework for Autoregressive Distributed Lag (ARDL) modeling
and cointegration analysis of e-commerce metrics.
"""

from .data_handler import load_data, prepare_data, test_stationarity
from .ardl_models import select_optimal_lags, estimate_ardl, convert_to_uecm, perform_bounds_test
from .analysis import (
    calculate_elasticity,
    calculate_half_life,
    calculate_dynamic_multipliers,
    perform_hypothesis_test
)
from .diagnostics import test_serial_correlation, test_stability
from .utils import setup_logger, save_results

__version__ = "1.0.0"
__author__ = "ARDL Analysis Framework"

__all__ = [
    'load_data',
    'prepare_data',
    'test_stationarity',
    'select_optimal_lags',
    'estimate_ardl',
    'convert_to_uecm',
    'perform_bounds_test',
    'calculate_elasticity',
    'calculate_half_life',
    'calculate_dynamic_multipliers',
    'perform_hypothesis_test',
    'test_serial_correlation',
    'test_stability',
    'setup_logger',
    'save_results'
]