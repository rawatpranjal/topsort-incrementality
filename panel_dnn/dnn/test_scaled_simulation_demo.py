#!/usr/bin/env python3
"""
Quick demo of the scaled simulation with smaller dataset for testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heterogeneous_treatment_effects_simulation_scaled import run_simulation

if __name__ == "__main__":
    print("Running DEMO with 1M observations (instead of 10M)")
    print("="*60)

    # Run just linear-linear case with 1M observations for quick demo
    results, history = run_simulation(
        beta_type='linear',
        g_type='linear',
        n_obs=1_000_000  # 1M instead of 10M for demo
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE - Key Results:")
    print("="*60)
    print(f"Y Correlation:    {results['y_metrics']['corr']:.4f}")
    print(f"Y R-squared:      {results['y_metrics']['r2']:.4f}")
    print(f"β(X) Correlation: {results['beta_metrics']['corr']:.4f}")
    print(f"g(X) Correlation: {results['g_metrics']['corr']:.4f}")
    print(f"\nTraining epochs:  {len(history['train_loss'])}")
    print(f"Best val loss:    {min(history['val_loss']):.4f}")