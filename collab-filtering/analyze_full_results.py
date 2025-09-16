#!/usr/bin/env python3
"""
Analyze full-scale ALS results and compute scores
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def analyze_full_factors():
    """Analyze the full-scale user and vendor factors"""
    
    print("="*80)
    print("FULL-SCALE ALS RESULTS ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load the full factors
    factors_file = Path('results/als_full_factors_20250913_081826.npz')
    data = np.load(factors_file)
    
    user_factors = data['user_factors']
    vendor_factors = data['vendor_factors']
    
    print(f"\nLoaded factors from: {factors_file}")
    print(f"User factors shape: {user_factors.shape}")
    print(f"Vendor factors shape: {vendor_factors.shape}")
    
    n_users, n_factors_u = user_factors.shape
    n_vendors, n_factors_v = vendor_factors.shape
    
    print(f"\nTotal possible scores: {n_users * n_vendors:,}")
    print(f"Memory for full matrix: {n_users * n_vendors * 4 / 1e9:.1f} GB")
    
    # Analyze user factors
    print("\n" + "="*80)
    print("USER FACTOR ANALYSIS")
    print("="*80)
    
    print("\nUser Factor Statistics:")
    print(f"  Mean: {user_factors.mean():.6f}")
    print(f"  Std: {user_factors.std():.6f}")
    print(f"  Min: {user_factors.min():.6f}")
    print(f"  Max: {user_factors.max():.6f}")
    print(f"  Sparsity (|x| < 0.01): {(np.abs(user_factors) < 0.01).mean():.2%}")
    print(f"  Zero values: {(user_factors == 0).mean():.2%}")
    
    # Factor importance
    user_factor_var = np.var(user_factors, axis=0)
    print("\nTop 10 User Factors by Variance:")
    for i, idx in enumerate(np.argsort(user_factor_var)[::-1][:10], 1):
        print(f"  {i:2d}. Factor {idx+1}: variance={user_factor_var[idx]:.6f}")
    
    # Analyze vendor factors
    print("\n" + "="*80)
    print("VENDOR FACTOR ANALYSIS")
    print("="*80)
    
    print("\nVendor Factor Statistics:")
    print(f"  Mean: {vendor_factors.mean():.6f}")
    print(f"  Std: {vendor_factors.std():.6f}")
    print(f"  Min: {vendor_factors.min():.6f}")
    print(f"  Max: {vendor_factors.max():.6f}")
    print(f"  Sparsity (|x| < 0.01): {(np.abs(vendor_factors) < 0.01).mean():.2%}")
    print(f"  Zero values: {(vendor_factors == 0).mean():.2%}")
    
    vendor_factor_var = np.var(vendor_factors, axis=0)
    print("\nTop 10 Vendor Factors by Variance:")
    for i, idx in enumerate(np.argsort(vendor_factor_var)[::-1][:10], 1):
        print(f"  {i:2d}. Factor {idx+1}: variance={vendor_factor_var[idx]:.6f}")
    
    # Compute score statistics on samples
    print("\n" + "="*80)
    print("SCORE STATISTICS (SAMPLED)")
    print("="*80)
    
    n_samples = 1000000
    print(f"\nSampling {n_samples:,} random user-vendor pairs...")
    
    # Random sampling
    sample_users = np.random.randint(0, n_users, n_samples)
    sample_vendors = np.random.randint(0, n_vendors, n_samples)
    
    # Compute scores
    scores = np.sum(user_factors[sample_users] * vendor_factors[sample_vendors], axis=1)
    
    print("\nScore Distribution:")
    print(f"  Mean: {scores.mean():.6f}")
    print(f"  Std: {scores.std():.6f}")
    print(f"  Min: {scores.min():.6f}")
    print(f"  Max: {scores.max():.6f}")
    
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    print("\nScore Percentiles:")
    for p in percentiles:
        val = np.percentile(scores, p)
        print(f"  {p:5.1f}%: {val:8.6f}")
    
    # Find extreme scores
    print("\n" + "="*80)
    print("EXTREME SCORES")
    print("="*80)
    
    # Get users and vendors with highest factor norms
    user_norms = np.linalg.norm(user_factors, axis=1)
    vendor_norms = np.linalg.norm(vendor_factors, axis=1)
    
    top_users = np.argsort(user_norms)[::-1][:10]
    top_vendors = np.argsort(vendor_norms)[::-1][:10]
    
    print("\nTop 10 Users by Factor Norm:")
    for i, user_idx in enumerate(top_users, 1):
        print(f"  {i:2d}. User {user_idx}: norm={user_norms[user_idx]:.4f}")
    
    print("\nTop 10 Vendors by Factor Norm:")
    for i, vendor_idx in enumerate(top_vendors, 1):
        print(f"  {i:2d}. Vendor {vendor_idx}: norm={vendor_norms[vendor_idx]:.4f}")
    
    # Compute scores for top user-vendor pairs
    print("\nScores for Top User × Top Vendor Combinations:")
    for i in range(min(5, len(top_users))):
        for j in range(min(5, len(top_vendors))):
            score = np.dot(user_factors[top_users[i]], vendor_factors[top_vendors[j]])
            print(f"  User {top_users[i]} × Vendor {top_vendors[j]}: {score:.4f}")
    
    # User and vendor coverage
    print("\n" + "="*80)
    print("FACTOR COVERAGE ANALYSIS")
    print("="*80)
    
    # How many users/vendors have meaningful factors
    user_has_signal = (np.abs(user_factors).max(axis=1) > 0.01)
    vendor_has_signal = (np.abs(vendor_factors).max(axis=1) > 0.01)
    
    print(f"\nUsers with meaningful factors (max |factor| > 0.01): {user_has_signal.sum():,} ({user_has_signal.mean():.1%})")
    print(f"Vendors with meaningful factors (max |factor| > 0.01): {vendor_has_signal.sum():,} ({vendor_has_signal.mean():.1%})")
    
    # Dominant factor analysis
    user_dominant = np.argmax(np.abs(user_factors), axis=1)
    vendor_dominant = np.argmax(np.abs(vendor_factors), axis=1)
    
    print("\nDominant Factor Distribution:")
    print("Users:")
    user_factor_counts = pd.Series(user_dominant).value_counts().head(10)
    for factor, count in user_factor_counts.items():
        print(f"  Factor {factor+1}: {count:,} users ({count/n_users*100:.2f}%)")
    
    print("\nVendors:")
    vendor_factor_counts = pd.Series(vendor_dominant).value_counts().head(10)
    for factor, count in vendor_factor_counts.items():
        print(f"  Factor {factor+1}: {count:,} vendors ({count/n_vendors*100:.2f}%)")
    
    # Save statistics
    stats = {
        'n_users': int(n_users),
        'n_vendors': int(n_vendors),
        'n_factors': int(n_factors_u),
        'total_possible_scores': int(n_users * n_vendors),
        'user_factor_stats': {
            'mean': float(user_factors.mean()),
            'std': float(user_factors.std()),
            'min': float(user_factors.min()),
            'max': float(user_factors.max()),
            'sparsity': float((np.abs(user_factors) < 0.01).mean())
        },
        'vendor_factor_stats': {
            'mean': float(vendor_factors.mean()),
            'std': float(vendor_factors.std()),
            'min': float(vendor_factors.min()),
            'max': float(vendor_factors.max()),
            'sparsity': float((np.abs(vendor_factors) < 0.01).mean())
        },
        'score_stats_sampled': {
            'n_samples': n_samples,
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'percentiles': {f'p{int(p)}': float(np.percentile(scores, p)) for p in percentiles}
        },
        'coverage': {
            'users_with_signal': int(user_has_signal.sum()),
            'users_with_signal_pct': float(user_has_signal.mean()),
            'vendors_with_signal': int(vendor_has_signal.sum()),
            'vendors_with_signal_pct': float(vendor_has_signal.mean())
        }
    }
    
    stats_file = Path('results') / f'full_als_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n\nStatistics saved to: {stats_file}")
    
    return user_factors, vendor_factors, stats

if __name__ == "__main__":
    user_factors, vendor_factors, stats = analyze_full_factors()