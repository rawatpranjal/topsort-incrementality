#!/usr/bin/env python3
"""
Full-scale ALS implementation for entire user-vendor dataset
Optimized for memory efficiency and speed
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import warnings
import sys
from datetime import datetime
from pathlib import Path
import json
import gc
from typing import Optional, Tuple
# import h5py  # Optional for HDF5 storage

warnings.filterwarnings('ignore')

class EfficientALS:
    """Memory-efficient ALS implementation for large-scale collaborative filtering"""
    
    def __init__(self, n_factors: int = 50, regularization: float = 0.01, 
                 iterations: int = 10, use_gpu: bool = False):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        
        self.user_factors = None
        self.vendor_factors = None
        self.n_users = 0
        self.n_vendors = 0
        
    def fit(self, matrix: csr_matrix, verbose: bool = True):
        """
        Fit ALS model using efficient sparse operations
        Uses Conjugate Gradient for solving linear systems
        """
        self.n_users, self.n_vendors = matrix.shape
        
        if verbose:
            print(f"Fitting ALS on {self.n_users:,} users × {self.n_vendors:,} vendors")
            print(f"Matrix density: {matrix.nnz / (self.n_users * self.n_vendors):.4%}")
            print(f"Non-zero entries: {matrix.nnz:,}")
        
        # Initialize factors with small random values
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (self.n_users, self.n_factors)).astype(np.float32)
        self.vendor_factors = np.random.normal(0, 0.01, (self.n_vendors, self.n_factors)).astype(np.float32)
        
        # Precompute regularization matrix
        reg_eye = self.regularization * np.eye(self.n_factors, dtype=np.float32)
        
        for iteration in range(self.iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.iterations}")
            
            # Update user factors (holding vendor factors fixed)
            if verbose:
                print("  Updating user factors...")
            # For users: we need matrix transposed, so each row is a vendor
            self._update_factors_batch(matrix.T.tocsr(), self.vendor_factors, 
                                      self.user_factors, reg_eye, verbose, is_user=True)
            
            # Update vendor factors (holding user factors fixed)
            if verbose:
                print("  Updating vendor factors...")
            # For vendors: matrix as is, each row is a user
            self._update_factors_batch(matrix, self.user_factors, 
                                      self.vendor_factors, reg_eye, verbose, is_user=False)
            
            # Calculate and print RMSE periodically
            if verbose and (iteration + 1) % 5 == 0:
                rmse = self._calculate_rmse_sample(matrix)
                print(f"  Sample RMSE: {rmse:.4f}")
        
        return self
    
    def _update_factors_batch(self, matrix: csr_matrix, fixed_factors: np.ndarray,
                              factors_to_update: np.ndarray, reg_eye: np.ndarray,
                              verbose: bool = False, is_user: bool = False):
        """
        Update factors in batches for memory efficiency
        Uses closed-form solution for each user/vendor
        """
        n_entities = factors_to_update.shape[0]
        n_other = fixed_factors.shape[0]
        batch_size = 1000
        
        # Ensure we don't go out of bounds
        n_rows = min(n_entities, matrix.shape[0])
        
        for batch_start in range(0, n_rows, batch_size):
            batch_end = min(batch_start + batch_size, n_rows)
            
            if verbose and batch_start % 10000 == 0:
                print(f"    Processing {batch_start:,} - {batch_end:,} / {n_rows:,}")
            
            for idx in range(batch_start, batch_end):
                # Get the sparse row
                start_idx = matrix.indptr[idx]
                end_idx = matrix.indptr[idx + 1]
                
                if start_idx == end_idx:  # No interactions
                    continue
                
                indices = matrix.indices[start_idx:end_idx]
                values = matrix.data[start_idx:end_idx].astype(np.float32)
                
                # Filter out invalid indices
                valid_mask = indices < n_other
                indices = indices[valid_mask]
                values = values[valid_mask]
                
                if len(indices) == 0:
                    continue
                
                # Solve: (X^T X + λI) w = X^T y
                X = fixed_factors[indices]
                XtX = X.T.dot(X) + reg_eye
                Xty = X.T.dot(values)
                
                try:
                    factors_to_update[idx] = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    # Use least squares if singular
                    factors_to_update[idx] = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    
    def _calculate_rmse_sample(self, matrix: csr_matrix, sample_size: int = 10000) -> float:
        """Calculate RMSE on a sample of entries for efficiency"""
        # Sample random entries
        coo = matrix.tocoo()
        n_entries = len(coo.data)
        
        if n_entries > sample_size:
            sample_idx = np.random.choice(n_entries, sample_size, replace=False)
            sample_users = coo.row[sample_idx]
            sample_vendors = coo.col[sample_idx]
            sample_values = coo.data[sample_idx]
        else:
            sample_users = coo.row
            sample_vendors = coo.col
            sample_values = coo.data
        
        # Calculate predictions
        predictions = np.sum(self.user_factors[sample_users] * 
                           self.vendor_factors[sample_vendors], axis=1)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - sample_values) ** 2))
        
        return rmse

class FullScaleDataProcessor:
    """Process full-scale user-vendor data"""
    
    def __init__(self, data_path: str = '../panel/archive'):
        self.data_path = Path(data_path)
        self.user_mapping = {}
        self.vendor_mapping = {}
        self.matrix = None
        
    def load_and_process_full_data(self, value_column: str = 'gmv') -> csr_matrix:
        """Load and process the full dataset"""
        print("="*80)
        print("LOADING FULL USER-VENDOR DATA")
        print("="*80)
        
        file_path = self.data_path / 'user_vendor_panel_pilot.parquet'
        
        print(f"Loading from: {file_path}")
        
        # Load in chunks to manage memory
        df = pd.read_parquet(file_path)
        
        print(f"Raw data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1e9:.2f} GB")
        
        # Rename columns
        df = df.rename(columns={
            'USER_ID': 'user_id',
            'VENDOR_ID': 'vendor_id',
            'TOTAL_REVENUE_VENDOR_PRODUCT': 'gmv',
            'TOTAL_CLICKS_PROMOTED': 'clicks',
            'TOTAL_PURCHASES_VENDOR_PRODUCT': 'purchases'
        })
        
        # Filter for positive values
        print(f"\nFiltering for positive {value_column}...")
        df_filtered = df[df[value_column] > 0].copy()
        
        # Free memory
        del df
        gc.collect()
        
        print(f"Filtered data shape: {df_filtered.shape}")
        
        # Aggregate by user-vendor (sum across weeks)
        print("\nAggregating by user-vendor pairs...")
        df_agg = df_filtered.groupby(['user_id', 'vendor_id'])[value_column].sum().reset_index()
        
        # Free memory
        del df_filtered
        gc.collect()
        
        print(f"Aggregated pairs: {len(df_agg):,}")
        
        # Create mappings
        print("\nCreating user and vendor mappings...")
        unique_users = df_agg['user_id'].unique()
        unique_vendors = df_agg['vendor_id'].unique()
        
        print(f"Unique users: {len(unique_users):,}")
        print(f"Unique vendors: {len(unique_vendors):,}")
        
        # Use efficient mapping with enumerate
        self.user_mapping = {user: idx for idx, user in enumerate(sorted(unique_users))}
        self.vendor_mapping = {vendor: idx for idx, vendor in enumerate(sorted(unique_vendors))}
        
        # Map to indices
        print("\nMapping to indices...")
        df_agg['user_idx'] = df_agg['user_id'].map(self.user_mapping)
        df_agg['vendor_idx'] = df_agg['vendor_id'].map(self.vendor_mapping)
        
        # Apply log transformation for GMV
        if value_column == 'gmv':
            print("Applying log transformation...")
            values = np.log1p(df_agg[value_column].values).astype(np.float32)
        else:
            values = df_agg[value_column].values.astype(np.float32)
        
        # Create sparse matrix
        print("\nCreating sparse matrix...")
        self.matrix = csr_matrix(
            (values, (df_agg['user_idx'].values, df_agg['vendor_idx'].values)),
            shape=(len(unique_users), len(unique_vendors)),
            dtype=np.float32
        )
        
        # Free memory
        del df_agg
        gc.collect()
        
        print(f"Matrix shape: {self.matrix.shape}")
        print(f"Non-zero entries: {self.matrix.nnz:,}")
        print(f"Density: {self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]):.4%}")
        print(f"Matrix memory usage: {self.matrix.data.nbytes / 1e6:.1f} MB")
        
        return self.matrix

class ScoreComputer:
    """Compute and save scores efficiently"""
    
    def __init__(self, als_model: EfficientALS, data_processor: FullScaleDataProcessor):
        self.als_model = als_model
        self.data_processor = data_processor
        
    def compute_scores_batch(self, batch_size: int = 1000, output_file: str = 'scores.npz'):
        """
        Compute scores in batches and save factors for efficiency
        Don't materialize full matrix - too large!
        """
        print("\n" + "="*80)
        print("COMPUTING SCORES IN BATCHES")
        print("="*80)
        
        n_users = self.als_model.n_users
        n_vendors = self.als_model.n_vendors
        
        print(f"Total user-vendor pairs: {n_users * n_vendors:,}")
        print(f"Estimated full matrix size: {n_users * n_vendors * 4 / 1e9:.1f} GB")
        
        # Save factors in compressed format instead of full matrix
        output_path = Path('results') / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        # Store factors - can reconstruct any score as needed
        np.savez_compressed(output_path,
                          user_factors=self.als_model.user_factors,
                          vendor_factors=self.als_model.vendor_factors,
                          n_users=n_users,
                          n_vendors=n_vendors,
                          n_factors=self.als_model.n_factors)
        
        print(f"Factors saved to: {output_path}")
        
        # Compute statistics on samples
        self._compute_score_statistics()
        
    def _compute_score_statistics(self, n_samples: int = 100000):
        """Compute statistics on sampled scores"""
        print("\nComputing score statistics on samples...")
        
        n_users = self.als_model.n_users
        n_vendors = self.als_model.n_vendors
        
        # Sample random user-vendor pairs
        sample_users = np.random.randint(0, n_users, n_samples)
        sample_vendors = np.random.randint(0, n_vendors, n_samples)
        
        # Compute scores for samples
        scores = np.sum(self.als_model.user_factors[sample_users] * 
                       self.als_model.vendor_factors[sample_vendors], axis=1)
        
        # Statistics
        stats = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(np.median(scores)),
            'percentiles': {
                p: float(np.percentile(scores, p))
                for p in [1, 5, 25, 50, 75, 95, 99, 99.9]
            }
        }
        
        print(f"Score statistics (n={n_samples:,} samples):")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        
        # Save statistics
        stats_file = Path('results') / f'score_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
        
        return stats
    
    def get_top_scores_per_user(self, n_top: int = 10, n_users: Optional[int] = None):
        """Get top N vendor recommendations for each user (or sample of users)"""
        print(f"\nGenerating top {n_top} recommendations per user...")
        
        if n_users is None:
            n_users = min(1000, self.als_model.n_users)  # Sample for demonstration
        
        user_indices = np.random.choice(self.als_model.n_users, n_users, replace=False)
        
        recommendations = []
        
        for user_idx in user_indices[:10]:  # Show first 10 as example
            # Compute scores for this user with all vendors
            user_scores = self.als_model.user_factors[user_idx].dot(self.als_model.vendor_factors.T)
            
            # Get top vendor indices
            top_vendor_indices = np.argsort(user_scores)[::-1][:n_top]
            
            # Get original IDs
            reverse_user_map = {v: k for k, v in self.data_processor.user_mapping.items()}
            reverse_vendor_map = {v: k for k, v in self.data_processor.vendor_mapping.items()}
            
            user_id = reverse_user_map.get(user_idx, f"user_{user_idx}")
            
            print(f"\nUser {user_id[:40]}...:")
            for rank, vendor_idx in enumerate(top_vendor_indices[:5], 1):
                vendor_id = reverse_vendor_map.get(vendor_idx, f"vendor_{vendor_idx}")
                score = user_scores[vendor_idx]
                print(f"  {rank}. Vendor {vendor_id[:40]}... (score={score:.4f})")
        
        return recommendations

def main():
    """Main execution for full-scale ALS"""
    print("="*80)
    print("FULL-SCALE ALS COLLABORATIVE FILTERING")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load and process full data
    processor = FullScaleDataProcessor()
    matrix = processor.load_and_process_full_data(value_column='gmv')
    
    # Fit ALS model
    print("\n" + "="*80)
    print("FITTING ALS MODEL")
    print("="*80)
    
    als = EfficientALS(n_factors=50, regularization=0.01, iterations=10)
    als.fit(matrix, verbose=True)
    
    # Save factors
    print("\nSaving factors...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save factors as NumPy arrays
    factors_file = Path('results') / f'als_full_factors_{timestamp}.npz'
    np.savez_compressed(factors_file,
                       user_factors=als.user_factors,
                       vendor_factors=als.vendor_factors)
    print(f"Factors saved to: {factors_file}")
    
    # Save mappings
    mappings_file = Path('results') / f'als_full_mappings_{timestamp}.json'
    
    # Convert mappings to JSON-serializable format (sample for size)
    sample_size = 1000
    sampled_user_mapping = dict(list(processor.user_mapping.items())[:sample_size])
    sampled_vendor_mapping = dict(list(processor.vendor_mapping.items())[:sample_size])
    
    with open(mappings_file, 'w') as f:
        json.dump({
            'n_users': len(processor.user_mapping),
            'n_vendors': len(processor.vendor_mapping),
            'user_mapping_sample': {str(k): v for k, v in sampled_user_mapping.items()},
            'vendor_mapping_sample': {str(k): v for k, v in sampled_vendor_mapping.items()},
            'note': f'Full mappings too large. Showing sample of {sample_size} entries.'
        }, f)
    print(f"Mappings saved to: {mappings_file}")
    
    # Compute scores and statistics
    scorer = ScoreComputer(als, processor)
    scorer.compute_scores_batch()
    
    # Generate sample recommendations
    scorer.get_top_scores_per_user(n_top=10, n_users=100)
    
    print("\n" + "="*80)
    print("FULL-SCALE ANALYSIS COMPLETE")
    print("="*80)
    print(f"User factors shape: {als.user_factors.shape}")
    print(f"Vendor factors shape: {als.vendor_factors.shape}")
    print(f"Total possible scores: {als.n_users * als.n_vendors:,}")
    
    return als, processor

if __name__ == "__main__":
    als_model, data_processor = main()