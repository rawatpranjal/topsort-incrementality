#!/usr/bin/env python3
"""
Collaborative Filtering using ALS on User-Vendor Interaction Matrix
Analyzes implicit feedback patterns in marketplace interactions
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

warnings.filterwarnings('ignore')

class DataLoader:
    """Load and prepare user-vendor interaction data"""
    
    def __init__(self, data_path: str = '../panel/archive'):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.interaction_matrix = None
        
    def load_user_vendor_data(self) -> pd.DataFrame:
        """Load user-vendor panel data"""
        print("="*80)
        print("LOADING USER-VENDOR INTERACTION DATA")
        print("="*80)
        
        # Try to load from panel data
        file_path = self.data_path / 'user_vendor_panel_pilot.parquet'
        
        if file_path.exists():
            print(f"Loading from: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"Loaded shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns[:10])}")
            
            # Rename columns to standard names
            df = df.rename(columns={
                'USER_ID': 'user_id',
                'VENDOR_ID': 'vendor_id',
                'WEEK': 'week',
                'TOTAL_REVENUE_VENDOR_PRODUCT': 'gmv',
                'TOTAL_CLICKS_PROMOTED': 'clicks',
                'TOTAL_PURCHASES_VENDOR_PRODUCT': 'purchases'
            })
            
            print(f"Renamed columns: {', '.join(df.columns)}")
            
            self.raw_data = df
            return df
        else:
            print(f"File not found: {file_path}")
            print("Generating synthetic data for demonstration...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic user-vendor interaction data"""
        np.random.seed(42)
        
        n_users = 10000
        n_vendors = 500
        n_weeks = 12
        sparsity = 0.95  # 95% sparse matrix
        
        print(f"Generating synthetic data:")
        print(f"  Users: {n_users}")
        print(f"  Vendors: {n_vendors}")
        print(f"  Weeks: {n_weeks}")
        print(f"  Sparsity: {sparsity:.1%}")
        
        data = []
        
        for week in range(n_weeks):
            n_interactions = int(n_users * n_vendors * (1 - sparsity) / n_weeks)
            
            users = np.random.choice(n_users, n_interactions)
            vendors = np.random.choice(n_vendors, n_interactions)
            
            # Generate different interaction types
            gmv = np.random.lognormal(4, 1.5, n_interactions)
            clicks = np.random.poisson(5, n_interactions)
            impressions = np.random.poisson(20, n_interactions)
            
            week_data = pd.DataFrame({
                'user_id': users,
                'vendor_id': vendors,
                'week': week,
                'gmv': gmv,
                'clicks': clicks,
                'impressions': impressions
            })
            
            data.append(week_data)
        
        df = pd.concat(data, ignore_index=True)
        
        # Aggregate by user-vendor pairs
        df = df.groupby(['user_id', 'vendor_id']).agg({
            'gmv': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'week': 'count'
        }).reset_index()
        
        df.rename(columns={'week': 'n_weeks'}, inplace=True)
        
        self.raw_data = df
        return df

class InteractionMatrixBuilder:
    """Build and analyze interaction matrices"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.matrices = {}
        self.user_mapping = {}
        self.vendor_mapping = {}
        
    def build_matrix(self, value_col: str = 'gmv', min_interactions: int = 5, 
                    sample_users: Optional[int] = 10000) -> csr_matrix:
        """Build sparse user-vendor interaction matrix"""
        print(f"\nBUILDING INTERACTION MATRIX FOR: {value_col}")
        print("-"*80)
        
        # Filter for minimum interactions  
        df_filtered = self.df[self.df[value_col] > 0].copy()
        
        # Sample users if requested (for computational efficiency)
        if sample_users and len(df_filtered['user_id'].unique()) > sample_users:
            print(f"Sampling {sample_users} users from {len(df_filtered['user_id'].unique())} total users")
            sampled_users = np.random.choice(df_filtered['user_id'].unique(), 
                                           sample_users, replace=False)
            df_filtered = df_filtered[df_filtered['user_id'].isin(sampled_users)]
        
        # Aggregate by user-vendor to handle duplicates
        df_filtered = df_filtered.groupby(['user_id', 'vendor_id'])[value_col].sum().reset_index()
        
        # Create user and vendor mappings
        unique_users = sorted(df_filtered['user_id'].unique())
        unique_vendors = sorted(df_filtered['vendor_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.vendor_mapping = {vendor: idx for idx, vendor in enumerate(unique_vendors)}
        
        print(f"Unique users: {len(unique_users)}")
        print(f"Unique vendors: {len(unique_vendors)}")
        print(f"Total interactions: {len(df_filtered)}")
        
        # Map to indices
        df_filtered['user_idx'] = df_filtered['user_id'].map(self.user_mapping)
        df_filtered['vendor_idx'] = df_filtered['vendor_id'].map(self.vendor_mapping)
        
        # Create sparse matrix
        row = df_filtered['user_idx'].values
        col = df_filtered['vendor_idx'].values
        data = df_filtered[value_col].values
        
        # Apply log transformation for GMV to reduce skewness
        if value_col == 'gmv':
            data = np.log1p(data)
            print("Applied log transformation to GMV values")
        
        matrix = csr_matrix((data, (row, col)), 
                           shape=(len(unique_users), len(unique_vendors)))
        
        # Calculate statistics
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        
        print(f"Matrix shape: {matrix.shape}")
        print(f"Non-zero entries: {matrix.nnz}")
        print(f"Density: {density:.4%}")
        print(f"Mean value (non-zero): {matrix.data.mean():.2f}")
        print(f"Std value (non-zero): {matrix.data.std():.2f}")
        
        self.matrices[value_col] = matrix
        return matrix
    
    def analyze_matrix_properties(self, matrix: csr_matrix, name: str = ""):
        """Analyze properties of interaction matrix"""
        print(f"\nMATRIX PROPERTIES: {name}")
        print("-"*80)
        
        # User statistics
        user_interactions = np.array((matrix > 0).sum(axis=1)).flatten()
        user_values = np.array(matrix.sum(axis=1)).flatten()
        
        print("User Statistics:")
        print(f"  Users with interactions: {(user_interactions > 0).sum()}")
        print(f"  Mean vendors per user: {user_interactions.mean():.2f}")
        print(f"  Median vendors per user: {np.median(user_interactions):.0f}")
        print(f"  Max vendors per user: {user_interactions.max()}")
        print(f"  Mean value per user: {user_values.mean():.2f}")
        
        # Vendor statistics
        vendor_interactions = np.array((matrix > 0).sum(axis=0)).flatten()
        vendor_values = np.array(matrix.sum(axis=0)).flatten()
        
        print("\nVendor Statistics:")
        print(f"  Vendors with interactions: {(vendor_interactions > 0).sum()}")
        print(f"  Mean users per vendor: {vendor_interactions.mean():.2f}")
        print(f"  Median users per vendor: {np.median(vendor_interactions):.0f}")
        print(f"  Max users per vendor: {vendor_interactions.max()}")
        print(f"  Mean value per vendor: {vendor_values.mean():.2f}")
        
        # Power law analysis
        user_dist = np.sort(user_interactions[user_interactions > 0])[::-1]
        vendor_dist = np.sort(vendor_interactions[vendor_interactions > 0])[::-1]
        
        print("\nDistribution Analysis:")
        print(f"  Top 10% users account for: {user_dist[:len(user_dist)//10].sum() / user_dist.sum():.1%} of interactions")
        print(f"  Top 10% vendors account for: {vendor_dist[:len(vendor_dist)//10].sum() / vendor_dist.sum():.1%} of interactions")

class ALSModel:
    """Alternating Least Squares for Matrix Factorization"""
    
    def __init__(self, n_factors: int = 50, regularization: float = 0.01, 
                 iterations: int = 20, random_state: int = 42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        
        self.user_factors = None
        self.vendor_factors = None
        self.training_error = []
        
    def fit(self, matrix: csr_matrix, verbose: bool = True):
        """Fit ALS model using alternating least squares"""
        print("\n" + "="*80)
        print("ALTERNATING LEAST SQUARES (ALS) FITTING")
        print("="*80)
        print(f"Factors: {self.n_factors}")
        print(f"Regularization: {self.regularization}")
        print(f"Iterations: {self.iterations}")
        print("-"*80)
        
        np.random.seed(self.random_state)
        
        n_users, n_vendors = matrix.shape
        
        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.vendor_factors = np.random.normal(0, 0.01, (n_vendors, self.n_factors))
        
        # Convert to COO format for efficient iteration
        matrix_coo = matrix.tocoo()
        
        for iteration in range(self.iterations):
            # Update user factors
            self._update_factors(matrix.T.tocsr(), self.vendor_factors, 
                                self.user_factors, self.regularization, update_users=True)
            
            # Update vendor factors
            self._update_factors(matrix, self.user_factors, 
                                self.vendor_factors, self.regularization, update_users=False)
            
            # Calculate training error
            error = self._calculate_error(matrix_coo)
            self.training_error.append(error)
            
            if verbose and (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1:3d}: RMSE = {error:.4f}")
        
        print(f"\nFinal RMSE: {self.training_error[-1]:.4f}")
        
        return self
    
    def _update_factors(self, matrix: csr_matrix, fixed_factors: np.ndarray,
                       factors_to_update: np.ndarray, regularization: float, 
                       update_users: bool = True):
        """Update one set of factors holding the other fixed"""
        n_factors = fixed_factors.shape[1]
        
        for idx in range(min(factors_to_update.shape[0], len(matrix.indptr) - 1)):
            # Get non-zero indices for this user/vendor
            start_idx = matrix.indptr[idx]
            end_idx = matrix.indptr[idx + 1]
            
            if start_idx == end_idx:  # No interactions
                continue
            
            indices = matrix.indices[start_idx:end_idx]
            values = matrix.data[start_idx:end_idx]
            
            # Ensure indices are within bounds
            if len(indices) > 0 and indices.max() >= fixed_factors.shape[0]:
                # Filter out invalid indices
                valid_mask = indices < fixed_factors.shape[0]
                indices = indices[valid_mask]
                values = values[valid_mask]
                if len(indices) == 0:
                    continue
            
            # Build system of equations
            A = fixed_factors[indices]
            AtA = A.T @ A + regularization * np.eye(n_factors)
            Atb = A.T @ values
            
            # Solve for factors
            try:
                factors_to_update[idx] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                # Use pseudoinverse if singular
                factors_to_update[idx] = np.linalg.pinv(AtA) @ Atb
    
    def _calculate_error(self, matrix_coo: coo_matrix) -> float:
        """Calculate RMSE on observed entries"""
        predictions = (self.user_factors[matrix_coo.row] * 
                      self.vendor_factors[matrix_coo.col]).sum(axis=1)
        
        errors = matrix_coo.data - predictions
        rmse = np.sqrt(np.mean(errors ** 2))
        
        return rmse
    
    def predict(self, user_idx: int, vendor_idx: int) -> float:
        """Predict interaction value for user-vendor pair"""
        return np.dot(self.user_factors[user_idx], self.vendor_factors[vendor_idx])
    
    def recommend_vendors(self, user_idx: int, n_recommendations: int = 10,
                         exclude_known: bool = True, known_vendors: set = None) -> List[Tuple[int, float]]:
        """Recommend top vendors for a user"""
        user_vector = self.user_factors[user_idx]
        scores = self.vendor_factors @ user_vector
        
        if exclude_known and known_vendors:
            for vendor_idx in known_vendors:
                scores[vendor_idx] = -np.inf
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(idx, scores[idx]) for idx in top_indices]
        
        return recommendations

class SVDModel:
    """Truncated SVD for comparison"""
    
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=n_components, 
                                 random_state=random_state)
        self.user_factors = None
        self.vendor_factors = None
        
    def fit(self, matrix: csr_matrix):
        """Fit SVD model"""
        print("\n" + "="*80)
        print("TRUNCATED SVD FITTING")
        print("="*80)
        print(f"Components: {self.n_components}")
        
        # Fit SVD
        self.user_factors = self.model.fit_transform(matrix)
        self.vendor_factors = self.model.components_.T
        
        # Calculate explained variance
        explained_var = self.model.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_var:.2%}")
        
        # Show singular values
        print(f"\nTop 10 singular values:")
        for i, val in enumerate(self.model.singular_values_[:10]):
            print(f"  Component {i+1}: {val:.2f}")
        
        return self

class NMFModel:
    """Non-negative Matrix Factorization for comparison"""
    
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = NMF(n_components=n_components, init='random',
                        random_state=random_state, max_iter=200)
        self.user_factors = None
        self.vendor_factors = None
        
    def fit(self, matrix: csr_matrix):
        """Fit NMF model"""
        print("\n" + "="*80)
        print("NON-NEGATIVE MATRIX FACTORIZATION")
        print("="*80)
        print(f"Components: {self.n_components}")
        
        # Ensure non-negative values
        matrix_nonneg = matrix.copy()
        matrix_nonneg.data = np.abs(matrix_nonneg.data)
        
        # Fit NMF
        self.user_factors = self.model.fit_transform(matrix_nonneg)
        self.vendor_factors = self.model.components_.T
        
        # Calculate reconstruction error
        reconstruction_err = self.model.reconstruction_err_
        print(f"Reconstruction error: {reconstruction_err:.2f}")
        
        return self

class ModelEvaluator:
    """Evaluate and compare collaborative filtering models"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_reconstruction(self, model, matrix: csr_matrix, model_name: str):
        """Evaluate model reconstruction error"""
        print(f"\nEVALUATING {model_name}")
        print("-"*80)
        
        # Get factors
        if hasattr(model, 'user_factors') and hasattr(model, 'vendor_factors'):
            U = model.user_factors
            V = model.vendor_factors
        else:
            print("Model doesn't have factors")
            return
        
        # Reconstruct matrix
        reconstructed = U @ V.T
        
        # Calculate errors on observed entries
        matrix_coo = matrix.tocoo()
        
        original_values = matrix_coo.data
        reconstructed_values = reconstructed[matrix_coo.row, matrix_coo.col]
        
        rmse = np.sqrt(mean_squared_error(original_values, reconstructed_values))
        mae = mean_absolute_error(original_values, reconstructed_values)
        
        # Correlation
        correlation = np.corrcoef(original_values, reconstructed_values)[0, 1]
        
        # Store results
        self.results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'n_factors': U.shape[1]
        }
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        # Analyze factor properties
        self.analyze_factors(U, V, model_name)
        
    def analyze_factors(self, U: np.ndarray, V: np.ndarray, model_name: str):
        """Analyze learned factors"""
        print(f"\nFACTOR ANALYSIS: {model_name}")
        print("-"*40)
        
        # User factors
        print("User Factors:")
        print(f"  Shape: {U.shape}")
        print(f"  Mean: {U.mean():.4f}")
        print(f"  Std: {U.std():.4f}")
        print(f"  Sparsity: {(np.abs(U) < 0.01).mean():.2%}")
        
        # Vendor factors
        print("\nVendor Factors:")
        print(f"  Shape: {V.shape}")
        print(f"  Mean: {V.mean():.4f}")
        print(f"  Std: {V.std():.4f}")
        print(f"  Sparsity: {(np.abs(V) < 0.01).mean():.2%}")
        
        # Factor importance (based on variance)
        user_var = np.var(U, axis=0)
        vendor_var = np.var(V, axis=0)
        
        print(f"\nTop 5 most important factors (by variance):")
        top_factors = np.argsort(user_var + vendor_var)[::-1][:5]
        for i, factor_idx in enumerate(top_factors, 1):
            print(f"  Factor {factor_idx+1}: User var={user_var[factor_idx]:.4f}, "
                  f"Vendor var={vendor_var[factor_idx]:.4f}")
    
    def compare_models(self):
        """Compare all evaluated models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        if not self.results:
            print("No models evaluated yet")
            return
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Factors': metrics['n_factors'],
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'Correlation': f"{metrics['correlation']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model = min(self.results.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest model (lowest RMSE): {best_model[0]}")

class ResultsManager:
    """Manage and save analysis results"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def save_results(self, evaluator: ModelEvaluator, als_model: ALSModel,
                    matrix_builder: InteractionMatrixBuilder):
        """Save all results to files"""
        
        # Save model comparison
        comparison_file = self.output_dir / f'model_comparison_{self.timestamp}.json'
        with open(comparison_file, 'w') as f:
            json.dump(evaluator.results, f, indent=2)
        print(f"\nModel comparison saved to: {comparison_file}")
        
        # Save factors
        factors_file = self.output_dir / f'als_factors_{self.timestamp}.npz'
        np.savez(factors_file,
                user_factors=als_model.user_factors,
                vendor_factors=als_model.vendor_factors,
                training_error=als_model.training_error)
        print(f"ALS factors saved to: {factors_file}")
        
        # Save mappings
        mappings_file = self.output_dir / f'mappings_{self.timestamp}.json'
        with open(mappings_file, 'w') as f:
            json.dump({
                'user_mapping': {str(k): v for k, v in matrix_builder.user_mapping.items()},
                'vendor_mapping': {str(k): v for k, v in matrix_builder.vendor_mapping.items()}
            }, f)
        print(f"Mappings saved to: {mappings_file}")
        
        # Generate recommendations sample
        self.generate_sample_recommendations(als_model, matrix_builder)
    
    def generate_sample_recommendations(self, als_model: ALSModel, 
                                       matrix_builder: InteractionMatrixBuilder):
        """Generate sample recommendations for analysis"""
        print("\n" + "="*80)
        print("SAMPLE RECOMMENDATIONS")
        print("="*80)
        
        matrix = matrix_builder.matrices.get('gmv')
        if matrix is None:
            return
        
        # Select sample users
        n_samples = min(5, len(matrix_builder.user_mapping))
        sample_users = np.random.choice(list(matrix_builder.user_mapping.values()), 
                                      n_samples, replace=False)
        
        recommendations = []
        
        for user_idx in sample_users:
            # Get known vendors for this user
            known_vendors = set(matrix[user_idx].indices)
            
            # Get recommendations
            recs = als_model.recommend_vendors(user_idx, n_recommendations=5,
                                              exclude_known=True, 
                                              known_vendors=known_vendors)
            
            # Reverse map to get original IDs
            user_id = [k for k, v in matrix_builder.user_mapping.items() if v == user_idx][0]
            
            print(f"\nUser {user_id} (index {user_idx}):")
            print(f"  Known vendors: {len(known_vendors)}")
            print("  Top 5 recommendations:")
            
            for vendor_idx, score in recs:
                vendor_id = [k for k, v in matrix_builder.vendor_mapping.items() 
                           if v == vendor_idx][0]
                print(f"    Vendor {vendor_id}: score={score:.4f}")
            
            recommendations.append({
                'user_id': str(user_id),
                'user_idx': int(user_idx),
                'known_vendors': len(known_vendors),
                'recommendations': [(str([k for k, v in matrix_builder.vendor_mapping.items() 
                                         if v == v_idx][0]), float(score)) 
                                   for v_idx, score in recs]
            })
        
        # Save recommendations
        recs_file = self.output_dir / f'sample_recommendations_{self.timestamp}.json'
        with open(recs_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nSample recommendations saved to: {recs_file}")

def main():
    """Main execution function"""
    print("="*80)
    print("COLLABORATIVE FILTERING WITH ALS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    loader = DataLoader()
    df = loader.load_user_vendor_data()
    
    # Build interaction matrices
    matrix_builder = InteractionMatrixBuilder(df)
    
    # Build GMV matrix (primary target)
    gmv_matrix = matrix_builder.build_matrix('gmv')
    matrix_builder.analyze_matrix_properties(gmv_matrix, "GMV")
    
    # Build clicks matrix if available
    if 'clicks' in df.columns:
        clicks_matrix = matrix_builder.build_matrix('clicks')
        matrix_builder.analyze_matrix_properties(clicks_matrix, "Clicks")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Fit ALS model
    als_model = ALSModel(n_factors=50, regularization=0.01, iterations=20)
    als_model.fit(gmv_matrix)
    evaluator.evaluate_reconstruction(als_model, gmv_matrix, "ALS (k=50)")
    
    # Try different number of factors
    for n_factors in [20, 100]:
        als_k = ALSModel(n_factors=n_factors, regularization=0.01, iterations=20)
        als_k.fit(gmv_matrix)
        evaluator.evaluate_reconstruction(als_k, gmv_matrix, f"ALS (k={n_factors})")
    
    # Fit SVD for comparison
    svd_model = SVDModel(n_components=50)
    svd_model.fit(gmv_matrix)
    evaluator.evaluate_reconstruction(svd_model, gmv_matrix, "SVD (k=50)")
    
    # Fit NMF for comparison
    nmf_model = NMFModel(n_components=50)
    nmf_model.fit(gmv_matrix)
    evaluator.evaluate_reconstruction(nmf_model, gmv_matrix, "NMF (k=50)")
    
    # Compare all models
    evaluator.compare_models()
    
    # Save results
    results_manager = ResultsManager()
    results_manager.save_results(evaluator, als_model, matrix_builder)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return evaluator, als_model, matrix_builder

if __name__ == "__main__":
    evaluator, als_model, matrix_builder = main()