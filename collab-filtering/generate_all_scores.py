#!/usr/bin/env python3
"""
Generate scores for all user-vendor pairs using the best model (SVD)
Extract and analyze user and vendor features
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class ScoreGenerator:
    """Generate and analyze user-vendor scores from collaborative filtering"""
    
    def __init__(self, data_path: str = '../panel/archive'):
        self.data_path = Path(data_path)
        self.user_factors = None
        self.vendor_factors = None
        self.user_mapping = None
        self.vendor_mapping = None
        self.scores_matrix = None
        
    def load_data_and_train_svd(self, n_components: int = 50, sample_users: int = 10000):
        """Load data and train SVD model (best performing)"""
        print("="*80)
        print("LOADING DATA AND TRAINING BEST MODEL (SVD)")
        print("="*80)
        
        # Load data
        file_path = self.data_path / 'user_vendor_panel_pilot.parquet'
        df = pd.read_parquet(file_path)
        
        # Rename columns
        df = df.rename(columns={
            'USER_ID': 'user_id',
            'VENDOR_ID': 'vendor_id',
            'TOTAL_REVENUE_VENDOR_PRODUCT': 'gmv'
        })
        
        # Filter and sample
        df_filtered = df[df['gmv'] > 0].copy()
        
        if sample_users and len(df_filtered['user_id'].unique()) > sample_users:
            print(f"Sampling {sample_users} users...")
            sampled_users = np.random.choice(df_filtered['user_id'].unique(), 
                                           sample_users, replace=False)
            df_filtered = df_filtered[df_filtered['user_id'].isin(sampled_users)]
        
        # Aggregate by user-vendor
        df_agg = df_filtered.groupby(['user_id', 'vendor_id'])['gmv'].sum().reset_index()
        
        # Create mappings
        unique_users = sorted(df_agg['user_id'].unique())
        unique_vendors = sorted(df_agg['vendor_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.vendor_mapping = {vendor: idx for idx, vendor in enumerate(unique_vendors)}
        
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_vendor_mapping = {idx: vendor for vendor, idx in self.vendor_mapping.items()}
        
        print(f"Users: {len(unique_users)}")
        print(f"Vendors: {len(unique_vendors)}")
        print(f"Interactions: {len(df_agg)}")
        
        # Create sparse matrix
        df_agg['user_idx'] = df_agg['user_id'].map(self.user_mapping)
        df_agg['vendor_idx'] = df_agg['vendor_id'].map(self.vendor_mapping)
        
        # Apply log transformation
        values = np.log1p(df_agg['gmv'].values)
        
        matrix = csr_matrix((values, 
                           (df_agg['user_idx'].values, df_agg['vendor_idx'].values)),
                          shape=(len(unique_users), len(unique_vendors)))
        
        # Train SVD
        print(f"\nTraining SVD with {n_components} components...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(matrix)
        self.vendor_factors = svd.components_.T
        
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_var:.2%}")
        
        return matrix
    
    def generate_all_scores(self):
        """Generate scores for all user-vendor pairs"""
        print("\n" + "="*80)
        print("GENERATING SCORES FOR ALL USER-VENDOR PAIRS")
        print("="*80)
        
        # Compute score matrix: U @ V^T
        self.scores_matrix = self.user_factors @ self.vendor_factors.T
        
        n_users, n_vendors = self.scores_matrix.shape
        total_pairs = n_users * n_vendors
        
        print(f"Score matrix shape: {self.scores_matrix.shape}")
        print(f"Total user-vendor pairs: {total_pairs:,}")
        print(f"Memory usage: {self.scores_matrix.nbytes / 1e6:.1f} MB")
        
        # Statistics
        print("\nScore Statistics:")
        print(f"  Mean: {self.scores_matrix.mean():.4f}")
        print(f"  Std: {self.scores_matrix.std():.4f}")
        print(f"  Min: {self.scores_matrix.min():.4f}")
        print(f"  Max: {self.scores_matrix.max():.4f}")
        print(f"  Median: {np.median(self.scores_matrix):.4f}")
        
        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        print("\nScore Percentiles:")
        for p in percentiles:
            val = np.percentile(self.scores_matrix, p)
            print(f"  {p:3d}%: {val:.4f}")
        
        return self.scores_matrix
    
    def analyze_user_features(self):
        """Analyze learned user features"""
        print("\n" + "="*80)
        print("USER FEATURE ANALYSIS")
        print("="*80)
        
        print(f"User factors shape: {self.user_factors.shape}")
        
        # Feature statistics
        print("\nFeature Statistics:")
        print(f"  Mean: {self.user_factors.mean():.4f}")
        print(f"  Std: {self.user_factors.std():.4f}")
        print(f"  Sparsity: {(np.abs(self.user_factors) < 0.01).mean():.2%}")
        
        # Top features by variance
        feature_var = np.var(self.user_factors, axis=0)
        top_features = np.argsort(feature_var)[::-1][:5]
        
        print("\nTop 5 Features by Variance:")
        for i, feat_idx in enumerate(top_features, 1):
            print(f"  Feature {feat_idx+1}: variance={feature_var[feat_idx]:.4f}")
        
        # User clustering
        print("\nUser Segmentation (by dominant feature):")
        dominant_feature = np.argmax(np.abs(self.user_factors), axis=1)
        feature_counts = pd.Series(dominant_feature).value_counts().head(10)
        for feat, count in feature_counts.items():
            print(f"  Feature {feat+1}: {count} users ({count/len(self.user_factors)*100:.1f}%)")
        
        # Find extreme users
        print("\nExtreme Users:")
        for i in range(min(3, self.user_factors.shape[1])):
            max_user_idx = np.argmax(self.user_factors[:, i])
            min_user_idx = np.argmin(self.user_factors[:, i])
            print(f"  Feature {i+1}:")
            print(f"    Max: User {self.reverse_user_mapping[max_user_idx][:50]}... (value={self.user_factors[max_user_idx, i]:.4f})")
            print(f"    Min: User {self.reverse_user_mapping[min_user_idx][:50]}... (value={self.user_factors[min_user_idx, i]:.4f})")
        
        return self.user_factors
    
    def analyze_vendor_features(self):
        """Analyze learned vendor features"""
        print("\n" + "="*80)
        print("VENDOR FEATURE ANALYSIS")
        print("="*80)
        
        print(f"Vendor factors shape: {self.vendor_factors.shape}")
        
        # Feature statistics
        print("\nFeature Statistics:")
        print(f"  Mean: {self.vendor_factors.mean():.4f}")
        print(f"  Std: {self.vendor_factors.std():.4f}")
        print(f"  Sparsity: {(np.abs(self.vendor_factors) < 0.01).mean():.2%}")
        
        # Top features by variance
        feature_var = np.var(self.vendor_factors, axis=0)
        top_features = np.argsort(feature_var)[::-1][:5]
        
        print("\nTop 5 Features by Variance:")
        for i, feat_idx in enumerate(top_features, 1):
            print(f"  Feature {feat_idx+1}: variance={feature_var[feat_idx]:.4f}")
        
        # Vendor clustering
        print("\nVendor Segmentation (by dominant feature):")
        dominant_feature = np.argmax(np.abs(self.vendor_factors), axis=1)
        feature_counts = pd.Series(dominant_feature).value_counts().head(10)
        for feat, count in feature_counts.items():
            print(f"  Feature {feat+1}: {count} vendors ({count/len(self.vendor_factors)*100:.1f}%)")
        
        # Find extreme vendors
        print("\nExtreme Vendors:")
        for i in range(min(3, self.vendor_factors.shape[1])):
            max_vendor_idx = np.argmax(self.vendor_factors[:, i])
            min_vendor_idx = np.argmin(self.vendor_factors[:, i])
            print(f"  Feature {i+1}:")
            print(f"    Max: Vendor {self.reverse_vendor_mapping[max_vendor_idx]} (value={self.vendor_factors[max_vendor_idx, i]:.4f})")
            print(f"    Min: Vendor {self.reverse_vendor_mapping[min_vendor_idx]} (value={self.vendor_factors[min_vendor_idx, i]:.4f})")
        
        return self.vendor_factors
    
    def get_top_pairs(self, n: int = 100):
        """Get top N user-vendor pairs by score"""
        print(f"\n" + "="*80)
        print(f"TOP {n} USER-VENDOR PAIRS BY SCORE")
        print("="*80)
        
        # Flatten and get top indices
        flat_scores = self.scores_matrix.flatten()
        top_indices = np.argpartition(flat_scores, -n)[-n:]
        top_indices = top_indices[np.argsort(flat_scores[top_indices])][::-1]
        
        # Convert to user-vendor pairs
        top_pairs = []
        for idx in top_indices:
            user_idx = idx // self.scores_matrix.shape[1]
            vendor_idx = idx % self.scores_matrix.shape[1]
            score = self.scores_matrix[user_idx, vendor_idx]
            
            top_pairs.append({
                'user_id': self.reverse_user_mapping[user_idx],
                'vendor_id': self.reverse_vendor_mapping[vendor_idx],
                'user_idx': user_idx,
                'vendor_idx': vendor_idx,
                'score': score
            })
        
        # Display top 10
        print("\nTop 10 pairs:")
        for i, pair in enumerate(top_pairs[:10], 1):
            print(f"{i:3d}. User {pair['user_id'][:30]}... → Vendor {pair['vendor_id'][:30]}...")
            print(f"     Score: {pair['score']:.4f}")
        
        return top_pairs
    
    def save_results(self, output_dir: str = 'results'):
        """Save all results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save factors
        np.savez(output_dir / f'svd_factors_{timestamp}.npz',
                user_factors=self.user_factors,
                vendor_factors=self.vendor_factors)
        print(f"\nFactors saved to: svd_factors_{timestamp}.npz")
        
        # Save mappings
        with open(output_dir / f'svd_mappings_{timestamp}.json', 'w') as f:
            json.dump({
                'user_mapping': {str(k): v for k, v in self.user_mapping.items()},
                'vendor_mapping': {str(k): v for k, v in self.vendor_mapping.items()}
            }, f)
        print(f"Mappings saved to: svd_mappings_{timestamp}.json")
        
        # Save score statistics
        score_stats = {
            'mean': float(self.scores_matrix.mean()),
            'std': float(self.scores_matrix.std()),
            'min': float(self.scores_matrix.min()),
            'max': float(self.scores_matrix.max()),
            'shape': list(self.scores_matrix.shape),
            'percentiles': {
                p: float(np.percentile(self.scores_matrix, p))
                for p in [1, 5, 25, 50, 75, 95, 99]
            }
        }
        
        with open(output_dir / f'score_statistics_{timestamp}.json', 'w') as f:
            json.dump(score_stats, f, indent=2)
        print(f"Score statistics saved to: score_statistics_{timestamp}.json")
        
        # Save top pairs
        top_pairs = self.get_top_pairs(1000)
        df_top = pd.DataFrame(top_pairs)
        df_top.to_csv(output_dir / f'top_1000_pairs_{timestamp}.csv', index=False)
        print(f"Top 1000 pairs saved to: top_1000_pairs_{timestamp}.csv")
        
    def export_score_samples(self, n_samples: int = 10000):
        """Export a sample of scores for analysis"""
        print(f"\n" + "="*80)
        print(f"EXPORTING SCORE SAMPLES")
        print("="*80)
        
        # Random sample of user-vendor pairs
        n_users, n_vendors = self.scores_matrix.shape
        
        sample_users = np.random.choice(n_users, min(n_samples, n_users), replace=False)
        sample_vendors = np.random.choice(n_vendors, min(100, n_vendors), replace=False)
        
        samples = []
        for user_idx in sample_users[:100]:  # Limit to 100 users for 10K pairs
            for vendor_idx in sample_vendors:
                samples.append({
                    'user_id': self.reverse_user_mapping[user_idx],
                    'vendor_id': self.reverse_vendor_mapping[vendor_idx],
                    'score': float(self.scores_matrix[user_idx, vendor_idx])
                })
        
        df_samples = pd.DataFrame(samples)
        
        print(f"Exported {len(df_samples)} score samples")
        print(f"Score distribution in sample:")
        print(df_samples['score'].describe())
        
        return df_samples

def main():
    """Main execution"""
    print("="*80)
    print("USER-VENDOR SCORE GENERATION AND FEATURE ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize generator
    generator = ScoreGenerator()
    
    # Load data and train SVD (best model)
    matrix = generator.load_data_and_train_svd(n_components=50, sample_users=10000)
    
    # Generate all scores
    scores = generator.generate_all_scores()
    
    # Analyze features
    user_features = generator.analyze_user_features()
    vendor_features = generator.analyze_vendor_features()
    
    # Get top pairs
    top_pairs = generator.get_top_pairs(100)
    
    # Export samples
    samples = generator.export_score_samples(10000)
    
    # Save everything
    generator.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return generator

if __name__ == "__main__":
    generator = main()