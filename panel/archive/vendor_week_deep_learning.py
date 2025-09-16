#!/usr/bin/env python3
"""
Deep Learning Replication of Vendor-Week Panel Fixed Effects Model

This script demonstrates how to replicate traditional econometric fixed effects
regression using deep learning with PyTorch. It learns the same parameters as
the R fixest model: log(revenue+1) ~ log(clicks+1) | vendor_id + week

Expected results:
- Main elasticity coefficient: ~0.6422
- Vendor and week fixed effects learned as embeddings
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FixedEffectsNet(nn.Module):
    """
    Neural network that replicates fixed effects regression.
    Uses embeddings to learn vendor and week fixed effects.
    """

    def __init__(self, n_vendors, n_weeks, embedding_dim=10):
        super(FixedEffectsNet, self).__init__()

        # Embeddings act as learned fixed effects
        self.vendor_embedding = nn.Embedding(n_vendors, embedding_dim)
        self.week_embedding = nn.Embedding(n_weeks, embedding_dim)

        # Main regression coefficient for log(clicks)
        self.click_coefficient = nn.Linear(1, 1, bias=False)

        # Combine embeddings and click effect
        # Input: embedding_dim * 2 (vendor + week) + 1 (click effect)
        self.combiner = nn.Linear(embedding_dim * 2 + 1, 1)

        # Initialize weights
        nn.init.normal_(self.vendor_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.week_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.click_coefficient.weight, mean=0.5, std=0.1)
        nn.init.zeros_(self.combiner.bias)

    def forward(self, vendor_ids, week_ids, log_clicks):
        # Get embeddings (fixed effects)
        vendor_fe = self.vendor_embedding(vendor_ids)
        week_fe = self.week_embedding(week_ids)

        # Apply main click effect
        click_effect = self.click_coefficient(log_clicks.unsqueeze(-1))

        # Concatenate all effects
        combined = torch.cat([vendor_fe, week_fe, click_effect], dim=1)

        # Final prediction
        output = self.combiner(combined)
        return output.squeeze()


class SimplifiedFixedEffectsNet(nn.Module):
    """
    Simplified version that more directly mirrors the econometric model:
    y = β*X + vendor_FE + week_FE + ε
    """

    def __init__(self, n_vendors, n_weeks):
        super(SimplifiedFixedEffectsNet, self).__init__()

        # Each vendor and week gets a single fixed effect (scalar)
        self.vendor_fe = nn.Embedding(n_vendors, 1)
        self.week_fe = nn.Embedding(n_weeks, 1)

        # Main regression coefficient β for log(clicks)
        self.beta = nn.Parameter(torch.tensor([0.5]))

        # Initialize fixed effects near zero
        nn.init.normal_(self.vendor_fe.weight, mean=0, std=0.01)
        nn.init.normal_(self.week_fe.weight, mean=0, std=0.01)

    def forward(self, vendor_ids, week_ids, log_clicks):
        # Get fixed effects
        vendor_effect = self.vendor_fe(vendor_ids).squeeze()
        week_effect = self.week_fe(week_ids).squeeze()

        # Linear model: y = β*X + vendor_FE + week_FE
        prediction = self.beta * log_clicks + vendor_effect + week_effect
        return prediction


def load_and_prepare_data(filepath='vendor_panel_full_history_clicks_only.parquet'):
    """Load and prepare the vendor-week panel data."""

    print("Loading vendor-week panel data...")
    df = pd.read_parquet(filepath)

    # Convert types
    df['revenue_dollars'] = df['revenue_dollars'].astype(float)
    df['clicks'] = df['clicks'].astype(int)
    df['week'] = pd.to_datetime(df['week'])

    # Create log-transformed variables (matching the R model)
    df['log_revenue_plus_1'] = np.log1p(df['revenue_dollars'])
    df['log_clicks_plus_1'] = np.log1p(df['clicks'])

    # IMPORTANT: Use ALL data including zero clicks to match R feols exactly
    # The R model uses all observations, not just clicks > 0
    df_clean = df.copy()

    # Encode vendor_id and week as integers for embeddings
    vendor_encoder = LabelEncoder()
    week_encoder = LabelEncoder()

    df_clean['vendor_id_encoded'] = vendor_encoder.fit_transform(df_clean['vendor_id'])
    df_clean['week_encoded'] = week_encoder.fit_transform(df_clean['week'])

    print(f"Data shape: {df_clean.shape}")
    print(f"Unique vendors: {df_clean['vendor_id'].nunique()}")
    print(f"Unique weeks: {df_clean['week'].nunique()}")
    print(f"Date range: {df_clean['week'].min()} to {df_clean['week'].max()}")
    print(f"Including ALL observations (with zero clicks) to match R feols")

    return df_clean, vendor_encoder, week_encoder


def create_dataloaders(df, batch_size=1024, train_split=0.8):
    """Create PyTorch dataloaders for training and validation."""

    # Prepare tensors
    vendor_ids = torch.LongTensor(df['vendor_id_encoded'].values)
    week_ids = torch.LongTensor(df['week_encoded'].values)
    log_clicks = torch.FloatTensor(df['log_clicks_plus_1'].values)
    log_revenue = torch.FloatTensor(df['log_revenue_plus_1'].values)

    # Create dataset
    dataset = TensorDataset(vendor_ids, week_ids, log_clicks, log_revenue)

    # Split into train and validation
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, n_epochs=100, lr=0.01):
    """Train the fixed effects neural network with detailed progress tracking."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Tracking arrays
    train_losses = []
    val_losses = []
    beta_history = []
    lr_history = []
    vendor_fe_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    week_fe_stats = {'mean': [], 'std': [], 'min': [], 'max': []}

    print(f"\nTraining on {device}...")
    print("="*50)
    print(f"Expected β (from R fixest): 0.6422")
    print("="*50)

    # Main training loop with epoch progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Training Progress", position=0)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0

        # Batch loop with progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}",
                          leave=False, position=1)

        for batch in batch_pbar:
            vendor_ids, week_ids, log_clicks, log_revenue = [b.to(device) for b in batch]

            # Forward pass
            predictions = model(vendor_ids, week_ids, log_clicks)
            loss = criterion(predictions, log_revenue)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                vendor_ids, week_ids, log_clicks, log_revenue = [b.to(device) for b in batch]
                predictions = model(vendor_ids, week_ids, log_clicks)
                loss = criterion(predictions, log_revenue)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lr_history.append(current_lr)

        # Track parameters
        if isinstance(model, SimplifiedFixedEffectsNet):
            beta = model.beta.item()
            beta_history.append(beta)

            # Track fixed effects statistics
            vendor_fe = model.vendor_fe.weight.detach().cpu().numpy().flatten()
            week_fe = model.week_fe.weight.detach().cpu().numpy().flatten()

            vendor_fe_stats['mean'].append(np.mean(vendor_fe))
            vendor_fe_stats['std'].append(np.std(vendor_fe))
            vendor_fe_stats['min'].append(np.min(vendor_fe))
            vendor_fe_stats['max'].append(np.max(vendor_fe))

            week_fe_stats['mean'].append(np.mean(week_fe))
            week_fe_stats['std'].append(np.std(week_fe))
            week_fe_stats['min'].append(np.min(week_fe))
            week_fe_stats['max'].append(np.max(week_fe))

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'β': f'{beta:.4f}',
                'Δβ': f'{abs(beta - 0.6422):.4f}',
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'lr': f'{current_lr:.5f}'
            })

        # Update learning rate
        scheduler.step(avg_val_loss)

    # Final summary
    if isinstance(model, SimplifiedFixedEffectsNet):
        final_beta = model.beta.item()
        print(f"\n" + "="*50)
        print(f"Training Complete!")
        print(f"Final β: {final_beta:.4f} (Target: 0.6422, Error: {abs(final_beta - 0.6422):.4f})")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")
        print("="*50)

    return train_losses, val_losses, beta_history, vendor_fe_stats, week_fe_stats


def extract_and_compare_results(model, df_clean, vendor_encoder, week_encoder):
    """Extract learned parameters and compare with R model results."""

    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)

    if isinstance(model, SimplifiedFixedEffectsNet):
        # Extract main elasticity coefficient
        beta_learned = model.beta.item()
        print(f"\nMain Elasticity Coefficient (β):")
        print(f"  Deep Learning: {beta_learned:.4f}")
        print(f"  R fixest:      0.6422")
        print(f"  Difference:    {abs(beta_learned - 0.6422):.4f}")

        # Extract vendor fixed effects
        vendor_effects = model.vendor_fe.weight.detach().cpu().numpy().flatten()

        print(f"\nVendor Fixed Effects Summary:")
        print(f"  Number of vendors: {len(vendor_effects)}")
        print(f"  Mean:     {np.mean(vendor_effects):.4f}")
        print(f"  Std Dev:  {np.std(vendor_effects):.4f}")
        print(f"  Min:      {np.min(vendor_effects):.4f}")
        print(f"  Max:      {np.max(vendor_effects):.4f}")

        # Extract week fixed effects
        week_effects = model.week_fe.weight.detach().cpu().numpy().flatten()

        print(f"\nWeek Fixed Effects Summary:")
        print(f"  Number of weeks: {len(week_effects)}")
        print(f"  Mean:     {np.mean(week_effects):.4f}")
        print(f"  Std Dev:  {np.std(week_effects):.4f}")
        print(f"  Min:      {np.min(week_effects):.4f}")
        print(f"  Max:      {np.max(week_effects):.4f}")

        # Calculate vendor-specific total effects (β + vendor_FE)
        vendor_total_effects = beta_learned + vendor_effects

        print(f"\nVendor-Specific Total Effects (β + vendor_FE):")
        print(f"  Mean:     {np.mean(vendor_total_effects):.4f}")
        print(f"  Std Dev:  {np.std(vendor_total_effects):.4f}")
        print(f"  Min:      {np.min(vendor_total_effects):.4f}")
        print(f"  Median:   {np.median(vendor_total_effects):.4f}")
        print(f"  Max:      {np.max(vendor_total_effects):.4f}")

        # Compare with R model vendor effects distribution
        print(f"\nR lme4 Model Vendor Effects (for comparison):")
        print(f"  Min:      0.1942")
        print(f"  Median:   0.7788")
        print(f"  Mean:     0.7721")
        print(f"  Max:      1.2652")

        return {
            'beta': beta_learned,
            'vendor_effects': vendor_effects,
            'week_effects': week_effects,
            'vendor_total_effects': vendor_total_effects,
            'vendor_encoder': vendor_encoder,
            'week_encoder': week_encoder
        }

    return None


def save_fixed_effects_to_csv(results, df_clean):
    """Save fixed effects to CSV files for comparison with R model."""

    if results is None:
        return

    print("\n" + "="*50)
    print("SAVING FIXED EFFECTS TO CSV")
    print("="*50)

    # Save vendor fixed effects
    vendor_ids = results['vendor_encoder'].classes_
    vendor_fe_df = pd.DataFrame({
        'vendor_id': vendor_ids,
        'fixed_effect': results['vendor_effects']
    })
    vendor_fe_df.to_csv('dl_vendor_fixed_effects.csv', index=False)
    print("✅ Saved dl_vendor_fixed_effects.csv")

    # Save week fixed effects
    week_ids = results['week_encoder'].classes_
    week_fe_df = pd.DataFrame({
        'week': week_ids,
        'fixed_effect': results['week_effects']
    })
    week_fe_df.to_csv('dl_week_fixed_effects.csv', index=False)
    print("✅ Saved dl_week_fixed_effects.csv")

    # Save model diagnostics
    diagnostics_df = pd.DataFrame([{
        'model': 'DeepLearning',
        'beta': results['beta'],
        'n_vendor_fe': len(results['vendor_effects']),
        'n_week_fe': len(results['week_effects']),
        'vendor_fe_mean': np.mean(results['vendor_effects']),
        'vendor_fe_std': np.std(results['vendor_effects']),
        'week_fe_mean': np.mean(results['week_effects']),
        'week_fe_std': np.std(results['week_effects'])
    }])
    diagnostics_df.to_csv('dl_model_diagnostics.csv', index=False)
    print("✅ Saved dl_model_diagnostics.csv")

    return vendor_fe_df, week_fe_df


def calculate_model_diagnostics(model, df_clean, device='cpu'):
    """Calculate R², RMSE, and other diagnostics for the trained model."""

    print("\n" + "="*50)
    print("CALCULATING MODEL DIAGNOSTICS")
    print("="*50)

    model.eval()

    # Prepare data
    vendor_ids = torch.LongTensor(df_clean['vendor_id_encoded'].values).to(device)
    week_ids = torch.LongTensor(df_clean['week_encoded'].values).to(device)
    log_clicks = torch.FloatTensor(df_clean['log_clicks_plus_1'].values).to(device)
    log_revenue = torch.FloatTensor(df_clean['log_revenue_plus_1'].values).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(vendor_ids, week_ids, log_clicks)

    # Calculate metrics
    residuals = log_revenue.cpu().numpy() - predictions.cpu().numpy()
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_revenue.cpu().numpy() - np.mean(log_revenue.cpu().numpy()))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))

    # Calculate log-likelihood (approximate)
    n = len(residuals)
    sigma2 = np.var(residuals)
    log_lik = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2) - 1/(2*sigma2) * ss_res

    print(f"R²: {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Log-likelihood: {log_lik:.2f}")

    # Save residuals and fitted values
    residuals_df = pd.DataFrame({
        'residuals': residuals,
        'fitted': predictions.cpu().numpy()
    })
    residuals_df.to_csv('dl_residuals_fitted.csv', index=False)
    print("✅ Saved dl_residuals_fitted.csv")

    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'log_likelihood': log_lik,
        'residuals': residuals,
        'fitted': predictions.cpu().numpy()
    }


def plot_training_history(train_losses, val_losses, beta_history=None):
    """Plot comprehensive training history including parameter evolution."""

    if beta_history:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.7)
        axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Beta evolution
        axes[0, 1].plot(beta_history, label='β (learned)', color='blue', linewidth=2)
        axes[0, 1].axhline(y=0.6422, color='red', linestyle='--',
                          label='β (R fixest) = 0.6422', linewidth=2)
        axes[0, 1].fill_between(range(len(beta_history)),
                               [0.6422 - 0.01]*len(beta_history),
                               [0.6422 + 0.01]*len(beta_history),
                               alpha=0.2, color='red', label='±0.01 tolerance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('β Coefficient')
        axes[0, 1].set_title('Beta Coefficient Convergence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Beta error over time
        beta_errors = [abs(b - 0.6422) for b in beta_history]
        axes[1, 0].semilogy(beta_errors, color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('|β - 0.6422| (log scale)')
        axes[1, 0].set_title('Beta Convergence Error (Log Scale)')
        axes[1, 0].grid(True, alpha=0.3)

        # Convergence speed
        axes[1, 1].plot(np.diff(beta_history), color='purple', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Δβ (change per epoch)')
        axes[1, 1].set_title('Beta Change Rate')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Deep Learning Fixed Effects Model: Training Dynamics', fontsize=16)
        plt.tight_layout()
        plt.savefig('vendor_week_dl_training.png', dpi=100)
        print(f"\nTraining plot saved to 'vendor_week_dl_training.png'")
        plt.show()
    else:
        # Simple plot if no beta history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History: Fixed Effects Deep Learning Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('vendor_week_dl_training.png', dpi=100)
        print(f"\nTraining plot saved to 'vendor_week_dl_training.png'")
        plt.show()


def plot_fixed_effects_evolution(vendor_fe_stats, week_fe_stats):
    """Plot how fixed effects statistics evolve during training."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vendor fixed effects evolution
    epochs = range(len(vendor_fe_stats['mean']))
    axes[0].plot(epochs, vendor_fe_stats['mean'], label='Mean', color='blue')
    axes[0].fill_between(epochs,
                         [m - s for m, s in zip(vendor_fe_stats['mean'], vendor_fe_stats['std'])],
                         [m + s for m, s in zip(vendor_fe_stats['mean'], vendor_fe_stats['std'])],
                         alpha=0.3, color='blue', label='±1 Std Dev')
    axes[0].plot(epochs, vendor_fe_stats['min'], '--', label='Min', color='red', alpha=0.5)
    axes[0].plot(epochs, vendor_fe_stats['max'], '--', label='Max', color='green', alpha=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Fixed Effect Value')
    axes[0].set_title('Vendor Fixed Effects Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Week fixed effects evolution
    axes[1].plot(epochs, week_fe_stats['mean'], label='Mean', color='blue')
    axes[1].fill_between(epochs,
                         [m - s for m, s in zip(week_fe_stats['mean'], week_fe_stats['std'])],
                         [m + s for m, s in zip(week_fe_stats['mean'], week_fe_stats['std'])],
                         alpha=0.3, color='blue', label='±1 Std Dev')
    axes[1].plot(epochs, week_fe_stats['min'], '--', label='Min', color='red', alpha=0.5)
    axes[1].plot(epochs, week_fe_stats['max'], '--', label='Max', color='green', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Fixed Effect Value')
    axes[1].set_title('Week Fixed Effects Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vendor_week_dl_fe_evolution.png', dpi=100)
    print(f"Fixed effects evolution saved to 'vendor_week_dl_fe_evolution.png'")
    plt.show()

def plot_effects_distribution(results):
    """Plot the distribution of learned effects."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Vendor fixed effects
    axes[0].hist(results['vendor_effects'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Vendor Fixed Effects Distribution')
    axes[0].set_xlabel('Fixed Effect Value')
    axes[0].set_ylabel('Count')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Week fixed effects
    axes[1].hist(results['week_effects'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_title('Week Fixed Effects Distribution')
    axes[1].set_xlabel('Fixed Effect Value')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Total vendor effects (β + vendor_FE)
    axes[2].hist(results['vendor_total_effects'], bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(results['beta'], color='red', linestyle='--',
                    label=f'β = {results["beta"]:.4f}')
    axes[2].set_title('Vendor Total Effects (β + Vendor FE)')
    axes[2].set_xlabel('Total Effect Value')
    axes[2].set_ylabel('Count')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('vendor_week_dl_effects.png', dpi=100)
    print(f"Effects distribution saved to 'vendor_week_dl_effects.png'")
    plt.show()


def main():
    """Main execution function."""

    print("="*50)
    print("DEEP LEARNING REPLICATION OF FIXED EFFECTS MODEL")
    print("="*50)

    # Load and prepare data
    df_clean, vendor_encoder, week_encoder = load_and_prepare_data()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(df_clean, batch_size=2048)

    # Initialize model
    n_vendors = df_clean['vendor_id_encoded'].nunique()
    n_weeks = df_clean['week_encoded'].nunique()

    # Use the simplified model that directly mirrors econometric specification
    model = SimplifiedFixedEffectsNet(n_vendors, n_weeks)

    print(f"\nModel Architecture:")
    print(f"  Vendor embeddings: {n_vendors} x 1")
    print(f"  Week embeddings: {n_weeks} x 1")
    print(f"  Main coefficient β: 1 learnable parameter")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    train_losses, val_losses, beta_history, vendor_fe_stats, week_fe_stats = train_model(
        model, train_loader, val_loader,
        n_epochs=50, lr=0.01
    )

    # Extract and compare results
    results = extract_and_compare_results(model, df_clean, vendor_encoder, week_encoder)

    # Calculate diagnostics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diagnostics = calculate_model_diagnostics(model, df_clean, device)

    # Save fixed effects to CSV
    if results:
        save_fixed_effects_to_csv(results, df_clean)

    # Plot results
    plot_training_history(train_losses, val_losses, beta_history)
    if results:
        plot_effects_distribution(results)
        plot_fixed_effects_evolution(vendor_fe_stats, week_fe_stats)

    # Save model
    torch.save(model.state_dict(), 'vendor_week_dl_model.pth')
    print(f"\nModel saved to 'vendor_week_dl_model.pth'")

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

    return model, results


if __name__ == "__main__":
    model, results = main()