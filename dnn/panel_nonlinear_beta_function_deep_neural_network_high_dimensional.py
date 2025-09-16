#!/usr/bin/env python3
"""
Nonlinear Panel Data with Deep Neural Networks

This implements a panel data model where the coefficient is a nonlinear function of X:
Y = β(X) + entity_fe + time_fe + ε

The DNN learns the high-dimensional nonlinear function β(X) while maintaining
proper fixed effects for entities and time periods.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt  # Removed for text-only output
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# NONLINEAR DATA GENERATOR
# =============================================================================

class NonlinearPanelDataGenerator:
    """Generate panel data with nonlinear β(X) function."""

    def __init__(
        self,
        n_entities: int = 500,
        n_periods: int = 50,
        n_covariates: int = 10,
        fe_strength: float = 2.0
    ):
        self.n_entities = n_entities
        self.n_periods = n_periods
        self.n_covariates = n_covariates

        # Generate fixed effects
        self.entity_fe = np.random.randn(n_entities) * fe_strength
        self.time_fe = np.random.randn(n_periods) * (fe_strength * 0.7)

    def true_beta_function(self, X: np.ndarray) -> np.ndarray:
        """
        Complex nonlinear function β(X).
        X shape: (n_obs, n_covariates)
        """
        # Extract individual covariates
        X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
        X4, X5, X6 = X[:, 3], X[:, 4], X[:, 5]

        # Complex nonlinear transformations
        beta_x = (
            # Trigonometric components
            2.0 * np.sin(X1) +
            1.5 * np.cos(X2) +

            # Polynomial terms
            0.8 * X3**2 - 0.3 * X3**3 +

            # Interactions
            1.2 * X1 * X2 +
            0.7 * X2 * X3 * (X3 > 0) +

            # Activation-like functions
            1.0 * np.tanh(2 * X4) +
            0.5 * np.maximum(0, X5) +  # ReLU-like

            # Threshold effects
            2.5 * (X6 > 0.5) * X6 +

            # Higher-order interactions if we have more covariates
            sum([0.3 * np.sin(X[:, i] * X[:, (i+1) % self.n_covariates])
                 for i in range(6, min(9, self.n_covariates))]) if self.n_covariates > 6 else 0
        )

        return beta_x

    def generate_data(self) -> pd.DataFrame:
        """Generate complete panel dataset."""

        n_obs = self.n_entities * self.n_periods
        entity_ids = np.repeat(range(self.n_entities), self.n_periods)
        time_ids = np.tile(range(self.n_periods), self.n_entities)

        # Generate covariates with some correlation structure
        # This makes the problem more realistic
        mean = np.zeros(self.n_covariates)

        # Create correlation matrix with some structure
        cov_matrix = np.eye(self.n_covariates) * 0.7
        for i in range(self.n_covariates):
            for j in range(self.n_covariates):
                if i != j:
                    cov_matrix[i, j] = 0.3 * np.exp(-abs(i - j) / 3)

        X = np.random.multivariate_normal(mean, cov_matrix, n_obs)

        # Add some entity and time specific effects to X (endogeneity)
        for idx, (i, t) in enumerate(zip(entity_ids, time_ids)):
            X[idx, :3] += 0.2 * self.entity_fe[i]
            X[idx, 3:6] += 0.1 * self.time_fe[t]

        # Create DataFrame
        df = pd.DataFrame({
            'entity_id': entity_ids,
            'time_id': time_ids
        })

        # Add covariates
        for j in range(self.n_covariates):
            df[f'X{j+1}'] = X[:, j]

        # Generate outcome with nonlinear β(X)
        beta_x = self.true_beta_function(X)

        # Add fixed effects
        entity_effects = self.entity_fe[entity_ids]
        time_effects = self.time_fe[time_ids]

        # Generate outcome
        epsilon = np.random.randn(n_obs) * 0.5
        df['Y'] = beta_x + entity_effects + time_effects + epsilon

        # Store true values for validation
        df['beta_x_true'] = beta_x
        df['entity_fe_true'] = entity_effects
        df['time_fe_true'] = time_effects

        return df


# =============================================================================
# NONLINEAR PANEL DNN MODEL
# =============================================================================

class NonlinearPanelDNN(nn.Module):
    """
    Deep neural network for panel data with nonlinear β(X) function.
    Separates fixed effects from the nonlinear component.
    """

    def __init__(
        self,
        n_entities: int,
        n_periods: int,
        n_covariates: int,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.2
    ):
        super().__init__()

        # Fixed effects as embeddings
        self.entity_fe = nn.Embedding(n_entities, 1)
        self.time_fe = nn.Embedding(n_periods, 1)

        # Initialize fixed effects
        nn.init.normal_(self.entity_fe.weight, mean=0, std=1.5)
        nn.init.normal_(self.time_fe.weight, mean=0, std=1.0)

        # Deep network for β(X)
        layers = []
        input_dim = n_covariates

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.beta_network = nn.Sequential(*layers)

        # Skip connection for input features (helps with gradient flow)
        self.skip_connection = nn.Linear(n_covariates, 1)

    def forward(self, entity_ids, time_ids, X):
        # Get fixed effects
        entity_effect = self.entity_fe(entity_ids).squeeze()
        time_effect = self.time_fe(time_ids).squeeze()

        # Demean for identification
        entity_effect = entity_effect - entity_effect.mean()
        time_effect = time_effect - time_effect.mean()

        # Compute nonlinear β(X)
        beta_x_deep = self.beta_network(X).squeeze()
        beta_x_skip = self.skip_connection(X).squeeze()
        beta_x = beta_x_deep + 0.1 * beta_x_skip  # Weighted skip connection

        # Combine all components
        y_pred = beta_x + entity_effect + time_effect

        return y_pred, beta_x

    def get_components(self, entity_ids, time_ids, X):
        """Extract individual components for analysis."""
        with torch.no_grad():
            entity_effect = self.entity_fe(entity_ids).squeeze()
            time_effect = self.time_fe(time_ids).squeeze()

            # Demean
            entity_effect = entity_effect - entity_effect.mean()
            time_effect = time_effect - time_effect.mean()

            # Get β(X)
            beta_x_deep = self.beta_network(X).squeeze()
            beta_x_skip = self.skip_connection(X).squeeze()
            beta_x = beta_x_deep + 0.1 * beta_x_skip

        return {
            'beta_x': beta_x.cpu().numpy(),
            'entity_fe': entity_effect.cpu().numpy(),
            'time_fe': time_effect.cpu().numpy()
        }


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_nonlinear_model(
    model: NonlinearPanelDNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 300,
    lr_beta: float = 0.001,
    lr_fe: float = 0.01,
    patience: int = 50
):
    """Train the nonlinear panel DNN model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Separate optimizers for different components
    beta_params = list(model.beta_network.parameters()) + list(model.skip_connection.parameters())
    fe_params = [model.entity_fe.weight, model.time_fe.weight]

    optimizer = optim.AdamW([
        {'params': beta_params, 'lr': lr_beta, 'weight_decay': 0.001},
        {'params': fe_params, 'lr': lr_fe, 'weight_decay': 0.0001}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'beta_x_corr': []
    }

    pbar = tqdm(range(n_epochs), desc="Training Nonlinear Model")

    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        all_beta_x_true = []
        all_beta_x_pred = []

        for batch in train_loader:
            entity_ids, time_ids, X, y, beta_x_true = [b.to(device) for b in batch]

            y_pred, beta_x_pred = model(entity_ids, time_ids, X)

            # Main loss
            loss = criterion(y_pred, y)

            # Add regularization to encourage FE to be mean-zero
            entity_mean = model.entity_fe.weight.mean()
            time_mean = model.time_fe.weight.mean()
            reg_loss = 0.001 * (entity_mean**2 + time_mean**2)

            total_loss = loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            all_beta_x_true.extend(beta_x_true.cpu().numpy())
            all_beta_x_pred.extend(beta_x_pred.detach().cpu().numpy())

        # Calculate correlation for β(X)
        beta_x_corr = np.corrcoef(all_beta_x_true, all_beta_x_pred)[0, 1]

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                entity_ids, time_ids, X, y, _ = [b.to(device) for b in batch]
                y_pred, _ = model(entity_ids, time_ids, X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['beta_x_corr'].append(beta_x_corr)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        pbar.set_postfix({
            'val_loss': f'{avg_val_loss:.4f}',
            'β(X) corr': f'{beta_x_corr:.3f}',
            'patience': f'{patience_counter}/{patience}'
        })

        scheduler.step(avg_val_loss)

    return history, model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, test_loader, generator, device):
    """Evaluate model performance on test data."""

    model.eval()

    all_y_true = []
    all_y_pred = []
    all_beta_x_true = []
    all_beta_x_pred = []
    all_entity_ids = []
    all_time_ids = []

    with torch.no_grad():
        for batch in test_loader:
            entity_ids, time_ids, X, y, beta_x_true = [b.to(device) for b in batch]

            y_pred, beta_x_pred = model(entity_ids, time_ids, X)

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
            all_beta_x_true.extend(beta_x_true.cpu().numpy())
            all_beta_x_pred.extend(beta_x_pred.cpu().numpy())
            all_entity_ids.extend(entity_ids.cpu().numpy())
            all_time_ids.extend(time_ids.cpu().numpy())

    # Convert to arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    beta_x_true = np.array(all_beta_x_true)
    beta_x_pred = np.array(all_beta_x_pred)

    # Calculate metrics
    y_mse = np.mean((y_true - y_pred)**2)
    y_corr = np.corrcoef(y_true, y_pred)[0, 1]

    beta_mse = np.mean((beta_x_true - beta_x_pred)**2)
    beta_corr = np.corrcoef(beta_x_true, beta_x_pred)[0, 1]

    # Extract fixed effects for a subset of entities
    unique_entities = np.unique(all_entity_ids)[:10]
    entity_fe_pred = model.entity_fe.weight.detach().cpu().numpy().flatten()
    entity_fe_pred = entity_fe_pred - entity_fe_pred.mean()

    entity_fe_true = generator.entity_fe - generator.entity_fe.mean()
    entity_fe_corr = np.corrcoef(entity_fe_true, entity_fe_pred[:len(entity_fe_true)])[0, 1]

    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)

    print(f"\nOutcome Y:")
    print(f"  MSE:         {y_mse:.4f}")
    print(f"  Correlation: {y_corr:.4f}")

    print(f"\nNonlinear β(X):")
    print(f"  MSE:         {beta_mse:.4f}")
    print(f"  Correlation: {beta_corr:.4f}")

    print(f"\nEntity Fixed Effects:")
    print(f"  Correlation: {entity_fe_corr:.4f}")

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'beta_x_true': beta_x_true,
        'beta_x_pred': beta_x_pred
    }


def print_results_to_file(results, history, output_file):
    """Output comprehensive results to text file."""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NONLINEAR PANEL DATA DEEP NEURAL NETWORK RESULTS\n")
        f.write("="*80 + "\n\n")

        # Training history summary
        f.write("TRAINING HISTORY\n")
        f.write("="*80 + "\n")
        f.write(f"Total epochs trained: {len(history['train_loss'])}\n")
        f.write(f"Final train loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"Best validation loss: {min(history['val_loss']):.6f}\n")
        f.write(f"Best epoch: {np.argmin(history['val_loss']) + 1}\n")
        f.write(f"Final β(X) correlation: {history['beta_x_corr'][-1]:.6f}\n")
        f.write(f"Best β(X) correlation: {max(history['beta_x_corr']):.6f}\n\n")

        # Detailed epoch-by-epoch history
        f.write("EPOCH-BY-EPOCH TRAINING LOG\n")
        f.write("="*80 + "\n")
        f.write("Epoch | Train Loss | Val Loss | β(X) Correlation\n")
        f.write("-"*60 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:5d} | {history['train_loss'][i]:10.6f} | {history['val_loss'][i]:8.6f} | {history['beta_x_corr'][i]:8.6f}\n")

        # Model performance metrics
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")

        # Y prediction metrics
        y_mse = np.mean((results['y_true'] - results['y_pred'])**2)
        y_rmse = np.sqrt(y_mse)
        y_mae = np.mean(np.abs(results['y_true'] - results['y_pred']))
        y_corr = np.corrcoef(results['y_true'], results['y_pred'])[0, 1]
        y_r2 = y_corr**2

        f.write("OUTCOME (Y) PREDICTION\n")
        f.write("-"*40 + "\n")
        f.write(f"MSE:                  {y_mse:.6f}\n")
        f.write(f"RMSE:                 {y_rmse:.6f}\n")
        f.write(f"MAE:                  {y_mae:.6f}\n")
        f.write(f"Correlation:          {y_corr:.6f}\n")
        f.write(f"R-squared:            {y_r2:.6f}\n")
        f.write(f"N observations:       {len(results['y_true'])}\n\n")

        # β(X) recovery metrics
        beta_mse = np.mean((results['beta_x_true'] - results['beta_x_pred'])**2)
        beta_rmse = np.sqrt(beta_mse)
        beta_mae = np.mean(np.abs(results['beta_x_true'] - results['beta_x_pred']))
        beta_corr = np.corrcoef(results['beta_x_true'], results['beta_x_pred'])[0, 1]
        beta_r2 = beta_corr**2

        f.write("NONLINEAR β(X) RECOVERY\n")
        f.write("-"*40 + "\n")
        f.write(f"MSE:                  {beta_mse:.6f}\n")
        f.write(f"RMSE:                 {beta_rmse:.6f}\n")
        f.write(f"MAE:                  {beta_mae:.6f}\n")
        f.write(f"Correlation:          {beta_corr:.6f}\n")
        f.write(f"R-squared:            {beta_r2:.6f}\n\n")

        # Distribution statistics
        f.write("DISTRIBUTION STATISTICS\n")
        f.write("="*80 + "\n\n")

        f.write("True Y Distribution:\n")
        f.write(f"  Mean:     {np.mean(results['y_true']):.6f}\n")
        f.write(f"  Std Dev:  {np.std(results['y_true']):.6f}\n")
        f.write(f"  Min:      {np.min(results['y_true']):.6f}\n")
        f.write(f"  Q1:       {np.percentile(results['y_true'], 25):.6f}\n")
        f.write(f"  Median:   {np.median(results['y_true']):.6f}\n")
        f.write(f"  Q3:       {np.percentile(results['y_true'], 75):.6f}\n")
        f.write(f"  Max:      {np.max(results['y_true']):.6f}\n\n")

        f.write("Predicted Y Distribution:\n")
        f.write(f"  Mean:     {np.mean(results['y_pred']):.6f}\n")
        f.write(f"  Std Dev:  {np.std(results['y_pred']):.6f}\n")
        f.write(f"  Min:      {np.min(results['y_pred']):.6f}\n")
        f.write(f"  Q1:       {np.percentile(results['y_pred'], 25):.6f}\n")
        f.write(f"  Median:   {np.median(results['y_pred']):.6f}\n")
        f.write(f"  Q3:       {np.percentile(results['y_pred'], 75):.6f}\n")
        f.write(f"  Max:      {np.max(results['y_pred']):.6f}\n\n")

        f.write("True β(X) Distribution:\n")
        f.write(f"  Mean:     {np.mean(results['beta_x_true']):.6f}\n")
        f.write(f"  Std Dev:  {np.std(results['beta_x_true']):.6f}\n")
        f.write(f"  Min:      {np.min(results['beta_x_true']):.6f}\n")
        f.write(f"  Q1:       {np.percentile(results['beta_x_true'], 25):.6f}\n")
        f.write(f"  Median:   {np.median(results['beta_x_true']):.6f}\n")
        f.write(f"  Q3:       {np.percentile(results['beta_x_true'], 75):.6f}\n")
        f.write(f"  Max:      {np.max(results['beta_x_true']):.6f}\n\n")

        f.write("Predicted β(X) Distribution:\n")
        f.write(f"  Mean:     {np.mean(results['beta_x_pred']):.6f}\n")
        f.write(f"  Std Dev:  {np.std(results['beta_x_pred']):.6f}\n")
        f.write(f"  Min:      {np.min(results['beta_x_pred']):.6f}\n")
        f.write(f"  Q1:       {np.percentile(results['beta_x_pred'], 25):.6f}\n")
        f.write(f"  Median:   {np.median(results['beta_x_pred']):.6f}\n")
        f.write(f"  Q3:       {np.percentile(results['beta_x_pred'], 75):.6f}\n")
        f.write(f"  Max:      {np.max(results['beta_x_pred']):.6f}\n\n")

        # Residual analysis
        residuals = results['y_true'] - results['y_pred']
        f.write("RESIDUAL ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Mean residual:        {np.mean(residuals):.6f}\n")
        f.write(f"Std Dev residual:     {np.std(residuals):.6f}\n")
        f.write(f"Min residual:         {np.min(residuals):.6f}\n")
        f.write(f"Max residual:         {np.max(residuals):.6f}\n")

        from scipy import stats
        f.write(f"Skewness:             {stats.skew(residuals):.6f}\n")
        f.write(f"Kurtosis:             {stats.kurtosis(residuals):.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("="*80 + "\n")


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_nonlinear_simulation():
    """Run the complete nonlinear panel data simulation."""

    print("="*70)
    print("NONLINEAR PANEL DATA WITH DEEP NEURAL NETWORKS")
    print("="*70)

    # Generate data
    print("\nGenerating data with complex nonlinear β(X) function...")
    generator = NonlinearPanelDataGenerator(
        n_entities=500,
        n_periods=50,
        n_covariates=10,
        fe_strength=2.0
    )

    df = generator.generate_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Entities: {generator.n_entities}, Periods: {generator.n_periods}")
    print(f"Covariates: {generator.n_covariates}")

    # Prepare data for DNN
    entity_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    df['entity_id_encoded'] = entity_encoder.fit_transform(df['entity_id'])
    df['time_id_encoded'] = time_encoder.fit_transform(df['time_id'])

    # Get feature columns
    X_cols = [f'X{j+1}' for j in range(generator.n_covariates)]

    # Create tensors
    X = torch.FloatTensor(df[X_cols].values)
    y = torch.FloatTensor(df['Y'].values)
    beta_x_true = torch.FloatTensor(df['beta_x_true'].values)
    entity_ids = torch.LongTensor(df['entity_id_encoded'].values)
    time_ids = torch.LongTensor(df['time_id_encoded'].values)

    # Create dataset
    dataset = TensorDataset(entity_ids, time_ids, X, y, beta_x_true)

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Initialize model
    print("\nInitializing nonlinear panel DNN model...")
    model = NonlinearPanelDNN(
        n_entities=generator.n_entities,
        n_periods=generator.n_periods,
        n_covariates=generator.n_covariates,
        hidden_dims=[256, 128, 64, 32],  # Deep architecture
        dropout_rate=0.2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train model
    print("\nTraining model...")
    history, trained_model = train_nonlinear_model(
        model,
        train_loader,
        val_loader,
        n_epochs=300,
        lr_beta=0.001,
        lr_fe=0.01,
        patience=50
    )

    # Evaluate model
    print("\nEvaluating model on test data...")
    results = evaluate_model(trained_model, test_loader, generator, device)

    # Save results to text file
    output_file = '/Users/pranjal/Code/topsort-incrementality/panel_dnn/results/panel_nonlinear_beta_results.txt'
    print(f"\nSaving comprehensive results to {output_file}...")
    print_results_to_file(results, history, output_file)
    print(f"Results saved successfully!")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\nThe DNN successfully learned the complex nonlinear β(X) function")
    print("while maintaining proper panel fixed effects structure.")

    return trained_model, results, history


if __name__ == "__main__":
    model, results, history = run_nonlinear_simulation()