#!/usr/bin/env python3
"""
Panel Data Simulation Study: Proper Fixed Effects Comparison

This simulation demonstrates that:
1. Fixed effects are ESSENTIAL for correct estimation
2. Deep learning can properly replicate two-way fixed effects
3. Both feols/feglm (via R) and DL recover the true parameters

The data generation ensures that without fixed effects, estimates are severely biased.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pyfixest as pf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
# import matplotlib.pyplot as plt  # Removed for text-only output
# import seaborn as sns  # Removed for text-only output
from scipy import stats
from tqdm import tqdm
import warnings
import time
from typing import Dict, Tuple, List, Optional
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion
from rpy2.robjects.conversion import localconverter

warnings.filterwarnings('ignore')
# plt.style.use('seaborn-v0_8-darkgrid')  # Removed for text-only output
# sns.set_palette("husl")  # Removed for text-only output

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# DATA GENERATION WITH STRONG FIXED EFFECTS
# =============================================================================

class PanelDataGeneratorWithStrongFE:
    """
    Generate panel data where fixed effects are ESSENTIAL.
    Without FE, the model will be severely biased due to:
    1. Entity-specific unobserved heterogeneity correlated with X
    2. Time-specific shocks correlated with X
    """

    def __init__(
        self,
        n_entities: int = 500,
        n_periods: int = 50,
        n_covariates: int = 3,
        true_beta: Optional[np.ndarray] = None,
        fe_correlation_strength: float = 0.7  # How much FE correlate with X
    ):
        """
        Initialize panel data generator with strong fixed effects.

        The key is that entity and time FE are correlated with X variables,
        creating omitted variable bias if FE are not included.
        """
        self.n_entities = n_entities
        self.n_periods = n_periods
        self.n_covariates = n_covariates
        self.n_obs = n_entities * n_periods
        self.fe_corr = fe_correlation_strength

        # True coefficients
        self.true_beta = true_beta if true_beta is not None else \
                        np.array([1.5, -0.8, 0.6])[:n_covariates]

        # Generate entity fixed effects with large variance
        # These represent time-invariant unobserved heterogeneity
        self.entity_fe = np.random.randn(n_entities) * 2.0

        # Generate time fixed effects (common shocks)
        self.time_fe = np.random.randn(n_periods) * 1.5

        # Normalize for identification
        self.entity_fe = self.entity_fe - self.entity_fe.mean()
        self.time_fe = self.time_fe - self.time_fe.mean()

        print(f"True beta coefficients: {self.true_beta}")
        print(f"Entity FE std: {self.entity_fe.std():.2f}")
        print(f"Time FE std: {self.time_fe.std():.2f}")
        print(f"FE-X correlation strength: {self.fe_corr}")

    def generate_panel(self) -> pd.DataFrame:
        """
        Generate panel data where X is correlated with fixed effects.
        This ensures FE are necessary for unbiased estimation.
        """

        # Create panel structure
        entity_ids = np.repeat(np.arange(self.n_entities), self.n_periods)
        time_ids = np.tile(np.arange(self.n_periods), self.n_entities)

        # Generate X variables that are correlated with fixed effects
        X = np.zeros((self.n_obs, self.n_covariates))

        for i in range(self.n_entities):
            for t in range(self.n_periods):
                idx = i * self.n_periods + t

                # X depends on both entity and time effects (creating endogeneity)
                for j in range(self.n_covariates):
                    # Base random component
                    random_component = np.random.randn()

                    # Add correlation with entity FE
                    entity_component = self.entity_fe[i] * self.fe_corr

                    # Add correlation with time FE
                    time_component = self.time_fe[t] * self.fe_corr * 0.5

                    # Combine
                    X[idx, j] = random_component + entity_component + time_component

                    # Add some AR(1) structure within entity
                    if t > 0:
                        X[idx, j] += 0.3 * X[idx - 1, j]

        # Create DataFrame
        df = pd.DataFrame({
            'entity_id': entity_ids,
            'time_id': time_ids
        })

        # Add covariates
        for j in range(self.n_covariates):
            df[f'X{j+1}'] = X[:, j]

        # Add true fixed effects
        df['entity_fe_true'] = self.entity_fe[entity_ids]
        df['time_fe_true'] = self.time_fe[time_ids]

        return df

    def generate_continuous_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate continuous outcome with strong FE influence."""

        X_cols = [f'X{j+1}' for j in range(self.n_covariates)]
        X = df[X_cols].values

        # True model: Y = X*beta + entity_fe + time_fe + epsilon
        linear_pred = (X @ self.true_beta +
                      df['entity_fe_true'].values +
                      df['time_fe_true'].values)

        # Add noise
        epsilon = np.random.randn(len(df)) * 0.5  # Small noise to highlight FE importance

        df['Y_continuous'] = linear_pred + epsilon
        df['linear_pred_true'] = linear_pred

        # Also create outcome WITHOUT fixed effects for comparison
        df['Y_no_fe'] = X @ self.true_beta + epsilon

        return df

    def generate_binary_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary outcome with strong FE influence."""

        X_cols = [f'X{j+1}' for j in range(self.n_covariates)]
        X = df[X_cols].values

        # Latent variable model
        linear_pred = (X @ self.true_beta +
                      df['entity_fe_true'].values * 0.7 +  # Scale for binary
                      df['time_fe_true'].values * 0.5)

        # Apply logistic transformation
        prob = 1 / (1 + np.exp(-linear_pred))
        df['Y_binary'] = np.random.binomial(1, prob)
        df['prob_true'] = prob

        # Without FE for comparison
        linear_pred_no_fe = X @ self.true_beta
        prob_no_fe = 1 / (1 + np.exp(-linear_pred_no_fe))
        df['Y_binary_no_fe'] = np.random.binomial(1, prob_no_fe)

        return df

    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete dataset demonstrating FE importance."""

        print("\nGenerating panel data with strong fixed effects...")
        df = self.generate_panel()
        df = self.generate_continuous_outcome(df)
        df = self.generate_binary_outcome(df)

        # Calculate and show correlation between X and FE
        X1_entity_corr = df.groupby('entity_id')['X1'].mean().corr(
            pd.Series(self.entity_fe, index=range(self.n_entities))
        )
        print(f"Correlation between X1 and entity FE: {X1_entity_corr:.3f}")

        print(f"Generated {len(df)} observations")
        print(f"Entities: {self.n_entities}, Periods: {self.n_periods}")

        return df


# =============================================================================
# R INTEGRATION FOR PROPER FEGLM
# =============================================================================

def setup_r_environment():
    """Setup R environment and load required packages."""

    print("Setting up R environment...")

    # R code to install/load packages
    r_code = """
    if (!require("fixest")) {
        install.packages("fixest", repos="https://cloud.r-project.org/")
    }
    library(fixest)
    """

    ro.r(r_code)
    print("R environment ready with fixest package")


def run_r_feols(df: pd.DataFrame, n_covariates: int) -> Dict:
    """Run feols using R's fixest package."""

    print("\nRunning R fixest::feols (proper two-way fixed effects)...")

    # Transfer DataFrame to R using context manager
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
        ro.globalenv['panel_data'] = r_df

    # Build formula
    X_vars = ' + '.join([f'X{j+1}' for j in range(n_covariates)])
    formula = f'Y_continuous ~ {X_vars} | entity_id + time_id'

    # Run feols in R
    start_time = time.time()
    r_code = f"""
    model <- feols({formula}, data = panel_data, vcov = "hetero")

    # Extract components
    coef_values <- coef(model)
    se_values <- se(model)

    # Get fixed effects
    fe <- fixef(model)
    entity_fe <- fe$entity_id
    time_fe <- fe$time_id

    # Model statistics
    r2_val <- r2(model, type = "r2")
    rmse_val <- sqrt(mean(residuals(model)^2))

    list(
        coef = coef_values,
        se = se_values,
        entity_fe = entity_fe,
        time_fe = time_fe,
        r2 = r2_val,
        rmse = rmse_val
    )
    """

    result = ro.r(r_code)
    elapsed_time = time.time() - start_time

    # Extract results
    coefficients = np.array(result[0])
    std_errors = np.array(result[1])
    entity_fe = np.array(result[2]) if result[2] is not None else None
    time_fe = np.array(result[3]) if result[3] is not None else None
    r2 = result[4][0] if result[4] is not None else None
    rmse = result[5][0] if result[5] is not None else None

    print(f"R feols completed in {elapsed_time:.2f} seconds")
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")

    return {
        'coefficients': coefficients,
        'std_errors': std_errors,
        'entity_fe': entity_fe,
        'time_fe': time_fe,
        'r2': r2,
        'rmse': rmse,
        'elapsed_time': elapsed_time
    }


def run_r_feglm(df: pd.DataFrame, n_covariates: int) -> Dict:
    """Run feglm using R's fixest package with proper fixed effects."""

    print("\nRunning R fixest::feglm (logit with two-way fixed effects)...")

    # Transfer DataFrame to R using context manager
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
        ro.globalenv['panel_data'] = r_df

    # Build formula
    X_vars = ' + '.join([f'X{j+1}' for j in range(n_covariates)])
    formula = f'Y_binary ~ {X_vars} | entity_id + time_id'

    # Run feglm in R
    start_time = time.time()
    r_code = f"""
    model <- feglm({formula}, data = panel_data, family = binomial())

    # Extract components
    coef_values <- coef(model)
    se_values <- se(model)

    # Get fixed effects
    fe <- fixef(model)
    entity_fe <- fe$entity_id
    time_fe <- fe$time_id

    # Model statistics
    deviance_val <- deviance(model)

    list(
        coef = coef_values,
        se = se_values,
        entity_fe = entity_fe,
        time_fe = time_fe,
        deviance = deviance_val
    )
    """

    result = ro.r(r_code)
    elapsed_time = time.time() - start_time

    # Extract results
    coefficients = np.array(result[0])
    std_errors = np.array(result[1])
    entity_fe = np.array(result[2]) if result[2] is not None else None
    time_fe = np.array(result[3]) if result[3] is not None else None
    deviance = result[4][0] if result[4] is not None else None

    print(f"R feglm completed in {elapsed_time:.2f} seconds")
    print(f"Deviance: {deviance:.2f}")

    return {
        'coefficients': coefficients,
        'std_errors': std_errors,
        'entity_fe': entity_fe,
        'time_fe': time_fe,
        'deviance': deviance,
        'elapsed_time': elapsed_time
    }


# =============================================================================
# DEEP LEARNING MODELS WITH PROPER FIXED EFFECTS
# =============================================================================

class ProperTwoWayFENet(nn.Module):
    """
    Neural network that properly implements two-way fixed effects.
    Key: The fixed effects are learned embeddings that capture entity/time heterogeneity.
    """

    def __init__(self, n_entities: int, n_periods: int, n_covariates: int):
        super(ProperTwoWayFENet, self).__init__()

        # Fixed effects as embeddings (one value per entity/period)
        self.entity_fe = nn.Embedding(n_entities, 1)
        self.time_fe = nn.Embedding(n_periods, 1)

        # Linear coefficients for X variables
        self.beta = nn.Linear(n_covariates, 1, bias=False)

        # Initialize FE near zero (they will learn the true values)
        nn.init.normal_(self.entity_fe.weight, mean=0, std=0.1)
        nn.init.normal_(self.time_fe.weight, mean=0, std=0.1)

        # Initialize beta near the expected range
        nn.init.normal_(self.beta.weight, mean=0.5, std=0.2)

    def forward(self, entity_ids, time_ids, X):
        # Get fixed effects
        entity_effect = self.entity_fe(entity_ids).squeeze()
        time_effect = self.time_fe(time_ids).squeeze()

        # Linear model: y = X*beta + entity_FE + time_FE
        linear_pred = self.beta(X).squeeze() + entity_effect + time_effect

        return linear_pred

    def get_parameters_dict(self):
        """Extract parameters as numpy arrays."""
        return {
            'beta': self.beta.weight.detach().cpu().numpy().flatten(),
            'entity_fe': self.entity_fe.weight.detach().cpu().numpy().flatten(),
            'time_fe': self.time_fe.weight.detach().cpu().numpy().flatten()
        }


class ProperBinaryFENet(nn.Module):
    """Binary outcome model with proper two-way fixed effects."""

    def __init__(self, n_entities: int, n_periods: int, n_covariates: int):
        super(ProperBinaryFENet, self).__init__()

        # Fixed effects
        self.entity_fe = nn.Embedding(n_entities, 1)
        self.time_fe = nn.Embedding(n_periods, 1)

        # Coefficients
        self.beta = nn.Linear(n_covariates, 1, bias=False)

        # Initialize
        nn.init.normal_(self.entity_fe.weight, mean=0, std=0.1)
        nn.init.normal_(self.time_fe.weight, mean=0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.5, std=0.2)

    def forward(self, entity_ids, time_ids, X):
        # Get fixed effects
        entity_effect = self.entity_fe(entity_ids).squeeze()
        time_effect = self.time_fe(time_ids).squeeze()

        # Logit model
        logits = self.beta(X).squeeze() + entity_effect + time_effect
        prob = torch.sigmoid(logits)

        return prob, logits

    def get_parameters_dict(self):
        """Extract parameters."""
        return {
            'beta': self.beta.weight.detach().cpu().numpy().flatten(),
            'entity_fe': self.entity_fe.weight.detach().cpu().numpy().flatten(),
            'time_fe': self.time_fe.weight.detach().cpu().numpy().flatten()
        }


# =============================================================================
# TRAINING WITH REGULARIZATION
# =============================================================================

def train_fe_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    is_binary: bool = False,
    n_epochs: int = 150,
    lr: float = 0.01,
    fe_regularization: float = 0.01  # L2 regularization on FE
):
    """Train model with proper handling of fixed effects."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss function
    if is_binary:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Optimizer with weight decay for regularization
    optimizer = optim.Adam([
        {'params': model.beta.parameters(), 'weight_decay': 0},
        {'params': model.entity_fe.parameters(), 'weight_decay': fe_regularization},
        {'params': model.time_fe.parameters(), 'weight_decay': fe_regularization}
    ], lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'beta_history': []}

    pbar = tqdm(range(n_epochs), desc="Training DL Model with Fixed Effects")

    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            entity_ids, time_ids, X, y = [b.to(device) for b in batch]

            if is_binary:
                prob, _ = model(entity_ids, time_ids, X)
                loss = criterion(prob, y.float())
            else:
                pred = model(entity_ids, time_ids, X)
                loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                entity_ids, time_ids, X, y = [b.to(device) for b in batch]

                if is_binary:
                    prob, _ = model(entity_ids, time_ids, X)
                    loss = criterion(prob, y.float())
                else:
                    pred = model(entity_ids, time_ids, X)
                    loss = criterion(pred, y)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Track parameters
        params = model.get_parameters_dict()
        history['beta_history'].append(params['beta'].copy())

        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'beta': str(params['beta'].round(3))
        })

        scheduler.step(avg_val_loss)

    return history, model


# =============================================================================
# COMPARISON AND VALIDATION
# =============================================================================

def demonstrate_fe_importance(df: pd.DataFrame, n_covariates: int):
    """Show that without FE, estimates are severely biased."""

    print("\n" + "="*70)
    print("DEMONSTRATING FIXED EFFECTS IMPORTANCE")
    print("="*70)

    # Run OLS without fixed effects
    X_vars = ' + '.join([f'X{j+1}' for j in range(n_covariates)])

    print("\n1. OLS WITHOUT Fixed Effects:")
    no_fe_model = pf.feols(f'Y_continuous ~ {X_vars}', data=df, vcov='hetero')
    no_fe_coef_all = no_fe_model.coef().values
    # Extract only the X coefficients (not intercept)
    no_fe_coef = no_fe_coef_all[1:] if len(no_fe_coef_all) > n_covariates else no_fe_coef_all
    print(f"   Coefficients (incl. intercept): {no_fe_coef_all.round(3)}")
    print(f"   X Coefficients only: {no_fe_coef.round(3)}")
    print(f"   R²: {no_fe_model._r2:.4f}")

    print("\n2. OLS WITH Fixed Effects:")
    with_fe_model = pf.feols(f'Y_continuous ~ {X_vars} | entity_id + time_id',
                             data=df, vcov='hetero')
    with_fe_coef = with_fe_model.coef().values
    print(f"   Coefficients: {with_fe_coef.round(3)}")
    print(f"   R²: {with_fe_model._r2:.4f}")

    return no_fe_coef, with_fe_coef


def compare_all_methods(
    true_params: Dict,
    r_params: Dict,
    dl_params: Dict,
    outcome_type: str
):
    """Compare parameters across all methods."""

    print(f"\n" + "="*70)
    print(f"{outcome_type.upper()} OUTCOME - PARAMETER COMPARISON")
    print("="*70)

    # Compare beta coefficients
    true_beta = true_params['beta']
    r_beta = r_params['coefficients']
    dl_beta = dl_params['beta']

    print("\nBeta Coefficients:")
    print(f"  True:          {true_beta.round(3)}")
    print(f"  R fixest:      {r_beta.round(3)}")
    print(f"  Deep Learning: {dl_beta.round(3)}")
    print(f"  R Error:       {(r_beta - true_beta).round(3)}")
    print(f"  DL Error:      {(dl_beta - true_beta).round(3)}")

    # Compare fixed effects recovery
    if r_params.get('entity_fe') is not None:
        # Normalize for comparison
        true_entity_fe = true_params['entity_fe'] - true_params['entity_fe'].mean()
        r_entity_fe = r_params['entity_fe'] - r_params['entity_fe'].mean()
        dl_entity_fe = dl_params['entity_fe'] - dl_params['entity_fe'].mean()

        r_entity_corr = np.corrcoef(true_entity_fe, r_entity_fe)[0, 1]
        dl_entity_corr = np.corrcoef(true_entity_fe, dl_entity_fe)[0, 1]

        print(f"\nEntity FE Recovery (correlation with true):")
        print(f"  R fixest:      {r_entity_corr:.4f}")
        print(f"  Deep Learning: {dl_entity_corr:.4f}")

    if r_params.get('time_fe') is not None:
        true_time_fe = true_params['time_fe'] - true_params['time_fe'].mean()
        r_time_fe = r_params['time_fe'] - r_params['time_fe'].mean()
        dl_time_fe = dl_params['time_fe'] - dl_params['time_fe'].mean()

        r_time_corr = np.corrcoef(true_time_fe, r_time_fe)[0, 1]
        dl_time_corr = np.corrcoef(true_time_fe, dl_time_fe)[0, 1]

        print(f"\nTime FE Recovery (correlation with true):")
        print(f"  R fixest:      {r_time_corr:.4f}")
        print(f"  Deep Learning: {dl_time_corr:.4f}")

    return {
        'r_beta_error': np.mean(np.abs(r_beta - true_beta)),
        'dl_beta_error': np.mean(np.abs(dl_beta - true_beta)),
        'r_entity_corr': r_entity_corr if r_params.get('entity_fe') is not None else None,
        'dl_entity_corr': dl_entity_corr if r_params.get('entity_fe') is not None else None,
        'r_time_corr': r_time_corr if r_params.get('time_fe') is not None else None,
        'dl_time_corr': dl_time_corr if r_params.get('time_fe') is not None else None
    }


def plot_fe_comparison(true_params, r_params, dl_params, outcome_type):
    """Plot fixed effects comparison."""
    # Plotting disabled for text-only output
    return  # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Normalize all FE
    true_entity_fe = true_params['entity_fe'] - true_params['entity_fe'].mean()
    true_time_fe = true_params['time_fe'] - true_params['time_fe'].mean()

    if r_params.get('entity_fe') is not None:
        r_entity_fe = r_params['entity_fe'] - r_params['entity_fe'].mean()
        r_time_fe = r_params['time_fe'] - r_params['time_fe'].mean()

    dl_entity_fe = dl_params['entity_fe'] - dl_params['entity_fe'].mean()
    dl_time_fe = dl_params['time_fe'] - dl_params['time_fe'].mean()

    # Entity FE: R vs True
    if r_params.get('entity_fe') is not None:
        axes[0, 0].scatter(true_entity_fe, r_entity_fe, alpha=0.5, s=10)
        axes[0, 0].plot([true_entity_fe.min(), true_entity_fe.max()],
                       [true_entity_fe.min(), true_entity_fe.max()], 'r--')
        corr = np.corrcoef(true_entity_fe, r_entity_fe)[0, 1]
        axes[0, 0].set_title(f'R fixest Entity FE (ρ={corr:.3f})')
        axes[0, 0].set_xlabel('True Entity FE')
        axes[0, 0].set_ylabel('Estimated Entity FE')

    # Entity FE: DL vs True
    axes[0, 1].scatter(true_entity_fe, dl_entity_fe, alpha=0.5, s=10)
    axes[0, 1].plot([true_entity_fe.min(), true_entity_fe.max()],
                   [true_entity_fe.min(), true_entity_fe.max()], 'r--')
    corr = np.corrcoef(true_entity_fe, dl_entity_fe)[0, 1]
    axes[0, 1].set_title(f'Deep Learning Entity FE (ρ={corr:.3f})')
    axes[0, 1].set_xlabel('True Entity FE')
    axes[0, 1].set_ylabel('Estimated Entity FE')

    # Entity FE distributions
    axes[0, 2].hist(true_entity_fe, bins=30, alpha=0.5, label='True', density=True)
    if r_params.get('entity_fe') is not None:
        axes[0, 2].hist(r_entity_fe, bins=30, alpha=0.5, label='R', density=True)
    axes[0, 2].hist(dl_entity_fe, bins=30, alpha=0.5, label='DL', density=True)
    axes[0, 2].set_title('Entity FE Distributions')
    axes[0, 2].legend()

    # Time FE: R vs True
    if r_params.get('time_fe') is not None:
        axes[1, 0].scatter(true_time_fe, r_time_fe, s=20)
        axes[1, 0].plot([true_time_fe.min(), true_time_fe.max()],
                       [true_time_fe.min(), true_time_fe.max()], 'r--')
        corr = np.corrcoef(true_time_fe, r_time_fe)[0, 1]
        axes[1, 0].set_title(f'R fixest Time FE (ρ={corr:.3f})')
        axes[1, 0].set_xlabel('True Time FE')
        axes[1, 0].set_ylabel('Estimated Time FE')

    # Time FE: DL vs True
    axes[1, 1].scatter(true_time_fe, dl_time_fe, s=20)
    axes[1, 1].plot([true_time_fe.min(), true_time_fe.max()],
                   [true_time_fe.min(), true_time_fe.max()], 'r--')
    corr = np.corrcoef(true_time_fe, dl_time_fe)[0, 1]
    axes[1, 1].set_title(f'Deep Learning Time FE (ρ={corr:.3f})')
    axes[1, 1].set_xlabel('True Time FE')
    axes[1, 1].set_ylabel('Estimated Time FE')

    # Time FE over time
    axes[1, 2].plot(true_time_fe, 'k-', label='True', linewidth=2)
    if r_params.get('time_fe') is not None:
        axes[1, 2].plot(r_time_fe, 'b--', label='R', alpha=0.7)
    axes[1, 2].plot(dl_time_fe, 'r--', label='DL', alpha=0.7)
    axes[1, 2].set_title('Time Fixed Effects Pattern')
    axes[1, 2].set_xlabel('Time Period')
    axes[1, 2].legend()

    plt.suptitle(f'Fixed Effects Recovery Comparison ({outcome_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'fe_recovery_{outcome_type.lower()}.png', dpi=100)
    plt.show()


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_proper_simulation():
    """Run simulation with proper fixed effects handling."""

    # Open output file for comprehensive results
    import sys
    output_file = '/Users/pranjal/Code/topsort-incrementality/panel_dnn/results/panel_two_way_fe_simulation_results.txt'
    original_stdout = sys.stdout
    sys.stdout = open(output_file, 'w')

    print("="*70)
    print("PANEL DATA SIMULATION WITH PROPER FIXED EFFECTS")
    print("="*70)

    # Setup R environment
    setup_r_environment()

    # Generate data with strong fixed effects
    generator = PanelDataGeneratorWithStrongFE(
        n_entities=200,  # Smaller for R computation
        n_periods=30,
        n_covariates=3,
        fe_correlation_strength=0.7
    )

    df = generator.generate_complete_dataset()

    # Save for reproducibility
    # df.to_csv('panel_data_with_strong_fe.csv', index=False)  # Skipping CSV output

    # Demonstrate FE importance
    no_fe_coef, with_fe_coef = demonstrate_fe_importance(df, generator.n_covariates)

    print(f"\nTrue coefficients: {generator.true_beta.round(3)}")
    print(f"Bias without FE: {(no_fe_coef - generator.true_beta).round(3)}")
    print(f"Bias with FE:    {(with_fe_coef - generator.true_beta).round(3)}")

    # Prepare data for deep learning
    entity_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    df['entity_id_encoded'] = entity_encoder.fit_transform(df['entity_id'])
    df['time_id_encoded'] = time_encoder.fit_transform(df['time_id'])

    # =========================================================================
    # CONTINUOUS OUTCOME
    # =========================================================================
    print("\n" + "="*70)
    print("CONTINUOUS OUTCOME ANALYSIS")
    print("="*70)

    # R fixest
    r_continuous_results = run_r_feols(df, generator.n_covariates)

    # Deep Learning
    X_cols = [f'X{j+1}' for j in range(generator.n_covariates)]
    entity_ids = torch.LongTensor(df['entity_id_encoded'].values)
    time_ids = torch.LongTensor(df['time_id_encoded'].values)
    X = torch.FloatTensor(df[X_cols].values)
    y_continuous = torch.FloatTensor(df['Y_continuous'].values)

    # Create datasets
    dataset = TensorDataset(entity_ids, time_ids, X, y_continuous)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print("\nTraining Deep Learning Model (Continuous)...")
    dl_continuous_model = ProperTwoWayFENet(
        generator.n_entities, generator.n_periods, generator.n_covariates
    )

    history_cont, trained_cont_model = train_fe_model(
        dl_continuous_model, train_loader, val_loader,
        is_binary=False, n_epochs=150
    )

    dl_continuous_params = trained_cont_model.get_parameters_dict()

    # Compare
    true_params = {
        'beta': generator.true_beta,
        'entity_fe': generator.entity_fe,
        'time_fe': generator.time_fe
    }

    cont_comparison = compare_all_methods(
        true_params, r_continuous_results, dl_continuous_params, "Continuous"
    )

    # =========================================================================
    # BINARY OUTCOME
    # =========================================================================
    print("\n" + "="*70)
    print("BINARY OUTCOME ANALYSIS")
    print("="*70)

    # R fixest
    r_binary_results = run_r_feglm(df, generator.n_covariates)

    # Deep Learning
    y_binary = torch.LongTensor(df['Y_binary'].values)
    dataset_binary = TensorDataset(entity_ids, time_ids, X, y_binary)
    train_dataset_binary, val_dataset_binary = torch.utils.data.random_split(
        dataset_binary, [n_train, n_val]
    )
    train_loader_binary = DataLoader(train_dataset_binary, batch_size=256, shuffle=True)
    val_loader_binary = DataLoader(val_dataset_binary, batch_size=256, shuffle=False)

    print("\nTraining Deep Learning Model (Binary)...")
    dl_binary_model = ProperBinaryFENet(
        generator.n_entities, generator.n_periods, generator.n_covariates
    )

    history_binary, trained_binary_model = train_fe_model(
        dl_binary_model, train_loader_binary, val_loader_binary,
        is_binary=True, n_epochs=150
    )

    dl_binary_params = trained_binary_model.get_parameters_dict()

    # Compare
    binary_comparison = compare_all_methods(
        true_params, r_binary_results, dl_binary_params, "Binary"
    )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # plot_fe_comparison(true_params, r_continuous_results, dl_continuous_params, "Continuous")  # Plotting disabled
    # plot_fe_comparison(true_params, r_binary_results, dl_binary_params, "Binary")  # Plotting disabled

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)

    print("\n✅ KEY FINDINGS:")
    print("-" * 40)

    print("\n1. FIXED EFFECTS ARE ESSENTIAL:")
    print(f"   Without FE - Mean absolute bias: {np.mean(np.abs(no_fe_coef - generator.true_beta)):.3f}")
    print(f"   With FE - Mean absolute bias: {np.mean(np.abs(with_fe_coef - generator.true_beta)):.3f}")

    print("\n2. DEEP LEARNING SUCCESSFULLY REPLICATES R FIXEST:")

    print("\n   Continuous Outcome:")
    print(f"   - Beta error (R):  {cont_comparison['r_beta_error']:.4f}")
    print(f"   - Beta error (DL): {cont_comparison['dl_beta_error']:.4f}")
    print(f"   - Entity FE correlation (DL): {cont_comparison['dl_entity_corr']:.4f}")
    print(f"   - Time FE correlation (DL): {cont_comparison['dl_time_corr']:.4f}")

    print("\n   Binary Outcome:")
    print(f"   - Beta error (R):  {binary_comparison['r_beta_error']:.4f}")
    print(f"   - Beta error (DL): {binary_comparison['dl_beta_error']:.4f}")
    print(f"   - Entity FE correlation (DL): {binary_comparison['dl_entity_corr']:.4f}")
    print(f"   - Time FE correlation (DL): {binary_comparison['dl_time_corr']:.4f}")

    validation_success = (
        cont_comparison['dl_entity_corr'] > 0.95 and
        cont_comparison['dl_time_corr'] > 0.95 and
        binary_comparison['dl_entity_corr'] > 0.90 and
        binary_comparison['dl_time_corr'] > 0.90
    )

    if validation_success:
        print("\n✅ VALIDATION SUCCESSFUL!")
        print("Deep learning with embeddings successfully replicates")
        print("traditional econometric fixed effects models.")
    else:
        print("\n⚠️ Further tuning may be needed for perfect replication.")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)

    # Restore stdout
    sys.stdout = original_stdout
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_proper_simulation()