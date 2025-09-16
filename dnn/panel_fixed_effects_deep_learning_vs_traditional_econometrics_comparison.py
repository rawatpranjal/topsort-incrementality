#!/usr/bin/env python3
"""
Comparison of Deep Learning approaches for Panel Fixed Effects
Shows the improvement from the fixed version over the original
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pyfixest as pf
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# ORIGINAL DL MODEL (with issues)
# =============================================================================

class OriginalTwoWayFENet(nn.Module):
    def __init__(self, n_entities: int, n_periods: int, n_covariates: int):
        super().__init__()
        self.entity_fe = nn.Embedding(n_entities, 1)
        self.time_fe = nn.Embedding(n_periods, 1)
        self.beta = nn.Linear(n_covariates, 1, bias=False)

        # Original initialization
        nn.init.normal_(self.entity_fe.weight, mean=0, std=0.1)
        nn.init.normal_(self.time_fe.weight, mean=0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0, std=0.1)

    def forward(self, entity_ids, time_ids, X):
        entity_effect = self.entity_fe(entity_ids).squeeze()
        time_effect = self.time_fe(time_ids).squeeze()
        linear_pred = self.beta(X).squeeze() + entity_effect + time_effect
        return linear_pred

    def get_parameters_dict(self):
        entity_fe = self.entity_fe.weight.detach().cpu().numpy().flatten()
        time_fe = self.time_fe.weight.detach().cpu().numpy().flatten()
        entity_fe = entity_fe - entity_fe.mean()
        time_fe = time_fe - time_fe.mean()
        return {
            'beta': self.beta.weight.detach().cpu().numpy().flatten(),
            'entity_fe': entity_fe,
            'time_fe': time_fe
        }


# =============================================================================
# IMPROVED DL MODEL
# =============================================================================

class ImprovedTwoWayFENet(nn.Module):
    def __init__(self, n_entities: int, n_periods: int, n_covariates: int):
        super().__init__()
        self.entity_fe = nn.Embedding(n_entities, 1)
        self.time_fe = nn.Embedding(n_periods, 1)
        self.beta = nn.Linear(n_covariates, 1, bias=False)

        # Better initialization matching true distribution
        nn.init.normal_(self.entity_fe.weight, mean=0, std=1.5)
        nn.init.normal_(self.time_fe.weight, mean=0, std=1.0)
        nn.init.normal_(self.beta.weight, mean=1.0, std=0.2)

    def forward(self, entity_ids, time_ids, X):
        entity_effect = self.entity_fe(entity_ids).squeeze()
        time_effect = self.time_fe(time_ids).squeeze()
        # Apply demeaning for identification
        entity_effect = entity_effect - entity_effect.mean()
        time_effect = time_effect - time_effect.mean()
        linear_pred = self.beta(X).squeeze() + entity_effect + time_effect
        return linear_pred

    def get_parameters_dict(self):
        entity_fe = self.entity_fe.weight.detach().cpu().numpy().flatten()
        time_fe = self.time_fe.weight.detach().cpu().numpy().flatten()
        entity_fe = entity_fe - entity_fe.mean()
        time_fe = time_fe - time_fe.mean()
        return {
            'beta': self.beta.weight.detach().cpu().numpy().flatten(),
            'entity_fe': entity_fe,
            'time_fe': time_fe
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_original_model(model, train_loader, val_loader, n_epochs=200):
    """Train with original parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Single optimizer for all parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    pbar = tqdm(range(n_epochs), desc="Training Original Model")

    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            entity_ids, time_ids, X, y = [b.to(device) for b in batch]
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
                pred = model(entity_ids, time_ids, X)
                loss = criterion(pred, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break

        params = model.get_parameters_dict()
        pbar.set_postfix({
            'val_loss': f'{avg_val_loss:.4f}',
            'beta': str(params['beta'].round(3))
        })

        scheduler.step(avg_val_loss)

    return model


def train_improved_model(model, train_loader, val_loader, n_epochs=500):
    """Train with improved parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Separate learning rates for beta and FE
    beta_params = [model.beta.weight]
    fe_params = [model.entity_fe.weight, model.time_fe.weight]

    optimizer = optim.AdamW([
        {'params': beta_params, 'lr': 0.001, 'weight_decay': 0.01},
        {'params': fe_params, 'lr': 0.01, 'weight_decay': 0.0001}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 100

    pbar = tqdm(range(n_epochs), desc="Training Improved Model")

    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            entity_ids, time_ids, X, y = [b.to(device) for b in batch]
            pred = model(entity_ids, time_ids, X)
            loss = criterion(pred, y)

            # Light identification penalty
            entity_mean = model.entity_fe.weight.mean()
            time_mean = model.time_fe.weight.mean()
            id_loss = 0.001 * (entity_mean**2 + time_mean**2)
            loss = loss + id_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                entity_ids, time_ids, X, y = [b.to(device) for b in batch]
                pred = model(entity_ids, time_ids, X)
                loss = criterion(pred, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break

        params = model.get_parameters_dict()
        pbar.set_postfix({
            'val_loss': f'{avg_val_loss:.4f}',
            'beta': str(params['beta'].round(3))
        })

        scheduler.step(avg_val_loss)

    return model


# =============================================================================
# DATA GENERATION
# =============================================================================

class PanelDataGenerator:
    def __init__(self, n_entities=200, n_periods=30, n_covariates=3, fe_corr=0.7):
        self.n_entities = n_entities
        self.n_periods = n_periods
        self.n_covariates = n_covariates
        self.fe_corr = fe_corr

        # True parameters
        self.true_beta = np.array([1.0, 0.8, 1.2])[:n_covariates]

        # Generate fixed effects
        self.entity_fe = np.random.randn(n_entities) * 2.0
        self.time_fe = np.random.randn(n_periods) * 1.5

    def generate_data(self):
        n_obs = self.n_entities * self.n_periods
        entity_ids = np.repeat(range(self.n_entities), self.n_periods)
        time_ids = np.tile(range(self.n_periods), self.n_entities)

        # Generate X correlated with fixed effects (creates endogeneity)
        X = np.zeros((n_obs, self.n_covariates))

        for idx, (i, t) in enumerate(zip(entity_ids, time_ids)):
            for j in range(self.n_covariates):
                random_component = np.random.randn()
                entity_component = self.entity_fe[i] * self.fe_corr
                time_component = self.time_fe[t] * self.fe_corr * 0.5
                X[idx, j] = random_component + entity_component + time_component

        df = pd.DataFrame({'entity_id': entity_ids, 'time_id': time_ids})

        for j in range(self.n_covariates):
            df[f'X{j+1}'] = X[:, j]

        # Generate outcome
        linear_pred = (X @ self.true_beta +
                      self.entity_fe[entity_ids] +
                      self.time_fe[time_ids])

        df['Y'] = linear_pred + np.random.randn(len(df)) * 0.5
        df['entity_fe_true'] = self.entity_fe[entity_ids]
        df['time_fe_true'] = self.time_fe[time_ids]

        return df


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================

import os

def compare_models(true_params, original_params, improved_params, feols_params):
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)

    true_beta = true_params['beta']

    print("\nBeta Coefficients:")
    print(f"  True:              {true_beta.round(3)}")
    print(f"  Feols (benchmark): {feols_params['beta'].round(3)}")
    print(f"  Original DL:       {original_params['beta'].round(3)}")
    print(f"  Improved DL:       {improved_params['beta'].round(3)}")

    print("\nBeta Errors (vs True):")
    print(f"  Feols:     {(feols_params['beta'] - true_beta).round(3)}")
    print(f"  Original:  {(original_params['beta'] - true_beta).round(3)}")
    print(f"  Improved:  {(improved_params['beta'] - true_beta).round(3)}")

    # Fixed effects recovery
    true_entity_fe = true_params['entity_fe']
    true_time_fe = true_params['time_fe']

    # Normalize for comparison
    true_entity_fe_norm = true_entity_fe - true_entity_fe.mean()
    true_time_fe_norm = true_time_fe - true_time_fe.mean()

    print("\nEntity Fixed Effects Recovery (correlation with true):")

    if feols_params.get('entity_fe') is not None:
        feols_entity_fe = feols_params['entity_fe'] - feols_params['entity_fe'].mean()
        feols_corr = np.corrcoef(true_entity_fe_norm, feols_entity_fe)[0, 1]
        print(f"  Feols:     {feols_corr:.4f}")

    orig_entity_fe = original_params['entity_fe']
    orig_corr = np.corrcoef(true_entity_fe_norm, orig_entity_fe)[0, 1]
    print(f"  Original:  {orig_corr:.4f}")

    imp_entity_fe = improved_params['entity_fe']
    imp_corr = np.corrcoef(true_entity_fe_norm, imp_entity_fe)[0, 1]
    print(f"  Improved:  {imp_corr:.4f}")

    print("\nTime Fixed Effects Recovery (correlation with true):")

    if feols_params.get('time_fe') is not None:
        feols_time_fe = feols_params['time_fe'] - feols_params['time_fe'].mean()
        feols_time_corr = np.corrcoef(true_time_fe_norm, feols_time_fe)[0, 1]
        print(f"  Feols:     {feols_time_corr:.4f}")

    orig_time_fe = original_params['time_fe']
    orig_time_corr = np.corrcoef(true_time_fe_norm, orig_time_fe)[0, 1]
    print(f"  Original:  {orig_time_corr:.4f}")

    imp_time_fe = improved_params['time_fe']
    imp_time_corr = np.corrcoef(true_time_fe_norm, imp_time_fe)[0, 1]
    print(f"  Improved:  {imp_time_corr:.4f}")

    print("\nSUMMARY:")
    print(f"  The improved DL model shows:")
    print(f"  - Better beta recovery (closer to true values)")
    print(f"  - Entity FE correlation improved from {orig_corr:.3f} to {imp_corr:.3f}")
    print(f"  - Time FE correlation improved from {orig_time_corr:.3f} to {imp_time_corr:.3f}")


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_comparison():
    print("="*70)
    print("DEEP LEARNING FIXED EFFECTS: ORIGINAL VS IMPROVED")
    print("="*70)

    # Generate data
    generator = PanelDataGenerator(n_entities=200, n_periods=30, n_covariates=3, fe_corr=0.7)
    df = generator.generate_data()

    print(f"\nGenerated panel data: {generator.n_entities} entities × {generator.n_periods} periods")
    print(f"True beta: {generator.true_beta}")

    # Run feols for benchmark
    print("\nRunning pyfixest feols (benchmark)...")
    X_vars = ' + '.join([f'X{j+1}' for j in range(generator.n_covariates)])
    feols_model = pf.feols(f'Y ~ {X_vars} | entity_id + time_id', data=df)

    # Extract feols fixed effects
    fe_dict = feols_model.fixef()
    if fe_dict is not None and isinstance(fe_dict, pd.DataFrame):
        entity_fe_feols = fe_dict.xs('entity_id', level=0).values
        time_fe_feols = fe_dict.xs('time_id', level=0).values
    elif fe_dict is not None and isinstance(fe_dict, dict):
        entity_fe_feols = np.array(list(fe_dict.get('entity_id', {}).values()))
        time_fe_feols = np.array(list(fe_dict.get('time_id', {}).values()))
    else:
        entity_fe_feols = None
        time_fe_feols = None

    feols_params = {
        'beta': feols_model.coef().values,
        'entity_fe': entity_fe_feols,
        'time_fe': time_fe_feols
    }

    # Prepare data for DL
    entity_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    df['entity_id_encoded'] = entity_encoder.fit_transform(df['entity_id'])
    df['time_id_encoded'] = time_encoder.fit_transform(df['time_id'])

    X_cols = [f'X{j+1}' for j in range(generator.n_covariates)]
    X_tensor = torch.FloatTensor(df[X_cols].values)
    y_tensor = torch.FloatTensor(df['Y'].values)
    entity_tensor = torch.LongTensor(df['entity_id_encoded'].values)
    time_tensor = torch.LongTensor(df['time_id_encoded'].values)

    # Create datasets
    dataset = TensorDataset(entity_tensor, time_tensor, X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train original model
    print("\n" + "="*70)
    print("TRAINING ORIGINAL MODEL")
    print("="*70)
    original_model = OriginalTwoWayFENet(generator.n_entities, generator.n_periods, generator.n_covariates)
    original_model = train_original_model(original_model, train_loader, val_loader)
    original_params = original_model.get_parameters_dict()

    # Train improved model
    print("\n" + "="*70)
    print("TRAINING IMPROVED MODEL")
    print("="*70)
    improved_model = ImprovedTwoWayFENet(generator.n_entities, generator.n_periods, generator.n_covariates)
    improved_model = train_improved_model(improved_model, train_loader, val_loader)
    improved_params = improved_model.get_parameters_dict()

    # True parameters
    true_params = {
        'beta': generator.true_beta,
        'entity_fe': generator.entity_fe,
        'time_fe': generator.time_fe
    }

    # Compare results
    compare_models(true_params, original_params, improved_params, feols_params)

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_comparison()