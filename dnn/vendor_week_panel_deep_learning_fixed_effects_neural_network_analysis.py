#!/usr/bin/env python3
"""
Vendor-Week Panel Analysis using Deep Learning
Outputs all results to vendor_week_panel_dl_results.txt
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

# Redirect all output to file
output_file = open('results/vendor_week_panel_dl_results.txt', 'w')
original_stdout = sys.stdout
sys.stdout = output_file


class SimplifiedFixedEffectsNet(nn.Module):
    def __init__(self, n_vendors, n_weeks):
        super(SimplifiedFixedEffectsNet, self).__init__()
        self.vendor_fe = nn.Embedding(n_vendors, 1)
        self.week_fe = nn.Embedding(n_weeks, 1)
        self.beta = nn.Parameter(torch.tensor([0.5]))
        nn.init.normal_(self.vendor_fe.weight, mean=0, std=0.01)
        nn.init.normal_(self.week_fe.weight, mean=0, std=0.01)

    def forward(self, vendor_ids, week_ids, log_clicks):
        vendor_effect = self.vendor_fe(vendor_ids).squeeze()
        week_effect = self.week_fe(week_ids).squeeze()
        prediction = self.beta * log_clicks + vendor_effect + week_effect
        return prediction


print("DEEP LEARNING VENDOR-WEEK PANEL ANALYSIS")
print("="*80)

# Load data
print("LOADING DATA: vendor_panel_full_history_clicks_only.parquet")
df = pd.read_parquet('data/vendor_panel_full_history_clicks_only.parquet')

print(f"DATA SHAPE: {df.shape}")
print(f"COLUMNS: {list(df.columns)}")
print(f"MEMORY USAGE: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("="*80)

# Data preparation
df['revenue_dollars'] = df['revenue_dollars'].astype(float)
df['clicks'] = df['clicks'].astype(int)
df['week'] = pd.to_datetime(df['week'])
df['log_revenue_plus_1'] = np.log1p(df['revenue_dollars'])
df['log_clicks_plus_1'] = np.log1p(df['clicks'])

print("\nDATA SUMMARY STATISTICS")
print("="*80)
print(df.describe())
print("="*80)

print("\nDATE RANGE")
print(f"MIN DATE: {df['week'].min()}")
print(f"MAX DATE: {df['week'].max()}")
print(f"UNIQUE WEEKS: {df['week'].nunique()}")
print(f"UNIQUE VENDORS: {df['vendor_id'].nunique()}")
print("="*80)

print("\nZERO COUNTS")
print(f"ZERO CLICKS: {(df['clicks'] == 0).sum()} ({(df['clicks'] == 0).mean()*100:.2f}%)")
print(f"ZERO REVENUE: {(df['revenue_dollars'] == 0).sum()} ({(df['revenue_dollars'] == 0).mean()*100:.2f}%)")
print("="*80)

# Encode categorical variables
vendor_encoder = LabelEncoder()
week_encoder = LabelEncoder()
df['vendor_id_encoded'] = vendor_encoder.fit_transform(df['vendor_id'])
df['week_encoded'] = week_encoder.fit_transform(df['week'])

n_vendors = df['vendor_id_encoded'].nunique()
n_weeks = df['week_encoded'].nunique()

print("\nMODEL ARCHITECTURE")
print("="*80)
print(f"Model: SimplifiedFixedEffectsNet")
print(f"Vendor embeddings: {n_vendors} x 1")
print(f"Week embeddings: {n_weeks} x 1")
print(f"Beta parameter: 1 learnable scalar")
print(f"Total parameters: {n_vendors + n_weeks + 1}")
print("="*80)

# Create dataloaders
vendor_ids = torch.LongTensor(df['vendor_id_encoded'].values)
week_ids = torch.LongTensor(df['week_encoded'].values)
log_clicks = torch.FloatTensor(df['log_clicks_plus_1'].values)
log_revenue = torch.FloatTensor(df['log_revenue_plus_1'].values)

dataset = TensorDataset(vendor_ids, week_ids, log_clicks, log_revenue)
n_samples = len(dataset)
n_train = int(n_samples * 0.8)
n_val = n_samples - n_train

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

print(f"\nDATA SPLIT")
print(f"Total samples: {n_samples}")
print(f"Training samples: {n_train}")
print(f"Validation samples: {n_val}")
print(f"Batch size: 2048")
print("="*80)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimplifiedFixedEffectsNet(n_vendors, n_weeks).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

print(f"\nTRAINING CONFIGURATION")
print(f"Device: {device}")
print(f"Loss function: MSE")
print(f"Optimizer: Adam")
print(f"Initial learning rate: 0.01")
print(f"Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)")
print("="*80)

# Training
print("\nTRAINING PROGRESS")
print("="*80)
n_epochs = 50

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0
    n_batches_train = 0

    for batch in train_loader:
        vendor_ids_batch, week_ids_batch, log_clicks_batch, log_revenue_batch = [b.to(device) for b in batch]
        predictions = model(vendor_ids_batch, week_ids_batch, log_clicks_batch)
        loss = criterion(predictions, log_revenue_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batches_train += 1

    # Validation
    model.eval()
    val_loss = 0
    n_batches_val = 0

    with torch.no_grad():
        for batch in val_loader:
            vendor_ids_batch, week_ids_batch, log_clicks_batch, log_revenue_batch = [b.to(device) for b in batch]
            predictions = model(vendor_ids_batch, week_ids_batch, log_clicks_batch)
            loss = criterion(predictions, log_revenue_batch)
            val_loss += loss.item()
            n_batches_val += 1

    avg_train_loss = train_loss / n_batches_train
    avg_val_loss = val_loss / n_batches_val
    current_lr = optimizer.param_groups[0]['lr']
    beta_value = model.beta.item()

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1:3d}/{n_epochs} | Beta: {beta_value:.10f} | Train Loss: {avg_train_loss:.10f} | Val Loss: {avg_val_loss:.10f} | LR: {current_lr:.8f}")

print("="*80)

# Extract final parameters
print("\nFINAL MODEL PARAMETERS")
print("="*80)
final_beta = model.beta.item()
print(f"Beta coefficient: {final_beta:.10f}")

vendor_effects = model.vendor_fe.weight.detach().cpu().numpy().flatten()
week_effects = model.week_fe.weight.detach().cpu().numpy().flatten()

print(f"\nVENDOR FIXED EFFECTS SUMMARY")
print(f"Count: {len(vendor_effects)}")
print(f"Mean: {np.mean(vendor_effects):.10f}")
print(f"Std Dev: {np.std(vendor_effects):.10f}")
print(f"Min: {np.min(vendor_effects):.10f}")
print(f"Q1: {np.percentile(vendor_effects, 25):.10f}")
print(f"Median: {np.median(vendor_effects):.10f}")
print(f"Q3: {np.percentile(vendor_effects, 75):.10f}")
print(f"Max: {np.max(vendor_effects):.10f}")

print(f"\nWEEK FIXED EFFECTS SUMMARY")
print(f"Count: {len(week_effects)}")
print(f"Mean: {np.mean(week_effects):.10f}")
print(f"Std Dev: {np.std(week_effects):.10f}")
print(f"Min: {np.min(week_effects):.10f}")
print(f"Q1: {np.percentile(week_effects, 25):.10f}")
print(f"Median: {np.median(week_effects):.10f}")
print(f"Q3: {np.percentile(week_effects, 75):.10f}")
print(f"Max: {np.max(week_effects):.10f}")
print("="*80)

print("\nFIRST 20 VENDOR FIXED EFFECTS")
print("="*80)
vendor_ids_original = vendor_encoder.classes_
for i in range(min(20, len(vendor_effects))):
    print(f"{vendor_ids_original[i]}: {vendor_effects[i]:.10f}")

print("\nALL WEEK FIXED EFFECTS")
print("="*80)
week_ids_original = week_encoder.classes_
for i in range(len(week_effects)):
    print(f"{week_ids_original[i]}: {week_effects[i]:.10f}")

# Calculate diagnostics on full dataset
print("\nMODEL DIAGNOSTICS")
print("="*80)
model.eval()
all_vendor_ids = torch.LongTensor(df['vendor_id_encoded'].values).to(device)
all_week_ids = torch.LongTensor(df['week_encoded'].values).to(device)
all_log_clicks = torch.FloatTensor(df['log_clicks_plus_1'].values).to(device)
all_log_revenue = torch.FloatTensor(df['log_revenue_plus_1'].values).to(device)

with torch.no_grad():
    all_predictions = model(all_vendor_ids, all_week_ids, all_log_clicks)

residuals = (all_log_revenue.cpu().numpy() - all_predictions.cpu().numpy())
fitted_values = all_predictions.cpu().numpy()

ss_res = np.sum(residuals**2)
ss_tot = np.sum((all_log_revenue.cpu().numpy() - np.mean(all_log_revenue.cpu().numpy()))**2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))

print(f"Observations: {len(df)}")
print(f"R-squared: {r_squared:.10f}")
print(f"RMSE: {rmse:.10f}")
print(f"MAE: {mae:.10f}")

n = len(residuals)
sigma2 = np.var(residuals)
log_lik = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2) - 1/(2*sigma2) * ss_res
aic = -2*log_lik + 2*(n_vendors + n_weeks + 1)
bic = -2*log_lik + np.log(n)*(n_vendors + n_weeks + 1)

print(f"Log-likelihood: {log_lik:.10f}")
print(f"AIC: {aic:.10f}")
print(f"BIC: {bic:.10f}")

print("\nRESIDUAL STATISTICS")
print("="*80)
print(f"Mean: {np.mean(residuals):.10f}")
print(f"Std Dev: {np.std(residuals):.10f}")
print(f"Min: {np.min(residuals):.10f}")
print(f"Q1: {np.percentile(residuals, 25):.10f}")
print(f"Median: {np.median(residuals):.10f}")
print(f"Q3: {np.percentile(residuals, 75):.10f}")
print(f"Max: {np.max(residuals):.10f}")

from scipy import stats
print(f"Skewness: {stats.skew(residuals):.10f}")
print(f"Kurtosis: {stats.kurtosis(residuals):.10f}")

print("\nFITTED VALUES STATISTICS")
print("="*80)
print(f"Mean: {np.mean(fitted_values):.10f}")
print(f"Std Dev: {np.std(fitted_values):.10f}")
print(f"Min: {np.min(fitted_values):.10f}")
print(f"Q1: {np.percentile(fitted_values, 25):.10f}")
print(f"Median: {np.median(fitted_values):.10f}")
print(f"Q3: {np.percentile(fitted_values, 75):.10f}")
print(f"Max: {np.max(fitted_values):.10f}")

print("\nVENDOR TOTAL EFFECTS (BETA + VENDOR_FE)")
print("="*80)
vendor_total_effects = final_beta + vendor_effects
print(f"Mean: {np.mean(vendor_total_effects):.10f}")
print(f"Std Dev: {np.std(vendor_total_effects):.10f}")
print(f"Min: {np.min(vendor_total_effects):.10f}")
print(f"Q1: {np.percentile(vendor_total_effects, 25):.10f}")
print(f"Median: {np.median(vendor_total_effects):.10f}")
print(f"Q3: {np.percentile(vendor_total_effects, 75):.10f}")
print(f"Max: {np.max(vendor_total_effects):.10f}")

print("\nEND OF ANALYSIS")
print("="*80)

# Close output file
sys.stdout = original_stdout
output_file.close()

print(f"Results saved to results/vendor_week_panel_dl_results.txt")