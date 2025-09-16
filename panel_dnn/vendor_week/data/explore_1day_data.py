#!/usr/bin/env python3
import pandas as pd
import os

os.chdir('/Users/pranjal/Code/topsort-incrementality/panel_dnn/vendor_week/data')

print("=" * 60)
print("EXPLORING 1-DAY EXTRACTED DATA (2025-03-14)")
print("=" * 60)

# Load the 1-day data files
clicks = pd.read_parquet('vendor_daily_pulls/clicks/clicks_2025-03-14.parquet')
impressions = pd.read_parquet('vendor_daily_pulls/impressions/impressions_2025-03-14.parquet')
auctions = pd.read_parquet('vendor_daily_pulls/auctions/auctions_2025-03-14.parquet')

print("\n1. CLICKS DATA")
print("-" * 40)
print(f"Shape: {clicks.shape}")
print(f"Columns: {clicks.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(clicks.head())
print(f"\nData types:")
print(clicks.dtypes)
print(f"\nUnique vendors: {clicks['vendor_id'].nunique() if 'vendor_id' in clicks.columns else 'N/A'}")
print(f"Total clicks: {clicks['clicks'].sum() if 'clicks' in clicks.columns else len(clicks)}")

print("\n2. IMPRESSIONS DATA")
print("-" * 40)
print(f"Shape: {impressions.shape}")
print(f"Columns: {impressions.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(impressions.head())
print(f"\nData types:")
print(impressions.dtypes)
print(f"\nUnique vendors: {impressions['vendor_id'].nunique() if 'vendor_id' in impressions.columns else 'N/A'}")
print(f"Total impressions: {impressions['impressions'].sum() if 'impressions' in impressions.columns else len(impressions)}")

print("\n3. AUCTIONS DATA")
print("-" * 40)
print(f"Shape: {auctions.shape}")
print(f"Columns: {auctions.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(auctions.head())
print(f"\nData types:")
print(auctions.dtypes)
print(f"\nUnique vendors: {auctions['vendor_id'].nunique() if 'vendor_id' in auctions.columns else 'N/A'}")
print(f"Total auctions: {auctions['auction_count'].sum() if 'auction_count' in auctions.columns else len(auctions)}")

# Cross-table analysis
print("\n" + "=" * 60)
print("CROSS-TABLE ANALYSIS")
print("=" * 60)

# Find overlapping vendors
if 'vendor_id' in clicks.columns and 'vendor_id' in impressions.columns:
    clicks_vendors = set(clicks['vendor_id'].unique())
    impressions_vendors = set(impressions['vendor_id'].unique())
    auctions_vendors = set(auctions['vendor_id'].unique()) if 'vendor_id' in auctions.columns else set()

    print(f"\nVendor overlap:")
    print(f"Vendors in clicks only: {len(clicks_vendors - impressions_vendors)}")
    print(f"Vendors in impressions only: {len(impressions_vendors - clicks_vendors)}")
    print(f"Vendors in both clicks and impressions: {len(clicks_vendors & impressions_vendors)}")

    if auctions_vendors:
        print(f"Vendors in all three tables: {len(clicks_vendors & impressions_vendors & auctions_vendors)}")

# Check if there's aggregation or raw data
print("\n" + "=" * 60)
print("DATA STRUCTURE ANALYSIS")
print("=" * 60)

for name, df in [('clicks', clicks), ('impressions', impressions), ('auctions', auctions)]:
    print(f"\n{name.upper()}:")
    # Check if it's aggregated (has count columns) or raw
    count_cols = [col for col in df.columns if 'count' in col.lower() or col == 'clicks' or col == 'impressions']
    if count_cols:
        print(f"  Appears to be AGGREGATED data (found columns: {count_cols})")
        for col in count_cols:
            print(f"    {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}")
    else:
        print(f"  Appears to be RAW data (no count columns found)")

    # Check date columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'occurred' in col.lower() or 'created' in col.lower()]
    if date_cols:
        print(f"  Date columns: {date_cols}")
        for col in date_cols:
            if df[col].dtype == 'object':
                try:
                    dates = pd.to_datetime(df[col])
                    print(f"    {col}: {dates.min()} to {dates.max()}")
                except:
                    pass

# Comprehensive summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS FOR MARCH 14, 2025")
print("=" * 60)

# Clicks statistics
print("\n📊 CLICKS METRICS:")
print("-" * 40)
print(f"Total clicks: {clicks['clicks'].sum():,}")
print(f"Unique vendors: {clicks['vendor_id'].nunique():,}")
print(f"Unique users who clicked: {clicks['distinct_users_clicked'].sum():,}")
print(f"Unique products clicked: {clicks['distinct_products_clicked'].sum():,}")
print(f"Unique campaigns: {clicks['distinct_campaigns_clicked'].sum():,}")
print(f"\nPer-vendor statistics:")
print(f"  Avg clicks/vendor: {clicks['clicks'].mean():.1f}")
print(f"  Median clicks/vendor: {clicks['clicks'].median():.0f}")
print(f"  Max clicks (single vendor): {clicks['clicks'].max():,}")
print(f"  Vendors with 1 click: {(clicks['clicks'] == 1).sum():,} ({(clicks['clicks'] == 1).sum()/len(clicks)*100:.1f}%)")
print(f"  Vendors with >100 clicks: {(clicks['clicks'] > 100).sum():,}")

# Impressions statistics
print("\n📊 IMPRESSIONS METRICS:")
print("-" * 40)
print(f"Total impressions: {impressions['impressions'].sum():,}")
print(f"Unique vendors: {impressions['vendor_id'].nunique():,}")
print(f"Unique users impressed: {impressions['distinct_users_impressed'].sum():,}")
print(f"Unique products shown: {impressions['distinct_products_impressed'].sum():,}")
print(f"Unique campaigns: {impressions['distinct_campaigns_impressed'].sum():,}")
print(f"\nPer-vendor statistics:")
print(f"  Avg impressions/vendor: {impressions['impressions'].mean():.1f}")
print(f"  Median impressions/vendor: {impressions['impressions'].median():.0f}")
print(f"  Max impressions (single vendor): {impressions['impressions'].max():,}")
print(f"  Vendors with <10 impressions: {(impressions['impressions'] < 10).sum():,}")
print(f"  Vendors with >1000 impressions: {(impressions['impressions'] > 1000).sum():,}")

# Auctions statistics
print("\n📊 AUCTIONS METRICS:")
print("-" * 40)
print(f"Total bids: {auctions['bids'].sum():,}")
print(f"Total wins: {auctions['wins'].sum():,}")
print(f"Win rate: {auctions['wins'].sum()/auctions['bids'].sum()*100:.1f}%")
print(f"Unique vendors bidding: {len(auctions):,}")
print(f"Unique products bid on: {auctions['distinct_products_bid'].sum():,}")
print(f"Unique campaigns: {auctions['distinct_campaigns_bid'].sum():,}")
print(f"\nPer-vendor statistics:")
print(f"  Avg bids/vendor: {auctions['bids'].mean():.1f}")
print(f"  Avg wins/vendor: {auctions['wins'].mean():.1f}")
print(f"  Max wins (single vendor): {auctions['wins'].max():,}")
print(f"  Vendors with no wins: {(auctions['wins'] == 0).sum():,}")

# CTR calculation (clicks/impressions)
print("\n📊 CLICK-THROUGH RATE (CTR):")
print("-" * 40)
# Merge clicks and impressions on vendor_id
merged = impressions.merge(clicks[['vendor_id', 'clicks']], on='vendor_id', how='left')
merged['clicks'] = merged['clicks'].fillna(0)
merged['ctr'] = (merged['clicks'] / merged['impressions'] * 100)
print(f"Overall CTR: {clicks['clicks'].sum()/impressions['impressions'].sum()*100:.3f}%")
print(f"Average vendor CTR: {merged['ctr'].mean():.3f}%")
print(f"Median vendor CTR: {merged['ctr'].median():.3f}%")
print(f"Max vendor CTR: {merged['ctr'].max():.2f}%")
print(f"Vendors with 0% CTR: {(merged['ctr'] == 0).sum():,}")

# Distribution analysis
print("\n📊 DISTRIBUTION ANALYSIS:")
print("-" * 40)
print("\nClicks distribution (percentiles):")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {clicks['clicks'].quantile(p/100):.0f} clicks")

print("\nImpressions distribution (percentiles):")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {impressions['impressions'].quantile(p/100):.0f} impressions")

# Top performers
print("\n📊 TOP 5 VENDORS BY CLICKS:")
print("-" * 40)
top_clicks = clicks.nlargest(5, 'clicks')[['vendor_id', 'clicks', 'distinct_users_clicked']]
for idx, row in top_clicks.iterrows():
    print(f"  {row['vendor_id'][:8]}...: {row['clicks']:,} clicks from {row['distinct_users_clicked']:,} users")

print("\n📊 TOP 5 VENDORS BY IMPRESSIONS:")
print("-" * 40)
top_imp = impressions.nlargest(5, 'impressions')[['vendor_id', 'impressions', 'distinct_users_impressed']]
for idx, row in top_imp.iterrows():
    print(f"  {row['vendor_id'][:8]}...: {row['impressions']:,} impressions to {row['distinct_users_impressed']:,} users")

print("\n📊 TOP 5 VENDORS BY WIN RATE (min 100 bids):")
print("-" * 40)
auctions_filtered = auctions[auctions['bids'] >= 100].copy()
auctions_filtered['win_rate'] = auctions_filtered['wins'] / auctions_filtered['bids'] * 100
top_winners = auctions_filtered.nlargest(5, 'win_rate')[['vendor_id', 'bids', 'wins', 'win_rate']]
for idx, row in top_winners.iterrows():
    vendor_str = str(row['vendor_id'])[:8] if isinstance(row['vendor_id'], bytes) else row['vendor_id'][:8]
    print(f"  {vendor_str}...: {row['win_rate']:.1f}% ({row['wins']:,}/{row['bids']:,})")