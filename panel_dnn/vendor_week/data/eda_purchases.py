
#!/usr/bin/env python3
import polars as pl

import os
os.chdir('/Users/pranjal/Code/topsort-incrementality/panel_dnn/vendor_week/data')
df = pl.read_parquet('product_daily_purchases.parquet')

print('=' * 60)
print('PRODUCT DAILY PURCHASES - EDA')
print('=' * 60)

print(f'\nShape: {df.shape}')
print(f'Size: {df.estimated_size("mb"):.1f} MB in memory')

print('\nColumns:', df.columns)
print('\nData types:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')

print('\n' + '=' * 40)
print('BASIC STATISTICS')
print('=' * 40)
print(df.describe())

print('\n' + '=' * 40)
print('DATE RANGE')
print('=' * 40)
print(f"Min date: {df['date'].min()}")
print(f"Max date: {df['date'].max()}")
print(f"Unique dates: {df['date'].n_unique()}")

print('\n' + '=' * 40)
print('PRODUCT STATISTICS')
print('=' * 40)
print(f"Unique products: {df['product_id'].n_unique():,}")
products_per_day = df.group_by('date').agg(pl.col('product_id').n_unique())
print(f"Products per day (avg): {products_per_day['product_id'].mean():.0f}")

print('\n' + '=' * 40)
print('PRODUCT SELLING FREQUENCY')
print('=' * 40)
product_days = df.group_by('product_id').agg(pl.count().alias('days_sold'))
print(f"Average days a product sells: {product_days['days_sold'].mean():.2f}")
print(f"Median days a product sells: {product_days['days_sold'].median():.0f}")
print(f"Max days a product sells: {product_days['days_sold'].max()}")

# Distribution
for threshold in [1, 2, 3, 4, 5, 10, 30, 100]:
    count = product_days.filter(pl.col('days_sold') == threshold if threshold < 5 else pl.col('days_sold') >= threshold).shape[0]
    pct = count / product_days.shape[0] * 100
    op = '=' if threshold < 5 else '>='
    print(f"Products selling {op}{threshold} day(s): {count:,} ({pct:.1f}%)")

print('\n' + '=' * 40)
print('REVENUE STATISTICS')
print('=' * 40)
print(f"Total revenue: ${df['revenue_dollars'].sum():,.2f}")
daily_revenue = df.group_by('date').agg(pl.col('revenue_dollars').sum())
print(f"Daily revenue (avg): ${daily_revenue['revenue_dollars'].mean():,.2f}")
print(f"Revenue per product-day (avg): ${df['revenue_dollars'].mean():.2f}")

print('\n' + '=' * 40)
print('UNITS SOLD')
print('=' * 40)
print(f"Total units: {df['units_sold'].sum():,}")
print(f"Units per product-day (avg): {df['units_sold'].mean():.1f}")

units_per_product = df.group_by('product_id').agg(pl.col('units_sold').sum())
print(f"Average total units per product: {units_per_product['units_sold'].mean():.1f}")
print(f"Median total units per product: {units_per_product['units_sold'].median():.0f}")

print('\n' + '=' * 40)
print('PRICE ANALYSIS')
print('=' * 40)
print(f"Avg unit price (mean): ${df['avg_unit_price'].mean():.2f}")
print(f"Avg unit price (median): ${df['avg_unit_price'].median():.2f}")
print(f"Price range: ${df['avg_unit_price'].min():.2f} - ${df['avg_unit_price'].max():.2f}")

print('\n' + '=' * 40)
print('MISSING VALUES')
print('=' * 40)
print(df.null_count())

print('\n' + '=' * 40)
print('TOP 10 PRODUCTS BY REVENUE')
print('=' * 40)
top_products = (df.group_by('product_id')
                .agg(pl.col('revenue_dollars').sum().alias('total_revenue'))
                .sort('total_revenue', descending=True)
                .head(10))
for row in top_products.iter_rows():
    print(f'{row[0]}: ${row[1]:,.2f}')

print('\n' + '=' * 40)
print('DAILY TREND (LAST 10 DAYS)')
print('=' * 40)
daily = (df.group_by('date')
         .agg([
             pl.col('revenue_dollars').sum().alias('revenue'),
             pl.col('units_sold').sum().alias('units'),
             pl.col('product_id').n_unique().alias('products')
         ])
         .sort('date')
         .tail(10))
print(daily)

print('\n' + '=' * 40)
print('SAMPLE DATA (FIRST 5 ROWS)')
print('=' * 40)
print(df.head())

print('\n' + '=' * 60)
print('DEEP DIVE: MULTI-DAY SELLING PRODUCTS')
print('=' * 60)

# Find products that sell on multiple days
product_days = df.group_by('product_id').agg([
    pl.len().alias('days_sold'),
    pl.col('units_sold').sum().alias('total_units'),
    pl.col('revenue_dollars').sum().alias('total_revenue'),
    pl.col('avg_unit_price').mean().alias('avg_price'),
    pl.col('date').min().alias('first_sale'),
    pl.col('date').max().alias('last_sale')
])

multi_day_products = product_days.filter(pl.col('days_sold') > 1)
print(f"\nProducts selling multiple days: {multi_day_products.shape[0]:,} ({multi_day_products.shape[0]/product_days.shape[0]*100:.2f}%)")

# Top recurring products
print('\n' + '=' * 40)
print('TOP 20 MOST FREQUENT SELLERS')
print('=' * 40)
top_frequent = (multi_day_products
                .sort('days_sold', descending=True)
                .head(20))

for row in top_frequent.iter_rows():
    product_id, days, units, revenue, price, first, last = row
    days_span = (last - first).days + 1
    print(f"{product_id[:24]}...: {days} days over {days_span} day span | ${revenue:,.0f} revenue | {units} units")

# Analyze characteristics of multi-day sellers
print('\n' + '=' * 40)
print('MULTI-DAY SELLER CHARACTERISTICS')
print('=' * 40)

# Revenue comparison
single_day_revenue = product_days.filter(pl.col('days_sold') == 1)['total_revenue'].sum()
multi_day_revenue = multi_day_products['total_revenue'].sum()
print(f"Single-day product revenue: ${single_day_revenue:,.2f} ({single_day_revenue/df['revenue_dollars'].sum()*100:.1f}%)")
print(f"Multi-day product revenue: ${multi_day_revenue:,.2f} ({multi_day_revenue/df['revenue_dollars'].sum()*100:.1f}%)")

# Price comparison
single_day_avg_price = product_days.filter(pl.col('days_sold') == 1)['avg_price'].mean()
multi_day_avg_price = multi_day_products['avg_price'].mean()
print(f"\nSingle-day product avg price: ${single_day_avg_price:.2f}")
print(f"Multi-day product avg price: ${multi_day_avg_price:.2f}")

# Units comparison
single_day_avg_units = product_days.filter(pl.col('days_sold') == 1)['total_units'].mean()
multi_day_avg_units = multi_day_products['total_units'].mean()
print(f"\nSingle-day product avg units: {single_day_avg_units:.1f}")
print(f"Multi-day product avg units: {multi_day_avg_units:.1f}")

# Consistency analysis for top sellers
print('\n' + '=' * 40)
print('CONSISTENCY ANALYSIS (TOP 100-DAY+ SELLERS)')
print('=' * 40)
consistent_sellers = product_days.filter(pl.col('days_sold') >= 100).sort('total_revenue', descending=True)

if consistent_sellers.shape[0] > 0:
    for row in consistent_sellers.head(10).iter_rows():
        product_id, days, units, revenue, price, first, last = row
        days_span = (last - first).days + 1
        consistency = days / days_span * 100
        print(f"{product_id[:24]}...: Sold {days}/{days_span} days ({consistency:.1f}% consistency) | ${revenue:,.0f}")
else:
    print("No products with 100+ days of sales")

# Date span analysis
print('\n' + '=' * 40)
print('SALES PERSISTENCE PATTERNS')
print('=' * 40)

multi_day_with_span = multi_day_products.with_columns(
    ((pl.col('last_sale') - pl.col('first_sale')).dt.total_days() + 1).alias('date_span'),
    (pl.col('days_sold') / ((pl.col('last_sale') - pl.col('first_sale')).dt.total_days() + 1) * 100).alias('consistency_pct')
)

# Group by consistency levels
high_consistency = multi_day_with_span.filter(pl.col('consistency_pct') >= 50)
medium_consistency = multi_day_with_span.filter((pl.col('consistency_pct') >= 20) & (pl.col('consistency_pct') < 50))
low_consistency = multi_day_with_span.filter(pl.col('consistency_pct') < 20)

print(f"High consistency (≥50% days): {high_consistency.shape[0]:,} products")
print(f"Medium consistency (20-50%): {medium_consistency.shape[0]:,} products")
print(f"Low consistency (<20%): {low_consistency.shape[0]:,} products")

# Revenue by consistency
print(f"\nRevenue by consistency:")
print(f"  High: ${high_consistency['total_revenue'].sum():,.2f}")
print(f"  Medium: ${medium_consistency['total_revenue'].sum():,.2f}")
print(f"  Low: ${low_consistency['total_revenue'].sum():,.2f}")

# Examine specific high-revenue multi-day products
print('\n' + '=' * 40)
print('HIGH-REVENUE MULTI-DAY PRODUCTS')
print('=' * 40)
high_revenue_multi = (multi_day_with_span
                      .filter(pl.col('total_revenue') > 10000)
                      .sort('total_revenue', descending=True))

print(f"Products with >$10K revenue selling multiple days: {high_revenue_multi.shape[0]}")
for row in high_revenue_multi.head(10).iter_rows():
    product_id, days, units, revenue, price, first, last, span, consistency = row
    print(f"{product_id[:24]}...: ${revenue:,.0f} | {days} days | {consistency:.1f}% consistency | ${price:.2f} avg price")
