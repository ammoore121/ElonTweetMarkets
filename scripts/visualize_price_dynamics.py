"""
Visualize key price dynamics findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

# Paths
BASE_DIR = Path(r"G:\My Drive\AI_Projects\Ideas - In Progress\ElonTweetMarkets")
PRICE_HISTORY = BASE_DIR / "data/sources/polymarket/prices/price_history.parquet"
MARKET_CATALOG = BASE_DIR / "data/processed/market_catalog.parquet"
OUTPUT_DIR = BASE_DIR / "data/analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("Loading data...")
prices_df = pd.read_parquet(PRICE_HISTORY)
catalog_df = pd.read_parquet(MARKET_CATALOG)

# Convert timestamp
prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
prices_df = prices_df.sort_values(['event_id', 'token_id', 'timestamp'])

# Get event metadata
event_metadata = catalog_df.groupby('event_id').agg({
    'event_slug': 'first',
    'market_type': 'first',
    'is_resolved': 'first'
}).to_dict('index')

prices_df['market_type'] = prices_df['event_id'].map(
    lambda x: event_metadata.get(x, {}).get('market_type', 'unknown')
)
prices_df['resolved'] = prices_df['event_id'].map(
    lambda x: event_metadata.get(x, {}).get('is_resolved', False)
)

# Focus on resolved weekly markets
resolved_weekly = prices_df[(prices_df['market_type'] == 'weekly') & (prices_df['resolved'] == True)].copy()

print(f"Analyzing {len(resolved_weekly):,} resolved weekly price records")

# Calculate price changes
resolved_weekly = resolved_weekly.sort_values(['token_id', 'timestamp'])
resolved_weekly['price_change'] = resolved_weekly.groupby('token_id')['price'].diff()
resolved_weekly['hour'] = resolved_weekly['timestamp'].dt.hour

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CLOB Price Dynamics Analysis - Resolved Weekly Markets', fontsize=16, fontweight='bold')

# 1. Distribution of price changes
ax = axes[0, 0]
price_changes = resolved_weekly['price_change'].dropna()
price_changes_filtered = price_changes[(price_changes >= -0.2) & (price_changes <= 0.2)]  # Filter outliers
ax.hist(price_changes_filtered, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Price Change ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Price Changes\n(filtered to +/- $0.20)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics
median_change = price_changes.abs().median()
mean_change = price_changes.abs().mean()
pct_2c = (price_changes.abs() >= 0.02).sum() / len(price_changes) * 100
ax.text(0.98, 0.97, f'Median abs: ${median_change:.4f}\nMean abs: ${mean_change:.4f}\n>=2c: {pct_2c:.1f}%',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# 2. Hourly price volatility
ax = axes[0, 1]
hourly_stats = resolved_weekly.groupby('hour').agg({
    'price_change': lambda x: x.abs().mean()
}).reset_index()
hourly_stats.columns = ['hour', 'mean_abs_change']

ax.bar(hourly_stats['hour'], hourly_stats['mean_abs_change'], color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Hour of Day (UTC)', fontsize=12)
ax.set_ylabel('Mean Absolute Price Change ($)', fontsize=12)
ax.set_title('Price Volatility by Hour of Day', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3, axis='y')

# Highlight peak hours
peak_hours = hourly_stats.nlargest(5, 'mean_abs_change')['hour'].tolist()
ax.text(0.98, 0.97, f'Peak hours: {", ".join(map(str, peak_hours))}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# 3. Tail bucket analysis
ax = axes[1, 0]
tail_buckets = resolved_weekly[(resolved_weekly['price'] >= 0.03) & (resolved_weekly['price'] <= 0.15)].copy()
non_tail = resolved_weekly[(resolved_weekly['price'] < 0.03) | (resolved_weekly['price'] > 0.15)].copy()

tail_changes = tail_buckets['price_change'].dropna().abs()
non_tail_changes = non_tail['price_change'].dropna().abs()

data_to_plot = [non_tail_changes[non_tail_changes <= 0.1], tail_changes[tail_changes <= 0.1]]
ax.boxplot(data_to_plot, labels=['Non-Tail\n(<3c or >15c)', 'Tail\n(3-15c)'],
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
ax.set_ylabel('Absolute Price Change ($)', fontsize=12)
ax.set_title('Tail vs Non-Tail Bucket Volatility', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
tail_median = tail_changes.median()
non_tail_median = non_tail_changes.median()
ax.text(0.98, 0.97, f'Tail median: ${tail_median:.4f}\nNon-tail median: ${non_tail_median:.4f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# 4. Lifecycle price evolution
ax = axes[1, 1]
lifecycle_data = []

for token_id in resolved_weekly['token_id'].unique():
    token_prices = resolved_weekly[resolved_weekly['token_id'] == token_id].sort_values('timestamp')
    if len(token_prices) < 2:
        continue

    first_price = token_prices.iloc[0]['price']
    last_price = token_prices.iloc[-1]['price']
    change_pct = (last_price - first_price) / first_price * 100 if first_price > 0 else 0

    lifecycle_data.append({
        'first_price': first_price,
        'change_pct': change_pct
    })

lifecycle_df = pd.DataFrame(lifecycle_data)

# Bin by initial price
lifecycle_df['price_bucket'] = pd.cut(
    lifecycle_df['first_price'],
    bins=[0, 0.05, 0.10, 0.20, 1.0],
    labels=['0-5c', '5-10c', '10-20c', '20c+']
)

bucket_means = lifecycle_df.groupby('price_bucket')['change_pct'].mean()
bucket_stds = lifecycle_df.groupby('price_bucket')['change_pct'].std()

x = range(len(bucket_means))
ax.bar(x, bucket_means, yerr=bucket_stds, alpha=0.7, color='mediumseagreen',
       edgecolor='black', capsize=5, error_kw={'linewidth': 2})
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(bucket_means.index)
ax.set_xlabel('Initial Price Range', fontsize=12)
ax.set_ylabel('Mean Price Change (%)', fontsize=12)
ax.set_title('Price Evolution: Open to Close by Initial Price', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = OUTPUT_DIR / "price_dynamics_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {output_file}")

# Create a second figure for polling frequency analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Polling Frequency Opportunity Analysis', fontsize=16, fontweight='bold')

# Simulate polling frequencies
def count_opportunities(df, freq_hours, threshold=0.02):
    """Count opportunities where price dropped >= threshold."""
    opportunities = 0

    for token_id in df['token_id'].unique():
        token_data = df[df['token_id'] == token_id].sort_values('timestamp')
        if len(token_data) < 2:
            continue

        start_time = token_data.iloc[0]['timestamp']
        end_time = token_data.iloc[-1]['timestamp']
        current_time = start_time
        last_price = token_data.iloc[0]['price']

        while current_time <= end_time:
            snapshot = token_data[token_data['timestamp'] <= current_time]
            if len(snapshot) > 0:
                current_price = snapshot.iloc[-1]['price']
                if current_price - last_price <= -threshold:
                    opportunities += 1
                last_price = current_price
            current_time += timedelta(hours=freq_hours)

    return opportunities

print("\nSimulating polling frequencies...")
frequencies = [1, 2, 3, 6, 12, 24]
opportunities = []

for freq in frequencies:
    print(f"  {freq}hr...")
    n_opps = count_opportunities(resolved_weekly, freq)
    opportunities.append(n_opps)

# Plot 1: Absolute opportunities
ax = axes2[0]
ax.bar(range(len(frequencies)), opportunities, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xticks(range(len(frequencies)))
ax.set_xticklabels([f'{f}hr' for f in frequencies])
ax.set_xlabel('Polling Frequency', fontsize=12)
ax.set_ylabel('Number of Opportunities (>=2c price drop)', fontsize=12)
ax.set_title('Opportunities Captured by Polling Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, v in enumerate(opportunities):
    ax.text(i, v + max(opportunities)*0.02, str(v), ha='center', va='bottom', fontweight='bold')

# Plot 2: Incremental value
ax = axes2[1]
baseline = opportunities[-1]  # 24hr
incremental = [(opp - baseline) / baseline * 100 for opp in opportunities]

bars = ax.bar(range(len(frequencies)), incremental, alpha=0.7, color='coral', edgecolor='black')
ax.set_xticks(range(len(frequencies)))
ax.set_xticklabels([f'{f}hr' for f in frequencies])
ax.set_xlabel('Polling Frequency', fontsize=12)
ax.set_ylabel('Additional Opportunities vs 24hr (%)', fontsize=12)
ax.set_title('Incremental Value vs 24hr Baseline', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='red', linestyle='--', linewidth=2)

# Add values on bars
for i, v in enumerate(incremental):
    ax.text(i, v + max(incremental)*0.02, f'+{v:.0f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
output_file2 = OUTPUT_DIR / "polling_frequency_analysis.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {output_file2}")

print("\nVisualization complete!")
