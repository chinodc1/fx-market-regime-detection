# %%
# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta

# ----------------------------------------
# 1. Configuration
# ----------------------------------------
tickers = ['AUDUSD=X', 'CADUSD=X', 'CHFUSD=X', 'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'NOKUSD=X', 'NZDUSD=X', 'SEKUSD=X']
start, end = "2000-01-01", datetime.today().strftime('%Y-%m-%d')
window = 20

# ----------------------------------------
# 2. Download FX Price Data
# ----------------------------------------
print("Downloading FX and ES price data...")
data = yf.download(tickers, start=start, end=end, interval='1d', group_by='ticker', auto_adjust=True)
es_data = yf.download('ES=F', start=start, end=end, interval='1d', auto_adjust=True)

df = pd.DataFrame()
for ticker in tickers:
    df[ticker] = data[ticker]['Close']  # Use 'Close' because 'Adj Close' might not exist for FX

es_prices = es_data['Close'].reindex(df.index).ffill()

df.dropna(how='all', inplace=True)
df.ffill(inplace=True)
print("Prices data ready.")

# ----------------------------------------
# 3. Compute Returns and Standardize
# ----------------------------------------
log_returns = np.log(df / df.shift(1)).dropna()
stds = (log_returns - log_returns.mean()) / log_returns.std()

# ----------------------------------------
# 4. Build Feature Matrix: Avg Pairwise Correlations
# ----------------------------------------
corrs = stds.rolling(window).corr()
pairs = stds.columns
n = len(pairs)

corr_list = []
valid_dates = []

for date in stds.index[window:]:
    try:
        corr_matrix = corrs.loc[date]
        if isinstance(corr_matrix, pd.DataFrame):
            matrix = corr_matrix.values
            if not np.isnan(matrix).any() and matrix.shape == (n, n):
                upper_tri_values = matrix[np.triu_indices(n, k=1)]
                corr_list.append(upper_tri_values)
                valid_dates.append(date)
    except KeyError:
        continue

feature_matrix = np.vstack(corr_list)
feature_dates = pd.to_datetime(valid_dates)
avg_corrs_by_date = pd.Series([np.mean(row) for row in feature_matrix], index=feature_dates)

# ----------------------------------------
# 5. Hidden Markov Model for Regime Detection
# ----------------------------------------
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
model.fit(feature_matrix)
regimes = model.predict(feature_matrix)

regime_df = pd.DataFrame({'Date': feature_dates, 'Regime': regimes})
regime_df.set_index('Date', inplace=True)

# ----------------------------------------
# 6. Align for Plotting and Evaluation
# ----------------------------------------
common_dates = avg_corrs_by_date.index.intersection(regime_df.index)
avg_corrs_aligned = avg_corrs_by_date.loc[common_dates]
regimes_aligned = regime_df.loc[common_dates]

# Calculate and print statistics for the report
stats = {}
total_obs = len(regimes_aligned)
expected_durations = 1 / (1 - np.diag(model.transmat_))

for regime in [0, 1]:
    mask = regimes_aligned['Regime'] == regime
    values = avg_corrs_aligned[mask].values
    stats[regime] = {
        'avg_corr': np.nanmean(values),
        'percentage': (mask.sum() / total_obs) * 100,
        'avg_duration': expected_durations[regime]
    }

# Dynamically assign labels based on which regime has higher correlation
high_corr_idx = 1 if stats[1]['avg_corr'] > stats[0]['avg_corr'] else 0
for r in [0, 1]:
    label = 'High Correlation' if r == high_corr_idx else 'Low Correlation'
    stats[r]['label'] = label
    avg_corr, percentage = stats[r]['avg_corr'], stats[r]['percentage']
    print(f"  Regime {r} ({label}): Avg Corr: {avg_corr:.4f}, Prevalence: {percentage:.2f}%, Avg Duration: {expected_durations[r]:.1f} days")

print(f"Transition Matrix:\n{model.transmat_}")

# ----------------------------------------
# 7. Plotting and Analysis
# ----------------------------------------
latest_date = avg_corrs_aligned.index.max()
latest_regime = int(regimes_aligned.iloc[-1]['Regime'])
regime_label = stats[latest_regime]['label']

def plot_regime_analysis(avg_corrs, regimes, es_vals, title_suffix, save_path=None):
    """Helper function to create the dual-subplot visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Adjust linewidth for data density (thinner for longer periods)
    lw = 1.0 if len(avg_corrs) > 1000 else 1.5

    # Plot 1: FX Correlation
    ax1.plot(avg_corrs.index, avg_corrs.values, label='Avg Pairwise FX Correlation', color='black', linewidth=lw)
    ax1.set_ylabel("FX Correlations")
    ax1.set_title(f"FX Correlations and S&P 500 E-mini (ES=F) - {title_suffix}")
    ax1.grid(True, alpha=0.3)

    # Plot 2: ES Price Chart
    ax2.plot(es_vals.index, es_vals.values, label='ES Futures (S&P 500)', color='blue', linewidth=lw)
    ax2.set_ylabel("ES Price")
    ax2.grid(True, alpha=0.3)

    # Add regime spans
    changes = regimes['Regime'].ne(regimes['Regime'].shift()).cumsum()
    for _, group in regimes.groupby(changes):
        regime_val = group['Regime'].iloc[0]
        color = 'green' if regime_val == high_corr_idx else 'red'
        for ax in [ax1, ax2]:
            ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel("Date")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Close the figure to free up memory
    else:
        plt.show()

def plot_fx_only_analysis(avg_corrs, regimes, title_suffix, save_path=None):
    """Creates a visualization of FX correlations without the ES price chart."""
    fig, ax = plt.subplots(figsize=(15, 6))
    lw = 1.0 if len(avg_corrs) > 1000 else 1.5

    ax.plot(avg_corrs.index, avg_corrs.values, label='Avg Pairwise FX Correlation', color='black', linewidth=lw)
    ax.set_ylabel("FX Correlations")
    ax.set_title(f"FX Correlations (Regime Tracking) - {title_suffix}")
    ax.grid(True, alpha=0.3)

    # Add regime spans
    changes = regimes['Regime'].ne(regimes['Regime'].shift()).cumsum()
    for _, group in regimes.groupby(changes):
        regime_val = group['Regime'].iloc[0]
        color = 'green' if regime_val == high_corr_idx else 'red'
        ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15)

    ax.legend(loc='upper right')
    plt.xlabel("Date")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def plot_prevalence(stats, script_dir):
    """Purpose: Shows that neither regime dominates the sample via a Bar Chart."""
    labels = [f"Regime {r}\n({stats[r]['label']})" for r in stats]
    percentages = [stats[r]['percentage'] for r in stats]
    colors = ['green' if r == high_corr_idx else 'red' for r in stats]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, percentages, color=colors, alpha=0.7)
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Regime Prevalence Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Summary_Regime_Prevalence.png"), dpi=300)
    plt.close(fig)

def plot_duration_distribution(regimes, script_dir):
    """Purpose: Visualize persistence and clustering behavior via a Histogram."""
    regime_series = regimes['Regime']
    streaks = (regime_series != regime_series.shift()).cumsum()
    durations = regime_series.groupby(streaks).size()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(durations, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Duration (Days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Regime Duration Distribution (Persistence & Clustering)')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Summary_Duration_Distribution.png"), dpi=300)
    plt.close(fig)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Clean up existing plots to ensure "overwrite" behavior if dates have shifted
patterns = ["FX_Regimes_", "FX_Only_Regimes_", "Summary_"]
for f in os.listdir(script_dir):
    if any(f.startswith(p) for p in patterns) and f.endswith(".png"):
        try:
            os.remove(os.path.join(script_dir, f))
        except OSError:
            pass

# Determine the start and end of the full data for chunking
full_data_start = avg_corrs_aligned.index.min()
full_data_end = avg_corrs_aligned.index.max()

current_chunk_start = full_data_start

while current_chunk_start <= full_data_end:
    # Calculate the end of the current 5-year period
    current_chunk_end = current_chunk_start + pd.DateOffset(years=5) - pd.DateOffset(days=1) 
    if current_chunk_end > full_data_end:
        current_chunk_end = full_data_end # Adjust for the last partial chunk

    # Filter data for the current chunk
    chunk_idx = avg_corrs_aligned.index[(avg_corrs_aligned.index >= current_chunk_start) & (avg_corrs_aligned.index <= current_chunk_end)]
    
    if not chunk_idx.empty:
        title_suffix = f"{current_chunk_start.strftime('%Y-%m-%d')} to {current_chunk_end.strftime('%Y-%m-%d')}"
        
        # Save standard dual-chart (FX + ES)
        filename_dual = f"FX_Regimes_{current_chunk_start.strftime('%Y%m%d')}_{current_chunk_end.strftime('%Y%m%d')}.png"
        save_path_dual = os.path.join(script_dir, filename_dual)
        plot_regime_analysis(avg_corrs_aligned.loc[chunk_idx], regimes_aligned.loc[chunk_idx], 
                             es_prices.loc[chunk_idx], title_suffix, save_path=save_path_dual)
        
        # Save FX-only chart
        filename_only = f"FX_Only_Regimes_{current_chunk_start.strftime('%Y%m%d')}_{current_chunk_end.strftime('%Y%m%d')}.png"
        save_path_only = os.path.join(script_dir, filename_only)
        plot_fx_only_analysis(avg_corrs_aligned.loc[chunk_idx], regimes_aligned.loc[chunk_idx], 
                              title_suffix, save_path=save_path_only)

    current_chunk_start = current_chunk_start + pd.DateOffset(years=5)

plot_prevalence(stats, script_dir)
plot_duration_distribution(regimes_aligned, script_dir)

# Final output
print(f"Latest Date Recorded: {latest_date.strftime('%Y-%m-%d')}")
print(f"Current Regime: {latest_regime} ({regime_label})")
print(f"Summary Stats:")
for r in [0, 1]:
    print(f"  Regime {r}: Avg Corr {stats[r]['avg_corr']:.4f}, {stats[r]['percentage']:.2f}% time, Avg Duration {stats[r]['avg_duration']:.1f} days")
print(f"Transition Matrix (Probability of moving between states):")
print(model.transmat_)


# %%
