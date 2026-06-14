# FX Market Cohesion & Regime Detection

## Overview
This script identifies market regimes by analyzing the "cohesion" of the foreign exchange market. It defines cohesion as the average pairwise correlation across major USD currency pairs. By applying a **Hidden Markov Model (HMM)**, the tool segments historical data into two distinct states: **High Correlation** and **Low Correlation**.

## Methodology
1.  **Data Acquisition**: Downloads daily Close prices for 9 major currency pairs (AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK vs. USD) and S&P 500 E-mini futures (ES=F) using `yfinance`.
2.  **Feature Engineering**: 
    *   Computes rolling 20-day log returns.
    *   Calculates the average pairwise correlation across all pairs to create a "Market Cohesion" index.
3.  **Regime Modeling**: Uses a 2-state **Gaussian Hidden Markov Model** to automatically detect transitions between regimes without manual thresholding.
4.  **Labeling**: Dynamically assigns "High Correlation" and "Low Correlation" labels based on the statistical means of the detected hidden states.

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install yfinance pandas numpy matplotlib hmmlearn
```

## How to Use
Simply run the script from your terminal:
```bash
python FXRegimes.py
```

## Outputs
The script generates several visualizations to analyze the relationship between FX cohesion and equity performance:
*   **FX_Regimes_[Dates].png**: Dual-axis charts showing FX correlations and ES prices, shaded by the detected regime.
*   **FX_Only_Regimes_[Dates].png**: A focused view of FX correlation clusters over 5-year windows.
*   **Summary_Regime_Prevalence.png**: A bar chart showing the percentage of time the market spends in each state.
*   **Summary_Duration_Distribution.png**: A histogram showing the persistence (how many days a regime typically lasts) of market states.

## Key Statistics
The console output provides:
*   **Transition Matrix**: The probability of the market staying in the current regime vs. switching to another.
*   **Prevalence**: How often each regime occurs.
*   **Average Duration**: The expected length (in days) of each market environment.

---
**Note**: This tool is designed for quantitative research purposes. High FX correlation is often associated with "Risk-Off" environments or USD-driven volatility.
