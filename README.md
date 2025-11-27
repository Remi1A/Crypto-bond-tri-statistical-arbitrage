# Tri-statistical-Arbitrage — BTC, ETH & US 30Y

This project implements a **rotating tri-arbitrage backtest** between **BTC**, **ETH** and the **US 30Y Treasury yield**.  
For each asset chosen as the *target*, the two others are used as *hedges*.

The idea is:

- Fit an **OLS plane on price levels** to explain the target asset using the two others:
  \[
  \text{Target} \approx \alpha + \beta_1 \cdot \text{Feat1} + \beta_2 \cdot \text{Feat2}
  \]
- The **residual** of this plane is the spread \(S_t\).  
- Compute a **static z-score** of this spread and use it as a **mean-reversion signal**.
- On returns, fit another OLS:
  \[
  r_\text{Target} \approx a + \gamma_1 \cdot r_\text{Feat1} + \gamma_2 \cdot r_\text{Feat2}
  \]
  to build a **hedged portfolio**.
- Sweep multiple **z-score thresholds k** and evaluate performance (Sharpe, MDD, etc.), **including transaction costs**.

The code is written in Python and focuses on the post-ETH ETF period starting on `2024-07-23`.

> 🇫🇷 Note: some comments and prints are in French; the logic is fully understandable in English.

---

## Features

- Tri-arbitrage between **ETH, BTC, US30Y**:
  - Targets: `["ETH", "BTC", "US30Y"]`
  - Hedges: the two other assets each time.
- **Price plane (levels)**:
  - OLS regression Target ~ 1 + Feat1 + Feat2
  - Residuals used as spread \(S_t\)
  - Static z-score normalization
- **Hedged returns**:
  - OLS regression on returns to get hedge ratios (γ)
  - Daily P&L with **transaction costs** per asset
- **Backtest engine**:
  - Long/short entry when |z| ≥ k
  - Exit when z crosses back through 0
  - Sweep over thresholds `k` (e.g. 0.50 to 3.00 by 0.01)
  - Compute:
    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Max drawdown
    - Number of trades
- **Visualizations**:
  - Z-scores of BTC, ETH, US30Y (post split date)
  - Sharpe vs threshold for each target
  - Equity curves for the best threshold per target
  - 3D OLS plane of the best strategy with residual-based point sizes and colors
  - BTC spread “extremes”: rolling 95th percentile of |z| with trend and non-parametric tests
- **Diagnostics**:
  - Quick correlation matrix (BTC, ETH, US30Y) post-ETF
  - Printed OLS coefficients (β on levels, γ on returns) for the BTC strategy with a simple explanation
