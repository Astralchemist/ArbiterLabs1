# Q1500 Mean Reversion Strategy

## Overview
This strategy is a classic mean reversion algorithm originally designed for the Quantopian platform. It identifies high dollar-volume stocks (liquidity filter) and ranks them based on their recent 5-day returns. The hypothesis is that top-performing stocks from the last week will underperform this week, and vice-versa.

## Mathematical Foundation
- **Universe**: Q1500US (Top 1500 US stocks by market cap/liquidity).
- **Signal**: 5-day simple returns.
- **Ranking**:
    - **Long**: Bottom 10% of stocks (Worst performers).
    - **Short**: Top 10% of stocks (Best performers).
- **Weighting**: Equal weights for all positions.
- **Rebalancing**: Weekly (First trading day of the week).

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| lookback_window | 5 | Days to calculate returns |
| long_percentile | 10 | Percentile for long entries (0-10) |
| short_percentile | 90 | Percentile for short entries (90-100) |
| leverage | 1.0 | Gross leverage (0.5 long, 0.5 short) |

## Quick Start
```bash
cd mean_reversion/q1500_mean_reversion
pip install -r requirements.txt
python backtest.py
```

## Dependencies
- pandas
- numpy
- zipline-reloaded (optional, for full backtest)
- yfinance (for data)

## Author
- Original: Quantopian Community
- Adapted by: ArbiterLabs
