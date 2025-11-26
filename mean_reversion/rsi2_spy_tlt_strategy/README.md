# RSI2 SPY-TLT Strategy

## Overview
A mean-reversion strategy using RSI(2) on SPY (S&P 500) and TLT (Long-Term Treasuries).
It buys when RSI(2) is oversold and sells when overbought.

## Mathematical Foundation
- **Indicators**: 2-day RSI.
- **Logic**:
    - SPY: Buy if RSI(2) < 40, Sell if RSI(2) > 80.
    - TLT: Buy if RSI(2) < 30, Sell if RSI(2) > 60.
- **Allocation**: 50% SPY, 50% TLT (when signals trigger).

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| rsi_period | 2 | RSI Period |
| spy_os | 40 | SPY Oversold Threshold |
| spy_ob | 80 | SPY Overbought Threshold |
| tlt_os | 30 | TLT Oversold Threshold |
| tlt_ob | 60 | TLT Overbought Threshold |

## Quick Start
```bash
cd mean_reversion/rsi2_spy_tlt_strategy
pip install -r requirements.txt
python backtest.py
```

## Dependencies
- pandas
- pandas_ta
- yfinance

## Author
- Original: Quant Prophet, LLC (Kory Hoang)
- Adapted by: ArbiterLabs
