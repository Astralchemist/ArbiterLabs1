# XIV/VXX Ballistic Volatility Strategy

**Performance Metrics (Quantopian Backtest):**
- Return: 199%
- Sharpe Ratio: 1.78
- Max Drawdown: 8.9%

## Overview

This is a high-performance volatility trading strategy that trades XIV (inverse VIX) and VXX (long VIX) ETFs based on RSI signals calculated on 2-hour bars. The strategy uses a conservative approach with 20% allocation to volatility instruments and 80% to treasury bonds (IEI).

## Strategy Logic

### Core Concept
- **Long XIV**: When volatility is oversold (RSI crosses above 70)
- **Long VXX**: When volatility is overbought (RSI crosses below 85) and RSI5 < 70
- **Exit**: RSI-based reversal signals or stop-loss triggers

### Technical Indicators
- **RSI(2)**: Primary signal generator on 2-hour bars
- **RSI(5)**: Filter for VXX entries
- **Stop Loss**: 25% for XIV, 1% for VXX
- **Take Profit**: 50% for VXX

### Position Management
- 20% volatility allocation (XIV or VXX)
- 80% bond allocation (IEI/TLT)
- Panic button: Exit XIV if price drops >10% from recent high

## Modern Adaptation

### Changes from Quantopian Version
1. **Data Source**: Replaced `data.history()` with yfinance
2. **Scheduling**: Replaced `schedule_function()` with time-based logic
3. **Symbols**: Updated to modern equivalents (XIV delisted in 2018)
4. **Execution**: Simplified order management without Quantopian's order system

### Important Notes
⚠️ **XIV was delisted in February 2018** after the Volmageddon event. Modern alternatives:
- **SVXY**: ProShares Short VIX Short-Term Futures ETF (0.5x short)
- **VIXM**: ProShares VIX Mid-Term Futures ETF
- Consider using VIX futures directly

## Configuration

Key parameters in `config.yaml`:
```yaml
parameters:
  xiv_stop_loss_pct: 0.25
  vxx_stop_loss_pct: 0.01
  vxx_take_profit_pct: 0.50
  vol_allocation: 0.20
  bond_allocation: 0.80
  rsi_period: 2
  rsi_long_period: 5
```

## Usage

```python
import yaml
from strategy import XIVVXXStrategy
from backtest import run_backtest

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run backtest
results = run_backtest(config)
```

## Risk Warnings

⚠️ **High Risk Strategy**
- Volatility products are extremely volatile
- XIV experienced a 90%+ loss in a single day (Feb 5, 2018)
- Not suitable for risk-averse investors
- Requires active monitoring

## License

Educational purposes only. Not financial advice.
