# RSI2 Mean Reversion

Classic mean reversion strategy using RSI(2) on SPY and TLT.

## Performance
- Return: 195%
- Sharpe: 1.19
- Max Drawdown: 13.15%

## Strategy
Buys oversold assets and sells when overbought using 2-day RSI.

### Signals
- **SPY Buy**: RSI(2) < 40
- **SPY Sell**: RSI(2) > 80
- **TLT Buy**: RSI(2) < 30
- **TLT Sell**: RSI(2) > 60

### Position Sizing
- 50% allocation to each asset
- Optional 1-2x leverage

## Usage

```python
import yaml
from strategy import RSI2Strategy

with open('config.yaml') as f:
    config = yaml.safe_load(f)

strategy = RSI2Strategy(config)
```

## Configuration

Edit `config.yaml`:
- `equity_oversold/overbought`: RSI thresholds for SPY
- `bond_oversold/overbought`: RSI thresholds for TLT
- `leverage`: 1.0 for vanilla, 2.0 for higher returns

## Notes

Developed by Cesar Alvarez. RSI(2) is more sensitive than RSI(14) for short-term timing. Works best in mean-reverting markets.

## Files
- `strategy.py`: Core implementation
- `config.yaml`: Parameters

Educational purposes only. Not financial advice.
