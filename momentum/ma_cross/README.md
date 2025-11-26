# MA Crossover Strategy

Simple moving average crossover for educational purposes.

## Strategy
Basic trend-following using MA crossovers.

### Signals
- **Buy**: Short MA crosses above long MA
- **Sell**: Short MA crosses below long MA

### Default Parameters
- Short MA: 20 days
- Long MA: 60 days
- Asset: SPY

## Usage

```python
import yaml
from strategy import MACrossoverStrategy

with open('config.yaml') as f:
    config = yaml.safe_load(f)

strategy = MACrossoverStrategy(config)
```

## Configuration

Edit `config.yaml`:
- `short_window`: Fast moving average period
- `long_window`: Slow moving average period
- `symbol`: Asset to trade

## Learning Path

1. **Beginner**: Understand basic logic, run backtest
2. **Intermediate**: Add stop losses, position sizing
3. **Advanced**: Add filters, optimize parameters

## Notes

Simple strategy for learning. Moderate returns in trending markets, whipsaws in choppy conditions. Lagging indicator with late entries/exits.

## Files
- `strategy.py`: Core implementation
- `config.yaml`: Parameters

Educational purposes only. Not financial advice.
