# Core-Satellite Strategy

Strategic and tactical asset allocation combining static and momentum portfolios.

## Performance
- Return: 305%
- Sharpe: 1.21
- Max Drawdown: 13%

## Strategy
Combines fixed-weight core portfolio with momentum-based satellite portfolio.

### Core Portfolio (25%)
Static allocation rebalanced weekly:
- QQQ: 25%
- XLP: 25%
- TLT: 25%
- IEF: 25%

### Satellite Portfolio (75%)
Momentum rotation among XLV, XLY, TLT, GLD.

Entry: MA(20) > MA(200)

## Usage

```python
import yaml
from strategy import CoreSatelliteStrategy

with open('config.yaml') as f:
    config = yaml.safe_load(f)

strategy = CoreSatelliteStrategy(config)
```

## Configuration

Edit `config.yaml`:
- `core_weight`: Allocation to core portfolio
- `satellite_weight`: Allocation to satellite portfolio
- `ma_fast/ma_slow`: Moving average periods

## Notes

Weekly rebalancing on Monday. Diversified across asset classes with momentum overlay.

## Files
- `strategy.py`: Core implementation
- `config.yaml`: Parameters

Educational purposes only. Not financial advice.
