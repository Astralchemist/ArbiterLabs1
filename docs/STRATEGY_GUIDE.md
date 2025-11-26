# Strategy Development Guide

A comprehensive guide to building and deploying quantitative trading strategies in ArbiterLabs.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Strategy Architecture](#strategy-architecture)
3. [Implementation Steps](#implementation-steps)
4. [Testing Your Strategy](#testing-your-strategy)
5. [Best Practices](#best-practices)
6. [Common Pitfalls](#common-pitfalls)

---

## Getting Started

### Prerequisites
- Python 3.8+
- Basic understanding of quantitative finance
- Familiarity with pandas and numpy

### Setup
```bash
# Clone and install base dependencies
git clone https://github.com/yourusername/arbiterlabs.git
cd arbiterlabs
pip install -r requirements-base.txt
```

---

## Strategy Architecture

Every strategy in ArbiterLabs follows a standard structure:

```
strategy_name/
├── README.md           # Documentation
├── requirements.txt    # Dependencies
├── config.yaml         # Configuration
├── strategy.py         # Core logic
├── backtest.py         # Backtesting
├── live.py            # Live trading
├── optimize.py        # Parameter optimization (optional)
├── data/              # Sample data (optional)
├── results/           # Backtest results (optional)
└── tests/             # Unit tests
```

### Key Components

#### 1. Strategy Class (`strategy.py`)
The core of your strategy with these methods:
- `__init__()` - Initialize strategy parameters
- `generate_signals()` - Create buy/sell signals
- `calculate_position_size()` - Determine position sizing
- `execute_trade()` - Execute trades
- `check_exit_conditions()` - Manage exits

#### 2. Configuration (`config.yaml`)
All strategy parameters in one place:
```yaml
strategy:
  name: "your_strategy"
  version: "1.0.0"

parameters:
  # Your strategy parameters here

risk:
  max_position_size: 0.1
  max_drawdown_exit: 0.15

data:
  symbols: ["AAPL"]
  timeframe: "1d"
```

#### 3. Backtest Script (`backtest.py`)
Runs historical simulations to validate strategy performance.

#### 4. Live Trading Script (`live.py`)
Connects to brokers and executes trades in real-time.

---

## Implementation Steps

### Step 1: Choose a Template

Start with the strategy template:
```bash
cp -r _templates/strategy_template/ <category>/<your_strategy>/
cd <category>/<your_strategy>/
```

### Step 2: Define Your Edge

Document your strategy's edge hypothesis:
- What market inefficiency are you exploiting?
- Why should this work?
- What are the assumptions?

### Step 3: Implement Signal Generation

```python
def generate_signals(self, data: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals.

    Returns:
        Series with 1 (buy), -1 (sell), 0 (hold)
    """
    # Example: Simple moving average crossover
    data['SMA_fast'] = data['close'].rolling(20).mean()
    data['SMA_slow'] = data['close'].rolling(50).mean()

    signals = pd.Series(0, index=data.index)
    signals[data['SMA_fast'] > data['SMA_slow']] = 1
    signals[data['SMA_fast'] < data['SMA_slow']] = -1

    return signals
```

### Step 4: Add Risk Management

```python
def calculate_position_size(self, signal: int, current_price: float) -> float:
    """
    Calculate position size with risk management.
    """
    # Use shared utilities
    from _shared.risk.position_sizing import volatility_adjusted_size
    from _shared.risk.kelly_criterion import kelly_criterion

    # Calculate appropriate size
    size = volatility_adjusted_size(
        self.portfolio_value,
        target_volatility=0.15,
        asset_volatility=self.calculate_volatility()
    )

    return size
```

### Step 5: Implement Exit Logic

```python
def check_exit_conditions(self, data: pd.DataFrame, current_idx: int) -> bool:
    """
    Check if we should exit the position.
    """
    if self.position == 0:
        return False

    # Example: Exit on stop loss or take profit
    entry_price = self.entry_price
    current_price = data['close'].iloc[current_idx]

    pnl_pct = (current_price - entry_price) / entry_price

    # Stop loss at -2%
    if pnl_pct < -0.02:
        return True

    # Take profit at +5%
    if pnl_pct > 0.05:
        return True

    return False
```

### Step 6: Write Tests

```python
import unittest
from strategy import Strategy

class TestStrategy(unittest.TestCase):
    def test_signal_generation(self):
        # Test that signals are generated correctly
        pass

    def test_position_sizing(self):
        # Test position sizing logic
        pass
```

### Step 7: Run Backtest

```bash
python backtest.py
```

### Step 8: Optimize Parameters

```bash
python optimize.py --metric sharpe --trials 1000
```

---

## Testing Your Strategy

### Unit Tests
Test individual components:
```bash
python -m pytest tests/
```

### Walk-Forward Testing
Validate on out-of-sample data:
```python
# Split data into training and testing periods
train_data = data['2020':'2022']
test_data = data['2023':]

# Train on historical data
strategy.optimize(train_data)

# Test on unseen data
results = strategy.backtest(test_data)
```

### Paper Trading
Test in live market conditions without risking capital:
```bash
python live.py --mode paper
```

---

## Best Practices

### 1. Avoid Overfitting
- Use walk-forward analysis
- Limit the number of parameters
- Test on multiple time periods and assets
- Use simple strategies when possible

### 2. Transaction Costs
Always include realistic costs:
```yaml
execution:
  slippage_bps: 5      # 0.05% slippage
  commission_bps: 10   # 0.10% commission
```

### 3. Data Quality
- Check for missing data
- Handle corporate actions (splits, dividends)
- Validate data ranges
- Be careful with delisted stocks (survivorship bias)

### 4. Risk Management
- Never risk more than 2% per trade
- Use stop losses
- Diversify across strategies and assets
- Monitor drawdowns

### 5. Version Control
- Use git for all strategies
- Document all changes
- Tag stable versions

### 6. Logging
```python
from _shared.utils.logger import get_strategy_logger

logger = get_strategy_logger('my_strategy')
logger.info("Trade executed: ...")
```

---

## Common Pitfalls

### ❌ Lookahead Bias
Don't use future information:
```python
# WRONG - uses future data
data['signal'] = data['close'].shift(-1) > data['close']

# CORRECT - uses only past data
data['signal'] = data['close'] > data['close'].shift(1)
```

### ❌ Survivorship Bias
Don't backtest only on stocks that still exist today.

### ❌ Data Snooping
Don't repeatedly test on the same data until you find something that works.

### ❌ Ignoring Transaction Costs
Even small costs can eliminate edge:
```python
# Always include realistic costs
gross_return = 15%  # Before costs
slippage = 0.05%    # Per trade
commission = 0.10%  # Per trade
num_trades = 100    # Per year

# Net return after costs
net_return = 15% - (0.15% × 100) = 0%  # No profit!
```

### ❌ Overfitting
Too many parameters = curve fitting:
```python
# WRONG - too many parameters
params = {
    'sma1': 17, 'sma2': 42, 'sma3': 87,
    'rsi_period': 13, 'rsi_upper': 71.3,
    'macd_fast': 11, 'macd_slow': 27
}

# BETTER - keep it simple
params = {
    'lookback': 20,
    'threshold': 2.0
}
```

---

## Performance Metrics

Always calculate these metrics:

| Metric | Formula | Good Value |
|--------|---------|------------|
| Sharpe Ratio | (Return - RFR) / Std Dev | > 1.0 |
| Max Drawdown | Largest peak-to-trough | < 20% |
| Win Rate | Wins / Total Trades | > 50% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Calmar Ratio | Annual Return / Max DD | > 1.0 |

---

## Next Steps

1. Read [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md)
2. Review [Backtesting Best Practices](BACKTESTING_BEST_PRACTICES.md)
3. Study existing strategies in the repository
4. Start with simple strategies first
5. Paper trade before going live

---

## Resources

- **Books**:
  - "Quantitative Trading" by Ernest Chan
  - "Algorithmic Trading" by Jeffrey Bacidore
  - "Advances in Financial Machine Learning" by Marcos López de Prado

- **Papers**:
  - SSRN Finance eLibrary
  - arXiv Quantitative Finance

- **Communities**:
  - QuantConnect Community
  - QuantopianForums Archive
  - Reddit r/algotrading

---

**Remember**: Past performance does not guarantee future results. Always start with paper trading!
