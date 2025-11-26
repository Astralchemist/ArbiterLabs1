# Shared Utilities

Reusable components for all trading strategies in ArbiterLabs.

---

## 📁 Structure

```
_shared/
├── data_loaders/       # Data fetching from various sources
├── execution/          # Broker execution interfaces
├── metrics/            # Performance metrics calculations
├── risk/               # Risk management and position sizing
└── utils/              # General utilities
```

---

## 📊 Data Loaders

### Available Loaders

1. **Yahoo Finance** (`yfinance_loader.py`)
   - US stocks, ETFs, indices
   - Free historical data
   - No API key required

2. **Binance** (`binance_loader.py`)
   - Cryptocurrency spot and futures
   - Historical klines
   - Order book data
   - Current prices

3. **MetaTrader 5** (`mt5_loader.py`)
   - Forex pairs
   - Stocks, indices, commodities
   - Tick data support
   - Requires MT5 terminal

4. **CSV** (`csv_loader.py`)
   - Load from local files
   - Flexible column mapping
   - Data validation

### Usage Examples

```python
# Universal loader
from _shared.data_loaders import load_data

# Yahoo Finance
data = load_data('yfinance', 'AAPL', '2023-01-01', '2023-12-31', '1d')

# Binance
data = load_data('binance', 'BTCUSDT', '2023-01-01', '2023-12-31', '1h')

# MetaTrader 5
data = load_data('mt5', 'EURUSD', '2023-01-01', '2023-12-31', 'H1')

# CSV
data = load_data('csv', 'mydata.csv', date_column='Date')
```

### Individual Loaders

```python
# Yahoo Finance
from _shared.data_loaders import yfinance_loader
data = yfinance_loader.load_data(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31')

# Binance with API
from _shared.data_loaders import binance_loader
data = binance_loader.load_data('BTCUSDT', '2023-01-01', '2023-12-31', '1d')
current_price = binance_loader.get_current_price('BTCUSDT')

# MT5
from _shared.data_loaders import mt5_loader
data = mt5_loader.load_data('EURUSD', '2023-01-01', '2023-12-31', 'H1')
bid, ask, last = mt5_loader.get_current_price('EURUSD')
```

---

## 🔄 Execution Modules

### Broker Base Class

All brokers implement the `BrokerBase` interface:

```python
from _shared.execution.broker_base import BrokerBase, Order, OrderSide, OrderType

# Methods available on all brokers:
# - connect() / disconnect()
# - get_account()
# - get_positions()
# - place_order(order)
# - cancel_order(order_id)
# - close_position(symbol)
# - close_all_positions()
```

### Available Executors

1. **Alpaca** - US stocks trading
2. **Binance** - Cryptocurrency trading
3. **Paper Trader** - Simulated trading

### Usage Examples

```python
# Alpaca (US Stocks)
from _shared.execution.alpaca_executor import AlpacaExecutor

config = {
    'api_key': 'YOUR_KEY',
    'api_secret': 'YOUR_SECRET',
    'base_url': 'https://paper-api.alpaca.markets'
}

broker = AlpacaExecutor(config)
broker.connect()

# Place market order
order_id = broker.buy_market('AAPL', 10)

# Get account info
account = broker.get_account()
print(f"Equity: ${account.equity:,.2f}")

# Close all positions
broker.close_all_positions()
```

```python
# Binance (Crypto)
from _shared.execution.binance_executor import BinanceExecutor

config = {
    'api_key': 'YOUR_KEY',
    'api_secret': 'YOUR_SECRET',
    'testnet': True  # Paper trading
}

broker = BinanceExecutor(config)
broker.connect()

order_id = broker.buy_market('BTCUSDT', 0.01)
```

```python
# Paper Trading (Simulation)
from _shared.execution.paper_trader import PaperTrader

config = {
    'initial_balance': 100000,
    'slippage_bps': 5,
    'commission_bps': 10
}

broker = PaperTrader(config)
broker.connect()

# Set price function
broker.set_price_function(lambda symbol: data[symbol]['close'].iloc[-1])

# Trade normally
order_id = broker.buy_market('AAPL', 100)

# Get performance
summary = broker.get_performance_summary()
print(summary)
```

---

## 📈 Performance Metrics

Calculate standard trading metrics:

```python
from _shared.metrics.performance import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, win_rate, profit_factor,
    calculate_all_metrics
)

# Individual metrics
sharpe = sharpe_ratio(returns)
max_dd = max_drawdown(returns)

# All metrics at once
metrics = calculate_all_metrics(returns)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
```

---

## 🎯 Risk Management

### Position Sizing

```python
from _shared.risk.position_sizing import (
    fixed_position_size,
    volatility_adjusted_size,
    atr_based_size
)

# Fixed percentage
size = fixed_position_size(capital=100000, position_pct=0.1)

# Volatility adjusted
size = volatility_adjusted_size(
    capital=100000,
    target_volatility=0.15,
    asset_volatility=0.25
)

# ATR-based
shares = atr_based_size(
    capital=100000,
    price=150,
    atr=5.0,
    risk_per_trade=0.02
)
```

### Kelly Criterion

```python
from _shared.risk.kelly_criterion import (
    kelly_criterion,
    kelly_from_sharpe,
    half_kelly
)

# From win rate and win/loss ratio
kelly_pct = kelly_criterion(
    win_rate=0.55,
    win_loss_ratio=2.0
)

# From Sharpe ratio
kelly_pct = kelly_from_sharpe(sharpe_ratio=1.5)

# Half Kelly (more conservative)
kelly_pct = half_kelly(win_rate=0.55, win_loss_ratio=2.0)
```

---

## 🛠️ Utilities

### Logging

```python
from _shared.utils.logger import get_strategy_logger

logger = get_strategy_logger('my_strategy')
logger.info("Strategy started")
logger.warning("High drawdown detected")
logger.error("API connection failed")
```

### Configuration

```python
from _shared.utils.config_loader import load_config, save_config

# Load YAML config
config = load_config('config.yaml')

# Save config
save_config(config, 'output_config.yaml')

# Validate config
from _shared.utils.config_loader import validate_config
validate_config(config, ['strategy.name', 'data.symbols'])
```

---

## 📦 Installation

### Required Dependencies

```bash
# Base dependencies
pip install numpy pandas scipy matplotlib pyyaml

# Data loaders
pip install yfinance python-binance MetaTrader5 ccxt

# Execution
pip install alpaca-py python-binance

# Testing
pip install pytest pytest-cov
```

### Optional Dependencies

```bash
# For specific brokers/data sources
pip install interactive-brokers-api  # Interactive Brokers
pip install polygon-api-client       # Polygon.io
```

---

## 🧪 Testing

All modules include error handling and validation:

```python
# Data loaders handle missing dependencies gracefully
try:
    from _shared.data_loaders import binance_loader
    data = binance_loader.load_data(...)
except ImportError:
    print("Install python-binance: pip install python-binance")
```

---

## 📚 Examples

### Complete Strategy Example

```python
from _shared.data_loaders import load_data
from _shared.execution.paper_trader import PaperTrader
from _shared.metrics.performance import calculate_all_metrics
from _shared.risk.position_sizing import volatility_adjusted_size

# 1. Load data
data = load_data('yfinance', 'AAPL', '2023-01-01', '2023-12-31', '1d')

# 2. Setup paper trading
broker = PaperTrader({'initial_balance': 100000})
broker.connect()
broker.set_price_function(lambda s: data['close'].iloc[-1])

# 3. Generate signals (your strategy logic)
data['signal'] = 0
data.loc[data['close'] > data['close'].rolling(50).mean(), 'signal'] = 1

# 4. Execute trades
for i in range(len(data)):
    if data['signal'].iloc[i] == 1:
        # Calculate position size
        size = volatility_adjusted_size(
            capital=broker.cash,
            target_volatility=0.15,
            asset_volatility=data['close'].pct_change().std()
        )
        shares = int(size / data['close'].iloc[i])

        # Place order
        broker.buy_market('AAPL', shares)

# 5. Analyze results
summary = broker.get_performance_summary()
print(f"Total Return: {summary['total_return_pct']:.2f}%")
```

---

## 🤝 Contributing

To add new shared utilities:

1. Create module in appropriate subdirectory
2. Add comprehensive docstrings
3. Include usage examples
4. Write unit tests
5. Update this README

---

## 📝 Notes

- All data loaders return standardized DataFrames with OHLCV columns
- All executors implement the same `BrokerBase` interface
- Metrics assume returns as pandas Series or numpy arrays
- Risk functions return position sizes in dollars or shares

---

**Built for ArbiterLabs** - Production-ready quantitative trading utilities
