# Data Loaders Documentation

Complete reference for all available data loaders in ArbiterLabs.

---

## 📊 Available Data Sources

| Loader | Asset Classes | Free | API Key Required | Features |
|--------|---------------|------|------------------|----------|
| **yfinance** | Stocks, ETFs, Indices, Options | ✅ Yes | ❌ No | Historical, Real-time, Fundamentals, Options |
| **Binance** | Cryptocurrency | ✅ Yes | ⚠️ Optional* | Spot, Futures, Orderbook, Multi-exchange |
| **MetaTrader 5** | Forex, CFDs, Stocks | ✅ Yes | ❌ No | Tick data, Multiple timeframes |
| **CSV** | Any (local files) | ✅ Yes | ❌ No | Validation, Cleaning, Resampling |

*API key optional for public data, required for trading

---

## 🌐 Yahoo Finance (yfinance_loader.py)

### Features
- ✅ US & International stocks
- ✅ ETFs, Mutual Funds, Indices
- ✅ Cryptocurrencies
- ✅ Fundamental data
- ✅ Options chains
- ✅ Dividends & splits
- ✅ No API key required

### Basic Usage

```python
from _shared.data_loaders import yfinance_loader

# Single symbol
data = yfinance_loader.load_data('AAPL', '2023-01-01', '2023-12-31', '1d')

# Multiple symbols
data = yfinance_loader.load_data(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31')

# Recent data (using period)
data = yfinance_loader.load_recent_data('AAPL', period='1mo', interval='1d')
```

### Advanced Features

```python
# Get current price
price = yfinance_loader.get_current_price('AAPL')

# Company information
info = yfinance_loader.get_info('AAPL')
print(info['marketCap'], info['sector'])

# Fundamental data
fundamentals = yfinance_loader.get_fundamentals('AAPL')
print(fundamentals['financials'])  # Income statement
print(fundamentals['balance_sheet'])
print(fundamentals['cashflow'])

# Dividends and splits
dividends = yfinance_loader.get_dividends('AAPL')
splits = yfinance_loader.get_splits('AAPL')

# Options chain
options = yfinance_loader.get_options_chain('AAPL')
print(options['calls'])
print(options['puts'])

# Available expiration dates
dates = yfinance_loader.get_available_options_dates('AAPL')

# Bulk download (faster for multiple symbols)
data = yfinance_loader.download_bulk(
    ['AAPL', 'MSFT', 'GOOGL'],
    '2023-01-01',
    '2023-12-31',
    threads=True
)

# Validate symbol
is_valid = yfinance_loader.validate_symbol('AAPL')
```

### Available Intervals
- `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m` (intraday)
- `1h` (hourly)
- `1d` (daily)
- `5d` (5 days)
- `1wk` (weekly)
- `1mo`, `3mo` (monthly)

### Periods for Recent Data
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

---

## 🪙 Binance (binance_loader.py)

### Features
- ✅ Spot & Futures markets
- ✅ Historical klines
- ✅ Real-time prices
- ✅ Order book data
- ✅ Multi-exchange support (via CCXT)

### Basic Usage

```python
from _shared.data_loaders import binance_loader

# Historical data (using python-binance)
data = binance_loader.load_data(
    'BTCUSDT',
    '2023-01-01',
    '2023-12-31',
    interval='1h'
)

# Using CCXT (supports multiple exchanges)
data = binance_loader.load_data_ccxt(
    'BTC/USDT',
    '2023-01-01',
    '2023-12-31',
    interval='1h',
    exchange='binance'  # or 'coinbase', 'kraken', etc.
)
```

### Spot vs Futures

```python
# Spot market
spot_data = binance_loader.load_spot_data('BTCUSDT', interval='1h', lookback_days=30)

# Futures market
futures_data = binance_loader.load_futures_data('BTCUSDT', interval='1h', lookback_days=30)
```

### Real-time Data

```python
# Current price
price = binance_loader.get_current_price('BTCUSDT')

# Order book
orderbook = binance_loader.get_orderbook('BTCUSDT', limit=100)
print(orderbook['bids'])  # DataFrame with bids
print(orderbook['asks'])  # DataFrame with asks

# All symbols
symbols = binance_loader.get_all_symbols()
print(f"Total symbols: {len(symbols)}")
```

### Available Intervals
- `1m`, `3m`, `5m`, `15m`, `30m` (minutes)
- `1h`, `2h`, `4h`, `6h`, `8h`, `12h` (hours)
- `1d`, `3d` (days)
- `1w` (week)
- `1M` (month)

---

## 💱 MetaTrader 5 (mt5_loader.py)

### Requirements
- MetaTrader 5 terminal installed
- Terminal must be running

### Features
- ✅ Forex pairs
- ✅ Stocks, Indices, Commodities
- ✅ Tick-by-tick data
- ✅ Multiple timeframes
- ✅ Symbol information

### Basic Usage

```python
from _shared.data_loaders import mt5_loader

# Check connection
mt5_loader.check_mt5_connection()

# Historical data
data = mt5_loader.load_data('EURUSD', '2023-01-01', '2023-12-31', 'H1')

# Recent data
data = mt5_loader.load_recent_data('EURUSD', timeframe='H1', num_bars=1000)
```

### Real-time Data

```python
# Current prices
bid, ask, last = mt5_loader.get_current_price('EURUSD')

# Symbol information
info = mt5_loader.get_symbol_info('EURUSD')
print(info['spread'], info['point'], info['digits'])

# All symbols
all_symbols = mt5_loader.get_all_symbols()

# Forex symbols only
forex_symbols = mt5_loader.get_forex_symbols()
```

### Tick Data

```python
# Load tick data (use with caution - can be very large)
ticks = mt5_loader.load_tick_data('EURUSD', '2023-12-01', '2023-12-02')
```

### Available Timeframes
- `M1`, `M5`, `M15`, `M30` (minutes)
- `H1`, `H4` (hours)
- `D1` (daily)
- `W1` (weekly)
- `MN1` (monthly)

---

## 📄 CSV Files (csv_loader.py)

### Features
- ✅ Load local CSV files
- ✅ Data validation
- ✅ Data cleaning
- ✅ Column standardization
- ✅ Resampling
- ✅ Folder loading

### Basic Usage

```python
from _shared.data_loaders import csv_loader

# Single file
data = csv_loader.load_data('prices.csv', date_column='Date')

# Multiple files
file_paths = ['AAPL.csv', 'MSFT.csv', 'GOOGL.csv']
data_dict = csv_loader.load_multiple_csv(file_paths, date_column='Date')

# Load entire folder
data_dict = csv_loader.load_from_folder('data/stocks/', pattern='*.csv')
```

### Validation & Cleaning

```python
# Validate OHLCV columns
is_valid = csv_loader.validate_ohlcv_columns(data)

# Comprehensive validation
result = csv_loader.validate_ohlcv_data(data)
if not result['valid']:
    print(result['issues'])

# Clean data
clean_data = csv_loader.clean_ohlcv_data(
    data,
    drop_invalid=True,
    fill_missing=True
)
```

### Data Processing

```python
# Standardize column names
data = csv_loader.convert_column_names(data)

# Custom mapping
mapping = {'Open': 'open', 'Close': 'close', 'Volume': 'vol'}
data = csv_loader.convert_column_names(data, mapping)

# Resample to different frequency
daily_data = csv_loader.resample_data(data, freq='D')
weekly_data = csv_loader.resample_data(data, freq='W')
monthly_data = csv_loader.resample_data(data, freq='M')

# Merge multiple sources
merged = csv_loader.merge_multiple_sources(
    data_dict,
    column='close',
    join='inner'
)
```

### Saving Data

```python
# Save to CSV
csv_loader.save_to_csv(data, 'output/processed_data.csv', include_index=True)
```

---

## 🔄 Universal Loader

Use the universal loader to switch between data sources easily:

```python
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

---

## 📦 Installation

```bash
# Base dependencies
pip install pandas numpy

# Yahoo Finance
pip install yfinance

# Binance
pip install python-binance ccxt

# MetaTrader 5
pip install MetaTrader5
```

---

## 🎯 Best Practices

### 1. Error Handling

```python
try:
    data = yfinance_loader.load_data('AAPL', '2023-01-01', '2023-12-31')
except Exception as e:
    print(f"Failed to load data: {e}")
```

### 2. Data Validation

```python
# Always validate CSV data
result = csv_loader.validate_ohlcv_data(data)
if not result['valid']:
    data = csv_loader.clean_ohlcv_data(data)
```

### 3. API Keys from Environment

```python
import os

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

data = binance_loader.load_data('BTCUSDT', ..., api_key=api_key, api_secret=api_secret)
```

### 4. Caching for Development

```python
import os
from pathlib import Path

cache_file = 'cache/AAPL_2023.csv'

if Path(cache_file).exists():
    data = csv_loader.load_data(cache_file)
else:
    data = yfinance_loader.load_data('AAPL', '2023-01-01', '2023-12-31')
    csv_loader.save_to_csv(data, cache_file)
```

---

## 🐛 Troubleshooting

### Yahoo Finance

**Issue**: No data returned
```python
# Solution: Verify symbol
if yfinance_loader.validate_symbol('AAPL'):
    data = yfinance_loader.load_data('AAPL', ...)
```

### Binance

**Issue**: Rate limit exceeded
```python
# Solution: Add delays between requests
import time
time.sleep(1)
```

### MetaTrader 5

**Issue**: "MT5 initialization failed"
```python
# Solution: Check if MT5 terminal is running
if not mt5_loader.check_mt5_connection():
    print("Please start MetaTrader 5 terminal")
```

### CSV

**Issue**: Date parsing errors
```python
# Solution: Specify date format
data = pd.read_csv('file.csv', parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
```

---

## 📊 Output Format

All loaders return standardized DataFrames:

| Column | Type | Description |
|--------|------|-------------|
| index | datetime | Timestamp |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | float | Trading volume |

---

**Ready to load data from any source!** 🚀
