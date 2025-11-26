"""
Binance Data Loader

Fetches cryptocurrency data from Binance using python-binance and ccxt libraries.
Supports both spot and futures markets.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import os


def load_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Load historical data from Binance using python-binance.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        api_key: Binance API key (optional for public data)
        api_secret: Binance API secret (optional for public data)

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = load_data('BTCUSDT', '2023-01-01', '2023-12-31', '1d')
        >>> print(data.head())
    """
    try:
        from binance.client import Client
    except ImportError:
        raise ImportError(
            "python-binance not installed. Install with: pip install python-binance"
        )

    # Initialize client (can use without keys for public data)
    api_key = api_key or os.getenv('BINANCE_API_KEY', '')
    api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

    client = Client(api_key, api_secret)

    # Get historical klines
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_date,
        end_date
    )

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def load_data_ccxt(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    exchange: str = "binance"
) -> pd.DataFrame:
    """
    Load historical data using CCXT library (supports multiple exchanges).

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        exchange: Exchange name (binance, coinbase, kraken, etc.)

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = load_data_ccxt('BTC/USDT', '2023-01-01', '2023-12-31', '1d')
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt not installed. Install with: pip install ccxt"
        )

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange)
    exchange_instance = exchange_class({
        'enableRateLimit': True,
    })

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    # Fetch OHLCV data
    all_candles = []
    current_ts = start_ts

    while current_ts < end_ts:
        try:
            candles = exchange_instance.fetch_ohlcv(
                symbol,
                timeframe=interval,
                since=current_ts,
                limit=1000
            )

            if not candles:
                break

            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1

        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Filter by end date
    df = df[df.index <= end_date]

    return df


def load_spot_data(
    symbol: str,
    interval: str = "1h",
    lookback_days: int = 30,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Load recent spot market data.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval
        lookback_days: Number of days to look back
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    return load_data(
        symbol,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval,
        api_key,
        api_secret
    )


def load_futures_data(
    symbol: str,
    interval: str = "1h",
    lookback_days: int = 30,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Load recent futures market data from Binance Futures.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval
        lookback_days: Number of days to look back
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        DataFrame with OHLCV data
    """
    try:
        from binance.client import Client
    except ImportError:
        raise ImportError(
            "python-binance not installed. Install with: pip install python-binance"
        )

    api_key = api_key or os.getenv('BINANCE_API_KEY', '')
    api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

    client = Client(api_key, api_secret)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Get futures klines
    klines = client.futures_historical_klines(
        symbol,
        interval,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def get_current_price(
    symbol: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> float:
    """
    Get current price for a symbol.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        Current price as float
    """
    try:
        from binance.client import Client
    except ImportError:
        raise ImportError(
            "python-binance not installed. Install with: pip install python-binance"
        )

    api_key = api_key or os.getenv('BINANCE_API_KEY', '')
    api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

    client = Client(api_key, api_secret)

    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])


def get_orderbook(
    symbol: str,
    limit: int = 100,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> Dict:
    """
    Get order book for a symbol.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        Dictionary with 'bids' and 'asks' as DataFrames
    """
    try:
        from binance.client import Client
    except ImportError:
        raise ImportError(
            "python-binance not installed. Install with: pip install python-binance"
        )

    api_key = api_key or os.getenv('BINANCE_API_KEY', '')
    api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

    client = Client(api_key, api_secret)

    depth = client.get_order_book(symbol=symbol, limit=limit)

    bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
    asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])

    bids['price'] = bids['price'].astype(float)
    bids['quantity'] = bids['quantity'].astype(float)
    asks['price'] = asks['price'].astype(float)
    asks['quantity'] = asks['quantity'].astype(float)

    return {
        'bids': bids,
        'asks': asks,
        'timestamp': pd.Timestamp.now()
    }


def get_all_symbols(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> List[str]:
    """
    Get list of all trading symbols on Binance.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        List of symbol strings
    """
    try:
        from binance.client import Client
    except ImportError:
        raise ImportError(
            "python-binance not installed. Install with: pip install python-binance"
        )

    api_key = api_key or os.getenv('BINANCE_API_KEY', '')
    api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')

    client = Client(api_key, api_secret)

    exchange_info = client.get_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']

    return symbols
