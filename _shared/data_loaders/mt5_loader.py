"""
MetaTrader 5 Data Loader

Fetches forex, stocks, and futures data from MetaTrader 5 platform.
Requires MT5 terminal to be installed and running.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


def load_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "D1"
) -> pd.DataFrame:
    """
    Load historical data from MetaTrader 5.

    Args:
        symbol: Symbol name (e.g., 'EURUSD', 'GBPUSD', 'US30')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = load_data('EURUSD', '2023-01-01', '2023-12-31', 'D1')
        >>> print(data.head())
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    # Initialize MT5 connection
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        # Map timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        mt5_timeframe = timeframe_map[timeframe]

        # Convert dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Get rates
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol}. Error: {mt5.last_error()}")

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Rename columns to standard format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        })

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    finally:
        # Always shutdown MT5 connection
        mt5.shutdown()


def load_recent_data(
    symbol: str,
    timeframe: str = "H1",
    num_bars: int = 1000
) -> pd.DataFrame:
    """
    Load recent N bars of data.

    Args:
        symbol: Symbol name
        timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        num_bars: Number of bars to load

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }

        mt5_timeframe = timeframe_map[timeframe]

        # Get recent bars
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_bars)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df = df.rename(columns={'tick_volume': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    finally:
        mt5.shutdown()


def get_current_price(symbol: str) -> Tuple[float, float, float]:
    """
    Get current bid, ask, and last price.

    Args:
        symbol: Symbol name

    Returns:
        Tuple of (bid, ask, last) prices
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            raise ValueError(f"Failed to get tick for {symbol}")

        return tick.bid, tick.ask, tick.last

    finally:
        mt5.shutdown()


def get_symbol_info(symbol: str) -> dict:
    """
    Get symbol information (pip size, contract size, etc.).

    Args:
        symbol: Symbol name

    Returns:
        Dictionary with symbol information
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)

        if info is None:
            raise ValueError(f"Symbol {symbol} not found")

        return {
            'name': info.name,
            'description': info.description,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'contract_size': info.trade_contract_size,
            'min_volume': info.volume_min,
            'max_volume': info.volume_max,
            'volume_step': info.volume_step,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'currency_margin': info.currency_margin,
        }

    finally:
        mt5.shutdown()


def get_all_symbols() -> List[str]:
    """
    Get list of all available symbols.

    Returns:
        List of symbol names
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        symbols = mt5.symbols_get()

        if symbols is None:
            return []

        return [s.name for s in symbols]

    finally:
        mt5.shutdown()


def get_forex_symbols() -> List[str]:
    """
    Get list of forex symbols only.

    Returns:
        List of forex symbol names
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        symbols = mt5.symbols_get()

        if symbols is None:
            return []

        # Filter forex symbols (typically 6 characters like EURUSD)
        forex_symbols = [
            s.name for s in symbols
            if s.name.isalpha() and len(s.name) == 6
        ]

        return forex_symbols

    finally:
        mt5.shutdown()


def load_tick_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load tick-by-tick data.

    Args:
        symbol: Symbol name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with tick data

    Warning:
        Tick data can be very large. Use with caution.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 not installed. Install with: pip install MetaTrader5"
        )

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Get ticks
        ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            raise ValueError(f"No tick data returned for {symbol}")

        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    finally:
        mt5.shutdown()


def check_mt5_connection() -> bool:
    """
    Check if MT5 terminal is running and accessible.

    Returns:
        True if connected, False otherwise
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 library not installed")
        return False

    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            print(f"Connected to MT5 Terminal: {terminal_info.name}")
            print(f"Build: {terminal_info.build}")
            return True
        return False

    finally:
        mt5.shutdown()
