"""
Data Loaders Package

Unified interface for loading data from various sources.
"""

from typing import Optional
import pandas as pd


def load_data(
    source: str,
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    **kwargs
) -> pd.DataFrame:
    """
    Universal data loader that routes to appropriate source.

    Args:
        source: Data source ('yfinance', 'binance', 'mt5', 'csv')
        symbol: Symbol/ticker to load
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Time interval/timeframe
        **kwargs: Additional arguments for specific loaders

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> # Load from Yahoo Finance
        >>> data = load_data('yfinance', 'AAPL', '2023-01-01', '2023-12-31')

        >>> # Load from Binance
        >>> data = load_data('binance', 'BTCUSDT', '2023-01-01', '2023-12-31', '1h')

        >>> # Load from MT5
        >>> data = load_data('mt5', 'EURUSD', '2023-01-01', '2023-12-31', 'H1')

        >>> # Load from CSV
        >>> data = load_data('csv', 'mydata.csv', date_column='Date')
    """
    source = source.lower()

    if source == 'yfinance' or source == 'yahoo':
        from . import yfinance_loader
        if isinstance(symbol, list):
            return yfinance_loader.load_multiple_symbols(
                symbol, start_date, end_date, interval
            )
        else:
            return yfinance_loader.load_data(
                [symbol], start_date, end_date, interval
            )

    elif source == 'binance':
        from . import binance_loader
        return binance_loader.load_data(
            symbol, start_date, end_date, interval,
            kwargs.get('api_key'), kwargs.get('api_secret')
        )

    elif source == 'mt5' or source == 'metatrader':
        from . import mt5_loader
        return mt5_loader.load_data(
            symbol, start_date, end_date, interval
        )

    elif source == 'csv':
        from . import csv_loader
        return csv_loader.load_data(
            symbol,  # symbol is file_path for CSV
            kwargs.get('date_column', 'date'),
            kwargs.get('parse_dates', True),
            kwargs.get('index_col')
        )

    else:
        raise ValueError(
            f"Unknown data source: {source}. "
            f"Supported sources: yfinance, binance, mt5, csv"
        )


__all__ = [
    'load_data',
    'yfinance_loader',
    'binance_loader',
    'mt5_loader',
    'csv_loader',
]
