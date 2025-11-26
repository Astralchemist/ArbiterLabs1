"""
Yahoo Finance Data Loader

Fetches historical market data using yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta


def load_data(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Load historical data from Yahoo Finance.

    Args:
        symbols: Ticker symbol or list of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        auto_adjust: Adjust OHLC for splits and dividends

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> data = load_data('AAPL', '2023-01-01', '2023-12-31')
        >>> data = load_data(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31', '1d')
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    if len(symbols) == 1:
        ticker = yf.Ticker(symbols[0])
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust
        )
        if not auto_adjust:
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
        else:
            data.columns = data.columns.str.lower()

        data = data[['open', 'high', 'low', 'close', 'volume']]
    else:
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by='ticker'
        )

    return data


def load_multiple_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols as separate DataFrames.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    data_dict = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        df.columns = df.columns.str.lower()
        data_dict[symbol] = df[['open', 'high', 'low', 'close', 'volume']]
    return data_dict


def load_recent_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Load recent data using period instead of dates.

    Args:
        symbol: Ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval

    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    data.columns = data.columns.str.lower()
    return data[['open', 'high', 'low', 'close', 'volume']]


def get_current_price(symbol: str) -> float:
    """
    Get current/latest price for a symbol.

    Args:
        symbol: Ticker symbol

    Returns:
        Current price
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d', interval='1m')
    if len(data) > 0:
        return float(data['Close'].iloc[-1])

    info = ticker.info
    return float(info.get('regularMarketPrice', 0))


def get_info(symbol: str) -> Dict:
    """
    Get company/asset information.

    Args:
        symbol: Ticker symbol

    Returns:
        Dictionary with symbol information
    """
    ticker = yf.Ticker(symbol)
    return ticker.info


def get_fundamentals(symbol: str) -> Dict[str, pd.DataFrame]:
    """
    Get fundamental data (financials, balance sheet, cash flow).

    Args:
        symbol: Ticker symbol

    Returns:
        Dictionary with financial DataFrames
    """
    ticker = yf.Ticker(symbol)

    return {
        'financials': ticker.financials,
        'quarterly_financials': ticker.quarterly_financials,
        'balance_sheet': ticker.balance_sheet,
        'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
        'cashflow': ticker.cashflow,
        'quarterly_cashflow': ticker.quarterly_cashflow,
    }


def get_dividends(symbol: str) -> pd.Series:
    """
    Get dividend history.

    Args:
        symbol: Ticker symbol

    Returns:
        Series with dividend dates and amounts
    """
    ticker = yf.Ticker(symbol)
    return ticker.dividends


def get_splits(symbol: str) -> pd.Series:
    """
    Get stock split history.

    Args:
        symbol: Ticker symbol

    Returns:
        Series with split dates and ratios
    """
    ticker = yf.Ticker(symbol)
    return ticker.splits


def get_options_chain(symbol: str, date: Optional[str] = None) -> Dict:
    """
    Get options chain data.

    Args:
        symbol: Ticker symbol
        date: Expiration date (YYYY-MM-DD), if None uses nearest expiration

    Returns:
        Dictionary with calls and puts DataFrames
    """
    ticker = yf.Ticker(symbol)

    if date is None:
        dates = ticker.options
        if len(dates) == 0:
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
        date = dates[0]

    opt = ticker.option_chain(date)

    return {
        'calls': opt.calls,
        'puts': opt.puts
    }


def get_available_options_dates(symbol: str) -> List[str]:
    """
    Get available options expiration dates.

    Args:
        symbol: Ticker symbol

    Returns:
        List of expiration dates
    """
    ticker = yf.Ticker(symbol)
    return list(ticker.options)


def download_bulk(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    threads: int = True
) -> pd.DataFrame:
    """
    Bulk download multiple symbols efficiently.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        threads: Use threading for faster downloads

    Returns:
        Multi-index DataFrame with all symbols
    """
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval=interval,
        threads=threads,
        group_by='ticker'
    )

    return data


def validate_symbol(symbol: str) -> bool:
    """
    Check if symbol is valid on Yahoo Finance.

    Args:
        symbol: Ticker symbol

    Returns:
        True if valid, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return 'regularMarketPrice' in info or 'currentPrice' in info
    except:
        return False
