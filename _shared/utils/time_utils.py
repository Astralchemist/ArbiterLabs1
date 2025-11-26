"""
Time Utilities

Helper functions for time and date operations in trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Union, List, Optional


def is_market_open(
    timestamp: datetime,
    market: str = 'us_stock',
    holidays: Optional[List[datetime]] = None
) -> bool:
    """
    Check if market is open at given timestamp.

    Args:
        timestamp: Timestamp to check
        market: Market type ('us_stock', 'crypto', 'forex')
        holidays: List of holiday dates

    Returns:
        True if market is open
    """
    if market == 'crypto':
        return True

    if timestamp.weekday() >= 5:
        return False

    if holidays and timestamp.date() in [h.date() for h in holidays]:
        return False

    if market == 'us_stock':
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= timestamp.time() <= market_close

    elif market == 'forex':
        if timestamp.weekday() == 4 and timestamp.time() >= time(17, 0):
            return False
        if timestamp.weekday() == 6:
            return False
        return True

    return True


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    market: str = 'us_stock'
) -> pd.DatetimeIndex:
    """
    Get trading days between dates.

    Args:
        start_date: Start date
        end_date: End date
        market: Market type

    Returns:
        DatetimeIndex of trading days
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    all_days = pd.date_range(start_date, end_date, freq='D')

    if market == 'crypto':
        return all_days

    trading_days = all_days[all_days.weekday < 5]

    return trading_days


def get_next_trading_day(
    date: Union[str, datetime],
    market: str = 'us_stock'
) -> datetime:
    """
    Get next trading day.

    Args:
        date: Current date
        market: Market type

    Returns:
        Next trading day
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    if market == 'crypto':
        return date + timedelta(days=1)

    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day


def get_previous_trading_day(
    date: Union[str, datetime],
    market: str = 'us_stock'
) -> datetime:
    """
    Get previous trading day.

    Args:
        date: Current date
        market: Market type

    Returns:
        Previous trading day
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    if market == 'crypto':
        return date - timedelta(days=1)

    prev_day = date - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    return prev_day


def time_to_market_close(
    timestamp: datetime,
    market: str = 'us_stock'
) -> timedelta:
    """
    Calculate time remaining until market close.

    Args:
        timestamp: Current timestamp
        market: Market type

    Returns:
        Time remaining as timedelta
    """
    if market == 'crypto':
        return timedelta(0)

    if market == 'us_stock':
        close_time = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        if timestamp > close_time:
            next_day = get_next_trading_day(timestamp)
            close_time = next_day.replace(hour=16, minute=0)

        return close_time - timestamp

    return timedelta(0)


def time_to_market_open(
    timestamp: datetime,
    market: str = 'us_stock'
) -> timedelta:
    """
    Calculate time until market opens.

    Args:
        timestamp: Current timestamp
        market: Market type

    Returns:
        Time until open as timedelta
    """
    if market == 'crypto':
        return timedelta(0)

    if market == 'us_stock':
        open_time = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        if timestamp > open_time:
            next_day = get_next_trading_day(timestamp)
            open_time = next_day.replace(hour=9, minute=30)

        return open_time - timestamp

    return timedelta(0)


def get_market_session(timestamp: datetime) -> str:
    """
    Determine market session.

    Args:
        timestamp: Timestamp to check

    Returns:
        Session name ('pre_market', 'regular', 'after_hours', 'closed')
    """
    t = timestamp.time()

    if time(4, 0) <= t < time(9, 30):
        return 'pre_market'
    elif time(9, 30) <= t < time(16, 0):
        return 'regular'
    elif time(16, 0) <= t < time(20, 0):
        return 'after_hours'
    else:
        return 'closed'


def resample_to_timeframe(
    data: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.

    Args:
        data: OHLCV DataFrame
        timeframe: Target timeframe ('1H', '4H', '1D', etc.)

    Returns:
        Resampled DataFrame
    """
    resampled = pd.DataFrame()
    resampled['open'] = data['open'].resample(timeframe).first()
    resampled['high'] = data['high'].resample(timeframe).max()
    resampled['low'] = data['low'].resample(timeframe).min()
    resampled['close'] = data['close'].resample(timeframe).last()
    resampled['volume'] = data['volume'].resample(timeframe).sum()

    return resampled.dropna()


def align_timestamps(
    *dataframes: pd.DataFrame,
    method: str = 'inner'
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames by timestamp.

    Args:
        *dataframes: DataFrames to align
        method: Join method ('inner', 'outer', 'left', 'right')

    Returns:
        List of aligned DataFrames
    """
    if len(dataframes) == 0:
        return []

    if len(dataframes) == 1:
        return list(dataframes)

    base_index = dataframes[0].index
    for df in dataframes[1:]:
        if method == 'inner':
            base_index = base_index.intersection(df.index)
        elif method == 'outer':
            base_index = base_index.union(df.index)

    aligned = [df.reindex(base_index) for df in dataframes]

    return aligned


def trading_days_between(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    market: str = 'us_stock'
) -> int:
    """
    Count trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        market: Market type

    Returns:
        Number of trading days
    """
    trading_days = get_trading_days(start_date, end_date, market)
    return len(trading_days)


def get_month_end_dates(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> pd.DatetimeIndex:
    """
    Get month-end dates in range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        DatetimeIndex of month-end dates
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    months = pd.date_range(start_date, end_date, freq='M')
    return months


def get_quarter_end_dates(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> pd.DatetimeIndex:
    """
    Get quarter-end dates in range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        DatetimeIndex of quarter-end dates
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    quarters = pd.date_range(start_date, end_date, freq='Q')
    return quarters


def is_month_end(date: datetime, tolerance_days: int = 3) -> bool:
    """
    Check if date is near month end.

    Args:
        date: Date to check
        tolerance_days: Days tolerance

    Returns:
        True if near month end
    """
    last_day = (date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    days_diff = abs((last_day - date).days)

    return days_diff <= tolerance_days


def get_us_market_holidays(year: int) -> List[datetime]:
    """
    Get US stock market holidays for a year.

    Args:
        year: Year

    Returns:
        List of holiday dates
    """
    holidays = [
        datetime(year, 1, 1),
        datetime(year, 7, 4),
        datetime(year, 12, 25),
    ]

    return holidays
