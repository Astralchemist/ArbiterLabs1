"""
Drawdown Metrics

Calculate and analyze portfolio drawdowns.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Series or array of returns

    Returns:
        Maximum drawdown as decimal (e.g., 0.25 = 25% drawdown)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return abs(drawdown.min())


def drawdown_series(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Calculate drawdown series over time.

    Args:
        returns: Series or array of returns

    Returns:
        Series of drawdowns
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return drawdown


def max_drawdown_duration(returns: Union[pd.Series, np.ndarray]) -> int:
    """
    Calculate maximum drawdown duration in periods.

    Args:
        returns: Series or array of returns

    Returns:
        Duration in number of periods
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    is_drawdown = drawdown < 0
    duration = 0
    max_duration = 0
    current_duration = 0

    for dd in is_drawdown:
        if dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def underwater_curve(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Calculate underwater (drawdown) curve.

    Args:
        returns: Series or array of returns

    Returns:
        Series of underwater values (negative = in drawdown)
    """
    return drawdown_series(returns) * 100


def calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (Annualized Return / Max Drawdown).

    Args:
        returns: Series or array of returns
        periods_per_year: Number of periods in a year

    Returns:
        Calmar Ratio
    """
    if len(returns) == 0:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    max_dd = max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    return annual_return / max_dd


def avg_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate average drawdown.

    Args:
        returns: Series or array of returns

    Returns:
        Average drawdown as decimal
    """
    dd_series = drawdown_series(returns)
    negative_dds = dd_series[dd_series < 0]

    if len(negative_dds) == 0:
        return 0.0

    return abs(negative_dds.mean())


def avg_drawdown_duration(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate average drawdown duration.

    Args:
        returns: Series or array of returns

    Returns:
        Average duration in periods
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    is_drawdown = drawdown < 0
    durations = []
    current_duration = 0

    for dd in is_drawdown:
        if dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return np.mean(durations) if durations else 0


def recovery_time(
    returns: Union[pd.Series, np.ndarray],
    from_date: Optional[pd.Timestamp] = None
) -> int:
    """
    Calculate time to recover from max drawdown.

    Args:
        returns: Series or array of returns
        from_date: Start date for calculation

    Returns:
        Recovery time in periods
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd_idx = drawdown.idxmin()

    recovery_idx = cumulative[max_dd_idx:].index[cumulative[max_dd_idx:] >= running_max[max_dd_idx]][0]

    return (recovery_idx - max_dd_idx).days if hasattr(recovery_idx, 'days') else int(recovery_idx - max_dd_idx)


def ulcer_index(returns: Union[pd.Series, np.ndarray], period: int = 14) -> float:
    """
    Calculate Ulcer Index (downside risk measure).

    Args:
        returns: Series or array of returns
        period: Lookback period

    Returns:
        Ulcer Index value
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(period).max()
    drawdown_pct = ((cumulative - rolling_max) / rolling_max) * 100

    ulcer = np.sqrt((drawdown_pct ** 2).rolling(period).mean().iloc[-1])

    return ulcer if not np.isnan(ulcer) else 0.0


def pain_index(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Pain Index (average of drawdowns).

    Args:
        returns: Series or array of returns

    Returns:
        Pain Index value
    """
    dd_series = abs(drawdown_series(returns))
    return dd_series.mean()


def pain_ratio(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """
    Calculate Pain Ratio (Return / Pain Index).

    Args:
        returns: Series or array of returns
        periods_per_year: Number of periods in a year

    Returns:
        Pain Ratio
    """
    annual_return = returns.mean() * periods_per_year
    pain = pain_index(returns)

    if pain == 0:
        return 0.0

    return annual_return / pain
