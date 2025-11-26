"""
Performance Metrics

Calculate trading strategy performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Union


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year (252 for daily)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (focuses on downside deviation).

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (Return / Max Drawdown).

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


def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Series or array of returns

    Returns:
        Win rate as decimal (0-1)
    """
    if len(returns) == 0:
        return 0.0

    winning_trades = (returns > 0).sum()
    return winning_trades / len(returns)


def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Series or array of returns

    Returns:
        Profit factor
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if gains > 0 else 0.0

    return gains / losses


def total_return(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate total cumulative return.

    Args:
        returns: Series or array of returns

    Returns:
        Total return as decimal
    """
    return (1 + returns).prod() - 1


def annual_return(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized return.

    Args:
        returns: Series or array of returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized return as decimal
    """
    if len(returns) == 0:
        return 0.0

    total_ret = total_return(returns)
    n_periods = len(returns)

    return (1 + total_ret) ** (periods_per_year / n_periods) - 1


def calculate_all_metrics(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate all performance metrics.

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Dictionary with all metrics
    """
    return {
        'total_return': total_return(returns),
        'annual_return': annual_return(returns, periods_per_year),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown(returns),
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
    }
