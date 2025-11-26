"""
Risk-Adjusted Performance Metrics

Calculate various risk-adjusted return metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

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
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio (downside deviation).

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        target_return: Minimum acceptable return

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < target_return]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Information Ratio
    """
    active_returns = returns - benchmark_returns

    if active_returns.std() == 0:
        return 0.0

    return np.sqrt(periods_per_year) * active_returns.mean() / active_returns.std()


def omega_ratio(
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega Ratio.

    Args:
        returns: Series or array of returns
        threshold: Threshold return

    Returns:
        Omega Ratio
    """
    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns < threshold]

    gains = returns_above.sum() if len(returns_above) > 0 else 0
    losses = returns_below.sum() if len(returns_below) > 0 else 0

    if losses == 0:
        return np.inf if gains > 0 else 0.0

    return gains / losses


def treynor_ratio(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Treynor Ratio.

    Args:
        returns: Strategy returns
        market_returns: Market returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Treynor Ratio
    """
    beta = calculate_beta(returns, market_returns)

    if beta == 0:
        return 0.0

    excess_return = returns.mean() - (risk_free_rate / periods_per_year)

    return (excess_return * periods_per_year) / beta


def calculate_beta(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate Beta.

    Args:
        returns: Strategy returns
        market_returns: Market returns

    Returns:
        Beta value
    """
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)

    if market_variance == 0:
        return 0.0

    return covariance / market_variance


def calculate_alpha(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's Alpha.

    Args:
        returns: Strategy returns
        market_returns: Market returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Alpha value
    """
    beta = calculate_beta(returns, market_returns)

    rf_per_period = risk_free_rate / periods_per_year
    strategy_return = returns.mean()
    market_return = market_returns.mean()

    alpha = strategy_return - (rf_per_period + beta * (market_return - rf_per_period))

    return alpha * periods_per_year


def sterling_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sterling Ratio.

    Args:
        returns: Series or array of returns
        periods_per_year: Number of periods in a year

    Returns:
        Sterling Ratio
    """
    from .drawdown import avg_drawdown

    annual_return = returns.mean() * periods_per_year
    avg_dd = avg_drawdown(returns)

    if avg_dd == 0:
        return 0.0

    return annual_return / avg_dd


def burke_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Burke Ratio.

    Args:
        returns: Series or array of returns
        periods_per_year: Number of periods in a year

    Returns:
        Burke Ratio
    """
    from .drawdown import drawdown_series

    annual_return = returns.mean() * periods_per_year
    dd_series = abs(drawdown_series(returns))
    dd_squared_sum = (dd_series ** 2).sum()

    if dd_squared_sum == 0:
        return 0.0

    return annual_return / np.sqrt(dd_squared_sum)


def tail_ratio(returns: Union[pd.Series, np.ndarray], percentile: float = 0.05) -> float:
    """
    Calculate Tail Ratio (95th percentile / 5th percentile).

    Args:
        returns: Series or array of returns
        percentile: Percentile for tails

    Returns:
        Tail Ratio
    """
    upper = np.percentile(returns, (1 - percentile) * 100)
    lower = abs(np.percentile(returns, percentile * 100))

    if lower == 0:
        return 0.0

    return upper / lower


def value_at_risk(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series or array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value
    """
    return -np.percentile(returns, (1 - confidence_level) * 100)


def conditional_value_at_risk(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).

    Args:
        returns: Series or array of returns
        confidence_level: Confidence level

    Returns:
        CVaR value
    """
    var = value_at_risk(returns, confidence_level)
    return -returns[returns <= -var].mean()


def gain_to_pain_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Gain-to-Pain Ratio.

    Args:
        returns: Series or array of returns

    Returns:
        Gain-to-Pain Ratio
    """
    total_gain = returns[returns > 0].sum()
    total_pain = abs(returns[returns < 0].sum())

    if total_pain == 0:
        return np.inf if total_gain > 0 else 0.0

    return total_gain / total_pain
