"""
Kelly Criterion Position Sizing

Optimal position sizing based on edge and odds.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion position size.

    Kelly% = (Win_Rate * Win_Loss_Ratio - (1 - Win_Rate)) / Win_Loss_Ratio

    Args:
        win_rate: Historical win rate (0-1)
        win_loss_ratio: Average win / Average loss
        max_kelly_fraction: Maximum fraction of Kelly to use (for safety)

    Returns:
        Recommended position size as fraction of capital (0-1)
    """
    if win_loss_ratio <= 0:
        return 0

    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    kelly = max(0, min(kelly, 1)) * max_kelly_fraction

    return kelly


def kelly_from_sharpe(
    sharpe_ratio: float,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly position size from Sharpe ratio.

    For normally distributed returns:
    Kelly% = Sharpe / Volatility ≈ Sharpe (when vol = 1)

    Args:
        sharpe_ratio: Sharpe ratio of strategy
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Position size as fraction of capital
    """
    if sharpe_ratio <= 0:
        return 0

    kelly = sharpe_ratio / (sharpe_ratio**2 + 1)
    kelly = min(kelly, 1) * max_kelly_fraction

    return kelly


def discrete_kelly(
    prob_up: float,
    payoff_up: float,
    payoff_down: float,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Discrete Kelly criterion for binary outcomes.

    Args:
        prob_up: Probability of winning (0-1)
        payoff_up: Payoff if winning (e.g., 2 means 2x return)
        payoff_down: Payoff if losing (e.g., -1 means lose everything)
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Position size as fraction of capital
    """
    prob_down = 1 - prob_up
    expected_value = prob_up * payoff_up + prob_down * payoff_down

    if expected_value <= 0:
        return 0

    kelly = expected_value / abs(payoff_down)
    kelly = max(0, min(kelly, 1)) * max_kelly_fraction

    return kelly


def half_kelly(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate half-Kelly position size (more conservative).

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / Average loss

    Returns:
        Half-Kelly position size
    """
    return kelly_criterion(win_rate, win_loss_ratio, max_kelly_fraction=0.5)


def quarter_kelly(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate quarter-Kelly position size (very conservative).

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / Average loss

    Returns:
        Quarter-Kelly position size
    """
    return kelly_criterion(win_rate, win_loss_ratio, max_kelly_fraction=0.25)


def kelly_from_returns(
    returns: Union[pd.Series, np.ndarray],
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly position size from return series.

    Args:
        returns: Series or array of trade returns
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Kelly position size as fraction
    """
    returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0

    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return 0

    win_loss_ratio = avg_win / avg_loss

    return kelly_criterion(win_rate, win_loss_ratio, max_kelly_fraction)


def continuous_kelly(
    mean_return: float,
    variance: float,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly for continuous returns (geometric Brownian motion).

    Args:
        mean_return: Expected return per period
        variance: Variance of returns
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Kelly position size as fraction
    """
    if variance == 0:
        return 0

    kelly = mean_return / variance
    kelly = max(0, min(kelly, 1)) * max_kelly_fraction

    return kelly


def multi_asset_kelly(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    max_leverage: float = 1.0
) -> np.ndarray:
    """
    Calculate Kelly optimal weights for multiple assets.

    Args:
        expected_returns: Array of expected returns
        covariance_matrix: Covariance matrix of returns
        max_leverage: Maximum total leverage

    Returns:
        Array of Kelly optimal weights
    """
    try:
        inv_cov = np.linalg.inv(covariance_matrix)
        kelly_weights = inv_cov @ expected_returns

        total_leverage = np.abs(kelly_weights).sum()
        if total_leverage > max_leverage:
            kelly_weights = kelly_weights * (max_leverage / total_leverage)

        return kelly_weights

    except np.linalg.LinAlgError:
        return np.zeros_like(expected_returns)


def kelly_with_drawdown_constraint(
    win_rate: float,
    win_loss_ratio: float,
    max_drawdown: float = 0.2,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly with maximum drawdown constraint.

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / Average loss
        max_drawdown: Maximum acceptable drawdown (0-1)
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Kelly position size constrained by drawdown
    """
    kelly = kelly_criterion(win_rate, win_loss_ratio, max_kelly_fraction)

    expected_drawdown = kelly * (1 - win_rate) * 3

    if expected_drawdown > max_drawdown:
        kelly = kelly * (max_drawdown / expected_drawdown)

    return kelly


def empirical_kelly(
    trades: Union[pd.Series, List[float], np.ndarray],
    bins: int = 100,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate empirical Kelly from trade distribution.

    Args:
        trades: Array of trade returns
        bins: Number of bins for histogram
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Empirical Kelly position size
    """
    trades = np.array(trades)

    if len(trades) == 0:
        return 0

    max_f = 0
    max_growth = float('-inf')

    for f in np.linspace(0, 1, bins):
        if f == 0:
            continue

        cumulative_growth = 0
        for trade in trades:
            cumulative_growth += np.log(1 + f * trade)

        avg_growth = cumulative_growth / len(trades)

        if avg_growth > max_growth:
            max_growth = avg_growth
            max_f = f

    return max_f * max_kelly_fraction


def kelly_from_odds(
    decimal_odds: float,
    prob_win: float,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly from betting odds and win probability.

    Args:
        decimal_odds: Decimal odds (e.g., 2.5 means 1.5:1 profit)
        prob_win: Estimated probability of winning (0-1)
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Kelly position size as fraction
    """
    b = decimal_odds - 1
    kelly = (b * prob_win - (1 - prob_win)) / b

    kelly = max(0, min(kelly, 1)) * max_kelly_fraction

    return kelly


def fractional_kelly_grid(
    win_rate: float,
    win_loss_ratio: float,
    fractions: Optional[List[float]] = None
) -> dict:
    """
    Calculate Kelly for multiple fractions.

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / Average loss
        fractions: List of fractions to test

    Returns:
        Dictionary mapping fraction to Kelly size
    """
    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    results = {}
    for fraction in fractions:
        results[fraction] = kelly_criterion(win_rate, win_loss_ratio, fraction)

    return results


def dynamic_kelly(
    recent_trades: Union[pd.Series, List[float]],
    lookback: int = 50,
    max_kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly using recent trade history (dynamic).

    Args:
        recent_trades: Recent trade returns
        lookback: Number of recent trades to consider
        max_kelly_fraction: Maximum fraction to use

    Returns:
        Dynamic Kelly position size
    """
    trades = pd.Series(recent_trades)

    if len(trades) == 0:
        return 0

    recent = trades.tail(lookback)

    return kelly_from_returns(recent, max_kelly_fraction)
