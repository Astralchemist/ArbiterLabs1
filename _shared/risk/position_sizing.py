"""
Position Sizing Methods

Various position sizing techniques for risk management.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


def fixed_position_size(
    capital: float,
    position_pct: float = 0.1
) -> float:
    """
    Calculate fixed percentage position size.

    Args:
        capital: Total capital
        position_pct: Percentage of capital to risk (0-1)

    Returns:
        Position size in dollars
    """
    return capital * position_pct


def volatility_adjusted_size(
    capital: float,
    target_volatility: float,
    asset_volatility: float,
    max_position_pct: float = 0.2
) -> float:
    """
    Calculate position size adjusted for volatility.

    Args:
        capital: Total capital
        target_volatility: Target portfolio volatility
        asset_volatility: Asset's historical volatility
        max_position_pct: Maximum position size as % of capital

    Returns:
        Position size in dollars
    """
    if asset_volatility == 0:
        return 0

    position_size = capital * (target_volatility / asset_volatility)
    max_size = capital * max_position_pct

    return min(position_size, max_size)


def risk_parity_size(
    capital: float,
    asset_volatilities: np.ndarray,
    num_assets: int
) -> np.ndarray:
    """
    Calculate risk parity position sizes.

    Args:
        capital: Total capital
        asset_volatilities: Array of asset volatilities
        num_assets: Number of assets

    Returns:
        Array of position sizes
    """
    inv_volatilities = 1 / asset_volatilities
    weights = inv_volatilities / inv_volatilities.sum()
    position_sizes = capital * weights

    return position_sizes


def max_drawdown_adjusted(
    capital: float,
    current_drawdown: float,
    max_allowed_drawdown: float = 0.15,
    base_position_pct: float = 0.1
) -> float:
    """
    Adjust position size based on current drawdown.

    Args:
        capital: Total capital
        current_drawdown: Current portfolio drawdown (0-1)
        max_allowed_drawdown: Maximum allowed drawdown before reducing size
        base_position_pct: Base position size percentage

    Returns:
        Adjusted position size
    """
    if current_drawdown >= max_allowed_drawdown:
        return 0

    reduction_factor = 1 - (current_drawdown / max_allowed_drawdown)
    adjusted_pct = base_position_pct * reduction_factor

    return capital * adjusted_pct


def atr_based_size(
    capital: float,
    price: float,
    atr: float,
    risk_per_trade: float = 0.02,
    atr_multiplier: float = 2.0
) -> int:
    """
    Calculate position size based on ATR (Average True Range).

    Args:
        capital: Total capital
        price: Current asset price
        atr: Average True Range value
        risk_per_trade: Risk per trade as % of capital
        atr_multiplier: ATR multiplier for stop loss

    Returns:
        Number of shares/contracts
    """
    risk_amount = capital * risk_per_trade
    stop_distance = atr * atr_multiplier

    if stop_distance == 0:
        return 0

    shares = risk_amount / stop_distance
    return int(shares)


def percent_risk_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade: float = 0.02
) -> int:
    """
    Calculate position size based on percent risk per trade.

    Args:
        capital: Total capital
        entry_price: Entry price
        stop_loss_price: Stop loss price
        risk_per_trade: Risk per trade as % of capital

    Returns:
        Number of shares/contracts
    """
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        return 0

    shares = risk_amount / risk_per_share
    return int(shares)


def equal_weight_size(
    capital: float,
    num_positions: int
) -> float:
    """
    Calculate equal weight position size.

    Args:
        capital: Total capital
        num_positions: Number of positions to hold

    Returns:
        Position size in dollars
    """
    return capital / num_positions


def inverse_volatility_size(
    capital: float,
    volatilities: Union[np.ndarray, List[float]],
    leverage: float = 1.0
) -> np.ndarray:
    """
    Calculate position sizes inversely proportional to volatility.

    Args:
        capital: Total capital
        volatilities: Array of asset volatilities
        leverage: Portfolio leverage

    Returns:
        Array of position sizes
    """
    volatilities = np.array(volatilities)
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    position_sizes = capital * weights * leverage

    return position_sizes


def target_dollar_volatility(
    capital: float,
    price: float,
    volatility: float,
    target_vol: float = 1000
) -> int:
    """
    Size position to achieve target dollar volatility.

    Args:
        capital: Total capital
        price: Current price
        volatility: Asset volatility (std dev of returns)
        target_vol: Target daily dollar volatility

    Returns:
        Number of shares
    """
    if volatility == 0:
        return 0

    shares = target_vol / (price * volatility)
    return int(shares)


def optimal_f(
    capital: float,
    largest_loss: float,
    f: float = 0.25
) -> float:
    """
    Calculate position size using Optimal F.

    Args:
        capital: Total capital
        largest_loss: Largest historical loss
        f: Fraction of optimal f to use (0-1)

    Returns:
        Position size in dollars
    """
    if largest_loss == 0:
        return 0

    optimal_fraction = capital / abs(largest_loss)
    position_size = capital * (optimal_fraction * f)

    return position_size


def martingale_size(
    capital: float,
    base_size: float,
    consecutive_losses: int,
    multiplier: float = 2.0,
    max_multiplier: float = 8.0
) -> float:
    """
    Calculate martingale position size (use with extreme caution).

    Args:
        capital: Total capital
        base_size: Base position size
        consecutive_losses: Number of consecutive losses
        multiplier: Size multiplier per loss
        max_multiplier: Maximum multiplier cap

    Returns:
        Position size in dollars

    Warning:
        Martingale is extremely risky. Use only for educational purposes.
    """
    size_multiplier = min(multiplier ** consecutive_losses, max_multiplier)
    position_size = min(base_size * size_multiplier, capital * 0.5)

    return position_size


def anti_martingale_size(
    capital: float,
    base_size: float,
    consecutive_wins: int,
    multiplier: float = 1.5,
    max_multiplier: float = 4.0
) -> float:
    """
    Calculate anti-martingale position size (increase on wins).

    Args:
        capital: Total capital
        base_size: Base position size
        consecutive_wins: Number of consecutive wins
        multiplier: Size multiplier per win
        max_multiplier: Maximum multiplier cap

    Returns:
        Position size in dollars
    """
    size_multiplier = min(multiplier ** consecutive_wins, max_multiplier)
    position_size = min(base_size * size_multiplier, capital * 0.3)

    return position_size


def correlation_adjusted_size(
    capital: float,
    base_size: float,
    correlation: float,
    max_correlation: float = 0.7
) -> float:
    """
    Adjust position size based on correlation with existing positions.

    Args:
        capital: Total capital
        base_size: Base position size
        correlation: Correlation with existing positions (-1 to 1)
        max_correlation: Correlation threshold for reduction

    Returns:
        Adjusted position size
    """
    if abs(correlation) >= max_correlation:
        reduction = abs(correlation) / max_correlation
        return base_size * (1 - reduction * 0.5)

    return base_size


def market_regime_size(
    capital: float,
    base_pct: float,
    regime: str,
    regime_multipliers: Optional[dict] = None
) -> float:
    """
    Adjust position size based on market regime.

    Args:
        capital: Total capital
        base_pct: Base position percentage
        regime: Market regime ('bull', 'bear', 'sideways', 'volatile')
        regime_multipliers: Custom multipliers per regime

    Returns:
        Position size in dollars
    """
    if regime_multipliers is None:
        regime_multipliers = {
            'bull': 1.2,
            'bear': 0.5,
            'sideways': 0.8,
            'volatile': 0.6
        }

    multiplier = regime_multipliers.get(regime, 1.0)
    position_size = capital * base_pct * multiplier

    return position_size


def expected_value_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_risk_pct: float = 0.05
) -> float:
    """
    Size position based on expected value.

    Args:
        capital: Total capital
        win_rate: Historical win rate (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive)
        max_risk_pct: Maximum risk percentage

    Returns:
        Position size in dollars
    """
    expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    if expected_value <= 0:
        return 0

    ev_ratio = expected_value / avg_loss
    risk_pct = min(ev_ratio * 0.01, max_risk_pct)

    return capital * risk_pct
