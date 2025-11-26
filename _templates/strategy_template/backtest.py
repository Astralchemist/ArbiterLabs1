"""
Backtesting Script

Run this script to backtest the strategy on historical data.

Usage:
    python backtest.py
    python backtest.py --config custom_config.yaml
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from strategy import Strategy


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """
    Load historical data based on config settings.

    Returns:
        DataFrame with OHLCV data
    """
    # TODO: Implement data loading logic
    # This is a placeholder - replace with actual data loading
    dates = pd.date_range(
        start=config['data']['start_date'],
        end=config['data']['end_date'],
        freq='D'
    )

    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    return data


def run_backtest(strategy: Strategy, data: pd.DataFrame) -> dict:
    """
    Execute backtest and calculate performance metrics.

    Args:
        strategy: Strategy instance
        data: Historical market data

    Returns:
        Dictionary containing performance metrics
    """
    signals = strategy.generate_signals(data)

    # Simple backtest loop
    equity_curve = []
    trades = []

    for i, (timestamp, row) in enumerate(data.iterrows()):
        if i < 20:  # Skip initial period for indicator warmup
            continue

        signal = signals.iloc[i]
        if signal != 0:
            strategy.execute_trade(signal, row['close'], timestamp)
            trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': row['close']
            })

        equity_curve.append(strategy.portfolio_value)

    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    metrics = {
        'total_return': (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
        'max_drawdown': ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100,
        'num_trades': len(trades),
        'win_rate': 0  # TODO: Calculate win rate
    }

    return metrics, equity_curve, trades


def main():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load data
    print(f"Loading data for {config['data']['symbols']}...")
    data = load_data(config)

    # Initialize strategy
    strategy = Strategy(config)

    # Run backtest
    print("Running backtest...")
    metrics, equity_curve, trades = run_backtest(strategy, data)

    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
