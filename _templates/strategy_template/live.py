"""
Live Trading Script

Run this script to execute the strategy in live or paper trading mode.

Usage:
    python live.py --mode paper
    python live.py --mode live
"""

import argparse
import yaml
import time
import pandas as pd
from datetime import datetime
from strategy import Strategy


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_live_data(config: dict) -> pd.DataFrame:
    """
    Fetch live market data.

    Returns:
        DataFrame with recent OHLCV data
    """
    # TODO: Implement live data fetching
    # This is a placeholder
    print("Fetching live data...")
    return pd.DataFrame()


def execute_live_trade(signal: int, price: float, mode: str):
    """
    Execute trade in live or paper mode.

    Args:
        signal: Trading signal
        price: Execution price
        mode: 'paper' or 'live'
    """
    if mode == 'paper':
        print(f"[PAPER] Trade executed: Signal={signal}, Price={price}")
    else:
        # TODO: Implement actual broker execution
        print(f"[LIVE] Trade executed: Signal={signal}, Price={price}")


def main():
    parser = argparse.ArgumentParser(description='Run strategy live')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode: paper or live')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize strategy
    strategy = Strategy(config)

    print(f"Starting live trading in {args.mode.upper()} mode...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Fetch current data
            data = fetch_live_data(config)

            if not data.empty:
                # Generate signals
                signals = strategy.generate_signals(data)
                current_signal = signals.iloc[-1]
                current_price = data['close'].iloc[-1]

                # Execute if signal present
                if current_signal != 0:
                    execute_live_trade(current_signal, current_price, args.mode)
                    strategy.execute_trade(current_signal, current_price, datetime.now())

                print(f"[{datetime.now()}] Portfolio Value: ${strategy.portfolio_value:,.2f}")

            # Sleep before next iteration
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        print(f"Final Portfolio Value: ${strategy.portfolio_value:,.2f}")


if __name__ == "__main__":
    main()
