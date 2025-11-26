"""
Backtest engine for XIV/VXX Volatility Strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from strategy import XIVVXXVolatilityStrategy


def download_data(symbols, start_date, end_date, interval='1h'):
    """Download data from yfinance."""
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, 
                           interval=interval, progress=False)
            if not df.empty:
                data[symbol] = df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    return data


def calculate_metrics(equity_curve):
    """Calculate performance metrics."""
    returns = equity_curve.pct_change().dropna()
    
    # Annualized return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = len(equity_curve) / 252
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Sharpe ratio (assuming 252 trading days)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Max drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return {
        'Total Return': f"{total_return * 100:.2f}%",
        'CAGR': f"{cagr * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown * 100:.2f}%",
        'Win Rate': f"{win_rate * 100:.2f}%",
        'Final Value': f"${equity_curve.iloc[-1]:,.2f}"
    }


def run_backtest(config):
    """Run backtest for XIV/VXX strategy."""
    
    print("=" * 60)
    print("XIV/VXX Volatility Strategy Backtest")
    print("=" * 60)
    
    # Extract config
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    
    symbols = [
        config['symbols']['volatility_short'],
        config['symbols']['volatility_long'],
        config['symbols']['bond']
    ]
    
    print(f"\nDownloading data for {symbols}...")
    print(f"Period: {start_date} to {end_date}")
    
    # Download data
    data = download_data(symbols, start_date, end_date, interval=config['data']['interval'])
    
    if len(data) < 3:
        print("Error: Could not download all required data")
        return None
    
    # Initialize strategy
    strategy = XIVVXXVolatilityStrategy(config)
    
    # Get common date range
    dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
    
    if len(dates) < 100:
        print("Error: Insufficient data points")
        return None
    
    print(f"Running backtest on {len(dates)} data points...")
    
    # Backtest loop
    equity_curve = []
    trades = []
    
    for i, date in enumerate(dates):
        if i < 50:  # Need some history for indicators
            continue
        
        # Get data up to current date
        xiv_hist = data[config['symbols']['volatility_short']].loc[:date]
        vxx_hist = data[config['symbols']['volatility_long']].loc[:date]
        bond_hist = data[config['symbols']['bond']].loc[:date]
        
        # Resample to 2-hour bars
        xiv_2h = strategy.resample_to_2hour(xiv_hist.tail(400))
        vxx_2h = strategy.resample_to_2hour(vxx_hist.tail(400))
        
        # Generate signals
        signals = strategy.generate_signals(xiv_2h, vxx_2h)
        
        # Check panic exit
        if strategy.check_panic_exit(xiv_hist.tail(120)):
            if strategy.xiv_position > 0:
                trade = strategy.execute_trade(
                    config['symbols']['volatility_short'],
                    strategy.xiv_position,
                    xiv_hist['Close'].iloc[-1],
                    'sell',
                    date
                )
                trades.append(trade)
        
        # Check stop losses
        if strategy.check_xiv_stop_loss(xiv_hist['Close'].iloc[-1]):
            if strategy.xiv_position > 0:
                trade = strategy.execute_trade(
                    config['symbols']['volatility_short'],
                    strategy.xiv_position,
                    xiv_hist['Close'].iloc[-1],
                    'sell',
                    date
                )
                trades.append(trade)
        
        vxx_stop, reason = strategy.check_vxx_stops(vxx_hist['Close'].iloc[-1])
        if vxx_stop:
            if strategy.vxx_position > 0:
                trade = strategy.execute_trade(
                    config['symbols']['volatility_long'],
                    strategy.vxx_position,
                    vxx_hist['Close'].iloc[-1],
                    'sell',
                    date
                )
                trades.append(trade)
        
        # Execute signals
        if signals['xiv_buy']:
            shares = strategy.calculate_position_size(
                xiv_hist['Close'].iloc[-1],
                config['parameters']['vol_allocation']
            )
            trade = strategy.execute_trade(
                config['symbols']['volatility_short'],
                shares,
                xiv_hist['Close'].iloc[-1],
                'buy',
                date
            )
            trades.append(trade)
        
        if signals['vxx_buy']:
            shares = strategy.calculate_position_size(
                vxx_hist['Close'].iloc[-1],
                config['parameters']['vol_allocation']
            )
            trade = strategy.execute_trade(
                config['symbols']['volatility_long'],
                shares,
                vxx_hist['Close'].iloc[-1],
                'buy',
                date
            )
            trades.append(trade)
        
        # Calculate portfolio value
        pv = strategy.get_portfolio_value(
            xiv_hist['Close'].iloc[-1],
            vxx_hist['Close'].iloc[-1],
            bond_hist['Close'].iloc[-1]
        )
        
        equity_curve.append({'date': date, 'value': pv})
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    metrics = calculate_metrics(equity_df['value'])
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value}")
    
    print(f"\nTotal Trades: {len(trades_df)}")
    
    # Plot results
    if config['reporting']['plot_results']:
        plot_results(equity_df, trades_df, config)
    
    return {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics
    }


def plot_results(equity_df, trades_df, config):
    """Plot backtest results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Equity curve
    ax1.plot(equity_df.index, equity_df['value'], label='Strategy', linewidth=2)
    ax1.set_title('XIV/VXX Volatility Strategy - Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    cummax = equity_df['value'].cummax()
    drawdown = (equity_df['value'] - cummax) / cummax * 100
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xiv_vxx_backtest_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'xiv_vxx_backtest_results.png'")
    plt.show()


if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run backtest
    results = run_backtest(config)
