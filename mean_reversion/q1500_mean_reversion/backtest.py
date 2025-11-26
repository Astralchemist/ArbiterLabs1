import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from strategy import Q1500MeanReversion
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest():
    config = load_config()
    
    print("--- Starting Backtest ---")
    print(f"Strategy: {config['strategy']['name']}")
    
    # 1. Data Fetching
    symbols = config['data']['symbols']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}...")
    # Using yfinance to download data
    # Note: For a real Q1500 strategy, you need a dynamic universe of 1500 stocks.
    # This example uses a static list from config.
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']
    
    if data.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Initialize Strategy
    strategy = Q1500MeanReversion(
        lookback=config['parameters']['lookback_window'],
        long_pct=config['parameters']['long_percentile'],
        short_pct=config['parameters']['short_percentile']
    )
    
    # 3. Simulation Loop (Weekly Rebalance)
    # Resample to weekly to simulate weekly rebalancing
    weekly_data = data.resample('W-MON').last()
    
    portfolio_values = [100000.0] # Start with $100k
    current_cash = 100000.0
    current_positions = pd.Series(0.0, index=symbols)
    
    print("Running simulation...")
    
    # Iterate through weeks
    for i in range(len(weekly_data) - 1):
        # Current date (decision time)
        current_date = weekly_data.index[i]
        
        # Get history up to this point for signal generation
        # We need daily data for the lookback calculation
        history = data.loc[:current_date]
        
        # Generate target weights
        target_weights = strategy.generate_signals(history)
        
        # Calculate next week's return
        next_date = weekly_data.index[i+1]
        
        # Prices at rebalance (Close of current_date)
        entry_prices = data.loc[current_date]
        # Prices at next rebalance (Close of next_date)
        exit_prices = data.loc[next_date]
        
        # Calculate portfolio return for this period
        # Simple approximation: assume we rebalance perfectly at Close
        
        # Position values = Total Equity * Target Weights
        equity = portfolio_values[-1]
        target_position_values = target_weights * equity
        
        # Calculate PnL
        # PnL = Position * (Exit - Entry) / Entry
        # But wait, weights are % of equity.
        # PnL = Equity * Weight * (Exit/Entry - 1)
        
        period_returns = (exit_prices / entry_prices) - 1
        period_pnl = (target_position_values * period_returns).sum()
        
        new_equity = equity + period_pnl
        portfolio_values.append(new_equity)
        
    # 4. Results
    results = pd.Series(portfolio_values, index=weekly_data.index)
    print(f"Final Equity: ${results.iloc[-1]:.2f}")
    
    # Calculate Sharpe
    returns = results.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (52**0.5) # Annualized
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Plot
    if not os.path.exists('results'):
        os.makedirs('results')
        
    plt.figure(figsize=(10, 6))
    results.plot(title=f"Equity Curve - {config['strategy']['name']}")
    plt.savefig('results/equity_curve.png')
    print("Equity curve saved to results/equity_curve.png")

if __name__ == "__main__":
    run_backtest()
