import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from strategy import RSI2Strategy
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest():
    config = load_config()
    
    print("--- Starting Backtest ---")
    print(f"Strategy: {config['strategy']['name']}")
    
    # 1. Data Fetching
    equity_symbol = config['symbols']['equity']
    bond_symbol = config['symbols']['bond']
    symbols = [equity_symbol, bond_symbol]
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"Fetching data for {symbols} from {start_date} to {end_date}...")
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']
    
    if data.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Initialize Strategy
    strategy = RSI2Strategy(config)
    
    # 3. Simulation Loop
    # We iterate daily
    portfolio_values = []
    
    print("Running simulation...")
    
    for date, prices in data.iterrows():
        # Get prices for this day
        # Note: In a real backtest, we'd use 'Open' for execution if signal came from previous 'Close'
        # Here we simplify and assume we trade at 'Close' based on 'Close' (Lookahead bias warning in real life!)
        # To fix lookahead, we should calculate signal on day T and trade on T+1 Open.
        # For this simple example, we'll calculate signal on T and assume we trade at T Close (which is possible if we run MOC orders).
        
        equity_price = prices[equity_symbol]
        bond_price = prices[bond_symbol]
        
        # Update strategy state (calculate indicators, check alerts)
        # We need history up to this point
        history = data.loc[:date]
        
        # Pass history to strategy (we need to modify strategy to accept history or just current price if it maintains state)
        # The current strategy implementation calculates RSI on the whole series passed to set_alerts.
        # So we pass the history.
        
        strategy.set_alerts(pd.DataFrame({'Close': history[equity_symbol]}), 
                            pd.DataFrame({'Close': history[bond_symbol]}))
        
        # Execute trades
        strategy.rebalance(equity_price, bond_price, date)
        
        # Record value
        val = strategy.get_portfolio_value(equity_price, bond_price)
        portfolio_values.append(val)
        
    # 4. Results
    results = pd.Series(portfolio_values, index=data.index)
    print(f"Final Equity: ${results.iloc[-1]:.2f}")
    
    # Calculate Sharpe
    returns = results.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252**0.5) # Annualized
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
