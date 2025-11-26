import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from strategy import RSI2SpyTltStrategy
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest():
    config = load_config()
    print(f"--- Starting Backtest: {config['strategy']['name']} ---")
    
    spy_symbol = config['data']['symbols']['equity']
    tlt_symbol = config['data']['symbols']['bond']
    symbols = [spy_symbol, tlt_symbol]
    
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"Fetching data for {symbols}...")
    data = yf.download(symbols, start=start, end=end, progress=False)['Adj Close']
    
    if data.empty:
        print("No data.")
        return
        
    strategy = RSI2SpyTltStrategy(config)
    portfolio_values = []
    
    print("Running simulation...")
    for date, row in data.iterrows():
        spy_price = row[spy_symbol]
        tlt_price = row[tlt_symbol]
        
        # History for signal
        history = data.loc[:date]
        
        # Generate signals
        strategy.generate_signals(
            pd.DataFrame({'Close': history[spy_symbol]}),
            pd.DataFrame({'Close': history[tlt_symbol]})
        )
        
        # Trade
        strategy.rebalance(spy_price, tlt_price, date)
        
        # Record
        portfolio_values.append(strategy.get_portfolio_value(spy_price, tlt_price))
        
    # Results
    results = pd.Series(portfolio_values, index=data.index)
    print(f"Final Equity: ${results.iloc[-1]:.2f}")
    
    returns = results.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252**0.5)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.figure(figsize=(10, 6))
    results.plot(title=f"Equity Curve - {config['strategy']['name']}")
    plt.savefig('results/equity_curve.png')
    print("Saved equity_curve.png")

if __name__ == "__main__":
    run_backtest()
