import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from strategy import ParabolicSARStrategy
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest():
    config = load_config()
    print(f"--- Starting Backtest: {config['strategy']['name']} ---")
    
    symbol = config['data']['symbols'][0]
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"Fetching data for {symbol}...")
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    if data.empty:
        print("No data.")
        return
        
    strategy = ParabolicSARStrategy(config)
    
    # Generate signals for whole history
    trends, sars = strategy.generate_signals(data)
    
    # Simulate trading
    portfolio_values = []
    
    # Fix short selling cash logic in strategy first? 
    # For this simple backtest, let's just track the strategy's signal performance vs buy & hold.
    
    data['SAR'] = sars
    data['Trend'] = trends
    
    # Vectorized PnL
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Trend'].shift(1) # Trade at Open based on yesterday Trend? Or Close?
    # Strategy generates signal at Close. We trade next Open?
    # Let's assume we trade at Close for simplicity (Trend[i] determines position for next day return?)
    # If Trend[i] is 1, we hold Long for i+1.
    
    data['Equity'] = (1 + data['Strategy_Returns']).cumprod() * 100000
    
    print(f"Final Equity: ${data['Equity'].iloc[-1]:.2f}")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.figure(figsize=(10, 6))
    data['Equity'].plot(title=f"Equity Curve - {config['strategy']['name']}")
    plt.savefig('results/equity_curve.png')
    print("Saved equity_curve.png")
    
    # Plot SAR
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'][-100:], label='Price')
    plt.plot(data['SAR'][-100:], label='SAR', linestyle='--')
    plt.legend()
    plt.savefig('results/sar_chart.png')

if __name__ == "__main__":
    run_backtest()
