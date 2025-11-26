import pandas as pd
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from strategy import Strategy
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest():
    config = load_config()
    print(f"--- Starting Backtest: {{config['strategy']['name']}} ---")
    
    symbol = config['data']['symbols'][0]
    start = config['data']['start_date']
    end = config['data']['end_date']
    
    print(f"Fetching data for {{symbol}}...")
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    if data.empty:
        print("No data.")
        return
        
    strategy = Strategy(config)
    
    # Placeholder simulation
    print("Running simulation...")
    
    # Results
    print("Final Equity: $100000.00 (Placeholder)")
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # plt.figure(figsize=(10, 6))
    # plt.plot([100000] * 100)
    # plt.savefig('results/equity_curve.png')
    # print("Saved equity_curve.png")

if __name__ == "__main__":
    run_backtest()
