import argparse
import yaml
import time

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_live(mode="paper"):
    config = load_config()
    print(f"Starting Live Trading in {{mode.upper()}} mode")
    print(f"Strategy: {{config['strategy']['name']}}")
    
    print("Connecting to broker...")
    print("Listening for market data...")
    try:
        while True:
            print("Heartbeat...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()
    
    run_live(args.mode)
