# Deployment Guide

A comprehensive guide to deploying quantitative trading strategies in production.

---

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Paper Trading](#paper-trading)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Live Trading](#live-trading)
5. [Monitoring](#monitoring)
6. [Risk Management](#risk-management)
7. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

Before going live, verify:

### Strategy Validation
- [ ] Backtested on minimum 2 years of data
- [ ] Walk-forward analysis completed
- [ ] Out-of-sample Sharpe ratio > 1.0
- [ ] Maximum drawdown < 20%
- [ ] Tested across multiple market regimes
- [ ] Sensitivity analysis shows robustness
- [ ] Monte Carlo simulation results acceptable

### Code Quality
- [ ] All unit tests passing
- [ ] Code reviewed by another developer
- [ ] No hardcoded credentials
- [ ] Logging implemented
- [ ] Error handling for all edge cases
- [ ] Configuration externalized to config files

### Risk Controls
- [ ] Position size limits configured
- [ ] Maximum drawdown kill switch implemented
- [ ] Daily loss limits set
- [ ] Exposure limits per asset/sector
- [ ] Emergency shutdown procedure documented

### Operational
- [ ] Paper trading completed (minimum 1 month)
- [ ] Broker account funded
- [ ] API credentials secured
- [ ] Monitoring dashboards setup
- [ ] Alert system configured
- [ ] Documentation completed

---

## Paper Trading

**Never skip paper trading!**

### Setup Paper Trading

```bash
# Run strategy in paper mode
python live.py --mode paper --config config.yaml
```

### What to Monitor

1. **Signal Generation**
   - Signals match backtest expectations?
   - No unexpected edge cases?

2. **Order Execution**
   - Orders filled at expected prices?
   - Slippage within acceptable range?

3. **Risk Management**
   - Position sizes correct?
   - Stop losses working?

4. **System Stability**
   - No crashes or errors?
   - Runs continuously without intervention?

### Paper Trading Duration

| Strategy Frequency | Minimum Paper Trading |
|-------------------|----------------------|
| High Frequency    | 2 weeks             |
| Intraday          | 1 month             |
| Daily             | 2-3 months          |
| Weekly+           | 6 months            |

### Paper Trading Metrics

Compare paper trading to backtest:

```python
def validate_paper_trading(paper_results, backtest_results):
    """
    Verify paper trading matches expectations
    """
    # Sharpe ratio should be within 30% of backtest
    sharpe_diff = abs(paper_results['sharpe'] - backtest_results['sharpe'])
    sharpe_threshold = 0.3 * backtest_results['sharpe']

    if sharpe_diff > sharpe_threshold:
        print("⚠️ WARNING: Paper trading Sharpe significantly different")
        return False

    # Win rate should be similar (within 10%)
    win_rate_diff = abs(paper_results['win_rate'] - backtest_results['win_rate'])

    if win_rate_diff > 0.10:
        print("⚠️ WARNING: Win rate differs from backtest")
        return False

    return True
```

---

## Infrastructure Setup

### Cloud vs Local

**Cloud Hosting** (Recommended):
- ✅ 99.9% uptime
- ✅ Automatic scaling
- ✅ Disaster recovery
- ❌ Monthly costs

**Local Machine**:
- ✅ No recurring costs
- ✅ Full control
- ❌ Requires maintenance
- ❌ Single point of failure

### Recommended Stack

```yaml
Infrastructure:
  Compute: AWS EC2 / Google Cloud Compute / DigitalOcean
  Database: PostgreSQL / TimescaleDB
  Message Queue: Redis / RabbitMQ
  Monitoring: Prometheus + Grafana
  Logging: ELK Stack (Elasticsearch, Logstash, Kibana)

Code:
  Language: Python 3.8+
  Scheduler: cron / systemd / supervisor
  Process Manager: supervisor / pm2
```

### AWS Example Setup

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-xxxxx \
  --instance-type t3.medium \
  --key-name your-key \
  --security-groups trading-sg

# 2. Install dependencies
sudo apt update
sudo apt install python3.8 python3-pip postgresql redis-server

# 3. Clone strategy
git clone https://github.com/yourusername/arbiterlabs.git
cd arbiterlabs/mean_reversion/pairs_trading_cointegration

# 4. Install requirements
pip3 install -r requirements.txt

# 5. Setup supervisor
sudo apt install supervisor
sudo cp deployment/supervisor_config.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
```

### Supervisor Configuration

```ini
# /etc/supervisor/conf.d/trading_strategy.conf
[program:pairs_trading]
command=/usr/bin/python3 /home/ubuntu/arbiterlabs/mean_reversion/pairs_trading_cointegration/live.py --mode live
directory=/home/ubuntu/arbiterlabs/mean_reversion/pairs_trading_cointegration
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/pairs_trading.log
environment=PYTHONUNBUFFERED=1
```

---

## Live Trading

### Configuration

```yaml
# config.live.yaml
strategy:
  name: "pairs_trading"
  mode: "live"

execution:
  broker: "alpaca"  # alpaca, interactive_brokers, binance
  api_key_env: "ALPACA_API_KEY"
  api_secret_env: "ALPACA_API_SECRET"
  base_url: "https://api.alpaca.markets"  # or paper URL

risk:
  max_position_size: 0.05  # Conservative: 5% per position
  max_drawdown_exit: 0.10  # Exit all if 10% drawdown
  daily_loss_limit: 0.02   # Stop trading if lose 2% in one day
  max_positions: 5

monitoring:
  alert_email: "your-email@example.com"
  alert_phone: "+1234567890"
  heartbeat_interval: 300  # seconds
```

### Secure Credentials

**Use environment variables**:

```bash
# .env file (NEVER commit this!)
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_token_here
```

```python
# Load in code
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
```

### Starting Live Trading

```bash
# Final checks
python -m pytest tests/  # Run all tests

# Start in tmux/screen (so it keeps running)
tmux new -s trading
python live.py --mode live --config config.live.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t trading
```

### Gradual Ramp-Up

**Start small and scale gradually**:

```
Week 1: 10% of target capital
Week 2: 25% of target capital
Week 3: 50% of target capital
Week 4+: 100% of target capital (if all going well)
```

---

## Monitoring

### System Health Metrics

Monitor these continuously:

1. **Strategy Performance**
   - PnL (daily, weekly, cumulative)
   - Sharpe ratio (rolling)
   - Drawdown (current)
   - Win rate
   - Number of trades

2. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk space
   - Network latency
   - API rate limits

3. **Data Quality**
   - Data feed uptime
   - Missing data points
   - Stale prices
   - Delayed quotes

### Alerting

```python
def setup_alerts(config):
    """
    Configure alerting system
    """
    # Email alerts
    email_handler = EmailAlert(
        smtp_server='smtp.gmail.com',
        from_addr=config['alert_email'],
        to_addrs=[config['alert_email']]
    )

    # SMS/Phone alerts
    sms_handler = TwilioAlert(
        account_sid=config['twilio_sid'],
        auth_token=config['twilio_token'],
        from_number=config['alert_phone']
    )

    # Alert conditions
    alerts = [
        Alert('high_drawdown', threshold=0.05, handler=sms_handler),
        Alert('large_loss', threshold=0.02, handler=sms_handler),
        Alert('system_error', severity='critical', handler=sms_handler),
        Alert('daily_summary', schedule='18:00', handler=email_handler),
    ]

    return alerts
```

### Monitoring Dashboard

```python
# Using Grafana + Prometheus

# metrics.py
from prometheus_client import Counter, Gauge, Histogram

# Define metrics
trades_total = Counter('trades_total', 'Total number of trades')
pnl_gauge = Gauge('pnl_current', 'Current PnL')
drawdown_gauge = Gauge('drawdown_current', 'Current drawdown')
latency_histogram = Histogram('api_latency_seconds', 'API latency')

# Update metrics
def record_trade(pnl):
    trades_total.inc()
    pnl_gauge.set(pnl)

def record_api_call(duration):
    latency_histogram.observe(duration)
```

### Daily Checklist

Every trading day:

- [ ] Check overnight performance
- [ ] Verify data feeds connected
- [ ] Review any alerts/errors
- [ ] Check position sizes
- [ ] Verify cash balance
- [ ] Monitor drawdown
- [ ] Check market regime
- [ ] Review upcoming news/events

---

## Risk Management

### Circuit Breakers

**Automatic shutdown triggers**:

```python
class CircuitBreaker:
    def __init__(self, config):
        self.max_daily_loss = config['daily_loss_limit']
        self.max_drawdown = config['max_drawdown_exit']
        self.start_balance = self.get_current_balance()

    def check(self):
        """Check if any circuit breaker triggered"""
        current_balance = self.get_current_balance()

        # Daily loss limit
        daily_loss = (current_balance - self.start_balance) / self.start_balance
        if daily_loss < -self.max_daily_loss:
            self.trigger_shutdown("Daily loss limit exceeded")
            return False

        # Maximum drawdown
        drawdown = self.calculate_current_drawdown()
        if drawdown > self.max_drawdown:
            self.trigger_shutdown("Maximum drawdown exceeded")
            return False

        return True

    def trigger_shutdown(self, reason):
        """Emergency shutdown"""
        print(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")

        # Close all positions
        self.close_all_positions()

        # Send alerts
        self.send_alert(reason, severity='CRITICAL')

        # Stop strategy
        sys.exit(1)
```

### Position Limits

```python
def enforce_position_limits(portfolio, new_trade, config):
    """
    Check if new trade violates position limits
    """
    # Max per position
    if new_trade.size > config['max_position_size']:
        print("❌ Trade exceeds max position size")
        return False

    # Max total positions
    if len(portfolio.positions) >= config['max_positions']:
        print("❌ Already at max number of positions")
        return False

    # Concentration limits (no more than 20% in one sector)
    sector_exposure = portfolio.get_sector_exposure(new_trade.symbol)
    if sector_exposure > 0.20:
        print("❌ Sector concentration limit exceeded")
        return False

    return True
```

---

## Troubleshooting

### Common Issues

**1. Strategy Not Generating Signals**
```python
# Debug signal generation
def debug_signals(data):
    print(f"Data shape: {data.shape}")
    print(f"Last close: {data['close'].iloc[-1]}")
    print(f"Indicator values:")
    print(f"  SMA: {data['sma'].iloc[-1]}")
    print(f"  RSI: {data['rsi'].iloc[-1]}")

    signals = generate_signals(data)
    print(f"Signal generated: {signals.iloc[-1]}")
```

**2. Orders Not Filling**
- Check if market is open
- Verify sufficient buying power
- Check order type (market vs limit)
- Review slippage settings

**3. Data Feed Issues**
```python
def check_data_feed():
    """Validate data feed"""
    data = fetch_latest_data()

    # Check freshness
    latest_timestamp = data.index[-1]
    age = datetime.now() - latest_timestamp

    if age > timedelta(minutes=15):
        print("⚠️ WARNING: Stale data detected")
        return False

    # Check for gaps
    expected_bars = calculate_expected_bars()
    actual_bars = len(data)

    if actual_bars < expected_bars * 0.95:
        print("⚠️ WARNING: Missing data")
        return False

    return True
```

### Logging Best Practices

```python
import logging
from _shared.utils.logger import setup_logger

logger = setup_logger('trading_strategy', 'logs/trading.log', level='INFO')

# Log important events
logger.info("Strategy started")
logger.info(f"Trade executed: {trade}")
logger.warning("High drawdown detected")
logger.error("API connection failed", exc_info=True)
logger.critical("Circuit breaker triggered")
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Automated backups
#!/bin/bash

# Backup configuration
cp config.yaml backups/config_$(date +%Y%m%d).yaml

# Backup database
pg_dump trading_db > backups/db_$(date +%Y%m%d).sql

# Backup logs
tar -czf backups/logs_$(date +%Y%m%d).tar.gz logs/

# Upload to S3
aws s3 sync backups/ s3://your-bucket/backups/
```

### Failover Plan

1. **Primary system fails**
   → Alert sent
   → Manual inspection required
   → Secondary system activated

2. **Data feed fails**
   → Switch to backup feed
   → Continue trading

3. **Broker API down**
   → Queue orders
   → Execute when API returns
   → Or use backup broker

---

## Final Checklist

Before going live:

- [ ] Paper trading successful for recommended duration
- [ ] All tests passing
- [ ] Risk limits configured conservatively
- [ ] Circuit breakers tested
- [ ] Monitoring and alerts working
- [ ] Credentials secured
- [ ] Backup system ready
- [ ] Emergency procedures documented
- [ ] Small capital allocation for initial live run
- [ ] Ready to monitor closely for first 2 weeks

---

## Remember

> "The best trading strategy in the world is worthless if you can't execute it properly in production."

- Start small
- Monitor closely
- Be ready to shut down
- Learn from every issue
- Iterate and improve

**Good luck, and trade responsibly!** 🚀

---

## Resources

- **Brokers**:
  - [Alpaca](https://alpaca.markets) - Commission-free US stocks
  - [Interactive Brokers](https://www.interactivebrokers.com) - Global markets
  - [Binance](https://www.binance.com) - Cryptocurrency

- **Cloud Providers**:
  - [AWS](https://aws.amazon.com)
  - [Google Cloud](https://cloud.google.com)
  - [DigitalOcean](https://www.digitalocean.com)

- **Monitoring**:
  - [Grafana](https://grafana.com)
  - [Datadog](https://www.datadoghq.com)
  - [New Relic](https://newrelic.com)
