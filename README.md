# ArbiterLabs 🎯

> An open-source collection of production-ready quantitative trading strategies.

Each strategy is self-contained, documented, and deployable. Grab a folder, run it, profit (or learn why not).

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/arbiterlabs.git
cd arbiterlabs

# Install base dependencies
pip install -r requirements-base.txt

# Navigate to a strategy
cd mean_reversion/pairs_trading_cointegration

# Install strategy-specific dependencies
pip install -r requirements.txt

# Run backtest
python backtest.py
```

---

## 📚 Strategy Categories

### 📊 Mean Reversion
- **Pairs Trading (Cointegration)** - Statistical pairs trading using cointegration
- **Bollinger Mean Reversion** - Mean reversion using Bollinger Bands
- **Ornstein-Uhlenbeck** - Mean reversion based on OU process
- **Z-Score Mean Reversion** - Standard deviation-based mean reversion

### 🚀 Momentum
- **Dual Momentum** - Relative and absolute momentum strategies
- **Momentum Breakout** - Price momentum breakout strategies
- **RSI Divergence** - Trading RSI divergence signals
- **MACD Crossover Enhanced** - Advanced MACD-based momentum
- **Rate of Change Momentum** - ROC-based momentum trading
- **Relative Strength Rotation** - Sector/asset rotation based on RS

### 📈 Trend Following
- **Turtle Trading** - Classic Turtle Trading System
- **Moving Average Crossover** - MA-based trend following
- **Adaptive Moving Average** - Dynamic MA adjustments
- **Supertrend Strategy** - Supertrend indicator system
- **Donchian Breakout** - Channel breakout strategy
- **Keltner Channel Breakout** - Volatility-based breakouts
- **Parabolic SAR Trend** - SAR-based trend trading

### 🔀 Statistical Arbitrage
- **Pairs Trading (ML)** - Machine learning enhanced pairs
- **Basket Trading** - Multi-asset statistical arbitrage
- **Index Arbitrage** - Index vs constituents arbitrage
- **ETF Arbitrage** - ETF creation/redemption arbitrage
- **Cross-Exchange Arbitrage** - Cross-exchange price differences

### 🏦 Market Making
- **Basic Market Maker** - Simple bid-ask market making
- **Avellaneda-Stoikov** - Optimal market making model
- **Inventory-Based MM** - Inventory risk management
- **Adaptive Spread MM** - Dynamic spread adjustment

### 🤖 Machine Learning
- **Random Forest Classifier** - RF-based signal generation
- **LSTM Price Prediction** - Recurrent neural networks
- **XGBoost Signal Generator** - Gradient boosting signals
- **Reinforcement Learning (DQN)** - Deep Q-learning trading
- **Transformer Price Forecast** - Transformer models
- **Ensemble Voting Strategy** - Combined ML predictions

### 📉 Options
- **Delta Neutral Hedging** - Delta-neutral option positions
- **Iron Condor Systematic** - Automated iron condor strategy
- **Volatility Arbitrage** - Trading implied vs realized vol
- **Gamma Scalping** - Delta hedging for gamma profit
- **Covered Call Wheel** - Systematic covered call writing

### 💨 Volatility
- **Volatility Breakout** - Trading volatility expansions
- **GARCH Volatility Trading** - GARCH model-based trading
- **VIX Mean Reversion** - VIX-based strategies
- **Implied vs Realized** - IV-RV spread trading
- **Volatility Regime Switching** - Regime detection strategies

### 💰 Smart Money Concepts
- **Order Block Strategy** - Institutional order blocks
- **Fair Value Gap Trading** - FVG identification and trading
- **Liquidity Sweep** - Liquidity grab patterns
- **Market Structure Break** - BOS/CHoCH trading
- **Optimal Trade Entry** - OTE Fibonacci entries
- **Institutional Candle Patterns** - Smart money patterns

### ⚡ High Frequency
- **Order Flow Imbalance** - Microstructure imbalances
- **Microstructure Alpha** - Market microstructure signals
- **Latency Arbitrage** - Speed-based arbitrage
- **Queue Position Strategy** - Order book positioning

### 📰 Sentiment
- **News Sentiment NLP** - Natural language processing
- **Social Media Sentiment** - Twitter/Reddit analysis
- **Fear & Greed Index** - Market sentiment indicators
- **Put/Call Ratio Sentiment** - Options-based sentiment

### 📅 Seasonal/Calendar
- **Day of Week Effect** - Weekday anomalies
- **Month-End Rebalancing** - Month-end flows
- **Earnings Drift** - Post-earnings momentum
- **Holiday Effect** - Pre/post-holiday patterns
- **Sector Rotation (Seasonal)** - Calendar-based rotation

### 🌐 Multi-Asset
- **Risk Parity Portfolio** - Equal risk contribution
- **Black-Litterman Allocation** - Views-based allocation
- **Hierarchical Risk Parity** - HRP portfolio construction
- **Momentum Across Assets** - Cross-asset momentum

### 🧪 Experimental
- **Genetic Algorithm Evolved** - GA-optimized strategies
- **Neural Architecture Search** - AutoML for trading
- **Alternative Data Signals** - Non-traditional data sources
- **Quantum-Inspired Optimization** - Quantum algorithms

---

## 📖 Documentation

- [Strategy Development Guide](docs/STRATEGY_GUIDE.md)
- [Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md)
- [Backtesting Best Practices](docs/BACKTESTING_BEST_PRACTICES.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps
1. Fork the repository
2. Copy `_templates/strategy_template/` to appropriate category
3. Implement your strategy
4. Include backtest results (minimum 2 years of data)
5. Write tests
6. Submit a pull request

---

## 📋 Strategy Quality Standards

Every strategy must include:
- ✅ Clear mathematical explanation
- ✅ Realistic backtest (no lookahead bias)
- ✅ Risk management implementation
- ✅ Unit tests
- ✅ Documentation with references
- ✅ Sample data or clear data source instructions

---

## ⚠️ Disclaimer

This repository is for **educational purposes only**. All strategies are provided as-is with no guarantees. Past performance does not indicate future results. Always paper trade before deploying real capital. Trading involves substantial risk of loss.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

## 📬 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/arbiterlabs/issues)
- Discussions: [Join the community](https://github.com/yourusername/arbiterlabs/discussions)

---

**Built with ❤️ by the quantitative trading community**
