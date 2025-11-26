# ArbiterLabs - Repository Architecture

## 🎯 Vision
An open-source collection of production-ready quantitative trading strategies. Each strategy is self-contained, documented, and deployable. Grab a folder, run it, profit (or learn why not).

---

## 📁 Repository Structure

```
arbiterlabs/
│
├── README.md                       # Project overview, quick start, contribution guide
├── LICENSE                         # MIT recommended for max adoption
├── CONTRIBUTING.md                 # How to add new strategies
├── requirements-base.txt           # Shared dependencies (numpy, pandas, etc.)
├── .gitignore
│
├── _templates/                     # Strategy template for contributors
│   ├── strategy_template/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── config.yaml
│   │   ├── strategy.py
│   │   ├── backtest.py
│   │   ├── live.py
│   │   └── tests/
│   │       └── test_strategy.py
│
├── _shared/                        # Optional shared utilities (strategies can copy what they need)
│   ├── data_loaders/
│   │   ├── yfinance_loader.py
│   │   ├── binance_loader.py
│   │   ├── mt5_loader.py
│   │   └── csv_loader.py
│   ├── risk/
│   │   ├── position_sizing.py
│   │   ├── kelly_criterion.py
│   │   └── risk_parity.py
│   ├── execution/
│   │   ├── broker_base.py
│   │   ├── alpaca_executor.py
│   │   ├── binance_executor.py
│   │   └── paper_trader.py
│   ├── metrics/
│   │   ├── performance.py          # Sharpe, Sortino, Calmar, etc.
│   │   ├── drawdown.py
│   │   └── risk_adjusted.py
│   └── utils/
│       ├── logger.py
│       ├── config_loader.py
│       └── time_utils.py
│
├── docs/
│   ├── STRATEGY_GUIDE.md           # How to build a strategy
│   ├── MATHEMATICAL_FOUNDATIONS.md # The quant taxonomy reference
│   ├── BACKTESTING_BEST_PRACTICES.md
│   └── DEPLOYMENT.md
│
│
│   #=============================================================
│   #  STRATEGIES - ORGANIZED BY CATEGORY
│   #=============================================================
│
├── mean_reversion/
│   │
│   ├── pairs_trading_cointegration/
│   │   ├── README.md               # Strategy explanation, math, expected performance
│   │   ├── requirements.txt        # Strategy-specific deps
│   │   ├── config.yaml             # Parameters, symbols, timeframes
│   │   ├── strategy.py             # Core logic (signals, entries, exits)
│   │   ├── backtest.py             # Self-contained backtester
│   │   ├── optimize.py             # Parameter optimization
│   │   ├── live.py                 # Live trading script
│   │   ├── data/
│   │   │   └── sample_data.csv     # Sample data for quick testing
│   │   ├── results/
│   │   │   ├── backtest_results.json
│   │   │   └── equity_curve.png
│   │   └── tests/
│   │       └── test_strategy.py
│   │
│   ├── bollinger_mean_reversion/
│   │   └── ... (same structure)
│   │
│   ├── ornstein_uhlenbeck/
│   │   └── ...
│   │
│   └── zscore_mean_reversion/
│       └── ...
│
├── momentum/
│   ├── dual_momentum/
│   ├── momentum_breakout/
│   ├── rsi_divergence/
│   ├── macd_crossover_enhanced/
│   ├── rate_of_change_momentum/
│   └── relative_strength_rotation/
│
├── trend_following/
│   ├── turtle_trading/
│   ├── moving_average_crossover/
│   ├── adaptive_moving_average/
│   ├── supertrend_strategy/
│   ├── donchian_breakout/
│   ├── keltner_channel_breakout/
│   └── parabolic_sar_trend/
│
├── statistical_arbitrage/
│   ├── pairs_trading_ml/
│   ├── basket_trading/
│   ├── index_arbitrage/
│   ├── etf_arbitrage/
│   └── cross_exchange_arb/
│
├── market_making/
│   ├── basic_market_maker/
│   ├── avellaneda_stoikov/
│   ├── inventory_based_mm/
│   └── adaptive_spread_mm/
│
├── machine_learning/
│   ├── random_forest_classifier/
│   ├── lstm_price_prediction/
│   ├── xgboost_signal_generator/
│   ├── reinforcement_learning_dqn/
│   ├── transformer_price_forecast/
│   └── ensemble_voting_strategy/
│
├── options/
│   ├── delta_neutral_hedging/
│   ├── iron_condor_systematic/
│   ├── volatility_arbitrage/
│   ├── gamma_scalping/
│   └── covered_call_wheel/
│
├── volatility/
│   ├── volatility_breakout/
│   ├── garch_volatility_trading/
│   ├── vix_mean_reversion/
│   ├── implied_vs_realized/
│   └── volatility_regime_switching/
│
├── smart_money_concepts/
│   ├── order_block_strategy/
│   ├── fair_value_gap_trading/
│   ├── liquidity_sweep/
│   ├── market_structure_break/
│   ├── optimal_trade_entry/
│   └── institutional_candle_patterns/
│
├── high_frequency/
│   ├── order_flow_imbalance/
│   ├── microstructure_alpha/
│   ├── latency_arbitrage/
│   └── queue_position_strategy/
│
├── sentiment/
│   ├── news_sentiment_nlp/
│   ├── social_media_sentiment/
│   ├── fear_greed_index/
│   └── put_call_ratio_sentiment/
│
├── seasonal_calendar/
│   ├── day_of_week_effect/
│   ├── month_end_rebalancing/
│   ├── earnings_drift/
│   ├── holiday_effect/
│   └── sector_rotation_seasonal/
│
├── multi_asset/
│   ├── risk_parity_portfolio/
│   ├── black_litterman_allocation/
│   ├── hierarchical_risk_parity/
│   └── momentum_across_assets/
│
└── experimental/
    ├── genetic_algorithm_evolved/
    ├── neural_architecture_search/
    ├── alternative_data_signals/
    └── quantum_inspired_optimization/
```

---

## 📄 Strategy Folder Standard Structure

Every strategy folder MUST contain:

```
strategy_name/
├── README.md           # REQUIRED - Strategy documentation
├── requirements.txt    # REQUIRED - Dependencies (pip install -r requirements.txt)
├── config.yaml         # REQUIRED - All parameters, easily editable
├── strategy.py         # REQUIRED - Core strategy class
├── backtest.py         # REQUIRED - Run: python backtest.py
├── live.py             # REQUIRED - Run: python live.py
├── optimize.py         # OPTIONAL - Parameter optimization
├── data/               # OPTIONAL - Sample data
├── results/            # OPTIONAL - Pre-computed results
├── tests/              # RECOMMENDED - Unit tests
└── notebooks/          # OPTIONAL - Jupyter analysis
```

---

## 📋 README.md Template for Each Strategy

```markdown
# Strategy Name

## Overview
Brief description of the strategy logic.

## Mathematical Foundation
- Core equations/formulas
- Statistical assumptions
- Edge hypothesis

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| lookback  | 20      | Lookback period for calculation |
| threshold | 2.0     | Entry threshold (z-score) |

## Performance Summary
- Sharpe Ratio: X.XX
- Max Drawdown: XX%
- Win Rate: XX%
- Profit Factor: X.XX

## Quick Start
```bash
cd strategy_name
pip install -r requirements.txt
python backtest.py
```

## Data Requirements
- Asset classes: Equities/Forex/Crypto
- Timeframe: Daily/Hourly/etc.
- Minimum history: X bars

## Dependencies
- numpy
- pandas
- (strategy-specific deps)

## References
- Paper/book citations
- Original source if adapted

## Author
- Contributor name
- Date added
```

---

## 📋 config.yaml Template

```yaml
# Strategy Configuration
strategy:
  name: "pairs_trading_cointegration"
  version: "1.0.0"

# Trading Parameters
parameters:
  lookback_period: 60
  entry_zscore: 2.0
  exit_zscore: 0.5
  stop_loss_zscore: 3.5

# Risk Management
risk:
  max_position_size: 0.1        # 10% of portfolio
  max_drawdown_exit: 0.15       # Exit all if 15% drawdown
  position_sizing: "kelly"       # kelly, fixed, volatility_adjusted

# Data Configuration
data:
  symbols: ["AAPL", "MSFT"]
  timeframe: "1d"
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  data_source: "yfinance"        # yfinance, binance, csv

# Execution
execution:
  broker: "paper"                # paper, alpaca, binance
  slippage_bps: 5
  commission_bps: 10

# Logging
logging:
  level: "INFO"
  save_trades: true
  save_equity_curve: true
```

---

## 🚀 Usage Examples

### Quick Backtest
```bash
cd arbiterlabs/mean_reversion/pairs_trading_cointegration
pip install -r requirements.txt
python backtest.py
```

### With Custom Config
```bash
python backtest.py --config my_config.yaml
```

### Optimize Parameters
```bash
python optimize.py --metric sharpe --trials 1000
```

### Go Live (Paper First!)
```bash
python live.py --mode paper
python live.py --mode live  # When ready
```

---

## 🤝 Contributing

1. Fork the repo
2. Copy `_templates/strategy_template/` to appropriate category
3. Implement your strategy following the standard structure
4. Include backtest results with at least 2 years of data
5. Write tests
6. Submit PR

### Quality Checklist
- [ ] README.md complete with math explanation
- [ ] config.yaml with sensible defaults
- [ ] Backtest shows realistic results (no lookahead bias)
- [ ] Tests pass
- [ ] Code is clean and documented
- [ ] Sample data included (or clear data source instructions)
