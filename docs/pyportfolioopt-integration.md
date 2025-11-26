# PyPortfolioOpt Integration Summary

## Overview

Successfully integrated portfolio optimization algorithms from PyPortfolioOpt into ArbiterLabs.

## What Was Integrated

### 1. Core Modules (10 files)

Located in `_shared/portfolio_optimization/`:

#### Risk & Return Estimation
- **`expected_returns.py`** (283 lines)
  - Mean historical returns (geometric & arithmetic)
  - Exponentially-weighted mean returns
  - CAPM-based return estimates
  - Utility functions for price/return conversion

- **`risk_models.py`** (590 lines)
  - Sample covariance matrix
  - Exponentially-weighted covariance
  - Semicovariance (downside risk)
  - Covariance shrinkage (Ledoit-Wolf, OAS)
  - Positive semidefinite matrix fixes
  - Correlation/covariance conversions

#### Portfolio Optimization
- **`efficient_frontier/`** (4 modules)
  - `efficient_frontier.py` - Classic mean-variance optimization
  - `efficient_cvar.py` - CVaR optimization
  - `efficient_cdar.py` - Conditional drawdown optimization
  - `efficient_semivariance.py` - Downside deviation optimization

- **`hierarchical_portfolio.py`**
  - Hierarchical Risk Parity (HRP) implementation
  - Cluster-based portfolio construction

- **`black_litterman.py`**
  - Black-Litterman model for Bayesian portfolio construction
  - Market-implied returns calculation
  - Custom view integration

#### Supporting Modules
- **`base_optimizer.py`** - Abstract base class for optimizers
- **`discrete_allocation.py`** - Convert weights to share quantities
- **`objective_functions.py`** - Custom optimization objectives
- **`exceptions.py`** - Custom exception classes

### 2. Data Files (2 CSV files)

Located in `_shared/data/`:

- **`stock_prices.csv`** (8,797 rows)
  - 19 major stocks: AAPL, AMD, AMZN, BABA, BAC, BBY, GE, GM, GOOG, JPM, MA, META, PFE, RRC, SBUX, T, UAA, WMT, XOM
  - Date range: 1990-01-02 to present
  - Daily adjusted close prices

- **`spy_prices.csv`** (8,018 rows)
  - S&P 500 ETF (SPY) prices
  - Date range: 1993-01-29 to present
  - Useful as market benchmark

## Key Algorithms & Methods

### Return Forecasting
1. **Mean Historical Return**
   - Geometric mean (CAGR) or arithmetic mean
   - Annualized from daily returns
   - Most common baseline method

2. **Exponentially Weighted Mean**
   - Recent data weighted higher (default span=500)
   - Adapts to changing market conditions
   - Better for trending markets

3. **CAPM Return**
   - Beta-based market model
   - Risk-free rate + β × (market return - risk-free rate)
   - Theory-driven approach

### Risk Modeling
1. **Sample Covariance**
   - Standard historical covariance
   - Simple but unstable with limited data

2. **Exponential Covariance**
   - Time-decay weighting
   - Adapts to changing correlations
   - Default span=180 days

3. **Semicovariance**
   - Only considers downside deviations
   - Focuses on loss risk
   - Better for asymmetric returns

4. **Shrinkage Methods**
   - **Ledoit-Wolf**: Optimal shrinkage to constant variance
   - **Single Factor**: Shrink to single-factor model
   - **Constant Correlation**: Shrink to equal correlation
   - **OAS**: Oracle Approximating Shrinkage
   - Reduces estimation error, stabilizes optimization

### Portfolio Construction
1. **Mean-Variance Optimization**
   - Maximize Sharpe ratio
   - Minimize volatility
   - Target return with minimum risk
   - Efficient frontier computation

2. **CVaR Optimization**
   - Minimize Conditional Value at Risk
   - Focus on tail risk
   - Better for skewed return distributions

3. **CDaR Optimization**
   - Minimize Conditional Drawdown at Risk
   - Drawdown-focused risk management
   - Good for equity strategies

4. **Hierarchical Risk Parity (HRP)**
   - Cluster assets by correlation
   - Allocate based on cluster risk
   - No matrix inversion required
   - Robust to estimation error
   - Works well with many assets

5. **Black-Litterman**
   - Start with market equilibrium (CAPM)
   - Add subjective views with confidence
   - Produces stable, intuitive portfolios
   - Avoids extreme concentrated positions

## Use Cases in ArbiterLabs

### 1. Multi-Strategy Allocation
Optimize capital allocation across different trading strategies:
```python
# Combine momentum, mean-reversion, trend-following strategies
strategy_returns = pd.DataFrame({
    'momentum': momentum_strategy_returns,
    'mean_reversion': mean_reversion_returns,
    'trend_following': trend_following_returns
})

mu = expected_returns.mean_historical_return(strategy_returns)
S = risk_models.sample_cov(strategy_returns)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
```

### 2. Asset Selection
Build optimal portfolio from universe of signals:
```python
# Select from 50 candidate assets
asset_prices = load_asset_prices()
mu = expected_returns.ema_historical_return(asset_prices, span=200)
S = risk_models.exp_cov(asset_prices, span=180)

ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: w >= 0)  # Long-only
weights = ef.max_sharpe()
```

### 3. Risk Budgeting
Allocate risk across uncorrelated strategies:
```python
# Use HRP for diversification
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
```

### 4. Tactical Allocation with Views
Incorporate market views into systematic allocation:
```python
# Black-Litterman with macro views
bl = BlackLittermanModel(S, pi=market_implied_returns)
bl.add_view('AAPL', 0.15, confidence=0.3)  # Expect 15% return, 30% confident
posterior_mu = bl.bl_returns()
```

## Integration Benefits

1. **Professional-grade optimization**: Battle-tested algorithms from academic literature
2. **Multiple risk measures**: Variance, CVaR, CDaR, semivariance
3. **Robust estimation**: Shrinkage methods reduce estimation error
4. **Flexibility**: Support for constraints, custom objectives
5. **No reinventing the wheel**: Proven implementations ready to use

## File Structure

```
_shared/
├── portfolio_optimization/
│   ├── __init__.py
│   ├── README.md
│   ├── expected_returns.py
│   ├── risk_models.py
│   ├── hierarchical_portfolio.py
│   ├── black_litterman.py
│   ├── base_optimizer.py
│   ├── discrete_allocation.py
│   ├── objective_functions.py
│   ├── exceptions.py
│   └── efficient_frontier/
│       ├── __init__.py
│       ├── efficient_frontier.py
│       ├── efficient_cvar.py
│       ├── efficient_cdar.py
│       └── efficient_semivariance.py
└── data/
    ├── stock_prices.csv (19 major stocks, 1990-present)
    └── spy_prices.csv (SPY benchmark, 1993-present)
```

## Dependencies Required

Add to `requirements-base.txt`:
```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0  # For shrinkage methods
```

## Next Steps

1. **Create example strategies** using portfolio optimization
2. **Build multi-strategy allocator** combining existing strategies
3. **Add rebalancing logic** for periodic optimization
4. **Integrate with backtesting** framework
5. **Add portfolio analytics** (performance attribution, risk decomposition)

## References

- Original repository: https://github.com/PyPortfolio/PyPortfolioOpt
- Paper: Martin, R. A. (2021). PyPortfolioOpt: portfolio optimization in Python. Journal of Open Source Software, 6(61), 3066
- License: MIT (compatible with ArbiterLabs)

## Notes

- All modules are self-contained and can be used independently
- Default parameters assume daily price data (252 trading days/year)
- Adjust `frequency` parameter for different timeframes
- Shrinkage recommended when assets > 20
- HRP good alternative when mean-variance unstable
