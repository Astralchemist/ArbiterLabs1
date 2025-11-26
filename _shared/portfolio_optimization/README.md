# Portfolio Optimization Modules

Portfolio optimization algorithms and risk models integrated from PyPortfolioOpt.

## Overview

This directory contains portfolio construction and optimization tools for multi-asset strategies. These modules complement the individual trading strategies in ArbiterLabs by providing methods to optimally allocate capital across multiple assets.

## Modules

### Expected Returns (`expected_returns.py`)
Methods for estimating future asset returns:
- **Mean Historical Return**: Geometric (CAGR) or arithmetic mean
- **EMA Historical Return**: Exponentially-weighted returns (recent data weighted higher)
- **CAPM Return**: Capital Asset Pricing Model estimates using beta and market returns

### Risk Models (`risk_models.py`)
Covariance matrix estimation methods:
- **Sample Covariance**: Standard historical covariance
- **Exponential Covariance**: Time-decay weighted covariance
- **Semicovariance**: Downside risk focus (returns below benchmark)
- **Shrinkage Methods**: Ledoit-Wolf, Oracle Approximating Shrinkage (OAS)
  - Constant variance target
  - Single factor target
  - Constant correlation target

### Efficient Frontier (`efficient_frontier/`)
Mean-variance optimization:
- **Efficient Frontier**: Classic Markowitz optimization
- **Efficient CVaR**: Conditional Value at Risk optimization
- **Efficient CDaR**: Conditional Drawdown at Risk
- **Efficient Semivariance**: Downside deviation optimization

### Hierarchical Portfolio (`hierarchical_portfolio.py`)
- **Hierarchical Risk Parity (HRP)**: Cluster-based allocation using hierarchical clustering
- Addresses multicollinearity issues in traditional mean-variance

### Black-Litterman (`black_litterman.py`)
- Bayesian approach combining market equilibrium with investor views
- Market-implied prior returns
- Custom view integration

### Discrete Allocation (`discrete_allocation.py`)
- Convert portfolio weights to actual share quantities
- Handle discrete lot sizes and minimum positions

### Base Optimizer (`base_optimizer.py`)
- Abstract base class for all optimizers
- Common interface for portfolio construction

### Objective Functions (`objective_functions.py`)
- Custom optimization objectives
- Sharpe ratio, volatility, return targets
- L2 regularization, transaction costs

## Key Concepts

### Risk-Return Trade-off
All portfolio optimization methods balance expected returns against risk (volatility, drawdown, CVaR, etc.)

### Diversification
- Reduces portfolio-specific risk
- HRP provides diversification without requiring matrix inversion
- Efficient Frontier finds optimal risk-return combinations

### Shrinkage
Improves covariance estimation by shrinking sample covariance toward structured target:
- Reduces estimation error
- Stabilizes optimization
- Ledoit-Wolf provides optimal shrinkage intensity

### Black-Litterman
- Starts with market equilibrium (CAPM)
- Incorporates analyst views with confidence levels
- Produces stable, diversified portfolios

## Usage Example

```python
from portfolio_optimization import expected_returns, risk_models
from portfolio_optimization.efficient_frontier import EfficientFrontier
import pandas as pd

# Load price data
prices = pd.read_csv('_shared/data/stock_prices.csv', index_col='date', parse_dates=True)

# Calculate expected returns and covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Optimize for maximum Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print(cleaned_weights)
```

## Integration with ArbiterLabs

These modules can be used to:
1. **Multi-strategy allocation**: Optimize capital allocation across different trading strategies
2. **Asset universe selection**: Build optimal portfolios from strategy signals
3. **Risk management**: Estimate portfolio risk metrics
4. **Rebalancing**: Periodic portfolio optimization

## Data Requirements

- Historical price data (pandas DataFrame)
- Returns can be computed from prices
- Minimum: 50-100 observations for stable estimates
- More data improves covariance estimation

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn (for shrinkage methods)

## Notes

- Default frequency is 252 (trading days/year)
- Adjust for different timeframes (12 for monthly, 52 for weekly)
- Log returns vs simple returns: generally similar results
- Shrinkage recommended for portfolios with >20 assets
- HRP works well when assets have complex correlation structures
