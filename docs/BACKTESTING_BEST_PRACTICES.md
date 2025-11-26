# Backtesting Best Practices

A comprehensive guide to avoiding common pitfalls and conducting robust backtests.

---

## Table of Contents
1. [Core Principles](#core-principles)
2. [Common Biases](#common-biases)
3. [Data Quality](#data-quality)
4. [Realistic Assumptions](#realistic-assumptions)
5. [Validation Techniques](#validation-techniques)
6. [Reporting Standards](#reporting-standards)

---

## Core Principles

### 1. Out-of-Sample Testing is Mandatory

**Never optimize on the full dataset!**

```
Full Dataset: 2015-2024 (10 years)

✅ CORRECT:
├── Training: 2015-2020 (5 years) → Optimize parameters
├── Validation: 2021-2022 (2 years) → Validate approach
└── Test: 2023-2024 (2 years) → Final performance check

❌ WRONG:
└── Optimize on entire 2015-2024 period
```

### 2. Walk-Forward Analysis

Test strategy on rolling windows:

```python
# Example: 2-year training, 1-year testing
for year in range(2018, 2024):
    train_data = data[f'{year-2}':f'{year}']
    test_data = data[f'{year}':f'{year+1}']

    # Optimize on training data
    optimal_params = optimize(train_data)

    # Test on unseen data
    results = backtest(test_data, optimal_params)
```

### 3. Multiple Market Regimes

Test across different market conditions:
- Bull markets (2009-2020)
- Bear markets (2008, 2020, 2022)
- High volatility (2020 COVID crash)
- Low volatility (2017)
- Rising rates vs. falling rates

---

## Common Biases

### 1. Lookahead Bias

**Using future information in your signals**

❌ **WRONG**:
```python
# This uses tomorrow's price to make today's decision!
data['signal'] = (data['close'].shift(-1) > data['close']).astype(int)
```

✅ **CORRECT**:
```python
# Only uses past information
data['signal'] = (data['close'] > data['close'].shift(1)).astype(int)
```

**Check your code**:
- No `.shift(-n)` with negative values
- No accessing `iloc[i+1]` when deciding at `iloc[i]`
- Be careful with `rolling()` calculations

### 2. Survivorship Bias

**Only testing on stocks that survived to today**

❌ **WRONG**:
- Backtest on current S&P 500 constituents

✅ **CORRECT**:
- Use point-in-time constituent lists
- Include delisted stocks
- Account for bankruptcies

**Impact**: Can inflate returns by 2-3% annually!

```python
# Example: Point-in-time universe
def get_universe(date):
    """Get stocks that existed on this date"""
    return stocks_db.query(
        f"SELECT ticker FROM universe "
        f"WHERE start_date <= '{date}' AND end_date >= '{date}'"
    )
```

### 3. Data Snooping Bias

**Testing multiple strategies on same data until one works**

**Problem**: If you test 100 strategies, ~5 will look good by pure chance (p < 0.05)

**Solutions**:
- Use separate validation set
- Apply Bonferroni correction: `p_adjusted = p_value × n_tests`
- Pre-register your hypothesis
- Use cross-validation

### 4. Selection Bias

**Cherry-picking favorable time periods or assets**

❌ **WRONG**:
- "My strategy worked great in 2017-2020!"
- Only testing on tech stocks
- Removing "outlier" periods that hurt performance

✅ **CORRECT**:
- Test on full available history
- Test on multiple asset classes
- Include crisis periods

---

## Data Quality

### 1. Adjust for Corporate Actions

**Stock Splits**:
```python
# Prices must be split-adjusted
# If 2:1 split on Jan 1, 2020
# Pre-split: $100 → Post-split: $50
# Historical prices before split should be halved
```

**Dividends**:
```python
# For total return calculations
total_return = (price_change + dividends) / initial_price
```

**Mergers & Acquisitions**:
- Handle ticker changes
- Account for merger ratios

### 2. Handle Missing Data

```python
# Check for gaps
missing_days = data.index.to_series().diff()
large_gaps = missing_days[missing_gaps > pd.Timedelta('3 days')]

if len(large_gaps) > 0:
    print(f"Warning: {len(large_gaps)} large gaps in data")

# Forward fill with limit
data = data.fillna(method='ffill', limit=5)
```

### 3. Data Sanity Checks

```python
def validate_ohlcv(data):
    """Validate OHLCV data quality"""

    # Check 1: High >= Low
    assert (data['high'] >= data['low']).all(), "High < Low detected"

    # Check 2: Close within high/low
    assert (data['close'] <= data['high']).all(), "Close > High"
    assert (data['close'] >= data['low']).all(), "Close < Low"

    # Check 3: No negative prices
    assert (data[['open','high','low','close']] > 0).all().all()

    # Check 4: No extreme price jumps (> 50% in one day)
    returns = data['close'].pct_change()
    extreme_moves = returns[abs(returns) > 0.5]
    if len(extreme_moves) > 0:
        print(f"Warning: {len(extreme_moves)} extreme moves detected")

    return True
```

---

## Realistic Assumptions

### 1. Transaction Costs

**Always include**:

```python
# Per-trade costs
slippage = 0.05%        # Price impact
commission = 0.10%      # Broker fees
SEC_fees = 0.0008%      # Regulatory fees (US)
bid_ask_spread = 0.02%  # Half-spread per trade

total_cost = 0.17% per trade
```

**Example Impact**:
```
Strategy: 100 trades/year
Gross return: 15%
Transaction costs: 0.17% × 100 = 17%
Net return: -2%  ❌
```

**Slippage Model**:
```python
def calculate_slippage(order_size, avg_volume, volatility):
    """
    Market impact model
    """
    volume_participation = order_size / avg_volume

    # Square-root market impact
    slippage_bps = 10 * np.sqrt(volume_participation) * volatility

    return slippage_bps / 10000
```

### 2. Execution Delays

**Don't assume instant fills**:

```python
# Signal generated at close
signal_time = market_close  # 4:00 PM

# Realistic execution scenarios:

# 1. Next day open (most common)
execution_time = next_day_open  # 9:30 AM next day
execution_price = next_day_open_price

# 2. After-hours (if available)
execution_time = signal_time + 5_minutes
execution_price = after_hours_price + slippage

# 3. Intraday (for fast strategies)
execution_time = signal_time + 1_second
execution_price = mid_price + slippage
```

### 3. Order Sizes

**Consider market liquidity**:

```python
def check_liquidity(order_size, avg_volume):
    """
    Ensure order is tradeable
    """
    participation_rate = order_size / avg_volume

    if participation_rate > 0.05:  # > 5% of daily volume
        print("⚠️ Warning: Order too large")
        print("May experience significant slippage")
        return False

    return True
```

### 4. Capital Constraints

```python
# Maximum positions
max_positions = 10

# Available capital
cash_buffer = 0.1  # Keep 10% in cash
available_capital = portfolio_value * (1 - cash_buffer)

# Max per position
max_per_position = available_capital / max_positions
```

---

## Validation Techniques

### 1. Monte Carlo Simulation

**Randomize trade sequence to test robustness**:

```python
def monte_carlo_analysis(trades, n_simulations=1000):
    """
    Randomize trade order to see distribution of outcomes
    """
    results = []

    for _ in range(n_simulations):
        # Randomly shuffle trades
        shuffled_trades = trades.sample(frac=1)

        # Calculate cumulative return
        cumulative_return = (1 + shuffled_trades['return']).prod() - 1
        results.append(cumulative_return)

    # Analyze distribution
    percentile_5 = np.percentile(results, 5)
    median = np.percentile(results, 50)
    percentile_95 = np.percentile(results, 95)

    return {
        '5th_percentile': percentile_5,
        'median': median,
        '95th_percentile': percentile_95
    }
```

### 2. Sensitivity Analysis

**Test how sensitive results are to parameters**:

```python
def sensitivity_analysis(base_params):
    """
    Vary each parameter ±20% and observe impact
    """
    results = {}

    for param_name, param_value in base_params.items():
        # Test -20%, base, +20%
        test_values = [
            param_value * 0.8,
            param_value,
            param_value * 1.2
        ]

        sharpe_ratios = []
        for test_value in test_values:
            params = base_params.copy()
            params[param_name] = test_value

            backtest_result = run_backtest(params)
            sharpe_ratios.append(backtest_result['sharpe'])

        # Calculate sensitivity
        sensitivity = (sharpe_ratios[2] - sharpe_ratios[0]) / sharpe_ratios[1]
        results[param_name] = sensitivity

    return results

# High sensitivity (> 0.5) = fragile strategy
# Low sensitivity (< 0.2) = robust strategy
```

### 3. Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(data, n_splits=5):
    """
    Time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Optimize on train
        params = optimize(train_data)

        # Test on validation
        perf = backtest(test_data, params)
        results.append(perf)

    # Average performance across folds
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    std_sharpe = np.std([r['sharpe'] for r in results])

    return avg_sharpe, std_sharpe
```

### 4. Regime Analysis

**How does strategy perform in different market conditions?**

```python
def regime_analysis(returns, market_returns):
    """
    Analyze performance by market regime
    """
    # Define regimes
    bull_mask = market_returns > market_returns.quantile(0.67)
    bear_mask = market_returns < market_returns.quantile(0.33)

    # Performance by regime
    bull_sharpe = sharpe_ratio(returns[bull_mask])
    bear_sharpe = sharpe_ratio(returns[bear_mask])

    print(f"Bull market Sharpe: {bull_sharpe:.2f}")
    print(f"Bear market Sharpe: {bear_sharpe:.2f}")

    return {
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe
    }
```

---

## Reporting Standards

### Required Metrics

Every backtest report must include:

```python
def generate_backtest_report(returns, trades):
    """
    Generate comprehensive backtest report
    """
    report = {
        # Returns
        'total_return': total_return(returns),
        'annual_return': annual_return(returns),
        'monthly_return': returns.resample('M').sum().mean(),

        # Risk metrics
        'sharpe_ratio': sharpe_ratio(returns),
        'sortino_ratio': sortino_ratio(returns),
        'calmar_ratio': calmar_ratio(returns),
        'max_drawdown': max_drawdown(returns),
        'volatility': returns.std() * np.sqrt(252),

        # Trade statistics
        'num_trades': len(trades),
        'win_rate': (trades['pnl'] > 0).mean(),
        'profit_factor': profit_factor(trades),
        'avg_win': trades[trades['pnl'] > 0]['pnl'].mean(),
        'avg_loss': trades[trades['pnl'] < 0]['pnl'].mean(),
        'largest_win': trades['pnl'].max(),
        'largest_loss': trades['pnl'].min(),

        # Time analysis
        'avg_holding_period': trades['holding_period'].mean(),
        'max_holding_period': trades['holding_period'].max(),

        # Consistency
        'pct_profitable_months': (returns.resample('M').sum() > 0).mean(),
        'pct_profitable_years': (returns.resample('Y').sum() > 0).mean(),
    }

    return report
```

### Visualization Requirements

**1. Equity Curve**:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns)
plt.title('Strategy Equity Curve')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
```

**2. Drawdown Chart**:
```python
drawdowns = calculate_drawdowns(returns)
plt.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
plt.title('Drawdown Chart')
```

**3. Monthly Returns Heatmap**:
```python
import seaborn as sns

monthly_returns = returns.resample('M').sum()
heatmap_data = monthly_returns.pivot_table(...)
sns.heatmap(heatmap_data, annot=True, fmt='.1%')
```

---

## Checklist

Before publishing your backtest:

- [ ] Tested on at least 2 years of data
- [ ] Walk-forward analysis performed
- [ ] Transaction costs included (slippage + commissions)
- [ ] No lookahead bias (verified with code review)
- [ ] Survivorship bias addressed
- [ ] Multiple market regimes tested
- [ ] Sensitivity analysis completed
- [ ] Monte Carlo simulation run
- [ ] All required metrics calculated
- [ ] Charts and visualizations generated
- [ ] Code reviewed by another person
- [ ] README.md includes methodology and assumptions

---

## Final Thoughts

> "In theory, there is no difference between theory and practice. In practice, there is."
> — Yogi Berra

**Remember**:
- If it looks too good to be true, it probably is
- Real trading is always messier than backtests
- Always paper trade before risking real capital
- Be conservative with position sizing
- Monitor live performance vs. backtest expectations

---

## Resources

1. **"Advances in Financial Machine Learning"** by Marcos López de Prado
   - Chapter on backtesting pitfalls

2. **"Evidence-Based Technical Analysis"** by David Aronson
   - Statistical validation of trading strategies

3. **"Quantitative Trading"** by Ernest Chan
   - Practical backtesting examples

4. **Papers**:
   - "Pseudo-Mathematics and Financial Charlatanism" (Bailey & López de Prado)
   - "The Probability of Backtest Overfitting" (Bailey et al.)
