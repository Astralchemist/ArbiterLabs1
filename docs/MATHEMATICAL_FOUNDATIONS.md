# Mathematical Foundations

A reference guide to the mathematical concepts underlying quantitative trading strategies.

---

## Table of Contents
1. [Statistical Concepts](#statistical-concepts)
2. [Time Series Analysis](#time-series-analysis)
3. [Risk Metrics](#risk-metrics)
4. [Position Sizing](#position-sizing)
5. [Portfolio Theory](#portfolio-theory)
6. [Options Pricing](#options-pricing)

---

## Statistical Concepts

### Mean and Variance

**Mean (Expected Return)**:
```
μ = E[R] = Σ(p_i × r_i)
```

**Variance**:
```
σ² = E[(R - μ)²]
```

**Standard Deviation (Volatility)**:
```
σ = √(variance)
```

### Z-Score (Standardization)

Measures how many standard deviations a value is from the mean:
```
z = (x - μ) / σ
```

**Application**: Mean reversion strategies
- If z > 2: Asset may be overbought
- If z < -2: Asset may be oversold

### Correlation

Measures linear relationship between two assets:
```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
```

Range: -1 to +1
- ρ = +1: Perfect positive correlation
- ρ = 0: No correlation
- ρ = -1: Perfect negative correlation

### Cointegration

Two non-stationary time series that have a stationary linear combination:
```
Y_t = β × X_t + ε_t

where ε_t is stationary
```

**Test**: Augmented Dickey-Fuller (ADF) test on residuals

**Application**: Pairs trading strategies

---

## Time Series Analysis

### Stationarity

A stationary time series has:
- Constant mean over time
- Constant variance over time
- Autocovariance that depends only on lag

**Test**: ADF test
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(price_series)
p_value = result[1]

if p_value < 0.05:
    print("Series is stationary")
```

### Autocorrelation

Correlation of a time series with itself at different lags:
```
ACF(k) = Corr(X_t, X_{t-k})
```

**Application**: Identifies momentum or mean reversion

### Moving Averages

**Simple Moving Average (SMA)**:
```
SMA_t = (P_t + P_{t-1} + ... + P_{t-n+1}) / n
```

**Exponential Moving Average (EMA)**:
```
EMA_t = α × P_t + (1-α) × EMA_{t-1}

where α = 2/(n+1)
```

### Bollinger Bands

```
Middle Band = SMA(P, n)
Upper Band = SMA + k × σ
Lower Band = SMA - k × σ

Typically: n=20, k=2
```

---

## Risk Metrics

### Sharpe Ratio

Risk-adjusted return:
```
Sharpe = (R_p - R_f) / σ_p

R_p = Portfolio return
R_f = Risk-free rate
σ_p = Portfolio standard deviation
```

**Interpretation**:
- < 1.0: Poor
- 1.0 - 2.0: Good
- > 2.0: Excellent

### Sortino Ratio

Like Sharpe, but only penalizes downside volatility:
```
Sortino = (R_p - R_f) / σ_downside

σ_downside = √(E[(min(R_t - R_target, 0))²])
```

### Maximum Drawdown

Largest peak-to-trough decline:
```
DD_t = (Equity_t - Peak_t) / Peak_t

Max DD = min(DD_t) for all t
```

### Calmar Ratio

Annual return divided by maximum drawdown:
```
Calmar = Annual Return / |Max Drawdown|
```

### Value at Risk (VaR)

Maximum expected loss at a confidence level:
```
VaR(95%) = μ - 1.645 × σ  (for normal distribution)
```

**Example**: 95% VaR of $10,000 means:
- 95% confident losses won't exceed $10,000
- 5% chance of losing more than $10,000

### Conditional Value at Risk (CVaR)

Expected loss given that VaR threshold is exceeded:
```
CVaR = E[Loss | Loss > VaR]
```

---

## Position Sizing

### Fixed Fractional

```
Position Size = Capital × f

where f is fixed fraction (e.g., 0.1 = 10%)
```

### Kelly Criterion

Optimal position size for maximum long-term growth:
```
f* = (p × b - q) / b

p = probability of winning
q = probability of losing (1-p)
b = win/loss ratio
```

**Example**:
- Win rate = 60% (p = 0.6)
- Win/loss ratio = 2:1 (b = 2)
```
f* = (0.6 × 2 - 0.4) / 2 = 0.4 = 40%
```

**Warning**: Full Kelly is aggressive; use fractional Kelly (e.g., 0.25× or 0.5×)

### Volatility-Based Sizing

```
Position Size = Target Volatility / Asset Volatility × Capital
```

### ATR-Based Sizing

```
Position Size = Risk per Trade / (ATR × Multiplier)

Shares = Risk Amount / Stop Distance
```

---

## Portfolio Theory

### Modern Portfolio Theory (MPT)

**Portfolio Return**:
```
R_p = Σ(w_i × R_i)

w_i = weight of asset i
```

**Portfolio Variance**:
```
σ_p² = Σ Σ(w_i × w_j × σ_i × σ_j × ρ_{ij})
```

### Efficient Frontier

The set of optimal portfolios offering:
- Maximum return for given risk, OR
- Minimum risk for given return

### Capital Asset Pricing Model (CAPM)

```
E[R_i] = R_f + β_i × (E[R_m] - R_f)

β_i = Cov(R_i, R_m) / Var(R_m)
```

### Risk Parity

Allocate capital so each asset contributes equally to portfolio risk:
```
w_i ∝ 1/σ_i

Normalized: w_i = (1/σ_i) / Σ(1/σ_j)
```

---

## Options Pricing

### Black-Scholes Model

**Call Option Price**:
```
C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)

d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Put Option Price**:
```
P = K × e^(-rT) × N(-d₂) - S₀ × N(-d₁)
```

### The Greeks

**Delta (Δ)**: Sensitivity to underlying price
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma (Γ)**: Rate of change of delta
```
Γ = N'(d₁) / (S × σ × √T)
```

**Theta (Θ)**: Time decay
```
Θ = -[S × N'(d₁) × σ / (2√T)] - r × K × e^(-rT) × N(d₂)
```

**Vega (ν)**: Sensitivity to volatility
```
ν = S × √T × N'(d₁)
```

**Rho (ρ)**: Sensitivity to interest rate
```
ρ = K × T × e^(-rT) × N(d₂)
```

---

## Technical Indicators

### Relative Strength Index (RSI)

```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

**Interpretation**:
- RSI > 70: Overbought
- RSI < 30: Oversold

### MACD (Moving Average Convergence Divergence)

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Signal**: Buy when MACD crosses above signal line

### Average True Range (ATR)

```
TR = max[(H-L), |H-C_prev|, |L-C_prev|]
ATR = SMA(TR, n)  or  EMA(TR, n)

Typically: n=14
```

---

## Statistical Tests

### Augmented Dickey-Fuller (ADF)

Tests for unit root (non-stationarity):
```
H₀: Series has a unit root (non-stationary)
H₁: Series is stationary

If p-value < 0.05: Reject H₀ (stationary)
```

### Jarque-Bera Test

Tests for normality of returns:
```
JB = (n/6) × [S² + (K-3)²/4]

S = skewness
K = kurtosis
```

### Ljung-Box Test

Tests for autocorrelation:
```
H₀: No autocorrelation
If p-value < 0.05: Autocorrelation present
```

---

## Machine Learning Concepts

### Cross-Validation for Time Series

**Expanding Window**:
```
Train: [1, 2, 3, 4, 5] → Test: [6]
Train: [1, 2, 3, 4, 5, 6] → Test: [7]
```

**Rolling Window**:
```
Train: [1, 2, 3, 4, 5] → Test: [6]
Train: [2, 3, 4, 5, 6] → Test: [7]
```

### Overfitting Metrics

**In-Sample vs Out-of-Sample**:
```
Overfitting Index = 1 - (Out-of-Sample Sharpe / In-Sample Sharpe)

< 0.2: Acceptable
> 0.5: Severe overfitting
```

---

## Practical Applications

### Example: Z-Score Mean Reversion

```python
# Calculate z-score
mean = prices.rolling(20).mean()
std = prices.rolling(20).std()
z_score = (prices - mean) / std

# Generate signals
signals = pd.Series(0, index=prices.index)
signals[z_score > 2] = -1  # Sell when overbought
signals[z_score < -2] = 1  # Buy when oversold
```

### Example: Kelly Position Sizing

```python
win_rate = 0.55
win_loss_ratio = 2.0

kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
kelly_fraction = kelly * 0.25  # Use quarter Kelly for safety

position_size = portfolio_value * kelly_fraction
```

---

## References

1. **Statistics**: "Statistical Inference" by Casella & Berger
2. **Time Series**: "Analysis of Financial Time Series" by Tsay
3. **Portfolio Theory**: "Modern Portfolio Theory and Investment Analysis" by Elton et al.
4. **Options**: "Options, Futures, and Other Derivatives" by Hull
5. **Algorithmic Trading**: "Algorithmic Trading" by Chan

---

**Note**: These formulas provide theoretical foundations. Real-world implementation requires careful consideration of market microstructure, transaction costs, and risk management.
