# Contributing to ArbiterLabs

Thank you for your interest in contributing to ArbiterLabs! This document provides guidelines for adding new strategies and improving existing ones.

---

## 🎯 Contribution Philosophy

We value:
- **Quality over quantity** - One well-documented strategy beats ten half-baked ones
- **Reproducibility** - All results should be reproducible by others
- **Educational value** - Clear explanations help everyone learn
- **Realistic expectations** - No curve-fitted strategies or survivorship bias

---

## 📝 How to Add a New Strategy

### 1. Choose the Right Category

Place your strategy in the appropriate folder:
- `mean_reversion/` - Mean reversion strategies
- `momentum/` - Momentum-based strategies
- `trend_following/` - Trend following systems
- `statistical_arbitrage/` - Statistical arbitrage
- `market_making/` - Market making strategies
- `machine_learning/` - ML-based strategies
- `options/` - Options strategies
- `volatility/` - Volatility trading
- `smart_money_concepts/` - SMC strategies
- `high_frequency/` - HFT strategies
- `sentiment/` - Sentiment-based trading
- `seasonal_calendar/` - Calendar effects
- `multi_asset/` - Multi-asset allocation
- `experimental/` - Experimental/novel approaches

### 2. Use the Template

```bash
# Copy the template
cp -r _templates/strategy_template/ <category>/<your_strategy_name>/

# Navigate to your strategy folder
cd <category>/<your_strategy_name>/
```

### 3. Implement Your Strategy

Your strategy folder must contain:

#### Required Files:
- ✅ `README.md` - Complete documentation
- ✅ `requirements.txt` - All dependencies
- ✅ `config.yaml` - Configuration parameters
- ✅ `strategy.py` - Core strategy logic
- ✅ `backtest.py` - Backtesting script
- ✅ `live.py` - Live trading script
- ✅ `tests/test_strategy.py` - Unit tests

#### Optional Files:
- `optimize.py` - Parameter optimization
- `data/` - Sample data
- `results/` - Pre-computed backtest results
- `notebooks/` - Jupyter notebooks for analysis

### 4. Documentation Requirements

Your `README.md` must include:

1. **Overview**: What does the strategy do?
2. **Mathematical Foundation**:
   - Core equations/formulas
   - Statistical assumptions
   - Why should this work? (edge hypothesis)
3. **Parameters**: Table of all configurable parameters
4. **Performance Summary**: Backtest metrics
5. **Data Requirements**: What data is needed
6. **Quick Start**: How to run it
7. **References**: Academic papers and books

### 5. Code Quality Standards

- **Clean Code**: Follow PEP 8 for Python
- **Type Hints**: Use type annotations where appropriate
- **Docstrings**: Document all functions and classes
- **No Magic Numbers**: Use named constants or config parameters
- **Error Handling**: Proper exception handling
- **Logging**: Use logging instead of print statements

### 6. Backtesting Requirements

Your backtest must:
- ✅ Use **at least 2 years** of historical data
- ✅ Include **realistic transaction costs** (slippage + commissions)
- ✅ Avoid **lookahead bias** (no peeking into future data)
- ✅ Avoid **survivorship bias** (use delisted stocks if applicable)
- ✅ Show **equity curve** and **drawdown chart**
- ✅ Calculate standard metrics:
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Total Return

### 7. Testing Requirements

Write unit tests for:
- Strategy initialization
- Signal generation
- Position sizing
- Trade execution
- Risk management

Run tests:
```bash
python -m pytest tests/
```

### 8. Submit Your Pull Request

1. **Fork** the repository
2. **Create** a new branch: `git checkout -b feature/your-strategy-name`
3. **Commit** your changes: `git commit -am 'Add new strategy: Strategy Name'`
4. **Push** to the branch: `git push origin feature/your-strategy-name`
5. **Submit** a pull request

#### PR Checklist:
- [ ] README.md is complete
- [ ] Code follows style guidelines
- [ ] Backtest results included
- [ ] Tests pass
- [ ] No sensitive data (API keys, credentials)
- [ ] Dependencies documented in requirements.txt

---

## 🔍 Code Review Process

All contributions will be reviewed for:
1. **Correctness** - Does the code work as described?
2. **Quality** - Is the code clean and maintainable?
3. **Documentation** - Is it well-documented?
4. **Testing** - Are there adequate tests?
5. **Originality** - Is this a unique contribution?

---

## 🚫 What We Don't Accept

- Strategies without proper documentation
- Curve-fitted strategies optimized on test data
- Strategies with lookahead bias
- Incomplete implementations
- Code with hardcoded credentials
- Strategies that promote market manipulation
- Plagiarized code without attribution

---

## 💡 Contribution Ideas

Not sure what to contribute? Here are some ideas:

### New Strategies
- Implement a classic strategy from academic literature
- Add a modern twist to an existing strategy
- Port a strategy from another platform

### Improvements
- Add new data sources to `_shared/data_loaders/`
- Improve risk management in `_shared/risk/`
- Add new performance metrics
- Improve documentation
- Fix bugs

### Documentation
- Add tutorials
- Create video walkthroughs
- Write blog posts about strategies
- Improve existing docs

---

## 📚 Resources

- [QuantLib](https://www.quantlib.org/)
- [Quantopian Lectures](https://www.quantopian.com/lectures)
- [SSRN Finance Papers](https://www.ssrn.com/index.cfm/en/janda/financ/)
- [Algorithmic Trading: Winning Strategies](https://www.amazon.com/Algorithmic-Trading-Winning-Strategies-Rationale/dp/1118460146)

---

## 🆘 Getting Help

- **GitHub Issues**: Ask questions or report problems
- **Discussions**: Join the community discussion
- **Discord**: [Join our Discord server](#) (coming soon)

---

## 🎖️ Recognition

Contributors will be:
- Listed in the strategy README.md as authors
- Mentioned in release notes
- Eligible for "Top Contributor" badges

---

## 📜 Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Help newcomers
- Give credit where credit is due
- Follow academic integrity

---

Thank you for contributing to ArbiterLabs! Together, we're building the most comprehensive open-source quantitative trading library. 🚀
