"""Expected returns estimation methods for portfolio optimization."""

import warnings
import numpy as np
import pandas as pd


def _check_returns(returns):
    """Check for NaN and infinite values in returns."""
    if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
        warnings.warn("Some returns are NaN. Please check your price data.", UserWarning)
    if np.any(np.isinf(returns)):
        warnings.warn("Some returns are infinite. Please check your price data.", UserWarning)


def returns_from_prices(prices, log_returns=False):
    """Calculate returns from prices."""
    if log_returns:
        returns = np.log(1 + prices.pct_change(fill_method=None)).dropna(how="all")
    else:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns


def prices_from_returns(returns, log_returns=False):
    """Calculate pseudo-prices from returns (initial prices set to 1)."""
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = 1 + returns
    ret.iloc[0] = 1
    return ret.cumprod()


def return_model(prices, method="mean_historical_return", **kwargs):
    """Compute expected returns using specified model."""
    if method == "mean_historical_return":
        return mean_historical_return(prices, **kwargs)
    elif method == "ema_historical_return":
        return ema_historical_return(prices, **kwargs)
    elif method == "capm_return":
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError("Return model {} not implemented".format(method))


def mean_historical_return(prices, returns_data=False, compounding=True, frequency=252, log_returns=False):
    """Calculate annualized mean historical return (CAGR if compounding=True)."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency


def ema_historical_return(prices, returns_data=False, compounding=True, span=500, frequency=252, log_returns=False):
    """Calculate exponentially-weighted mean historical return."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency


def capm_return(prices, market_prices=None, returns_data=False, risk_free_rate=0.0, compounding=True, frequency=252, log_returns=False):
    """Compute expected returns using CAPM: R_i = R_f + β_i(E(R_m) - R_f)."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    market_returns = None

    if returns_data:
        returns = prices.copy()
        if market_prices is not None:
            market_returns = market_prices
    else:
        returns = returns_from_prices(prices, log_returns)
        if market_prices is not None:
            if not isinstance(market_prices, pd.DataFrame):
                warnings.warn("market prices are not in a dataframe", RuntimeWarning)
                market_prices = pd.DataFrame(market_prices)
            market_returns = returns_from_prices(market_prices, log_returns)

    # Use equally-weighted dataset as market proxy if not provided
    if market_returns is None:
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")

    _check_returns(returns)

    # Compute covariance matrix and betas
    cov = returns.cov()
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")

    # Calculate mean market return
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (frequency / returns["mkt"].count()) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency

    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)
