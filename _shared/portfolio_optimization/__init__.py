"""Portfolio Optimization Module - integrated from PyPortfolioOpt."""

from .expected_returns import (
    mean_historical_return,
    ema_historical_return,
    capm_return,
    returns_from_prices,
    prices_from_returns,
)

from .risk_models import (
    sample_cov,
    semicovariance,
    exp_cov,
    CovarianceShrinkage,
    cov_to_corr,
    corr_to_cov,
)

__version__ = "1.0.0"

__all__ = [
    "mean_historical_return",
    "ema_historical_return",
    "capm_return",
    "returns_from_prices",
    "prices_from_returns",
    "sample_cov",
    "semicovariance",
    "exp_cov",
    "CovarianceShrinkage",
    "cov_to_corr",
    "corr_to_cov",
]
