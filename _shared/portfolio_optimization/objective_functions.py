"""Objective functions for portfolio optimization."""

import cvxpy as cp
import numpy as np


def _objective_value(w, obj):
    """Return objective value or expression depending on w type."""
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj


def portfolio_variance(w, cov_matrix):
    """Calculate portfolio variance."""
    variance = cp.quad_form(w, cov_matrix, assume_PSD=True)
    return _objective_value(w, variance)


def portfolio_return(w, expected_returns, negative=True):
    """Calculate (negative) mean return of portfolio."""
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)


def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.0, negative=True):
    """Calculate (negative) Sharpe ratio."""
    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix, assume_PSD=True))
    sign = -1 if negative else 1
    sharpe = (mu - risk_free_rate) / sigma
    return _objective_value(w, sign * sharpe)


def L2_reg(w, gamma=1):
    """L2 regularization to increase number of nonzero weights."""
    L2_reg = gamma * cp.sum_squares(w)
    return _objective_value(w, L2_reg)


def quadratic_utility(w, expected_returns, cov_matrix, risk_aversion, negative=True):
    """Quadratic utility function: μ - (1/2)δw^TΣw."""
    sign = -1 if negative else 1
    mu = w @ expected_returns
    variance = cp.quad_form(w, cov_matrix, assume_PSD=True)

    risk_aversion_par = cp.Parameter(value=risk_aversion, name="risk_aversion", nonneg=True)
    utility = mu - 0.5 * risk_aversion_par * variance
    return _objective_value(w, sign * utility)


def transaction_cost(w, w_prev, k=0.001):
    """Simple transaction cost model (fixed percentage commission)."""
    return _objective_value(w, k * cp.norm(w - w_prev, 1))


def ex_ante_tracking_error(w, cov_matrix, benchmark_weights):
    """Calculate ex-ante tracking error: (w - w_b)^TΣ(w - w_b)."""
    relative_weights = w - benchmark_weights
    tracking_error = cp.quad_form(relative_weights, cov_matrix)
    return _objective_value(w, tracking_error)


def ex_post_tracking_error(w, historic_returns, benchmark_returns):
    """Calculate ex-post tracking error: Var(r - r_b)."""
    if not isinstance(historic_returns, np.ndarray):
        historic_returns = np.array(historic_returns)
    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    x_i = w @ historic_returns.T - benchmark_returns
    mean = cp.sum(x_i) / len(benchmark_returns)
    tracking_error = cp.sum_squares(x_i - mean)
    return _objective_value(w, tracking_error)
