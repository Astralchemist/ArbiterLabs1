"""Mean-semivariance efficient frontier optimization."""

import cvxpy as cp
import numpy as np

from .. import objective_functions
from .efficient_frontier import EfficientFrontier


class EfficientSemivariance(EfficientFrontier):
    """Mean-semivariance efficient frontier optimizer (downside deviation)."""

    def __init__(
        self,
        expected_returns,
        returns,
        frequency=252,
        benchmark=0,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for semideviation only.
        :type expected_returns: pd.Series, list, np.ndarray
        :param returns: (historic) returns for all your assets (no NaNs).
                                 See ``expected_returns.returns_from_prices``.
        :type returns: pd.DataFrame or np.array
        :param frequency: number of time periods in a year, defaults to 252 (the number
                          of trading days in a year). This must agree with the frequency
                          parameter used in your ``expected_returns``.
        :type frequency: int, optional
        :param benchmark: the return threshold to distinguish "downside" and "upside".
                          This should match the frequency of your ``returns``,
                          i.e this should be a benchmark daily returns if your
                          ``returns`` are also daily.
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`
        :type solver: str
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        """
        # Instantiate parent
        super().__init__(
            expected_returns=expected_returns,
            cov_matrix=np.zeros((returns.shape[1],) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        self.returns = self._validate_returns(returns)
        self.benchmark = benchmark
        self.frequency = frequency
        self._T = self.returns.shape[0]

    def min_volatility(self):
        raise NotImplementedError("Please use min_semivariance instead.")

    def max_sharpe(self, risk_free_rate=0.0):
        raise NotImplementedError("Method not available in EfficientSemivariance")

    def min_semivariance(self, market_neutral=False):
        """Minimize portfolio semivariance (downside deviation)."""
        p = cp.Variable(self._T, nonneg=True)
        n = cp.Variable(self._T, nonneg=True)
        self._objective = cp.sum(cp.square(n))

        for obj in self._additional_objectives:
            self._objective += obj

        B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
        self.add_constraint(lambda w: B @ w - p + n == 0)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        """Maximize quadratic utility using semivariance."""
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        update_existing_parameter = self.is_parameter_defined("risk_aversion")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("risk_aversion", risk_aversion)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            mu = objective_functions.portfolio_return(self._w, self.expected_returns)
            mu /= self.frequency
            risk_aversion_par = cp.Parameter(
                value=risk_aversion, name="risk_aversion", nonneg=True
            )
            self._objective = mu + 0.5 * risk_aversion_par * cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj

            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_semideviation, market_neutral=False):
        """Maximize return for target semideviation."""
        update_existing_parameter = self.is_parameter_defined("target_semivariance")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_semivariance", target_semideviation**2)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            for obj in self._additional_objectives:
                self._objective += obj

            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)

            target_semivariance = cp.Parameter(
                value=target_semideviation**2, name="target_semivariance", nonneg=True
            )
            self.add_constraint(
                lambda _: self.frequency * cp.sum(cp.square(n)) <= target_semivariance
            )
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """Minimize semideviation for target return."""
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if target_return > np.abs(self.expected_returns).max():
            raise ValueError(
                "target_return must be lower than the largest expected return"
            )

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            self._objective = cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(
                lambda w: cp.sum(w @ self.expected_returns) >= target_return_par
            )
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """Calculate expected return, semideviation, and Sortino ratio."""
        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )

        portfolio_returns = self.returns @ self.weights
        drops = np.fmin(portfolio_returns - self.benchmark, 0)
        semivariance = np.sum(np.square(drops)) / self._T * self.frequency
        semi_deviation = np.sqrt(semivariance)
        sortino_ratio = (mu - risk_free_rate) / semi_deviation

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual semi-deviation: {:.1f}%".format(100 * semi_deviation))
            print("Sortino Ratio: {:.2f}".format(sortino_ratio))

        return mu, semi_deviation, sortino_ratio
