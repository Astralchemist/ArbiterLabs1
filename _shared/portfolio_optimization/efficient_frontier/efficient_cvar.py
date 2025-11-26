"""Mean-CVaR efficient frontier optimization."""

import warnings

import cvxpy as cp
import numpy as np

from .. import objective_functions
from .efficient_frontier import EfficientFrontier


class EfficientCVaR(EfficientFrontier):
    """Mean-CVaR efficient frontier optimizer (Rockafellar & Ursayev 2001)."""

    def __init__(
        self,
        expected_returns,
        returns,
        beta=0.95,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for conditional value at risk only.
        :type expected_returns: pd.Series, list, np.ndarray
        :param returns: (historic) returns for all your assets (no NaNs).
                                 See ``expected_returns.returns_from_prices``.
        :type returns: pd.DataFrame or np.array
        :param beta: confidence level, defauls to 0.95 (i.e expected loss on the worst (1-beta) days).
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
        super().__init__(
            expected_returns=expected_returns,
            cov_matrix=np.zeros((returns.shape[1],) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        self.returns = self._validate_returns(returns)
        self._beta = self._validate_beta(beta)
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.returns))

    def set_weights(self, input_weights):
        raise NotImplementedError("Method not available in EfficientCVaR.")

    @staticmethod
    def _validate_beta(beta):
        if not (0 <= beta < 1):
            raise ValueError("beta must be between 0 and 1")
        if beta <= 0.2:
            warnings.warn(
                "Warning: beta is the confidence-level, not the quantile. Typical values are 80%, 90%, 95%.",
                UserWarning,
            )
        return beta

    def min_volatility(self):
        raise NotImplementedError("Please use min_cvar instead.")

    def max_sharpe(self, risk_free_rate=0.0):
        raise NotImplementedError("Method not available in EfficientCVaR.")

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        raise NotImplementedError("Method not available in EfficientCVaR.")

    def min_cvar(self, market_neutral=False):
        """Minimize portfolio CVaR."""
        self._objective = self._alpha + 1.0 / (
            len(self.returns) * (1 - self._beta)
        ) * cp.sum(self._u)

        for obj in self._additional_objectives:
            self._objective += obj

        self.add_constraint(lambda _: self._u >= 0.0)
        self.add_constraint(
            lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
        )

        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """Minimize CVaR for target return."""
        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            self._objective = self._alpha + 1.0 / (
                len(self.returns) * (1 - self._beta)
            ) * cp.sum(self._u)

            for obj in self._additional_objectives:
                self._objective += obj

            self.add_constraint(lambda _: self._u >= 0.0)
            self.add_constraint(
                lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
            )

            ret = self.expected_returns.T @ self._w
            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)

            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_cvar, market_neutral=False):
        """Maximize return for target CVaR."""
        update_existing_parameter = self.is_parameter_defined("target_cvar")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_cvar", target_cvar)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            for obj in self._additional_objectives:
                self._objective += obj

            cvar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(
                self._u
            )
            target_cvar_par = cp.Parameter(
                value=target_cvar, name="target_cvar", nonneg=True
            )

            self.add_constraint(lambda _: cvar <= target_cvar_par)
            self.add_constraint(lambda _: self._u >= 0.0)
            self.add_constraint(
                lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
            )

            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False):
        """Calculate expected return and CVaR."""
        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )

        cvar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(
            self._u
        )
        cvar_val = cvar.value

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Conditional Value at Risk: {:.2f}%".format(100 * cvar_val))

        return mu, cvar_val
