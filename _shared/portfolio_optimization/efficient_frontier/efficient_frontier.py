"""Mean-variance efficient frontier optimization."""

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from .. import base_optimizer, exceptions, objective_functions


class EfficientFrontier(base_optimizer.BaseConvexOptimizer):
    """Mean-variance efficient frontier optimizer."""

    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        # Inputs
        self.cov_matrix = self._validate_cov_matrix(cov_matrix)
        self.expected_returns = self._validate_expected_returns(expected_returns)
        self._max_return_value = None
        self._market_neutral = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # use integer labels
            tickers = list(range(num_assets))

        if expected_returns is not None and cov_matrix is not None:
            if cov_matrix.shape != (num_assets, num_assets):
                raise ValueError("Covariance matrix does not match expected returns")

        super().__init__(
            len(tickers),
            tickers,
            weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if cov_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError("cov_matrix is not a dataframe or array")

    def _validate_returns(self, returns):
        """Validate returns dataframe."""
        if not isinstance(returns, (pd.DataFrame, np.ndarray)):
            raise TypeError("returns should be a pd.DataFrame or np.ndarray")

        returns_df = pd.DataFrame(returns)
        if returns_df.isnull().values.any():
            warnings.warn(
                "Removing NaNs from returns",
                UserWarning,
            )
            returns_df = returns_df.dropna(axis=0, how="any")

        if self.expected_returns is not None:
            if returns_df.shape[1] != len(self.expected_returns):
                raise ValueError(
                    "returns columns do not match expected_returns. Please check your tickers."
                )

        return returns_df

    def _make_weight_sum_constraint(self, is_market_neutral):
        """Create weight sum constraint (1 or 0 for market neutral)."""
        if is_market_neutral:
            # Check and fix bounds
            portfolio_possible = np.any(self._lower_bounds < 0)
            if not portfolio_possible:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self._map_bounds_to_constraints((-1, 1))
                # Delete original constraints
                del self._constraints[0]
                del self._constraints[0]

            self.add_constraint(lambda w: cp.sum(w) == 0)
        else:
            self.add_constraint(lambda w: cp.sum(w) == 1)
        self._market_neutral = is_market_neutral

    def min_volatility(self):
        """Minimize portfolio volatility."""
        self._objective = objective_functions.portfolio_variance(
            self._w, self.cov_matrix
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def _max_return(self, return_value=True):
        """Helper to maximize return (internal use only)."""
        if self.expected_returns is None:
            raise ValueError("no expected returns provided")

        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )

        self.add_constraint(lambda w: cp.sum(w) == 1)

        res = self._solve_cvxpy_opt_problem()

        if return_value:
            return -self._opt.value
        else:
            return res

    def max_sharpe(self, risk_free_rate=0.0):
        """Maximize Sharpe ratio (tangency portfolio on efficient frontier)."""
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        if max(self.expected_returns) <= risk_free_rate:
            raise ValueError(
                "at least one of the assets must have an expected return exceeding the risk-free rate"
            )

        self._risk_free_rate = risk_free_rate

        # max_sharpe requires us to make a variable transformation.
        # Here we treat w as the transformed variable.
        self._objective = cp.quad_form(self._w, self.cov_matrix, assume_PSD=True)
        k = cp.Variable()

        # Note: objectives are not scaled by k. Hence there are subtle differences
        # between how these objectives work for max_sharpe vs min_volatility
        if len(self._additional_objectives) > 0:
            warnings.warn(
                "max_sharpe transforms the optimization problem so additional objectives may not work as expected."
            )
        for obj in self._additional_objectives:
            self._objective += obj

        new_constraints = []
        # Must rebuild the constraints
        for constr in self._constraints:
            if isinstance(constr, cp.constraints.nonpos.Inequality):
                # Either the first or second item is the expression
                if isinstance(
                    constr.args[0], cp.expressions.constants.constant.Constant
                ):
                    new_constraints.append(constr.args[1] >= constr.args[0] * k)
                else:
                    new_constraints.append(constr.args[0] <= constr.args[1] * k)
            elif isinstance(constr, cp.constraints.zero.Equality):
                new_constraints.append(constr.args[0] == constr.args[1] * k)
            else:
                raise TypeError(
                    "Please check that your constraints are in a suitable format"
                )

        # Transformed max_sharpe convex problem:
        self._constraints = [
            (self.expected_returns - risk_free_rate).T @ self._w == 1,
            cp.sum(self._w) == k,
            k >= 0,
        ] + new_constraints

        self._solve_cvxpy_opt_problem()
        # Inverse-transform
        self.weights = (self._w.value / k.value).round(16) + 0.0
        return self._make_output_weights()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        r"""Maximize quadratic utility: max w^T*mu - (delta/2)*w^T*Sigma*w."""
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        update_existing_parameter = self.is_parameter_defined("risk_aversion")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("risk_aversion", risk_aversion)
        else:
            self._objective = objective_functions.quadratic_utility(
                self._w,
                self.expected_returns,
                self.cov_matrix,
                risk_aversion=risk_aversion,
            )
            for obj in self._additional_objectives:
                self._objective += obj

            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_volatility, market_neutral=False):
        """Maximize return for target volatility."""
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("target_volatility should be a positive float")

        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.cov_matrix)))

        if target_volatility < global_min_volatility:
            raise ValueError(
                "The minimum volatility is {:.3f}. Please use a higher target_volatility".format(
                    global_min_volatility
                )
            )

        update_existing_parameter = self.is_parameter_defined("target_variance")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_variance", target_volatility**2)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            variance = objective_functions.portfolio_variance(self._w, self.cov_matrix)

            for obj in self._additional_objectives:
                self._objective += obj

            target_variance = cp.Parameter(
                name="target_variance", value=target_volatility**2, nonneg=True
            )
            self.add_constraint(lambda _: variance <= target_variance)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """Minimize volatility for target return (Markowitz portfolio)."""
        if not isinstance(target_return, float):
            raise ValueError("target_return should be a float")
        if not self._max_return_value:
            a = self.deepcopy()
            self._max_return_value = a._max_return()
        if target_return > self._max_return_value:
            raise ValueError(
                "target_return must be lower than the maximum possible return"
            )

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            self._objective = objective_functions.portfolio_variance(
                self._w, self.cov_matrix
            )
            ret = objective_functions.portfolio_return(
                self._w, self.expected_returns, negative=False
            )

            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """Calculate expected return, volatility, and Sharpe ratio."""
        if self._risk_free_rate is not None:
            if risk_free_rate != self._risk_free_rate:
                warnings.warn(
                    "The risk_free_rate provided to portfolio_performance is different"
                    " to the one used by max_sharpe. Using the previous value.",
                    UserWarning,
                )
            risk_free_rate = self._risk_free_rate

        return base_optimizer.portfolio_performance(
            self.weights,
            self.expected_returns,
            self.cov_matrix,
            verbose,
            risk_free_rate,
        )

    def _validate_market_neutral(self, market_neutral: bool) -> None:
        if self._market_neutral != market_neutral:
            raise exceptions.InstantiationError(
                "A new instance must be created when changing market_neutral."
            )
