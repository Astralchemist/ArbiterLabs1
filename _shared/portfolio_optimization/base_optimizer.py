"""Base optimizer classes for portfolio optimization."""

import collections
import copy
import json
import warnings
from collections.abc import Iterable
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.optimize as sco

from . import exceptions, objective_functions


class BaseOptimizer:
    """Base class for all portfolio optimizers."""

    def __init__(self, n_assets, tickers=None):
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.tickers = tickers
        self._risk_free_rate = None
        # Outputs
        self.weights = None

    def _make_output_weights(self, weights=None):
        """Convert weight array to ordered dict."""
        if weights is None:
            weights = self.weights

        # Convert numpy float64 to plain Python float
        weights = [float(w) for w in weights]

        return collections.OrderedDict(zip(self.tickers, weights))

    def set_weights(self, input_weights):
        """Set weights from dict."""
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=5):
        """Round weights and clip near-zeros."""
        if self.weights is None:
            raise AttributeError("Weights not yet computed")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)

    def save_weights_to_file(self, filename="weights.csv"):
        """Save weights to csv, json, or txt file."""
        clean_weights = self.clean_weights()

        ext = filename.split(".")[-1].lower()
        if ext == "csv":
            pd.Series(clean_weights).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean_weights, fp)
        elif ext == "txt":
            with open(filename, "w") as f:
                f.write(str(dict(clean_weights)))
        else:
            raise NotImplementedError("Only supports .txt .json .csv")


class BaseConvexOptimizer(BaseOptimizer):
    """Base class for cvxpy/scipy convex optimization."""

    def __init__(
        self,
        n_assets,
        tickers=None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        super().__init__(n_assets, tickers)

        # Optimization variables
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solver_options = solver_options if solver_options else {}
        self._map_bounds_to_constraints(weight_bounds)

    def deepcopy(self):
        """Return custom deep copy (cvxpy expressions don't support deepcopy)."""
        self_copy = copy.copy(self)
        self_copy._additional_objectives = [
            copy.copy(obj) for obj in self_copy._additional_objectives
        ]
        self_copy._constraints = [copy.copy(con) for con in self_copy._constraints]
        return self_copy

    def _map_bounds_to_constraints(self, test_bounds):
        """Convert bounds to cvxpy constraints."""
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else:
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds)
        self.add_constraint(lambda w: w <= self._upper_bounds)

    def is_parameter_defined(self, parameter_name: str) -> bool:
        is_defined = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name and not is_defined:
                    is_defined = True
                elif param.name() == parameter_name and is_defined:
                    raise exceptions.InstantiationError(
                        "Parameter name defined multiple times"
                    )
        return is_defined

    def update_parameter_value(self, parameter_name: str, new_value: float) -> None:
        if not self.is_parameter_defined(parameter_name):
            raise exceptions.InstantiationError("Parameter has not been defined")
        was_updated = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name:
                    param.value = new_value
                    was_updated = True
        if not was_updated:
            raise exceptions.InstantiationError("Parameter was not updated")

    def _solve_cvxpy_opt_problem(self):
        """Solve cvxpy problem and validate solution."""
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._objective.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            else:
                if not self._objective.id == self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization. "
                        "Please create a new instance instead."
                    )

                constr_ids = {const.id for const in self._constraints}
                if not constr_ids == self._initial_constraint_ids:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization. "
                        "Please create a new instance instead."
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        self.weights = self._w.value.round(16) + 0.0  # +0.0 removes signed zero
        return self._make_output_weights()

    def add_objective(self, new_objective, **kwargs):
        """Add convex objective term to optimization problem."""
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding objectives to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of objectives."
            )
        self._additional_objectives.append(new_objective(self._w, **kwargs))

    def add_constraint(self, new_constraint):
        """Add constraint to optimization problem (must satisfy DCP rules)."""
        if not callable(new_constraint):
            raise TypeError(
                "New constraint must be provided as a callable (e.g lambda function)"
            )
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of constraints."
            )
        self._constraints.append(new_constraint(self._w))

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        """
        Adds constraints on the sum of weights of different groups of assets.
        Most commonly, these will be sector constraints e.g portfolio's exposure to
        tech must be less than x%::

            sector_mapper = {
                "GOOG": "tech",
                "FB": "tech",,
                "XOM": "Oil/Gas",
                "RRC": "Oil/Gas",
                "MA": "Financials",
                "JPM": "Financials",
            }

            sector_lower = {"tech": 0.1}  # at least 10% to tech
            sector_upper = {
                "tech": 0.4, # less than 40% tech
                "Oil/Gas": 0.1 # less than 10% oil and gas
            }

        :param sector_mapper: dict that maps tickers to sectors
        :type sector_mapper: {str: str} dict
        :param sector_lower: lower bounds for each sector
        :type sector_lower: {str: float} dict
        :param sector_upper: upper bounds for each sector
        :type sector_upper: {str:float} dict
        """
        if np.any(self._lower_bounds < 0):
            warnings.warn(
                "Sector constraints may not produce reasonable results if shorts are allowed."
            )
        for sector in sector_upper:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
        for sector in sector_lower:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    def convex_objective(self, custom_objective, weights_sum_to_one=True, **kwargs):
        """Optimize custom convex objective function."""
        # custom_objective must have the right signature (w, **kwargs)
        self._objective = custom_objective(self._w, **kwargs)

        for obj in self._additional_objectives:
            self._objective += obj

        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)

        return self._solve_cvxpy_opt_problem()

    def nonconvex_objective(
        self,
        custom_objective,
        objective_args=None,
        weights_sum_to_one=True,
        constraints=None,
        solver="SLSQP",
        initial_guess=None,
    ):
        """Optimize nonconvex objective using scipy (may get stuck in local minima)."""
        # Sanitise inputs
        if not isinstance(objective_args, tuple):
            objective_args = (objective_args,)

        # Make scipy bounds
        bound_array = np.vstack((self._lower_bounds, self._upper_bounds)).T
        bounds = list(map(tuple, bound_array))

        if initial_guess is None:
            initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        # Construct constraints
        final_constraints = []
        if weights_sum_to_one:
            final_constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        if constraints is not None:
            final_constraints += constraints

        result = sco.minimize(
            custom_objective,
            x0=initial_guess,
            args=objective_args,
            method=solver,
            bounds=bounds,
            constraints=final_constraints,
        )
        self.weights = result["x"]
        return self._make_output_weights()


def portfolio_performance(
    weights, expected_returns, cov_matrix, verbose=False, risk_free_rate=0.0
):
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))

        for i, k in enumerate(tickers):
            if k in weights:
                new_weights[i] = weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        new_weights = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

    if expected_returns is not None:
        mu = objective_functions.portfolio_return(
            new_weights, expected_returns, negative=False
        )

        sharpe = objective_functions.sharpe_ratio(
            new_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            negative=False,
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
    else:
        if verbose:
            print("Annual volatility: {:.1f}%".format(100 * sigma))
        return None, sigma, None


def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
    """Recursively get all arguments from cvxpy expression."""
    if expression.args == []:
        return [expression]
    else:
        return list(_flatten([_get_all_args(arg) for arg in expression.args]))


def _flatten(alist: Iterable) -> Iterable:
    for v in alist:
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            yield from _flatten(v)
        else:
            yield v
