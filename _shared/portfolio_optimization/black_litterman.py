"""Black-Litterman model for posterior return estimates."""

import sys
import warnings

import numpy as np
import pandas as pd

from . import base_optimizer


def market_implied_prior_returns(
    market_caps, risk_aversion, cov_matrix, risk_free_rate=0.0
):
    r"""Compute prior returns implied by market weights: Pi = delta * Sigma * w_mkt."""
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn(
            "If cov_matrix is not a dataframe, market cap index must be aligned to cov_matrix",
            RuntimeWarning,
        )
    mcaps = pd.Series(market_caps)
    mkt_weights = mcaps / mcaps.sum()
    # Pi is excess returns so must add risk_free_rate to get return.
    return risk_aversion * cov_matrix.dot(mkt_weights) + risk_free_rate


def market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=0.0):
    r"""
    Calculate the market-implied risk-aversion parameter (i.e market price of risk)
    based on market prices. For example, if the market has excess returns of 10% a year
    with 5% variance, the risk-aversion parameter is 2, i.e you have to be compensated 2x
    the variance.

    .. math::

        \delta = \frac{R - R_f}{\sigma^2}

    :param market_prices: the (daily) prices of the market portfolio, e.g SPY.
    :type market_prices: pd.Series with DatetimeIndex.
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param risk_free_rate: annualised risk-free rate of borrowing/lending, defaults to 0.0.
    :type risk_free_rate: float, optional
    :raises TypeError: if market_prices cannot be parsed
    :return: market-implied risk aversion
    :rtype: float
    """
    if not isinstance(market_prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Please format market_prices as a pd.Series")
    market_prices = market_prices.squeeze()
    rets = market_prices.pct_change().dropna()
    r = rets.mean() * frequency
    var = rets.var() * frequency
    return (r - risk_free_rate) / var


class BlackLittermanModel(base_optimizer.BaseOptimizer):
    """Black-Litterman model for combining prior and views to generate posterior estimates."""

    def __init__(
        self,
        cov_matrix,
        pi=None,
        absolute_views=None,
        Q=None,
        P=None,
        omega=None,
        view_confidences=None,
        tau=0.05,
        risk_aversion=1,
        **kwargs,
    ):
        """
        :param cov_matrix: NxN covariance matrix of returns
        :type cov_matrix: pd.DataFrame or np.ndarray
        :param pi: Nx1 prior estimate of returns, defaults to None.
                   If pi="market", calculate a market-implied prior (requires market_caps
                   to be passed).
                   If pi="equal", use an equal-weighted prior.
        :type pi: np.ndarray, pd.Series, optional
        :param absolute_views: a collection of K absolute views on a subset of assets,
                               defaults to None. If this is provided, we do not need P, Q.
        :type absolute_views: pd.Series or dict, optional
        :param Q: Kx1 views vector, defaults to None
        :type Q: np.ndarray or pd.DataFrame, optional
        :param P: KxN picking matrix, defaults to None
        :type P: np.ndarray or pd.DataFrame, optional
        :param omega: KxK view uncertainty matrix (diagonal), defaults to None
                      Can instead pass "idzorek" to use Idzorek's method (requires
                      you to pass view_confidences). If omega="default" or None,
                      we set the uncertainty proportional to the variance.
        :type omega: np.ndarray or Pd.DataFrame, or string, optional
        :param view_confidences: Kx1 vector of percentage view confidences (between 0 and 1),
                                required to compute omega via Idzorek's method.
        :type view_confidences: np.ndarray, pd.Series, list, optional
        :param tau: the weight-on-views scalar (default is 0.05)
        :type tau: float, optional
        :param risk_aversion: risk aversion parameter, defaults to 1
        :type risk_aversion: positive float, optional
        :param market_caps: (kwarg) market caps for the assets, required if pi="market"
        :type market_caps: np.ndarray, pd.Series, optional
        :param risk_free_rate: (kwarg) risk_free_rate is needed in some methods
        :type risk_free_rate: float, defaults to 0.0
        """
        if sys.version_info[1] == 5:  # pragma: no cover
            warnings.warn(
                "When using python 3.5 you must explicitly construct the Black-Litterman inputs"
            )

        # Keep raw dataframes
        self._raw_cov_matrix = cov_matrix

        #  Initialise base optimizer
        if isinstance(cov_matrix, np.ndarray):
            self.cov_matrix = cov_matrix
            super().__init__(len(cov_matrix), list(range(len(cov_matrix))))
        else:
            self.cov_matrix = cov_matrix.values
            super().__init__(len(cov_matrix), cov_matrix.columns)

        #  Sanitise inputs
        if absolute_views is not None:
            self.Q, self.P = self._parse_views(absolute_views)
        else:
            self._set_Q_P(Q, P)
        self._set_risk_aversion(risk_aversion)
        self._set_pi(pi, **kwargs)
        self._set_tau(tau)
        # Make sure all dimensions work
        self._check_attribute_dimensions()

        self._set_omega(omega, view_confidences)

        # Private intermediaries
        self._tau_sigma_P = None
        self._A = None

        self.posterior_rets = None
        self.posterior_cov = None

    def _parse_views(self, absolute_views):
        """Construct views vector and picking matrix from absolute views dict/series."""
        if not isinstance(absolute_views, (dict, pd.Series)):
            raise TypeError("views should be a dict or pd.Series")
        # Coerce to series
        views = pd.Series(absolute_views)
        k = len(views)

        Q = np.zeros((k, 1))
        P = np.zeros((k, self.n_assets))

        for i, view_ticker in enumerate(views.keys()):
            try:
                Q[i] = views[view_ticker]
                P[i, list(self.tickers).index(view_ticker)] = 1
            except ValueError:
                #  Could make this smarter by just skipping
                raise ValueError("Providing a view on an asset not in the universe")
        return Q, P

    def _set_Q_P(self, Q, P):
        if isinstance(Q, (pd.Series, pd.DataFrame)):
            self.Q = Q.values.reshape(-1, 1)
        elif isinstance(Q, np.ndarray):
            self.Q = Q.reshape(-1, 1)
        else:
            raise TypeError("Q must be an array or dataframe")

        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif isinstance(P, np.ndarray):
            self.P = P
        elif len(self.Q) == self.n_assets:
            # If a view on every asset is provided, P defaults
            # to the identity matrix.
            self.P = np.eye(self.n_assets)
        else:
            raise TypeError("P must be an array or dataframe")

    def _set_pi(self, pi, **kwargs):
        if pi is None:
            warnings.warn("Running Black-Litterman with no prior.")
            self.pi = np.zeros((self.n_assets, 1))
        elif isinstance(pi, (pd.Series, pd.DataFrame)):
            self.pi = pi.values.reshape(-1, 1)
        elif isinstance(pi, np.ndarray):
            self.pi = pi.reshape(-1, 1)
        elif pi == "market":
            if "market_caps" not in kwargs:
                raise ValueError(
                    "Please pass a series/array of market caps via the market_caps keyword argument"
                )
            market_caps = kwargs.get("market_caps")
            risk_free_rate = kwargs.get("risk_free_rate", 0.0)

            market_prior = market_implied_prior_returns(
                market_caps, self.risk_aversion, self._raw_cov_matrix, risk_free_rate
            )
            self.pi = market_prior.values.reshape(-1, 1)
        elif pi == "equal":
            self.pi = np.ones((self.n_assets, 1)) / self.n_assets
        else:
            raise TypeError("pi must be an array or series")

    def _set_tau(self, tau):
        if tau <= 0 or tau > 1:
            raise ValueError("tau should be between 0 and 1")
        self.tau = tau

    def _set_risk_aversion(self, risk_aversion):
        if risk_aversion <= 0:
            raise ValueError("risk_aversion should be a positive float")
        self.risk_aversion = risk_aversion

    def _set_omega(self, omega, view_confidences):
        if isinstance(omega, pd.DataFrame):
            self.omega = omega.values
        elif isinstance(omega, np.ndarray):
            self.omega = omega
        elif omega == "idzorek":
            if view_confidences is None:
                raise ValueError(
                    "To use Idzorek's method, please supply a vector of percentage confidence levels for each view."
                )
            if not isinstance(view_confidences, np.ndarray):
                try:
                    view_confidences = np.array(view_confidences).reshape(-1, 1)
                    assert view_confidences.shape[0] == self.Q.shape[0]
                    assert np.issubdtype(view_confidences.dtype, np.number)
                except AssertionError:
                    raise ValueError(
                        "view_confidences should be a numpy 1D array or vector with the same length "
                        "as the number of views."
                    )

            self.omega = BlackLittermanModel.idzorek_method(
                view_confidences,
                self.cov_matrix,
                self.pi,
                self.Q,
                self.P,
                self.tau,
                self.risk_aversion,
            )
        elif omega is None or omega == "default":
            self.omega = BlackLittermanModel.default_omega(
                self.cov_matrix, self.P, self.tau
            )
        else:
            raise TypeError("self.omega must be a square array, dataframe, or string")

        K = len(self.Q)
        assert self.omega.shape == (K, K), "omega must have dimensions KxK"

    def _check_attribute_dimensions(self):
        """
        Helper method to ensure that all of the attributes created by the initialiser
        have the correct dimensions, to avoid linear algebra errors later on.

        :raises ValueError: if there are incorrect dimensions.
        """
        N = self.n_assets
        K = len(self.Q)
        assert self.pi.shape == (N, 1), "pi must have dimensions Nx1"
        assert self.P.shape == (K, N), "P must have dimensions KxN"
        assert self.cov_matrix.shape == (N, N), "cov_matrix must have shape NxN"

    @staticmethod
    def default_omega(cov_matrix, P, tau):
        """Calculate default omega (He & Litterman 1999): omega/tau ~ view portfolio variance."""
        return np.diag(np.diag(tau * P @ cov_matrix @ P.T))

    @staticmethod
    def idzorek_method(view_confidences, cov_matrix, pi, Q, P, tau, risk_aversion=1):
        """Create omega from view confidences using Idzorek method (Walters 2014)."""
        view_omegas = []
        for view_idx in range(len(Q)):
            conf = view_confidences[view_idx]

            if conf < 0 or conf > 1:
                raise ValueError("View confidences must be between 0 and 1")

            # Special handler to avoid dividing by zero.
            # If zero conf, return very big number as uncertainty
            if conf == 0:
                view_omegas.append(1e6)
                continue

            P_view = P[view_idx].reshape(1, -1)
            alpha = (1 - conf) / conf  # formula (44)
            omega = tau * alpha * P_view @ cov_matrix @ P_view.T  # formula (41)
            view_omegas.append(omega.item())

        return np.diag(view_omegas)

    def bl_returns(self):
        """Calculate posterior returns given views."""

        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T

        # Solve the linear system Ax = b to avoid inversion
        if self._A is None:
            self._A = (self.P @ self._tau_sigma_P) + self.omega
        b = self.Q - self.P @ self.pi
        try:
            solution = np.linalg.solve(self._A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                solution = np.linalg.lstsq(self._A, b, rcond=None)[0]
            else:
                raise e
        post_rets = self.pi + self._tau_sigma_P @ solution
        return pd.Series(post_rets.flatten(), index=self.tickers)

    def bl_cov(self):
        """Calculate posterior covariance (He & Litterman 2002, assumes diagonal omega)."""
        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        if self._A is None:
            self._A = (self.P @ self._tau_sigma_P) + self.omega

        b = self._tau_sigma_P.T
        try:
            M_solution = np.linalg.solve(self._A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                M_solution = np.linalg.lstsq(self._A, b, rcond=None)[0]
            else:
                raise e
        M = self.tau * self.cov_matrix - self._tau_sigma_P @ M_solution
        posterior_cov = self.cov_matrix + M
        return pd.DataFrame(posterior_cov, index=self.tickers, columns=self.tickers)

    def bl_weights(self, risk_aversion=None):
        r"""Compute weights from posterior returns: w = (delta * Sigma)^-1 * E(R)."""
        if risk_aversion is None:
            risk_aversion = self.risk_aversion

        self.posterior_rets = self.bl_returns()
        A = risk_aversion * self.cov_matrix
        b = self.posterior_rets
        try:
            weight_solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                weight_solution = np.linalg.lstsq(self._A, b, rcond=None)[0]
            else:
                raise e
        raw_weights = weight_solution
        self.weights = raw_weights / raw_weights.sum()
        return self._make_output_weights()

    def optimize(self, risk_aversion=None):
        """Alias for bl_weights."""
        return self.bl_weights(risk_aversion)

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """Calculate portfolio performance using BL posterior estimates."""
        if self.posterior_cov is None:
            self.posterior_cov = self.bl_cov()
        return base_optimizer.portfolio_performance(
            self.weights,
            self.posterior_rets,
            self.posterior_cov,
            verbose,
            risk_free_rate,
        )
