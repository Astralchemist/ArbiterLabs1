"""Risk models for estimating covariance matrices."""

import warnings
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from .expected_returns import returns_from_prices


def _is_positive_semidefinite(matrix):
    """Check if matrix is positive semidefinite."""
    try:
        # More efficient than checking eigenvalues
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """Fix non-positive semidefinite covariance matrix."""
    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn("The covariance matrix is non positive semidefinite. Amending eigenvalues.")

    q, V = np.linalg.eigh(matrix)

    if fix_method == "spectral":
        q = np.where(q > 0, q, 0)
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError("Method {} not implemented".format(fix_method))

    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        warnings.warn("Could not fix matrix. Please try a different risk model.", UserWarning)

    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix


def risk_matrix(prices, method="sample_cov", **kwargs):
    """Compute covariance matrix using specified risk model."""
    if method == "sample_cov":
        return sample_cov(prices, **kwargs)
    elif method == "semicovariance" or method == "semivariance":
        return semicovariance(prices, **kwargs)
    elif method == "exp_cov":
        return exp_cov(prices, **kwargs)
    elif method == "ledoit_wolf" or method == "ledoit_wolf_constant_variance":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf()
    elif method == "ledoit_wolf_single_factor":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(shrinkage_target="single_factor")
    elif method == "ledoit_wolf_constant_correlation":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(shrinkage_target="constant_correlation")
    elif method == "oracle_approximating":
        return CovarianceShrinkage(prices, **kwargs).oracle_approximating()
    else:
        raise NotImplementedError("Risk model {} not implemented".format(method))


def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, **kwargs):
    """Calculate annualized sample covariance matrix from asset prices/returns."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    return fix_nonpositive_semidefinite(returns.cov() * frequency, kwargs.get("fix_method", "spectral"))


def semicovariance(prices, returns_data=False, benchmark=0.000079, frequency=252, log_returns=False, **kwargs):
    """Estimate semicovariance matrix (covariance when returns < benchmark)."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    drops = np.fmin(returns - benchmark, 0)
    T = drops.shape[0]
    return fix_nonpositive_semidefinite((drops.T @ drops) / T * frequency, kwargs.get("fix_method", "spectral"))


def _pair_exp_cov(X, Y, span=180):
    """Calculate exponential covariance between two return time series."""
    covariation = (X - X.mean()) * (Y - Y.mean())
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    return covariation.ewm(span=span).mean().iloc[-1]


def exp_cov(prices, returns_data=False, span=180, frequency=252, log_returns=False, **kwargs):
    """Estimate exponentially-weighted covariance matrix."""
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    N = len(assets)

    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(returns.iloc[:, i], returns.iloc[:, j], span)
    cov = pd.DataFrame(S * frequency, columns=assets, index=assets)

    return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))


def min_cov_determinant(prices, returns_data=False, frequency=252, random_state=None, log_returns=False, **kwargs):  # pragma: no cover
    """Deprecated: minimum covariance determinant estimator."""
    warnings.warn("min_cov_determinant is deprecated and will be removed in v1.5")

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if not _check_soft_dependencies(["scikit-learn"], severity="none"):
        raise ImportError("scikit-learn is required to use min_cov_determinant. Please ensure that scikit-learn is installed in your environment, e.g via pip install scikit-learn")

    from sklearn.covariance import fast_mcd

    assets = prices.columns

    if returns_data:
        X = prices
    else:
        X = returns_from_prices(prices, log_returns)
    X = X.dropna().values
    raw_cov_array = fast_mcd(X, random_state=random_state)[1]
    cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * frequency
    return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))


def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix."""
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn("cov_matrix is not a dataframe", RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)


def corr_to_cov(corr_matrix, stdevs):
    """Convert correlation matrix to covariance matrix."""
    if not isinstance(corr_matrix, pd.DataFrame):
        warnings.warn("corr_matrix is not a dataframe", RuntimeWarning)
        corr_matrix = pd.DataFrame(corr_matrix)

    return corr_matrix * np.outer(stdevs, stdevs)


class CovarianceShrinkage:
    """Covariance shrinkage estimators (Ledoit-Wolf, Oracle Approximating)."""

    def __init__(self, prices, returns_data=False, frequency=252, log_returns=False):
        if not _check_soft_dependencies(["scikit-learn"], severity="none"):
            raise ImportError("scikit-learn is required to use CovarianceShrinkage. Please ensure that scikit-learn is installed in your environment, e.g via pip install scikit-learn")

        from sklearn import covariance

        self.covariance = covariance

        if not isinstance(prices, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            prices = pd.DataFrame(prices)

        self.frequency = frequency

        if returns_data:
            self.X = prices.dropna(how="all")
        else:
            self.X = returns_from_prices(prices, log_returns).dropna(how="all")

        self.S = self.X.cov().values
        self.delta = None

    def _format_and_annualize(self, raw_cov_array):
        """Annualize and format covariance matrix as DataFrame."""
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        return fix_nonpositive_semidefinite(cov, fix_method="spectral")

    def shrunk_covariance(self, delta=0.2):
        """Shrink sample covariance to identity matrix (scaled by average variance)."""
        self.delta = delta
        N = self.S.shape[1]
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)

    def ledoit_wolf(self, shrinkage_target="constant_variance"):
        """Calculate Ledoit-Wolf shrinkage estimate."""
        if shrinkage_target == "constant_variance":
            X = np.nan_to_num(self.X.values)
            shrunk_cov, self.delta = self.covariance.ledoit_wolf(X)
        elif shrinkage_target == "single_factor":
            shrunk_cov, self.delta = self._ledoit_wolf_single_factor()
        elif shrinkage_target == "constant_correlation":
            shrunk_cov, self.delta = self._ledoit_wolf_constant_correlation()
        else:
            raise NotImplementedError("Shrinkage target {} not recognised".format(shrinkage_target))

        return self._format_and_annualize(shrunk_cov)

    def _ledoit_wolf_single_factor(self):
        """Ledoit-Wolf shrinkage with Sharpe single-factor matrix target."""
        X = np.nan_to_num(self.X.values)

        t, n = np.shape(X)
        Xm = X - X.mean(axis=0)
        xmkt = Xm.mean(axis=1).reshape(t, 1)

        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        F = np.dot(betas, betas.T) / varmkt
        F[np.eye(n) == 1] = np.diag(sample)

        c = np.linalg.norm(sample - F, "fro") ** 2
        y = Xm**2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample**2)

        # Diagonal and off-diagonal terms
        rdiag = 1 / t * np.sum(y**2) - sum(np.diag(sample) ** 2)
        z = Xm * np.tile(xmkt, (n,))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(betas, (n,)) * sample
        roff1 = np.sum(v1 * np.tile(betas, (n,)).T) / varmkt - np.sum(np.diag(v1) * betas.T) / varmkt
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = np.sum(v3 * np.dot(betas, betas.T)) / varmkt**2 - np.sum(np.diag(v3).reshape(-1, 1) * betas**2) / varmkt**2
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        k = (p - r) / c
        delta = max(0, min(1, k / t))

        shrunk_cov = delta * F + (1 - delta) * sample
        return shrunk_cov, delta

    def _ledoit_wolf_constant_correlation(self):
        """Ledoit-Wolf shrinkage with constant correlation matrix target."""
        X = np.nan_to_num(self.X.values)
        t, n = np.shape(X)

        S = self.S

        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        Xm = X - X.mean(axis=0)
        y = Xm**2
        pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S**2
        pi_hat = np.sum(pi_mat)

        term1 = np.dot((Xm**3).T, Xm) / t
        help_ = np.dot(Xm.T, Xm) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(np.dot((1 / std), std.T) * theta_mat)

        gamma_hat = np.linalg.norm(S - F, "fro") ** 2

        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

        shrunk_cov = delta * F + (1 - delta) * S
        return shrunk_cov, delta

    def oracle_approximating(self):
        """Calculate Oracle Approximating Shrinkage estimate."""
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.oas(X)
        return self._format_and_annualize(shrunk_cov)
