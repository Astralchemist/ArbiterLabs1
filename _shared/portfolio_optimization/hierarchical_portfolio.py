"""Hierarchical Risk Parity (HRP) portfolio optimization."""

import collections
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from . import base_optimizer, risk_models


class HRPOpt(base_optimizer.BaseOptimizer):
    """Hierarchical Risk Parity portfolio optimizer."""

    def __init__(self, returns=None, cov_matrix=None):
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")

        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None

        if returns is None:
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """Compute variance per cluster."""
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_slice, weights))

    @staticmethod
    def _get_quasi_diag(link):
        """Sort clustered items by distance."""
        return sch.to_tree(link, rd=False).pre_order()

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """Compute HRP weights by recursively traversing hierarchical tree."""
        w = pd.Series(1.0, index=ordered_tickers)
        cluster_items = [ordered_tickers]

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            # Optimize locally for each pair
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha
                w[second_cluster] *= 1 - alpha
        return w

    def optimize(self, linkage_method="single"):
        """Construct HRP portfolio using hierarchical clustering."""
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        if self.returns is None:
            cov = self.cov_matrix
            corr = risk_models.cov_to_corr(self.cov_matrix).round(6)
        else:
            corr, cov = self.returns.corr(), self.returns.cov()

        # Compute distance matrix (avoid floating point issues)
        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)

        self.clusters = sch.linkage(dist, linkage_method)
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        weights = collections.OrderedDict(hrp.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0, frequency=252):
        """Calculate expected return, volatility, and Sharpe ratio."""
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return base_optimizer.portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)
