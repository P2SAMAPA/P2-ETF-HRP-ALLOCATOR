"""
Hierarchical Risk Parity (HRP) allocation model.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Dict, List

class HRPAllocator:
    """
    Hierarchical Risk Parity portfolio allocator.
    Uses hierarchical clustering and recursive bisection to generate robust weights.
    """

    def __init__(self, linkage_method: str = 'ward'):
        self.linkage_method = linkage_method
        self.linkage = None

    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute HRP weights from a DataFrame of returns.
        Returns a dictionary mapping ticker to weight.
        """
        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0}

        # Covariance and correlation
        cov = returns.cov()
        corr = returns.corr()

        # Distance matrix
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)

        # Hierarchical clustering
        self.linkage = sch.linkage(dist, method=self.linkage_method)

        # Quasi-diagonalization: order tickers according to dendrogram leaves
        ordered_indices = sch.leaves_list(self.linkage)
        ordered_tickers = [returns.columns[i] for i in ordered_indices]

        # Recursive bisection on the ordered covariance matrix
        cov_ordered = cov.loc[ordered_tickers, ordered_tickers]
        weights = self._recursive_bisection(cov_ordered)

        return dict(zip(ordered_tickers, weights))

    def _recursive_bisection(self, cov: pd.DataFrame) -> np.ndarray:
        """
        Recursively split the covariance matrix and compute inverse-variance weights.
        """
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])

        # Split cluster into two halves (standard HRP approximation)
        mid = n // 2
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))

        # Sub-covariance matrices
        cov_left = cov.iloc[left_indices, left_indices]
        cov_right = cov.iloc[right_indices, right_indices]

        # Inverse-variance weights within each sub-cluster
        w_left = self._inverse_variance_weights(cov_left)
        w_right = self._inverse_variance_weights(cov_right)

        # Cluster variances
        var_left = np.linalg.multi_dot((w_left, cov_left.values, w_left))
        var_right = np.linalg.multi_dot((w_right, cov_right.values, w_right))

        # Risk parity allocation between clusters
        alpha = var_right / (var_left + var_right)

        # Recurse and assemble full weight vector
        weights = np.zeros(n)
        weights[left_indices] = alpha * self._recursive_bisection(cov_left)
        weights[right_indices] = (1 - alpha) * self._recursive_bisection(cov_right)

        return weights

    def _inverse_variance_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Compute inverse-variance weights for a covariance matrix."""
        diag = np.diag(cov)
        # Avoid division by zero
        diag = np.where(diag < 1e-10, 1e-10, diag)
        ivp = 1.0 / diag
        return ivp / ivp.sum()
