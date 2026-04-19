"""
Hierarchical Risk Parity (HRP) allocation model.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import List, Dict

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
        
        cov = returns.cov()
        corr = returns.corr()
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)
        self.linkage = sch.linkage(dist, method=self.linkage_method)
        
        # Get the order of leaves from the dendrogram
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
        
        # Split cluster into two based on the linkage tree
        left_indices, right_indices = self._get_cluster_split(n)
        
        # Sub-covariance matrices
        cov_left = cov.iloc[left_indices, left_indices]
        cov_right = cov.iloc[right_indices, right_indices]
        
        # Inverse-variance weights within each sub-cluster
        w_left = self._inverse_variance_weights(cov_left)
        w_right = self._inverse_variance_weights(cov_right)
        
        # Cluster variances
        var_left = np.linalg.multi_dot((w_left, cov_left.values, w_left))
        var_right = np.linalg.multi_dot((w_right, cov_right.values, w_right))
        
        # Risk parity between clusters
        alpha = var_right / (var_left + var_right)
        
        # Recurse and assemble full weight vector
        weights = np.zeros(n)
        weights[left_indices] = alpha * self._recursive_bisection(cov_left)
        weights[right_indices] = (1 - alpha) * self._recursive_bisection(cov_right)
        
        return weights
    
    def _get_cluster_split(self, n: int) -> tuple:
        """
        Determine how to split a cluster of size n into two sub-clusters
        based on the top-most split in the linkage matrix.
        Returns two lists of indices (0-based relative to the cluster).
        """
        # The linkage matrix for the cluster can be extracted by looking at
        # the last (n-1) merges that involve only elements within the cluster.
        # A simpler approach: we can use the ordering from leaves_list and
        # split at the point where the distance is largest, but the canonical
        # HRP paper splits exactly at the top of the sub-dendrogram.
        
        # We'll use a helper function that recursively builds the split.
        # For simplicity, we split the ordered list into two halves.
        # This is a common approximation that works well in practice.
        mid = n // 2
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))
        return left_indices, right_indices
    
    def _inverse_variance_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Compute inverse-variance weights for a covariance matrix."""
        diag = np.diag(cov)
        # Avoid division by zero
        diag = np.where(diag < 1e-10, 1e-10, diag)
        ivp = 1.0 / diag
        return ivp / ivp.sum()
