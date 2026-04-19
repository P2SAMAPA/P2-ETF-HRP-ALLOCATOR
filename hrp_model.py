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
            # Single asset: 100% allocation
            return {returns.columns[0]: 1.0}
        
        # Compute covariance and correlation matrices
        cov = returns.cov()
        corr = returns.corr()
        
        # Distance matrix
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)
        
        # Hierarchical clustering
        self.linkage = sch.linkage(dist, method=self.linkage_method)
        
        # Quasi-diagonalization: reorder tickers based on clustering
        ordered_tickers = self._quasi_diagonalize(returns.columns.tolist())
        
        # Recursive bisection
        weights = self._recursive_bisection(cov, ordered_tickers)
        
        return dict(zip(ordered_tickers, weights))
    
    def _quasi_diagonalize(self, tickers: List[str]) -> List[str]:
        """Reorder tickers based on hierarchical clustering leaves."""
        leaves = sch.leaves_list(self.linkage)
        return [tickers[i] for i in leaves]
    
    def _recursive_bisection(self, cov: pd.DataFrame, tickers: List[str]) -> np.ndarray:
        """Recursively split cluster and compute inverse-variance weights."""
        if len(tickers) == 1:
            return np.array([1.0])
        
        # Split cluster into two based on linkage
        left, right = self._split_cluster(tickers)
        
        # Sub-covariance matrices
        cov_left = cov.loc[left, left]
        cov_right = cov.loc[right, right]
        
        # Inverse-variance weights within each cluster
        w_left = self._inverse_variance_weights(cov_left)
        w_right = self._inverse_variance_weights(cov_right)
        
        # Cluster variances
        var_left = np.linalg.multi_dot((w_left, cov_left, w_left))
        var_right = np.linalg.multi_dot((w_right, cov_right, w_right))
        
        # Allocation between clusters (risk parity)
        alpha = var_right / (var_left + var_right)
        
        # Recurse and combine
        return np.concatenate([
            alpha * self._recursive_bisection(cov, left),
            (1 - alpha) * self._recursive_bisection(cov, right)
        ])
    
    def _split_cluster(self, tickers: List[str]) -> tuple:
        """Split a cluster into two groups based on linkage."""
        # Build cluster mapping from linkage matrix
        n = len(tickers)
        cluster_map = {i: [i] for i in range(n)}
        
        for i, row in enumerate(self.linkage):
            left = int(row[0])
            right = int(row[1])
            new_cluster = n + i
            cluster_map[new_cluster] = cluster_map[left] + cluster_map[right]
        
        # The root cluster is the last one created
        root_cluster = cluster_map[max(cluster_map.keys())]
        
        # Split at the highest level
        last_row = self.linkage[-1]
        left_idx = int(last_row[0])
        right_idx = int(last_row[1])
        
        left_indices = cluster_map[left_idx] if left_idx in cluster_map else [left_idx]
        right_indices = cluster_map[right_idx] if right_idx in cluster_map else [right_idx]
        
        left_tickers = [tickers[i] for i in left_indices]
        right_tickers = [tickers[i] for i in right_indices]
        
        return left_tickers, right_tickers
    
    def _inverse_variance_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Compute inverse-variance weights for a covariance matrix."""
        ivp = 1.0 / np.diag(cov)
        return ivp / ivp.sum()
