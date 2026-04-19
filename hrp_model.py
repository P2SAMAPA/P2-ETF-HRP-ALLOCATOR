"""
Hierarchical Risk Parity (HRP) allocation model with optional return signals.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Dict, List, Tuple

class HRPAllocator:
    """
    Hierarchical Risk Parity portfolio allocator.
    Supports inverse-variance, Sharpe, mean-return, and return-over-variance weighting.
    """

    def __init__(self, linkage_method: str = 'ward', return_metric: str = 'sharpe', risk_free_rate: float = 0.0):
        self.linkage_method = linkage_method
        self.return_metric = return_metric
        self.risk_free_rate = risk_free_rate
        self.linkage = None
        self.original_tickers = None
        self.returns_used = None

    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute HRP weights from a DataFrame of returns.
        """
        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0}

        self.original_tickers = returns.columns.tolist()
        self.returns_used = returns.copy()

        cov = returns.cov()
        corr = returns.corr()
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)
        self.linkage = sch.linkage(dist, method=self.linkage_method)

        ordered_indices = sch.leaves_list(self.linkage)
        ordered_tickers = [self.original_tickers[i] for i in ordered_indices]

        cov_ordered = cov.loc[ordered_tickers, ordered_tickers]
        returns_ordered = returns[ordered_tickers]

        weights = self._recursive_bisection(cov_ordered, returns_ordered)
        return dict(zip(ordered_tickers, weights))

    def _recursive_bisection(self, cov: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])

        # Split cluster into two halves (standard HRP approximation)
        mid = n // 2
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))

        cov_left = cov.iloc[left_indices, left_indices]
        cov_right = cov.iloc[right_indices, right_indices]
        ret_left = returns.iloc[:, left_indices]
        ret_right = returns.iloc[:, right_indices]

        # Compute leaf weights within each sub-cluster using return-aware metric
        w_left = self._leaf_weights(cov_left, ret_left)
        w_right = self._leaf_weights(cov_right, ret_right)

        # Compute cluster-level "risk" or "score" for allocation between clusters
        score_left = self._cluster_score(cov_left, ret_left, w_left)
        score_right = self._cluster_score(cov_right, ret_right, w_right)

        # Allocate inversely proportional to score (higher score → lower weight)
        alpha = score_right / (score_left + score_right)

        weights = np.zeros(n)
        weights[left_indices] = alpha * self._recursive_bisection(cov_left, ret_left)
        weights[right_indices] = (1 - alpha) * self._recursive_bisection(cov_right, ret_right)

        return weights

    def _leaf_weights(self, cov: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute weight for each asset within a cluster using the selected return metric.
        """
        n = cov.shape[0]
        if self.return_metric == 'inverse_variance':
            diag = np.diag(cov)
            diag = np.where(diag < 1e-10, 1e-10, diag)
            w = 1.0 / diag
        else:
            # Compute metric per asset
            metrics = np.zeros(n)
            for i in range(n):
                asset_ret = returns.iloc[:, i].values
                vol = np.sqrt(cov.iloc[i, i])
                if vol < 1e-10:
                    vol = 1e-10
                mean_ret = np.mean(asset_ret) * 252  # annualized
                if self.return_metric == 'sharpe':
                    metrics[i] = (mean_ret - self.risk_free_rate) / (vol * np.sqrt(252))
                elif self.return_metric == 'mean_return':
                    metrics[i] = mean_ret
                elif self.return_metric == 'return_over_var':
                    metrics[i] = mean_ret / (cov.iloc[i, i] * 252)
                else:
                    raise ValueError(f"Unknown return metric: {self.return_metric}")
            # Shift to positive if needed (Sharpe can be negative)
            min_val = np.min(metrics)
            if min_val < 0:
                metrics = metrics - min_val + 1e-6
            w = metrics / np.sum(metrics)
        return w / w.sum()

    def _cluster_score(self, cov: pd.DataFrame, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Compute a scalar score for the cluster. For inverse_variance, it's variance.
        For return metrics, we use the inverse of the weighted metric (higher metric → lower score).
        """
        if self.return_metric == 'inverse_variance':
            return np.linalg.multi_dot((weights, cov.values, weights))
        else:
            # Compute weighted average of the asset metrics
            metrics = []
            for i in range(cov.shape[0]):
                asset_ret = returns.iloc[:, i].values
                vol = np.sqrt(cov.iloc[i, i])
                if vol < 1e-10:
                    vol = 1e-10
                mean_ret = np.mean(asset_ret) * 252
                if self.return_metric == 'sharpe':
                    m = (mean_ret - self.risk_free_rate) / (vol * np.sqrt(252))
                elif self.return_metric == 'mean_return':
                    m = mean_ret
                elif self.return_metric == 'return_over_var':
                    m = mean_ret / (cov.iloc[i, i] * 252)
                else:
                    m = 0.0
                metrics.append(m)
            metrics = np.array(metrics)
            # Shift if needed
            min_val = np.min(metrics)
            if min_val < 0:
                metrics = metrics - min_val + 1e-6
            weighted_metric = np.dot(weights, metrics)
            # Higher metric → lower score (inverse relationship)
            return 1.0 / (weighted_metric + 1e-6)

    def get_linkage_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """Return linkage matrix and original ticker order."""
        return self.linkage, self.original_tickers
