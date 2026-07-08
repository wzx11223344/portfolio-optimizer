"""
Black-Litterman 模型模块
=========================
Black-Litterman 模型将市场均衡收益（先验）与投资者观点（后验）结合，
生成更稳定、更合理的期望收益估计，有效解决了纯历史均值估计的不稳定性问题。

核心公式：
  先验（市场均衡收益）:  Π = δ * Σ * w_market
  后验期望收益:  E[R] = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P^T Ω^{-1} Q]
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class BlackLittermanOptimizer:
    """Black-Litterman 优化器"""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 market_caps: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.03,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        初始化 Black-Litterman 优化器

        参数:
            mean_returns:    年化历史均值收益率（用于参考）
            cov_matrix:      年化协方差矩阵
            market_caps:     各资产市值（用于计算市场均衡权重），
                             如果为None则使用等权重作为市场权重
            risk_free_rate:  无风险利率
            risk_aversion:   风险厌恶系数 δ（典型值2-4）
            tau:             观点不确定性缩放因子（典型值0.01-0.05）
        """
        self.tickers = list(mean_returns.index)
        self.n_assets = len(mean_returns)
        self.cov_matrix = cov_matrix.values
        self.mean_returns = mean_returns.values
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau

        # 计算市场均衡权重
        if market_caps is not None:
            # 使用市值加权
            self.market_weights = (market_caps.reindex(mean_returns.index).values /
                                   market_caps.reindex(mean_returns.index).values.sum())
        else:
            # 如果没有市值数据，使用等权重作为近似
            self.market_weights = np.ones(self.n_assets) / self.n_assets

        # 计算市场均衡收益（先验）
        self.prior_returns = self._compute_equilibrium_returns()

    def _compute_equilibrium_returns(self) -> np.ndarray:
        """
        计算市场均衡收益（逆优化）

        市场组合是均值-方差最优的，因此可以通过逆优化反解出隐含均衡收益：
          Π = δ * Σ * w_market

        返回:
            市场均衡收益数组
        """
        pi = self.risk_aversion * np.dot(self.cov_matrix, self.market_weights)
        return pi

    def get_equilibrium_returns(self) -> pd.Series:
        """获取市场均衡收益（先验）"""
        return pd.Series(self.prior_returns, index=self.tickers)

    def optimize(self, P: np.ndarray, Q: np.ndarray,
                 omega: Optional[np.ndarray] = None,
                 weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> dict:
        """
        Black-Litterman 优化主流程

        参数:
            P:    观点矩阵 (K x N)，每行代表一个观点
                  绝对观点: [1, 0, 0, ...] 表示资产1的收益为Q中的值
                  相对观点: [1, -1, 0, ...] 表示资产1比资产2表现好Q中的值
            Q:    观点收益向量 (K x 1)
            omega: 观点不确定性矩阵 (K x K)，如果为None则自动计算
            weight_bounds: 权重上下限

        返回:
            优化结果字典
        """
        # 步骤1：计算后验期望收益
        posterior_returns = self._compute_posterior_returns(P, Q, omega)

        # 步骤2：使用后验收益进行均值-方差优化
        result = self._mean_variance_optimize(
            posterior_returns, weight_bounds
        )

        result["method"] = "Black-Litterman组合"
        result["prior_returns"] = pd.Series(self.prior_returns, index=self.tickers)
        result["posterior_returns"] = pd.Series(posterior_returns, index=self.tickers)

        return result

    def _compute_posterior_returns(self, P: np.ndarray, Q: np.ndarray,
                                   omega: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算后验期望收益

        E[R] = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P^T Ω^{-1} Q]

        参数:
            P:     观点矩阵 (K x N)
            Q:     观点收益向量 (K,)
            omega: 观点不确定性矩阵 (K x K)

        返回:
            后验期望收益数组
        """
        # tau * Sigma：先验协方差
        tau_sigma = self.tau * self.cov_matrix

        # 如果未提供 omega，使用 He-Litterman 启发式方法自动计算：
        # Omega = diag(P * (tau * Sigma) * P^T)
        # 这样每个观点的不确定性与该观点涉及的资产风险成比例
        if omega is None:
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))

        # 后验期望收益计算
        # 使用矩阵运算避免直接求逆以提高数值稳定性
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        # 后验精度矩阵（逆协方差）
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P

        # 后验均值
        posterior_mean = (
            np.linalg.inv(posterior_precision) @
            (tau_sigma_inv @ self.prior_returns + P.T @ omega_inv @ Q)
        )

        return posterior_mean

    def _mean_variance_optimize(self, expected_returns: np.ndarray,
                                weight_bounds: Tuple[float, float]) -> dict:
        """
        使用 cvxpy 进行最大夏普比率优化

        参数:
            expected_returns: 后验期望收益
            weight_bounds:    权重上下限

        返回:
            优化结果字典
        """
        # 确保协方差矩阵正定
        cov = self.cov_matrix.copy()
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-8:
            cov = cov + np.eye(self.n_assets) * 1e-8

        # 最大夏普比率：变量替换法
        excess_returns = expected_returns - self.risk_free_rate

        # 检查是否存在正超额收益
        if np.all(excess_returns <= 0):
            # 如果所有超额收益都非正，退回到最小方差
            return self._min_variance(cov, weight_bounds, expected_returns)

        y = cp.Variable(self.n_assets)
        k = cp.Variable(nonneg=True)

        objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov)))

        constraints = [
            excess_returns @ y == 1,
            cp.sum(y) == k,
            y >= 0,
            y <= k * weight_bounds[1],
            y >= k * weight_bounds[0],
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if y.value is None or k.value is None or k.value < 1e-10:
            return self._min_variance(cov, weight_bounds, expected_returns)

        weights = np.array(y.value) / k.value
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        portfolio_return = float(np.dot(weights, expected_returns))
        portfolio_vol = float(np.sqrt(np.dot(weights, np.dot(cov, weights))))
        sharpe = float((portfolio_return - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0

        return {
            "weights": pd.Series(weights, index=self.tickers),
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
        }

    def _min_variance(self, cov: np.ndarray,
                      weight_bounds: Tuple[float, float],
                      expected_returns: np.ndarray) -> dict:
        """最小方差优化（备选方案）"""
        w = cp.Variable(self.n_assets)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        constraints = [
            cp.sum(w) == 1,
            w >= weight_bounds[0],
            w <= weight_bounds[1],
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = np.array(w.value) if w.value is not None else np.ones(self.n_assets) / self.n_assets
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        portfolio_return = float(np.dot(weights, expected_returns))
        portfolio_vol = float(np.sqrt(np.dot(weights, np.dot(cov, weights))))
        sharpe = float((portfolio_return - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0

        return {
            "weights": pd.Series(weights, index=self.tickers),
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
        }

    def create_default_views(self, historical_returns: pd.Series,
                             confidence: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据历史收益率自动生成默认观点

        将历史收益率排名转化为相对观点：
        - 表现最好的资产优于表现最差的资产

        参数:
            historical_returns: 历史年化收益率
            confidence:         观点置信度（0-1）

        返回:
            (P矩阵, Q向量, Omega矩阵)
        """
        sorted_idx = historical_returns.sort_values(ascending=False).index.tolist()

        views = []
        # 生成相邻排名的相对观点
        for i in range(len(sorted_idx) - 1):
            view = np.zeros(self.n_assets)
            idx_high = self.tickers.index(sorted_idx[i])
            idx_low = self.tickers.index(sorted_idx[i + 1])
            view[idx_high] = 1
            view[idx_low] = -1
            views.append(view)

        P = np.array(views)
        # Q值：相邻资产的收益率差
        Q = np.array([
            historical_returns[sorted_idx[i]] - historical_returns[sorted_idx[i + 1]]
            for i in range(len(sorted_idx) - 1)
        ])

        # Omega：基于置信度计算
        tau_sigma = self.tau * self.cov_matrix
        omega = np.diag(np.diag(P @ tau_sigma @ P.T)) / max(confidence, 0.01)

        return P, Q, omega
