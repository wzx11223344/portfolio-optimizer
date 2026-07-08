"""
风险平价优化模块 (Risk Parity)
==============================
风险平价策略使各资产对组合总风险的贡献相等，而非资金权重相等。
这避免了高风险资产在资金等权分配中占据过大风险敞口的问题。

核心概念：
  - 边际风险贡献 (MRC): MRC_i = (Σw)_i / sqrt(w^T Σ w)
  - 风险贡献 (RC):       RC_i = w_i * MRC_i
  - 目标:                RC_i = RC_j = (组合总风险 / N) for all i, j
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional


class RiskParityOptimizer:
    """风险平价优化器（等风险贡献）"""

    def __init__(self, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        初始化风险平价优化器

        参数:
            cov_matrix:     年化协方差矩阵
            risk_free_rate: 无风险利率
        """
        self.cov_matrix = cov_matrix.values
        self.tickers = list(cov_matrix.columns)
        self.n_assets = len(cov_matrix)
        self.risk_free_rate = risk_free_rate

        # 确保协方差矩阵正定
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if eigvals.min() < 1e-8:
            self.cov_matrix = self.cov_matrix + np.eye(self.n_assets) * 1e-8

    def optimize(self, weight_bounds: Tuple[float, float] = (0.01, 1.0),
                 mean_returns: Optional[pd.Series] = None) -> dict:
        """
        求解等风险贡献权重

        使用 scipy.optimize 最小化风险贡献之间的差异

        参数:
            weight_bounds: 权重上下限
            mean_returns:  期望收益率（用于计算夏普比率，可选）

        返回:
            优化结果字典
        """
        n = self.n_assets

        # 目标函数：最小化各资产风险贡献与目标（等风险贡献）之间的偏差
        def risk_parity_objective(w):
            """
            目标函数：最小化风险贡献的方差

            等风险贡献意味着每个资产的 RC_i 应该等于 sigma_p / N
            我们最小化 sum((RC_i - sigma_p/N)^2)
            """
            w = np.array(w)
            # 组合波动率
            portfolio_var = np.dot(w, np.dot(self.cov_matrix, w))
            if portfolio_var < 1e-20:
                return 1e10
            portfolio_vol = np.sqrt(portfolio_var)

            # 边际风险贡献
            mrc = np.dot(self.cov_matrix, w) / portfolio_vol
            # 风险贡献
            rc = w * mrc
            # 目标风险贡献（等风险贡献）
            target_rc = portfolio_vol / n

            # 最小化风险贡献与目标的偏差平方和
            return np.sum((rc - target_rc) ** 2)

        # 初始猜测：等权重
        w0 = np.ones(n) / n

        # 约束条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 权重之和=1
        ]

        # 权重边界
        bounds = [(weight_bounds[0], weight_bounds[1])] * n

        # 求解
        result = minimize(
            risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            print(f"警告: 风险平价优化可能未收敛: {result.message}")

        weights = result.x
        # 清理极小负值并归一化
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        # 计算结果指标
        portfolio_var = float(np.dot(weights, np.dot(self.cov_matrix, weights)))
        portfolio_vol = float(np.sqrt(portfolio_var))

        # 如果提供了期望收益率，计算夏普比率
        if mean_returns is not None:
            mu = mean_returns.reindex(self.tickers).values
            portfolio_return = float(np.dot(weights, mu))
            sharpe = float((portfolio_return - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0
        else:
            portfolio_return = 0.0
            sharpe = 0.0

        # 计算各资产的风险贡献
        risk_contributions = self.compute_risk_contributions(weights)

        return {
            "method": "风险平价组合",
            "weights": pd.Series(weights, index=self.tickers),
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
            "risk_contributions": risk_contributions,
        }

    def compute_risk_contributions(self, weights: np.ndarray) -> pd.Series:
        """
        计算各资产的风险贡献

        参数:
            weights: 权重数组

        返回:
            各资产风险贡献占比的 Series（总和为1）
        """
        weights = np.array(weights)
        portfolio_var = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        # 边际风险贡献: MRC_i = (Σw)_i / σ_p
        mrc = np.dot(self.cov_matrix, weights) / portfolio_vol
        # 风险贡献: RC_i = w_i * MRC_i
        rc = weights * mrc

        # 归一化为百分比
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        return pd.Series(rc_pct, index=self.tickers)

    def get_marginal_risk_contributions(self, weights: np.ndarray) -> pd.Series:
        """
        计算各资产的边际风险贡献

        边际风险贡献表示增加单位权重时组合风险的变化量

        参数:
            weights: 权重数组

        返回:
            边际风险贡献 Series
        """
        weights = np.array(weights)
        portfolio_var = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        mrc = np.dot(self.cov_matrix, weights) / portfolio_vol
        return pd.Series(mrc, index=self.tickers)
