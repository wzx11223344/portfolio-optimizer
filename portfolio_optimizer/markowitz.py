"""
均值-方差优化模块 (Markowitz Mean-Variance Optimization)
=========================================================
基于 Markowitz (1952) 现代投资组合理论，使用 cvxpy 进行凸优化求解：
  - 最大夏普比率组合（通过变量替换转化为二次规划）
  - 最小方差组合
  - 有效前沿计算
  - 支持权重上下限约束和做空限制
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class MarkowitzOptimizer:
    """均值-方差优化器"""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.03):
        """
        初始化优化器

        参数:
            mean_returns:    年化期望收益率 Series
            cov_matrix:      年化协方差矩阵 DataFrame
            risk_free_rate:  无风险利率（年化），默认3%
        """
        self.mean_returns = mean_returns.values
        self.cov_matrix = cov_matrix.values
        self.tickers = list(mean_returns.index)
        self.n_assets = len(mean_returns)
        self.risk_free_rate = risk_free_rate

        # 确保 Sigma 为正定矩阵（添加微小正则化项以保证数值稳定性）
        # 这是凸优化的必要条件
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if eigvals.min() < 1e-8:
            self.cov_matrix = self.cov_matrix + np.eye(self.n_assets) * 1e-8

    def min_variance(self, weight_bounds: Tuple[float, float] = (0.0, 1.0),
                     allow_short: bool = False) -> dict:
        """
        最小方差组合：在给定约束下最小化组合方差

        参数:
            weight_bounds: 权重上下限 (lower, upper)
            allow_short:   是否允许做空

        返回:
            包含权重、预期收益、波动率、夏普比率的字典
        """
        # 定义优化变量
        w = cp.Variable(self.n_assets)

        # 目标函数：最小化 w^T * Sigma * w
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(self.cov_matrix)))

        # 约束条件
        constraints = [cp.sum(w) == 1]  # 权重之和为1

        if not allow_short:
            # 不允许做空：权重非负
            constraints.append(w >= weight_bounds[0])

        constraints.append(w <= weight_bounds[1])

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is None:
            raise RuntimeError("最小方差优化求解失败，请检查约束条件")

        weights = np.array(w.value)
        # 清理数值误差导致的极小负值
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return self._build_result(weights, "最小方差组合")

    def max_sharpe(self, weight_bounds: Tuple[float, float] = (0.0, 1.0),
                   allow_short: bool = False) -> dict:
        """
        最大夏普比率组合

        通过变量替换将非凸的最大夏普比率问题转化为凸的二次规划问题：
        原问题: max (mu - rf)^T w / sqrt(w^T Sigma w), s.t. sum(w) = 1
        令 y = k*w, k > 0, 则:
          min y^T Sigma y
          s.t. (mu - rf)^T y = 1
               sum(y) = k (k >= 0)
        最优权重 w* = y* / sum(y*)

        参数:
            weight_bounds: 权重上下限 (lower, upper)
            allow_short:   是否允许做空

        返回:
            包含权重、预期收益、波动率、夏普比率的字典
        """
        # 超额收益
        excess_returns = self.mean_returns - self.risk_free_rate

        # 定义辅助变量 y 和标量 k
        y = cp.Variable(self.n_assets)
        k = cp.Variable(nonneg=True)  # k >= 0

        # 目标函数：最小化 y^T Sigma y
        objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(self.cov_matrix)))

        # 约束条件
        constraints = [
            excess_returns @ y == 1,  # (mu - rf)^T y = 1
            cp.sum(y) == k,            # sum(y) = k
        ]

        if not allow_short:
            # 不允许做空：y >= 0 (因为 k > 0, w = y/k, 所以 y >= 0 等价于 w >= 0)
            constraints.append(y >= 0)
            # 权重上限约束: w_i <= upper 等价于 y_i <= k * upper
            constraints.append(y <= k * weight_bounds[1])
            # 权重下限约束
            constraints.append(y >= k * weight_bounds[0])
        else:
            # 允许做空时仍有上限约束
            constraints.append(y <= k * weight_bounds[1])
            constraints.append(y >= k * weight_bounds[0])

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if y.value is None or k.value is None or k.value < 1e-10:
            # 如果转换方法失败，退回到网格搜索方法
            return self._max_sharpe_grid_search(weight_bounds, allow_short)

        # 还原权重: w = y / k
        weights = np.array(y.value) / k.value
        weights = np.maximum(weights, 0) if not allow_short else weights
        # 归一化
        weights = weights / weights.sum()

        return self._build_result(weights, "最大夏普比率组合")

    def _max_sharpe_grid_search(self, weight_bounds: Tuple[float, float],
                                allow_short: bool) -> dict:
        """
        最大夏普比率的网格搜索备选方案
        在有效前沿上搜索夏普比率最大的点
        """
        frontier = self.efficient_frontier(weight_bounds, allow_short, n_points=100)
        best_sharpe = -np.inf
        best_result = None

        for point in frontier:
            if point["sharpe_ratio"] > best_sharpe:
                best_sharpe = point["sharpe_ratio"]
                best_result = point

        if best_result is None:
            # 最终回退到等权重
            weights = np.ones(self.n_assets) / self.n_assets
            return self._build_result(weights, "最大夏普比率组合(等权重回退)")

        best_result["method"] = "最大夏普比率组合"
        return best_result

    def efficient_frontier(self, weight_bounds: Tuple[float, float] = (0.0, 1.0),
                           allow_short: bool = False,
                           n_points: int = 50) -> List[dict]:
        """
        计算有效前沿

        在不同的目标收益率水平下，最小化组合方差，得到有效前沿上的点

        参数:
            weight_bounds: 权重上下限
            allow_short:   是否允许做空
            n_points:      有效前沿上的点数

        返回:
            有效前沿上各点的列表，每个点包含权重、收益、波动率、夏普比率
        """
        # 确定目标收益率的范围
        # 最小目标收益率：各资产最小收益率
        # 最大目标收益率：各资产最大收益率
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()

        # 生成目标收益率序列
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target_ret in target_returns:
            try:
                result = self._min_variance_for_target(
                    target_ret, weight_bounds, allow_short
                )
                if result is not None:
                    frontier.append(result)
            except Exception:
                continue

        return frontier

    def _min_variance_for_target(self, target_return: float,
                                 weight_bounds: Tuple[float, float],
                                 allow_short: bool) -> Optional[dict]:
        """
        在给定目标收益率下最小化方差

        参数:
            target_return: 目标收益率
            weight_bounds: 权重上下限
            allow_short:   是否允许做空

        返回:
            优化结果字典，如果不可行则返回 None
        """
        w = cp.Variable(self.n_assets)

        # 目标函数：最小化组合方差
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(self.cov_matrix)))

        # 约束条件
        constraints = [
            cp.sum(w) == 1,                         # 权重之和为1
            self.mean_returns @ w >= target_return,  # 收益率不低于目标
        ]

        if not allow_short:
            constraints.append(w >= weight_bounds[0])

        constraints.append(w <= weight_bounds[1])

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is None:
            return None

        weights = np.array(w.value)
        weights = np.maximum(weights, 0) if not allow_short else weights
        sum_w = weights.sum()
        if sum_w > 0:
            weights = weights / sum_w

        return self._build_result(weights, "有效前沿点")

    def _build_result(self, weights: np.ndarray, method: str) -> dict:
        """
        构建优化结果字典

        参数:
            weights: 最优权重数组
            method:  方法名称

        返回:
            结果字典
        """
        # 组合预期收益率
        portfolio_return = float(np.dot(weights, self.mean_returns))
        # 组合方差
        portfolio_var = float(np.dot(weights, np.dot(self.cov_matrix, weights)))
        # 组合波动率（标准差）
        portfolio_vol = float(np.sqrt(portfolio_var))
        # 夏普比率
        sharpe = float((portfolio_return - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0

        return {
            "method": method,
            "weights": pd.Series(weights, index=self.tickers),
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
        }

    def get_efficient_frontier_data(self, weight_bounds: Tuple[float, float] = (0.0, 1.0),
                                    allow_short: bool = False,
                                    n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取有效前沿数据（用于绘图）

        返回:
            (returns_array, volatilities_array)
        """
        frontier = self.efficient_frontier(weight_bounds, allow_short, n_points)
        returns = np.array([p["expected_return"] for p in frontier])
        vols = np.array([p["volatility"] for p in frontier])
        return returns, vols
