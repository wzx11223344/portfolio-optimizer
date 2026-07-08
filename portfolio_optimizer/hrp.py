"""
层次风险平价模块 (Hierarchical Risk Parity, HRP)
==================================================
HRP 由 Marcos Lopez de Prado (2016) 提出，通过层次聚类和递归二分法分配权重，
无需估计收益率和协方差矩阵的逆矩阵，因此对参数估计误差更加鲁棒。

算法步骤：
  1. 计算相关性距离矩阵
  2. 使用层次聚类（ward linkage）对资产进行分组
  3. 准对角化：按聚类顺序重排协方差矩阵
  4. 递归二分法：不断将组合一分为二，分配权重
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
from typing import Optional


class HRPOptimizer:
    """层次风险平价优化器"""

    def __init__(self, cov_matrix: pd.DataFrame,
                 corr_matrix: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.03):
        """
        初始化 HRP 优化器

        参数:
            cov_matrix:     年化协方差矩阵
            corr_matrix:    相关系数矩阵（如果为None则从协方差矩阵推导）
            risk_free_rate: 无风险利率
        """
        self.cov_matrix = cov_matrix
        self.tickers = list(cov_matrix.columns)
        self.n_assets = len(cov_matrix)
        self.risk_free_rate = risk_free_rate

        # 如果未提供相关系数矩阵，从协方差矩阵推导
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
        else:
            # corr_ij = cov_ij / (sigma_i * sigma_j)
            std = np.sqrt(np.diag(cov_matrix.values))
            outer_std = np.outer(std, std)
            outer_std[outer_std == 0] = 1e-10
            corr_values = cov_matrix.values / outer_std
            # 限制在 [-1, 1] 范围内
            corr_values = np.clip(corr_values, -1.0, 1.0)
            np.fill_diagonal(corr_values, 1.0)
            self.corr_matrix = pd.DataFrame(
                corr_values,
                index=cov_matrix.index,
                columns=cov_matrix.columns,
            )

    def optimize(self, mean_returns: Optional[pd.Series] = None) -> dict:
        """
        执行 HRP 优化

        参数:
            mean_returns: 期望收益率（用于计算夏普比率，可选）

        返回:
            优化结果字典
        """
        # 步骤1：计算距离矩阵
        distances = self._compute_distance_matrix()

        # 步骤2：层次聚类
        link = self._hierarchical_clustering(distances)

        # 步骤3：获取准对角化排序
        sorted_indices = self._get_quasi_diag(link)

        # 步骤4：递归二分法分配权重
        weights = self._recursive_bisection(sorted_indices)

        # 构建结果
        weight_series = pd.Series(weights, index=self.tickers)

        # 计算组合指标
        cov = self.cov_matrix.values
        portfolio_var = float(np.dot(weights, np.dot(cov, weights)))
        portfolio_vol = float(np.sqrt(portfolio_var))

        if mean_returns is not None:
            mu = mean_returns.reindex(self.tickers).values
            portfolio_return = float(np.dot(weights, mu))
            sharpe = float((portfolio_return - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0
        else:
            portfolio_return = 0.0
            sharpe = 0.0

        return {
            "method": "层次风险平价组合(HRP)",
            "weights": weight_series,
            "expected_return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
            "linkage_matrix": link,
            "sorted_indices": sorted_indices,
        }

    def _compute_distance_matrix(self) -> pd.DataFrame:
        """
        计算相关性距离矩阵

        距离定义: d_ij = sqrt(0.5 * (1 - corr_ij))
        这个距离度量满足度量空间的公理

        返回:
            距离矩阵 DataFrame
        """
        corr = self.corr_matrix.values
        # 距离公式
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        return pd.DataFrame(
            dist,
            index=self.corr_matrix.index,
            columns=self.corr_matrix.columns,
        )

    def _hierarchical_clustering(self, distances: pd.DataFrame) -> np.ndarray:
        """
        执行层次聚类

        使用 Ward 连接法，最小化簇内方差

        参数:
            distances: 距离矩阵

        返回:
            聚类链接矩阵 (linkage matrix)
        """
        # 将距离矩阵转换为压缩形式（上三角向量）
        dist_condensed = squareform(distances.values, checks=False)

        # 使用 ward 连接法进行层次聚类
        link = linkage(dist_condensed, method="ward")

        return link

    def _get_quasi_diag(self, link: np.ndarray) -> list:
        """
        获取准对角化排序

        将层次聚类的树状结构展开为叶子节点的有序列表
        使得相关的资产在协方差矩阵中相邻排列

        参数:
            link: 聚类链接矩阵

        返回:
            排序后的资产索引列表
        """
        # 将链接矩阵转换为树结构并获取叶子节点的顺序
        link_copy = link.copy()
        # 转换为整数
        link_copy = link_copy.astype(int)

        # 获取叶子节点排序
        n = self.n_assets
        sorted_indices = self._get_cluster_order(link_copy, n)

        return sorted_indices

    def _get_cluster_order(self, link: np.ndarray, n: int) -> list:
        """
        递归获取聚类树叶子节点的排序

        参数:
            link: 链接矩阵
            n:    原始资产数量

        返回:
            排序后的叶子索引列表
        """
        # 使用 to_tree 将链接矩阵转换为树
        tree = to_tree(link)

        # 递归遍历树获取叶子顺序
        def get_leaves(node):
            if node.is_leaf():
                return [node.id]
            else:
                return get_leaves(node.left) + get_leaves(node.right)

        return get_leaves(tree)

    def _recursive_bisection(self, sorted_indices: list) -> np.ndarray:
        """
        递归二分法分配权重

        算法：
        1. 初始化所有资产等权重
        2. 将排序后的资产列表分为两半
        3. 根据每半的聚合风险（逆方差）分配权重
        4. 对每个子集递归执行步骤2-3

        参数:
            sorted_indices: 准对角化排序后的资产索引

        返回:
            权重数组
        """
        # 初始化权重
        weights = np.ones(self.n_assets)

        # 递归分配
        clusters = [sorted_indices]

        while len(clusters) > 0:
            # 弹出最后一个簇
            cluster = clusters.pop()

            # 如果簇只有一个元素，无需继续分割
            if len(cluster) <= 1:
                continue

            # 将簇分为两半
            mid = len(cluster) // 2
            left_cluster = cluster[:mid]
            right_cluster = cluster[mid:]

            # 计算左半和右半的聚合方差（使用逆方差作为权重）
            left_var = self._get_cluster_variance(left_cluster)
            right_var = self._get_cluster_variance(right_cluster)

            # 根据逆方差比例分配权重
            # 风险较小的子集获得更大权重
            alpha = 1.0 - left_var / (left_var + right_var)

            # 更新左半和右半的权重
            for i in left_cluster:
                weights[i] *= alpha
            for i in right_cluster:
                weights[i] *= (1.0 - alpha)

            # 将子簇加入待处理列表
            clusters.append(left_cluster)
            clusters.append(right_cluster)

        # 归一化
        weights = weights / weights.sum()

        return weights

    def _get_cluster_variance(self, indices: list) -> float:
        """
        计算子簇的聚合方差

        使用逆方差加权计算子组合的方差

        参数:
            indices: 子簇中的资产索引列表

        返回:
            子簇的聚合方差
        """
        cov = self.cov_matrix.values
        sub_cov = cov[np.ix_(indices, indices)]

        # 逆方差权重
        ivp = 1.0 / np.diag(sub_cov)
        ivp = ivp / ivp.sum()

        # 子组合方差
        cluster_var = np.dot(ivp, np.dot(sub_cov, ivp))

        return float(cluster_var)

    def get_dendrogram_data(self, link: np.ndarray) -> dict:
        """
        获取树状图数据（用于可视化）

        参数:
            link: 链接矩阵

        返回:
            树状图数据字典
        """
        # 使用 scipy 的 dendrogram 函数计算绘图数据
        ddg = dendrogram(
            link,
            labels=self.tickers,
            no_plot=True,
            orientation="left",
        )
        return ddg
