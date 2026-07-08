"""
组合绩效指标模块
==================
计算投资组合的各项绩效指标，包括：
  - 年化收益率、年化波动率
  - 夏普比率 (Sharpe Ratio)
  - 最大回撤 (Max Drawdown)
  - Calmar 比率
  - Sortino 比率
  - 信息比率 (Information Ratio)
  - VaR (95%)、CVaR (95%)
"""

import numpy as np
import pandas as pd
from typing import Optional


class PortfolioMetrics:
    """投资组合绩效指标计算"""

    TRADING_DAYS_PER_YEAR = 244

    def __init__(self, returns: pd.DataFrame, weights: pd.Series,
                 risk_free_rate: float = 0.03,
                 benchmark_returns: Optional[pd.Series] = None):
        """
        初始化绩效指标计算器

        参数:
            returns:          资产收益率 DataFrame（日频对数收益率）
            weights:          组合权重 Series
            risk_free_rate:   年化无风险利率
            benchmark_returns: 基准收益率 Series（日频），用于计算信息比率
        """
        self.returns = returns
        self.weights = weights.reindex(returns.columns).fillna(0).values
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns

        # 计算组合日收益率
        self.portfolio_returns = (returns.values * self.weights).sum(axis=1)
        self.portfolio_returns = pd.Series(
            self.portfolio_returns,
            index=returns.index,
        )

        # 日无风险利率
        self.daily_rf = risk_free_rate / self.TRADING_DAYS_PER_YEAR

    def annualized_return(self) -> float:
        """
        年化收益率

        使用几何平均年化：(累积收益)^(244/天数) - 1
        """
        cumulative = (1 + self.portfolio_returns).prod()
        n_days = len(self.portfolio_returns)
        if n_days == 0:
            return 0.0
        annual_ret = cumulative ** (self.TRADING_DAYS_PER_YEAR / n_days) - 1
        return float(annual_ret)

    def annualized_volatility(self) -> float:
        """年化波动率"""
        daily_vol = self.portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float(annual_vol)

    def sharpe_ratio(self) -> float:
        """
        夏普比率

        Sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
        vol = self.annualized_volatility()
        if vol == 0:
            return 0.0
        ret = self.annualized_return()
        return float((ret - self.risk_free_rate) / vol)

    def max_drawdown(self) -> float:
        """
        最大回撤

        从历史最高点到后续最低点的最大跌幅
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        # 运行最大值（历史最高点）
        running_max = cumulative.expanding().max()
        # 回撤
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        return float(max_dd)

    def calmar_ratio(self) -> float:
        """
        Calmar 比率

        Calmar = 年化收益率 / |最大回撤|
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0
        annual_ret = self.annualized_return()
        return float(annual_ret / abs(max_dd))

    def sortino_ratio(self) -> float:
        """
        Sortino 比率

        只考虑下行波动率（负收益的波动）
        Sortino = (年化收益 - 无风险利率) / 年化下行波动率
        """
        # 下行偏差：仅计算负收益
        downside_returns = self.portfolio_returns.copy()
        downside_returns[downside_returns > 0] = 0

        # 下行波动率
        daily_downside_vol = np.sqrt((downside_returns ** 2).mean())
        annual_downside_vol = daily_downside_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        if annual_downside_vol == 0:
            return 0.0

        annual_ret = self.annualized_return()
        return float((annual_ret - self.risk_free_rate) / annual_downside_vol)

    def information_ratio(self) -> float:
        """
        信息比率

        IR = (组合超额收益 - 基准超额收益) / 跟踪误差
        需要提供基准收益率
        """
        if self.benchmark_returns is None:
            return float("nan")

        # 对齐日期
        common_idx = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )
        port_ret = self.portfolio_returns.loc[common_idx]
        bench_ret = self.benchmark_returns.loc[common_idx]

        # 超额收益（相对于基准）
        excess = port_ret - bench_ret

        # 年化超额收益
        annual_excess = excess.mean() * self.TRADING_DAYS_PER_YEAR

        # 跟踪误差（超额收益的年化标准差）
        tracking_error = excess.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        if tracking_error == 0:
            return 0.0

        return float(annual_excess / tracking_error)

    def var_95(self) -> float:
        """
        VaR (Value at Risk) 95% 置信度

        在95%置信度下，组合日收益率的最坏情况
        使用历史模拟法
        """
        var = np.percentile(self.portfolio_returns, 5)
        return float(var)

    def cvar_95(self) -> float:
        """
        CVaR (Conditional Value at Risk) 95% 置信度

        又称 Expected Shortfall，表示在 VaR 之外的尾部平均损失
        """
        var = self.var_95()
        # 取所有低于 VaR 的收益率
        tail_losses = self.portfolio_returns[self.portfolio_returns <= var]
        if len(tail_losses) == 0:
            return float(var)
        cvar = tail_losses.mean()
        return float(cvar)

    def get_all_metrics(self) -> dict:
        """
        计算所有绩效指标

        返回:
            包含所有指标的字典
        """
        return {
            "年化收益率": self.annualized_return(),
            "年化波动率": self.annualized_volatility(),
            "夏普比率": self.sharpe_ratio(),
            "最大回撤": self.max_drawdown(),
            "Calmar比率": self.calmar_ratio(),
            "Sortino比率": self.sortino_ratio(),
            "信息比率": self.information_ratio(),
            "VaR(95%)": self.var_95(),
            "CVaR(95%)": self.cvar_95(),
        }

    def get_cumulative_returns(self) -> pd.Series:
        """获取累积收益率序列（用于绘图）"""
        return (1 + self.portfolio_returns).cumprod()
