"""
数据获取模块
============
使用 akshare 获取A股股票历史日线数据，计算对数收益率和协方差矩阵。
支持 Ledoit-Wolf 收缩估计以改善协方差矩阵的稳定性。
"""

import akshare as ak
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import List, Tuple, Optional


class DataFetcher:
    """A股数据获取与预处理"""

    # A股交易日每年约244天
    TRADING_DAYS_PER_YEAR = 244

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        初始化数据获取器

        参数:
            tickers:   股票代码列表，如 ['600519', '000858', '601318']
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date:   结束日期，格式 'YYYYMMDD'
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.price_data: Optional[pd.DataFrame] = None  # 收盘价面板
        self.returns: Optional[pd.DataFrame] = None     # 对数收益率面板

    def fetch_data(self, adjust: str = "qfq") -> pd.DataFrame:
        """
        从 akshare 获取A股历史日线数据

        参数:
            adjust: 复权方式，'qfq'=前复权, 'hfq'=后复权, ''=不复权

        返回:
            收盘价 DataFrame，列为股票代码，行为日期
        """
        price_dict = {}

        for ticker in self.tickers:
            try:
                # 使用 akshare 的 stock_zh_a_hist 接口获取A股历史数据
                # 该接口返回真实交易所数据，非随机生成
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period="daily",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust=adjust,
                )

                if df is not None and not df.empty:
                    # 确保日期列格式正确
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.set_index("日期")
                    # 提取收盘价
                    price_dict[ticker] = df["收盘"]
                else:
                    print(f"警告: 股票 {ticker} 未获取到数据，请检查代码是否正确")

            except Exception as e:
                print(f"错误: 获取股票 {ticker} 数据失败: {e}")

        if not price_dict:
            raise ValueError("未能获取任何股票数据，请检查股票代码和日期范围")

        # 合并为面板数据，对齐日期（内连接，只保留所有股票都有数据的交易日）
        self.price_data = pd.DataFrame(price_dict)
        self.price_data = self.price_data.dropna()

        # 计算对数收益率
        self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna()

        return self.price_data

    def get_returns(self) -> pd.DataFrame:
        """获取对数收益率 DataFrame"""
        if self.returns is None:
            self.fetch_data()
        return self.returns

    def get_prices(self) -> pd.DataFrame:
        """获取收盘价 DataFrame"""
        if self.price_data is None:
            self.fetch_data()
        return self.price_data

    def get_mean_returns(self, annualize: bool = True) -> pd.Series:
        """
        计算各资产的均值收益率

        参数:
            annualize: 是否年化

        返回:
            各资产的年化均值收益率 Series
        """
        returns = self.get_returns()
        mean_returns = returns.mean()
        if annualize:
            mean_returns = mean_returns * self.TRADING_DAYS_PER_YEAR
        return mean_returns

    def get_covariance_matrix(self, method: str = "sample",
                              annualize: bool = True) -> pd.DataFrame:
        """
        计算协方差矩阵

        参数:
            method:    'sample' = 样本协方差, 'ledoit_wolf' = Ledoit-Wolf收缩估计
            annualize: 是否年化

        返回:
            协方差矩阵 DataFrame
        """
        returns = self.get_returns()

        if method == "ledoit_wolf":
            # Ledoit-Wolf 收缩估计：将样本协方差向单位矩阵收缩
            # 可以显著降低估计误差，尤其在样本量不足时效果明显
            lw = LedoitWolf()
            lw.fit(returns.values)
            cov_matrix = lw.covariance_
        else:
            # 样本协方差矩阵
            cov_matrix = returns.cov().values

        cov_df = pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns,
        )

        if annualize:
            cov_df = cov_df * self.TRADING_DAYS_PER_YEAR

        return cov_df

    def get_correlation_matrix(self) -> pd.DataFrame:
        """获取相关系数矩阵"""
        returns = self.get_returns()
        return returns.corr()

    def get_summary(self) -> pd.DataFrame:
        """
        生成数据摘要统计

        返回:
            包含年化收益率、年化波动率、夏普比率(无风险利率=0)的 DataFrame
        """
        returns = self.get_returns()
        mean_returns = returns.mean() * self.TRADING_DAYS_PER_YEAR
        vol = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        sharpe = mean_returns / vol

        summary = pd.DataFrame({
            "年化收益率": mean_returns,
            "年化波动率": vol,
            "夏普比率(无风险利率=0)": sharpe,
        })
        return summary
