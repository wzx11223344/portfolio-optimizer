"""
HTML 报告生成模块
===================
使用 matplotlib 生成可视化图表，嵌入 HTML 报告中。
报告包含：
  - 有效前沿图
  - 权重饼图
  - 风险贡献条形图
  - 各优化方法对比表
"""

import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import List, Optional
from datetime import datetime

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ReportGenerator:
    """HTML 报告生成器"""

    def __init__(self):
        self.images = {}  # 存储图片的 base64 编码

    def _fig_to_base64(self, fig) -> str:
        """将 matplotlib Figure 转换为 base64 编码字符串"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return img_base64

    def plot_efficient_frontier(self, frontier_returns: np.ndarray,
                                frontier_vols: np.ndarray,
                                optimal_points: Optional[List[dict]] = None,
                                asset_returns: Optional[pd.Series] = None,
                                asset_vols: Optional[pd.Series] = None) -> str:
        """
        绘制有效前沿图

        参数:
            frontier_returns: 有效前沿收益率数组
            frontier_vols:    有效前沿波动率数组
            optimal_points:   最优组合点列表（如最大夏普、最小方差）
            asset_returns:    各资产收益率（用于标注）
            asset_vols:       各资产波动率（用于标注）

        返回:
            base64 编码的图片
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # 绘制有效前沿
        ax.plot(frontier_vols, frontier_returns, "b-", linewidth=2,
                label="有效前沿", zorder=1)

        # 标注各资产
        if asset_returns is not None and asset_vols is not None:
            ax.scatter(asset_vols.values, asset_returns.values,
                       color="gray", s=50, alpha=0.7, label="各资产", zorder=2)
            for ticker in asset_returns.index:
                ax.annotate(ticker,
                            (asset_vols[ticker], asset_returns[ticker]),
                            fontsize=8, alpha=0.7)

        # 标注最优组合
        if optimal_points:
            colors = ["red", "green", "orange", "purple"]
            markers = ["*", "D", "s", "^"]
            for i, point in enumerate(optimal_points):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                ax.scatter(point["volatility"], point["expected_return"],
                           color=color, s=200, marker=marker, zorder=5,
                           edgecolors="black", linewidth=1,
                           label=point.get("method", f"组合{i+1}"))

        ax.set_xlabel("年化波动率", fontsize=12)
        ax.set_ylabel("年化收益率", fontsize=12)
        ax.set_title("马科维茨有效前沿", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def plot_weights_pie(self, results: List[dict]) -> str:
        """
        绘制各优化方法的权重饼图

        参数:
            results: 优化结果列表

        返回:
            base64 编码的图片
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        # 色彩方案
        colors = plt.cm.Set3(np.linspace(0, 1, 20))

        for i, result in enumerate(results):
            weights = result["weights"]
            method = result.get("method", f"方法{i+1}")

            # 过滤掉权重接近0的资产
            mask = weights.abs() > 0.001
            w = weights[mask]

            axes[i].pie(w.values, labels=w.index, autopct="%1.1f%%",
                        colors=colors[:len(w)], startangle=90,
                        textprops={"fontsize": 9})
            axes[i].set_title(method, fontsize=12, fontweight="bold")

        fig.suptitle("各优化方法权重分配", fontsize=14, fontweight="bold", y=1.02)
        return self._fig_to_base64(fig)

    def plot_risk_contribution(self, results: List[dict],
                               cov_matrix: pd.DataFrame) -> str:
        """
        绘制风险贡献条形图

        参数:
            results:    优化结果列表
            cov_matrix: 协方差矩阵

        返回:
            base64 编码的图片
        """
        n_methods = len(results)
        tickers = list(cov_matrix.columns)
        n_assets = len(tickers)

        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for i, result in enumerate(results):
            weights = result["weights"].reindex(tickers).fillna(0).values
            method = result.get("method", f"方法{i+1}")

            # 计算风险贡献
            portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))
            if portfolio_var < 1e-20:
                rc = np.zeros(n_assets)
            else:
                portfolio_vol = np.sqrt(portfolio_var)
                mrc = np.dot(cov_matrix.values, weights) / portfolio_vol
                rc = weights * mrc
                rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

            x = np.arange(n_assets)
            axes[i].bar(x, rc_pct * 100, color=plt.cm.tab10(np.linspace(0, 0.9, n_assets)))
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
            axes[i].set_ylabel("风险贡献占比 (%)", fontsize=10)
            axes[i].set_title(method, fontsize=12, fontweight="bold")
            axes[i].axhline(y=100.0 / n_assets, color="red", linestyle="--",
                           alpha=0.5, label=f"等风险贡献={100/n_assets:.1f}%")
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")

        fig.suptitle("各资产风险贡献", fontsize=14, fontweight="bold", y=1.02)
        return self._fig_to_base64(fig)

    def plot_cumulative_returns(self, cumulative_returns_dict: dict) -> str:
        """
        绘制各组合的累积收益曲线

        参数:
            cumulative_returns_dict: {方法名: 累积收益Series}

        返回:
            base64 编码的图片
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for method, cum_ret in cumulative_returns_dict.items():
            ax.plot(cum_ret.index, cum_ret.values, linewidth=1.5, label=method)

        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("累积收益", fontsize=12)
        ax.set_title("各优化方法累积收益对比", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3)

        return self._fig_to_base64(fig)

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """
        绘制相关性热力图

        参数:
            corr_matrix: 相关系数矩阵

        返回:
            base64 编码的图片
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(corr_matrix.values, cmap="RdYlGn_r", vmin=-1, vmax=1,
                       aspect="auto")

        # 设置刻度
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_matrix.columns, fontsize=9)

        # 添加数值标注
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.values[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

        ax.set_title("资产相关系数矩阵", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

        return self._fig_to_base64(fig)

    def generate_html_report(self, tickers: List[str], start_date: str,
                             end_date: str, results: List[dict],
                             data_summary: pd.DataFrame,
                             images_dict: dict,
                             metrics_table: Optional[pd.DataFrame] = None) -> str:
        """
        生成完整的 HTML 报告

        参数:
            tickers:       股票代码列表
            start_date:    开始日期
            end_date:      结束日期
            results:       优化结果列表
            data_summary:  数据摘要统计
            images_dict:   图片字典 {名称: base64编码}
            metrics_table: 绩效指标对比表

        返回:
            HTML 字符串
        """
        # 构建图片 HTML
        def img_tag(b64, alt_text="", width="100%"):
            return f'<img src="data:image/png;base64,{b64}" alt="{alt_text}" style="width:{width};max-width:1000px;display:block;margin:20px auto;border:1px solid #ddd;border-radius:8px;"/>'

        # 构建对比表 HTML
        comparison_rows = ""
        for result in results:
            comparison_rows += f"""
            <tr>
                <td>{result.get('method', '')}</td>
                <td>{result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)</td>
                <td>{result['volatility']:.4f} ({result['volatility']*100:.2f}%)</td>
                <td>{result['sharpe_ratio']:.4f}</td>
            </tr>"""

        # 构建权重表 HTML
        weight_table_rows = ""
        all_tickers = set()
        for result in results:
            all_tickers.update(result["weights"].index)
        all_tickers = sorted(all_tickers)

        header_cells = "".join([f"<th>{t}</th>" for t in all_tickers])
        for result in results:
            weight_cells = ""
            for t in all_tickers:
                w = result["weights"].get(t, 0)
                weight_cells += f"<td>{w:.4f} ({w*100:.2f}%)</td>"
            weight_table_rows += f"""
            <tr>
                <td><strong>{result.get('method', '')}</strong></td>
                {weight_cells}
            </tr>"""

        # 构建数据摘要表 HTML
        summary_rows = ""
        for ticker, row in data_summary.iterrows():
            summary_rows += f"""
            <tr>
                <td>{ticker}</td>
                <td>{row['年化收益率']:.4f} ({row['年化收益率']*100:.2f}%)</td>
                <td>{row['年化波动率']:.4f} ({row['年化波动率']*100:.2f}%)</td>
                <td>{row['夏普比率(无风险利率=0)']:.4f}</td>
            </tr>"""

        # 构建绩效指标表 HTML
        metrics_html = ""
        if metrics_table is not None and not metrics_table.empty:
            metrics_header = "".join([f"<th>{c}</th>" for c in metrics_table.columns])
            metrics_rows = ""
            for idx, row in metrics_table.iterrows():
                cells = "".join([f"<td>{v:.4f}</td>" if isinstance(v, (int, float)) and not np.isnan(v) else f"<td>N/A</td>" for v in row])
                metrics_rows += f"<tr><td><strong>{idx}</strong></td>{cells}</tr>"
            metrics_html = f"""
            <h2>五、组合绩效指标对比</h2>
            <div class="table-container">
                <table>
                    <thead><tr><th>优化方法</th>{metrics_header}</tr></thead>
                    <tbody>{metrics_rows}</tbody>
                </table>
            </div>"""

        # 构建图片区域 HTML
        images_html = ""
        if "efficient_frontier" in images_dict:
            images_html += f"<h2>六、有效前沿</h2>{img_tag(images_dict['efficient_frontier'], '有效前沿')}"
        if "weights_pie" in images_dict:
            images_html += f"<h2>七、权重分配</h2>{img_tag(images_dict['weights_pie'], '权重饼图')}"
        if "risk_contribution" in images_dict:
            images_html += f"<h2>八、风险贡献分析</h2>{img_tag(images_dict['risk_contribution'], '风险贡献')}"
        if "correlation" in images_dict:
            images_html += f"<h2>九、资产相关性分析</h2>{img_tag(images_dict['correlation'], '相关性热力图')}"
        if "cumulative_returns" in images_dict:
            images_html += f"<h2>十、累积收益对比</h2>{img_tag(images_dict['cumulative_returns'], '累积收益')}"

        # 完整 HTML 模板
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>投资组合优化报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a237e;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        h2 {{
            color: #283593;
            margin: 30px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0;
            font-size: 22px;
        }}
        .info-bar {{
            background: #e8eaf6;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .info-item {{
            font-size: 14px;
        }}
        .info-item strong {{
            color: #1a237e;
        }}
        .table-container {{
            overflow-x: auto;
            margin: 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            min-width: 600px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        th {{
            background: #3f51b5;
            color: white;
            font-weight: bold;
        }}
        tbody tr:nth-child(even) {{
            background: #f5f5f5;
        }}
        tbody tr:hover {{
            background: #e8eaf6;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>投资组合优化报告</h1>
    <div class="info-bar">
        <span class="info-item"><strong>股票池:</strong> {', '.join(tickers)}</span>
        <span class="info-item"><strong>数据区间:</strong> {start_date} ~ {end_date}</span>
        <span class="info-item"><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>

    <h2>一、数据摘要</h2>
    <div class="table-container">
        <table>
            <thead>
                <tr><th>股票代码</th><th>年化收益率</th><th>年化波动率</th><th>夏普比率(rf=0)</th></tr>
            </thead>
            <tbody>{summary_rows}</tbody>
        </table>
    </div>

    <h2>二、优化结果对比</h2>
    <div class="table-container">
        <table>
            <thead>
                <tr><th>优化方法</th><th>预期收益率</th><th>波动率</th><th>夏普比率</th></tr>
            </thead>
            <tbody>{comparison_rows}</tbody>
        </table>
    </div>

    <h2>三、各方法权重分配</h2>
    <div class="table-container">
        <table>
            <thead>
                <tr><th>优化方法</th>{header_cells}</tr>
            </thead>
            <tbody>{weight_table_rows}</tbody>
        </table>
    </div>

    {metrics_html}

    {images_html}

    <div class="footer">
        <p>本报告由「投资组合优化大师」自动生成 | 基于真实A股数据(akshare)</p>
        <p>免责声明：本报告仅供学术研究和学习参考，不构成任何投资建议。投资有风险，入市需谨慎。</p>
    </div>
</div>
</body>
</html>"""

        return html
