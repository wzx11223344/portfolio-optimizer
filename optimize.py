#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
投资组合优化大师 - CLI 入口
=============================
命令行工具，支持多种投资组合优化方法。

用法示例:
    python optimize.py --tickers 600519,000858,601318 --method all --start 20240101 --end 20250101
    python optimize.py --tickers 600519,000858,601318 --method markowitz --start 20240101 --end 20250101
    python optimize.py --tickers 600519,000858,601318 --method risk_parity,hrp --start 20240101 --end 20250101
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio_optimizer.data import DataFetcher
from portfolio_optimizer.markowitz import MarkowitzOptimizer
from portfolio_optimizer.black_litterman import BlackLittermanOptimizer
from portfolio_optimizer.risk_parity import RiskParityOptimizer
from portfolio_optimizer.hrp import HRPOptimizer
from portfolio_optimizer.metrics import PortfolioMetrics
from portfolio_optimizer.report import ReportGenerator

console = Console()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="投资组合优化大师 - 多策略投资组合优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python optimize.py --tickers 600519,000858,601318 --method all --start 20240101 --end 20250101
  python optimize.py --tickers 600519,000858,601318 --method markowitz --start 20240101 --end 20250101
  python optimize.py --tickers 600519,000858,601318,000001,600036 --method risk_parity,hrp
        """,
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="股票代码列表，逗号分隔，如 600519,000858,601318",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help="优化方法: markowitz, bl, risk_parity, hrp, all（可逗号分隔选择多个）",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="20240101",
        help="开始日期，格式 YYYYMMDD，默认 20240101",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="20250101",
        help="结束日期，格式 YYYYMMDD，默认 20250101",
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=0.03,
        help="无风险利率（年化），默认 0.03 (3%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="报告输出目录，默认 output",
    )
    parser.add_argument(
        "--cov-method",
        type=str,
        default="sample",
        choices=["sample", "ledoit_wolf"],
        help="协方差矩阵估计方法: sample 或 ledoit_wolf，默认 sample",
    )
    parser.add_argument(
        "--weight-lower",
        type=float,
        default=0.0,
        help="权重下限，默认 0.0（不允许做空）",
    )
    parser.add_argument(
        "--weight-upper",
        type=float,
        default=1.0,
        help="权重上限，默认 1.0",
    )
    return parser.parse_args()


def run_markowitz(data_fetcher, mean_returns, cov_matrix, rf, weight_bounds):
    """运行均值-方差优化"""
    optimizer = MarkowitzOptimizer(mean_returns, cov_matrix, risk_free_rate=rf)

    results = []

    # 最大夏普比率
    console.print("  [cyan]计算最大夏普比率组合...[/cyan]")
    max_sharpe = optimizer.max_sharpe(weight_bounds=weight_bounds)
    results.append(max_sharpe)

    # 最小方差
    console.print("  [cyan]计算最小方差组合...[/cyan]")
    min_var = optimizer.min_variance(weight_bounds=weight_bounds)
    results.append(min_var)

    # 有效前沿数据
    console.print("  [cyan]计算有效前沿...[/cyan]")
    frontier = optimizer.efficient_frontier(weight_bounds=weight_bounds, n_points=50)

    return results, optimizer, frontier


def run_black_litterman(data_fetcher, mean_returns, cov_matrix, rf, weight_bounds):
    """运行 Black-Litterman 优化"""
    optimizer = BlackLittermanOptimizer(
        mean_returns, cov_matrix, risk_free_rate=rf
    )

    # 使用历史收益率自动生成观点
    console.print("  [cyan]生成投资者观点...[/cyan]")
    P, Q, omega = optimizer.create_default_views(mean_returns, confidence=0.5)

    # 打印观点信息
    console.print(f"  [dim]观点数量: {len(Q)}[/dim]")
    for i in range(len(Q)):
        view_assets = [data_fetcher.tickers[j] for j in range(len(P[i])) if P[i][j] != 0]
        view_weights = [P[i][j] for j in range(len(P[i])) if P[i][j] != 0]
        view_str = " + ".join([f"{w:+.0f}*{a}" for w, a in zip(view_weights, view_assets)])
        console.print(f"  [dim]观点{i+1}: {view_str} = {Q[i]:.4f}[/dim]")

    # 优化
    result = optimizer.optimize(P, Q, omega, weight_bounds=weight_bounds)
    return [result], optimizer, None


def run_risk_parity(cov_matrix, mean_returns, rf, weight_bounds):
    """运行风险平价优化"""
    optimizer = RiskParityOptimizer(cov_matrix, risk_free_rate=rf)
    result = optimizer.optimize(weight_bounds=weight_bounds, mean_returns=mean_returns)
    return [result], optimizer, None


def run_hrp(cov_matrix, corr_matrix, mean_returns, rf):
    """运行层次风险平价优化"""
    optimizer = HRPOptimizer(cov_matrix, corr_matrix, risk_free_rate=rf)
    result = optimizer.optimize(mean_returns=mean_returns)
    return [result], optimizer, None


def display_results(results, title="优化结果"):
    """在终端展示优化结果"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("优化方法", style="cyan", bold=True)
    table.add_column("预期收益率", justify="right")
    table.add_column("波动率", justify="right")
    table.add_column("夏普比率", justify="right")

    for result in results:
        ret_pct = result["expected_return"] * 100
        vol_pct = result["volatility"] * 100
        sharpe = result["sharpe_ratio"]

        # 夏普比率着色
        sharpe_str = f"{sharpe:.4f}"
        if sharpe > 1.0:
            sharpe_str = f"[green]{sharpe_str}[/green]"
        elif sharpe < 0:
            sharpe_str = f"[red]{sharpe_str}[/red]"

        table.add_row(
            result["method"],
            f"{ret_pct:.2f}%",
            f"{vol_pct:.2f}%",
            sharpe_str,
        )

    console.print(table)


def display_weights(results, tickers):
    """在终端展示权重分配"""
    table = Table(title="权重分配详情", box=box.ROUNDED)
    table.add_column("优化方法", style="cyan", bold=True)
    for t in tickers:
        table.add_column(t, justify="right")

    for result in results:
        row = [result["method"]]
        for t in tickers:
            w = result["weights"].get(t, 0)
            if w > 0.001:
                row.append(f"{w*100:.2f}%")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def main():
    """主函数"""
    args = parse_args()

    # 解析股票代码
    tickers = [t.strip() for t in args.tickers.split(",")]
    # 解析优化方法
    if args.method.lower() == "all":
        methods = ["markowitz", "bl", "risk_parity", "hrp"]
    else:
        methods = [m.strip().lower() for m in args.method.split(",")]

    # 打印标题
    console.print(Panel.fit(
        "[bold blue]投资组合优化大师[/bold blue]\n"
        f"[dim]多策略投资组合优化工具 v1.0.0[/dim]",
        border_style="blue",
    ))

    console.print(f"\n[bold]配置信息:[/bold]")
    console.print(f"  股票池:     {', '.join(tickers)}")
    console.print(f"  优化方法:   {', '.join(methods)}")
    console.print(f"  数据区间:   {args.start} ~ {args.end}")
    console.print(f"  无风险利率: {args.rf*100:.1f}%")
    console.print(f"  协方差方法: {args.cov_method}")
    console.print(f"  权重约束:   [{args.weight_lower}, {args.weight_upper}]")
    console.print()

    weight_bounds = (args.weight_lower, args.weight_upper)

    # 使用 rich 进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # 步骤1：获取数据
        task = progress.add_task("[cyan]获取A股历史数据...", total=None)
        data_fetcher = DataFetcher(tickers, args.start, args.end)
        try:
            data_fetcher.fetch_data()
        except Exception as e:
            console.print(f"\n[red]错误: 数据获取失败: {e}[/red]")
            sys.exit(1)

        prices = data_fetcher.get_prices()
        returns = data_fetcher.get_returns()
        mean_returns = data_fetcher.get_mean_returns(annualize=True)
        cov_matrix = data_fetcher.get_covariance_matrix(method=args.cov_method)
        corr_matrix = data_fetcher.get_correlation_matrix()
        data_summary = data_fetcher.get_summary()

        progress.update(task, completed=True, description=f"[green]数据获取完成 ({len(returns)} 个交易日)")

        # 打印数据摘要
        console.print("\n[bold]数据摘要:[/bold]")
        summary_table = Table(box=box.SIMPLE)
        summary_table.add_column("股票代码", style="cyan")
        summary_table.add_column("年化收益率", justify="right")
        summary_table.add_column("年化波动率", justify="right")
        summary_table.add_column("夏普比率", justify="right")
        for ticker, row in data_summary.iterrows():
            summary_table.add_row(
                ticker,
                f"{row['年化收益率']*100:.2f}%",
                f"{row['年化波动率']*100:.2f}%",
                f"{row['夏普比率(无风险利率=0)']:.4f}",
            )
        console.print(summary_table)
        console.print()

        # 步骤2：执行优化
        all_results = []
        markowitz_frontier = None
        markowitz_optimizer = None

        for method in methods:
            task = progress.add_task(f"[cyan]执行 {method} 优化...", total=None)

            try:
                if method == "markowitz":
                    results, opt, frontier = run_markowitz(
                        data_fetcher, mean_returns, cov_matrix, args.rf, weight_bounds
                    )
                    markowitz_frontier = frontier
                    markowitz_optimizer = opt

                elif method == "bl":
                    results, opt, _ = run_black_litterman(
                        data_fetcher, mean_returns, cov_matrix, args.rf, weight_bounds
                    )

                elif method == "risk_parity":
                    results, opt, _ = run_risk_parity(
                        cov_matrix, mean_returns, args.rf, weight_bounds
                    )

                elif method == "hrp":
                    results, opt, _ = run_hrp(
                        cov_matrix, corr_matrix, mean_returns, args.rf
                    )

                else:
                    console.print(f"  [yellow]未知方法: {method}，跳过[/yellow]")
                    progress.update(task, completed=True)
                    continue

                all_results.extend(results)
                progress.update(task, completed=True, description=f"[green]{method} 优化完成")

            except Exception as e:
                console.print(f"\n[red]{method} 优化失败: {e}[/red]")
                import traceback
                traceback.print_exc()
                progress.update(task, completed=True, description=f"[red]{method} 优化失败")

        # 步骤3：计算绩效指标
        console.print("\n[bold]优化结果对比:[/bold]")
        display_results(all_results)
        display_weights(all_results, tickers)

        # 计算各组合的绩效指标
        progress.add_task("[cyan]计算绩效指标...", total=None)
        metrics_data = {}
        cumulative_returns_dict = {}

        for result in all_results:
            try:
                pm = PortfolioMetrics(
                    returns, result["weights"],
                    risk_free_rate=args.rf,
                )
                metrics_data[result["method"]] = pm.get_all_metrics()
                cumulative_returns_dict[result["method"]] = pm.get_cumulative_returns()
            except Exception as e:
                console.print(f"  [yellow]绩效指标计算失败 ({result['method']}): {e}[/yellow]")

        metrics_table = pd.DataFrame(metrics_data).T if metrics_data else pd.DataFrame()

        # 步骤4：生成报告
        task = progress.add_task("[cyan]生成HTML报告...", total=None)

        report_gen = ReportGenerator()
        images_dict = {}

        # 有效前沿图
        if markowitz_frontier and markowitz_optimizer:
            try:
                frontier_returns = np.array([p["expected_return"] for p in markowitz_frontier])
                frontier_vols = np.array([p["volatility"] for p in markowitz_frontier])

                # 筛选有效前沿上的最优结果用于标注
                optimal_points = [r for r in all_results if "markowitz" in r.get("method", "").lower()
                                  or "夏普" in r.get("method", "")
                                  or "方差" in r.get("method", "")]

                asset_vols = np.sqrt(np.diag(cov_matrix.values))
                asset_vols_series = pd.Series(asset_vols, index=cov_matrix.columns)

                images_dict["efficient_frontier"] = report_gen.plot_efficient_frontier(
                    frontier_returns, frontier_vols,
                    optimal_points=optimal_points,
                    asset_returns=mean_returns,
                    asset_vols=asset_vols_series,
                )
            except Exception as e:
                console.print(f"  [yellow]有效前沿图生成失败: {e}[/yellow]")

        # 权重饼图
        try:
            images_dict["weights_pie"] = report_gen.plot_weights_pie(all_results)
        except Exception as e:
            console.print(f"  [yellow]权重饼图生成失败: {e}[/yellow]")

        # 风险贡献图
        try:
            images_dict["risk_contribution"] = report_gen.plot_risk_contribution(
                all_results, cov_matrix
            )
        except Exception as e:
            console.print(f"  [yellow]风险贡献图生成失败: {e}[/yellow]")

        # 相关性热力图
        try:
            images_dict["correlation"] = report_gen.plot_correlation_heatmap(corr_matrix)
        except Exception as e:
            console.print(f"  [yellow]相关性热力图生成失败: {e}[/yellow]")

        # 累积收益曲线
        try:
            if cumulative_returns_dict:
                images_dict["cumulative_returns"] = report_gen.plot_cumulative_returns(
                    cumulative_returns_dict
                )
        except Exception as e:
            console.print(f"  [yellow]累积收益图生成失败: {e}[/yellow]")

        # 生成 HTML
        html_content = report_gen.generate_html_report(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            results=all_results,
            data_summary=data_summary,
            images_dict=images_dict,
            metrics_table=metrics_table,
        )

        # 保存报告
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"portfolio_report_{timestamp}.html")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        progress.update(task, completed=True, description="[green]报告生成完成")

        # 完成
        console.print()
        console.print(Panel.fit(
            f"[bold green]优化完成！[/bold green]\n\n"
            f"报告已保存至: [blue]{os.path.abspath(report_path)}[/blue]\n"
            f"优化方法数量: {len(all_results)}\n"
            f"股票数量:     {len(tickers)}\n"
            f"交易日数:     {len(returns)}",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
