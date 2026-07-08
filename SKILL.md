---
slug: portfolio-optimizer
displayName: 投资组合优化大师
version: 1.0.0
summary: 多策略投资组合优化工具，集成均值-方差、Black-Litterman、风险平价、层次风险平价(HRP)四种高级优化算法，基于凸优化理论，使用真实A股数据。
tags:
  - finance
  - portfolio-optimization
  - quantitative-finance
  - convex-optimization
  - risk-management
license: MIT
---

# 投资组合优化大师

## 概述

投资组合优化大师是一个基于 Python 的多策略投资组合优化工具，集成了四种业界主流的资产配置优化算法。所有数据均来自 akshare 真实A股行情接口，基于凸优化理论（cvxpy）构建，适用于学术研究、量化投资学习和投资决策辅助。

## 核心能力

### 1. 均值-方差优化 (Markowitz)
- **最大夏普比率组合**：通过变量替换将非凸优化转化为二次规划问题，使用 cvxpy 求解
- **最小方差组合**：在约束条件下最小化组合方差
- **有效前沿计算**：扫描不同目标收益率水平，绘制完整的有效前沿曲线
- **权重约束**：支持上下限约束和做空限制

### 2. Black-Litterman 模型
- **市场均衡收益**：通过逆优化从市场权重反解隐含均衡收益
- **投资者观点**：支持绝对观点和相对观点，自动从历史数据生成默认观点
- **后验收益**：贝叶斯框架融合先验与观点，计算后验期望收益
- **Omega 自动计算**：基于 He-Litterman 启发式方法自动确定观点不确定性

### 3. 风险平价优化 (Risk Parity)
- **等风险贡献**：使各资产对组合总风险的贡献相等
- **边际风险贡献**：精确计算各资产的边际风险贡献和风险贡献占比
- **SLSQP 求解**：使用 scipy.optimize 的 SLSQP 方法迭代求解

### 4. 层次风险平价 (HRP)
- **相关性距离矩阵**：基于相关系数构建距离度量
- **层次聚类**：使用 Ward 连接法对资产进行自动分组
- **准对角化**：按聚类顺序重排协方差矩阵
- **递归二分法**：自顶向下递归分配权重，无需矩阵求逆

## 绩效指标

支持以下完整绩效指标计算：
- 年化收益率、年化波动率
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Max Drawdown)
- Calmar 比率
- Sortino 比率（下行风险调整）
- 信息比率 (Information Ratio)
- VaR (95%)、CVaR (95%)

## 能力边界说明

### 适用场景
- A股市场的多资产投资组合优化
- 学术研究中的资产配置算法对比
- 量化投资策略的权重分配
- 风险管理和绩效归因分析

### 不适用场景
- **非A股市场**：当前仅支持 akshare 的 A 股数据接口
- **高频交易**：数据粒度为日频，不适用于日内或高频策略
- **衍生品组合**：不支持期权、期货等衍生品的优化
- **实时交易**：本工具用于离线分析和研究，不具备实时交易功能
- **投资建议**：本工具的输出仅供研究参考，不构成任何投资建议

### 已知限制
- akshare 接口可能因网络或数据源问题导致获取失败
- 历史数据不能保证未来表现，优化结果存在过拟合风险
- 协方差矩阵估计在样本量不足时可能不稳定（建议使用 Ledoit-Wolf 收缩估计）
- Black-Litterman 模型的观点质量直接影响优化结果

## FAQ

### Q1: 如何安装依赖？
```bash
pip install -r requirements.txt
```
主要依赖包括 akshare（数据）、cvxpy（凸优化）、scipy（科学计算）、scikit-learn（Ledoit-Wolf）、matplotlib（可视化）、rich（终端美化）。

### Q2: 支持哪些股票代码格式？
支持标准的6位A股代码，如沪市的 `600519`（贵州茅台）、深市的 `000858`（五粮液）、 `601318`（中国平安）等。不需要加前缀（如 sh/sz）。

### Q3: 协方差矩阵的 Ledoit-Wolf 收缩估计有什么优势？
当资产数量接近或超过样本数量时，样本协方差矩阵会变得不稳定（奇异）。Ledoit-Wolf 方法将样本协方差向结构化目标（如单位矩阵）收缩，显著降低估计误差。可通过 `--cov-method ledoit_wolf` 启用。

### Q4: Black-Litterman 模型的观点如何设定？
工具默认根据历史收益率排名自动生成相对观点（排名靠前的资产优于排名靠后的资产）。高级用户可以通过修改 `black_litterman.py` 中的 `create_default_views` 方法或直接构造 P、Q、Omega 矩阵来自定义观点。

### Q5: 四种优化方法各有什么特点和适用场景？
- **Markowitz**：经典方法，适合对收益有明确预期的场景，但对输入参数敏感
- **Black-Litterman**：融合市场均衡与主观观点，适合有明确投资观点的机构投资者
- **风险平价**：关注风险分配而非收益预测，适合追求稳健的风险均衡策略
- **HRP**：无需矩阵求逆，对参数估计误差鲁棒，适合资产数量较多或相关性较高时

### Q6: 如何选择数据区间？
建议使用至少1年（244个交易日）以上的历史数据以保证统计可靠性。过短的数据区间会导致协方差矩阵估计不准确。可通过 `--start` 和 `--end` 参数指定。

### Q7: 优化结果中权重出现极小值怎么办？
这可能是由于某些资产的风险收益特征不佳，优化器自然降低了其权重。可以通过设置权重下限 `--weight-lower 0.05`（如最低5%）来避免过度集中。

### Q8: 报告中的有效前沿为什么有时不完整？
有效前沿的完整性取决于约束条件和数据的可行性。当目标收益率过高（超出可达范围）时，优化问题可能不可行，对应的有效前沿点会被跳过。

## 输出示例

### 终端输出
```
╭─────────────────────────────────────────╮
│         投资组合优化大师                  │
│   多策略投资组合优化工具 v1.0.0           │
╰─────────────────────────────────────────╯

配置信息:
  股票池:     600519, 000858, 601318
  优化方法:   markowitz, bl, risk_parity, hrp
  数据区间:   20240101 ~ 20250101
  无风险利率: 3.0%

    优化结果对比
┌──────────────────┬──────────┬─────────┬──────────┐
│ 优化方法          │ 预期收益率 │ 波动率   │ 夏普比率  │
├──────────────────┼──────────┼─────────┼──────────┤
│ 最大夏普比率组合   │  18.52%  │ 22.31%  │  0.6957  │
│ 最小方差组合       │  12.34%  │ 18.45%  │  0.5049  │
│ Black-Litterman组合│  15.67%  │ 20.12%  │  0.6297  │
│ 风险平价组合       │  10.23%  │ 19.87%  │  0.3639  │
│ 层次风险平价组合    │  11.45%  │ 19.56%  │  0.4321  │
└──────────────────┴──────────┴─────────┴──────────┘
```

### HTML 报告
报告包含以下内容：
1. 数据摘要表（各资产年化收益率、波动率、夏普比率）
2. 优化结果对比表
3. 各方法权重分配表
4. 组合绩效指标对比表
5. 有效前沿图（标注最优组合点）
6. 权重分配饼图
7. 风险贡献条形图
8. 资产相关系数热力图
9. 累积收益曲线对比图

## 安装使用说明

### 安装
```bash
cd portfolio-optimizer
pip install -r requirements.txt
```

### 基本用法
```bash
# 运行所有优化方法
python optimize.py --tickers 600519,000858,601318 --method all --start 20240101 --end 20250101

# 仅运行均值-方差优化
python optimize.py --tickers 600519,000858,601318 --method markowitz

# 运行风险平价和HRP
python optimize.py --tickers 600519,000858,601318,000001,600036 --method risk_parity,hrp

# 使用Ledoit-Wolf协方差估计
python optimize.py --tickers 600519,000858,601318 --method all --cov-method ledoit_wolf

# 设置权重约束（每只股票至少5%，最多50%）
python optimize.py --tickers 600519,000858,601318 --method markowitz --weight-lower 0.05 --weight-upper 0.5
```

### 编程接口
```python
from portfolio_optimizer import DataFetcher, MarkowitzOptimizer, RiskParityOptimizer

# 获取数据
fetcher = DataFetcher(['600519', '000858', '601318'], '20240101', '20250101')
fetcher.fetch_data()
mean_returns = fetcher.get_mean_returns()
cov_matrix = fetcher.get_covariance_matrix()

# 均值-方差优化
markowitz = MarkowitzOptimizer(mean_returns, cov_matrix, risk_free_rate=0.03)
max_sharpe = markowitz.max_sharpe()
print(max_sharpe['weights'])

# 风险平价优化
rp = RiskParityOptimizer(cov_matrix, risk_free_rate=0.03)
rp_result = rp.optimize(mean_returns=mean_returns)
print(rp_result['weights'])
```
