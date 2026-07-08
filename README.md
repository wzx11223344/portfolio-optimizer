# 投资组合优化大师 (Portfolio Optimizer Master)

多策略投资组合优化工具，集成均值-方差、Black-Litterman、风险平价、层次风险平价(HRP)四种高级优化算法，基于凸优化理论，使用真实A股数据。

## 功能特性

- **均值-方差优化 (Markowitz)**：最大夏普比率、最小方差、有效前沿，使用 cvxpy 凸优化
- **Black-Litterman 模型**：融合市场均衡收益与投资者观点，贝叶斯框架后验估计
- **风险平价优化 (Risk Parity)**：等风险贡献，使用 scipy.optimize SLSQP 求解
- **层次风险平价 (HRP)**：层次聚类 + 递归二分法，无需矩阵求逆，对参数估计鲁棒
- **完整绩效指标**：夏普比率、最大回撤、Calmar、Sortino、信息比率、VaR、CVaR
- **HTML 可视化报告**：有效前沿图、权重饼图、风险贡献条形图、相关性热力图、累积收益曲线
- **真实数据**：所有数据来自 akshare A股历史日线接口，无任何随机/伪造数据

## 项目结构

```
portfolio-optimizer/
├── optimize.py                    # CLI入口
├── portfolio_optimizer/
│   ├── __init__.py                # 包初始化
│   ├── data.py                    # 数据获取（akshare真实数据）
│   ├── markowitz.py               # 均值-方差优化（cvxpy凸优化）
│   ├── black_litterman.py         # Black-Litterman模型
│   ├── risk_parity.py             # 风险平价优化
│   ├── hrp.py                     # 层次风险平价（HRP）
│   ├── metrics.py                 # 组合绩效指标
│   └── report.py                  # HTML报告生成（matplotlib图表）
├── output/                        # 报告输出目录
├── SKILL.md                       # 技能描述文件
├── README.md                      # 本文件
└── requirements.txt               # Python依赖
```

## 安装

### 环境要求
- Python 3.8+
- 网络连接（用于获取 akshare 数据）

### 安装依赖

```bash
cd portfolio-optimizer
pip install -r requirements.txt
```

依赖包列表：
| 包名 | 用途 |
|------|------|
| akshare | A股历史数据获取 |
| cvxpy | 凸优化求解 |
| numpy | 数值计算 |
| pandas | 数据处理 |
| scipy | 科学计算、层次聚类、优化求解 |
| scikit-learn | Ledoit-Wolf 协方差收缩估计 |
| matplotlib | 图表生成 |
| rich | 终端美化输出 |

## 使用方法

### 命令行

```bash
# 运行所有优化方法（默认）
python optimize.py --tickers 600519,000858,601318 --method all --start 20240101 --end 20250101

# 仅运行均值-方差优化
python optimize.py --tickers 600519,000858,601318 --method markowitz

# 运行多种方法
python optimize.py --tickers 600519,000858,601318 --method markowitz,risk_parity

# 使用 Ledoit-Wolf 协方差估计
python optimize.py --tickers 600519,000858,601318 --method all --cov-method ledoit_wolf

# 设置权重约束
python optimize.py --tickers 600519,000858,601318 --method markowitz --weight-lower 0.05 --weight-upper 0.5

# 自定义无风险利率
python optimize.py --tickers 600519,000858,601318 --method all --rf 0.025
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tickers` | 股票代码列表（逗号分隔） | 必填 |
| `--method` | 优化方法：markowitz, bl, risk_parity, hrp, all | all |
| `--start` | 开始日期 YYYYMMDD | 20240101 |
| `--end` | 结束日期 YYYYMMDD | 20250101 |
| `--rf` | 无风险利率（年化） | 0.03 |
| `--output` | 报告输出目录 | output |
| `--cov-method` | 协方差估计方法：sample, ledoit_wolf | sample |
| `--weight-lower` | 权重下限 | 0.0 |
| `--weight-upper` | 权重上限 | 1.0 |

### 编程接口

```python
from portfolio_optimizer import DataFetcher, MarkowitzOptimizer, RiskParityOptimizer, HRPOptimizer

# 1. 获取数据
fetcher = DataFetcher(['600519', '000858', '601318'], '20240101', '20250101')
fetcher.fetch_data()
mean_returns = fetcher.get_mean_returns()
cov_matrix = fetcher.get_covariance_matrix(method='ledoit_wolf')

# 2. 均值-方差优化
markowitz = MarkowitzOptimizer(mean_returns, cov_matrix, risk_free_rate=0.03)
max_sharpe_result = markowitz.max_sharpe()
min_var_result = markowitz.min_variance()
frontier = markowitz.efficient_frontier(n_points=50)

print("最大夏普比率组合:")
print(f"  收益率: {max_sharpe_result['expected_return']:.4f}")
print(f"  波动率: {max_sharpe_result['volatility']:.4f}")
print(f"  夏普比率: {max_sharpe_result['sharpe_ratio']:.4f}")
print(f"  权重: {max_sharpe_result['weights']}")

# 3. 风险平价优化
rp = RiskParityOptimizer(cov_matrix, risk_free_rate=0.03)
rp_result = rp.optimize(mean_returns=mean_returns)

# 4. HRP 优化
corr_matrix = fetcher.get_correlation_matrix()
hrp = HRPOptimizer(cov_matrix, corr_matrix, risk_free_rate=0.03)
hrp_result = hrp.optimize(mean_returns=mean_returns)
```

## 优化算法说明

### 1. 均值-方差优化 (Markowitz)

基于 Markowitz (1952) 现代投资组合理论，在给定的期望收益和风险之间寻找最优权衡。

- **最大夏普比率**：通过变量替换 `y = k*w` 将非凸的分式规划转化为凸的二次规划
- **最小方差**：直接最小化 `w^T Σ w`
- **有效前沿**：扫描不同目标收益率，最小化方差

### 2. Black-Litterman 模型

解决纯历史均值估计不稳定的问题，将市场均衡收益（先验）与投资者观点结合。

```
先验: Π = δ * Σ * w_market
后验: E[R] = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P^T Ω^{-1} Q]
```

### 3. 风险平价 (Risk Parity)

使各资产对组合总风险的贡献相等。边际风险贡献 `MRC_i = (Σw)_i / σ_p`，风险贡献 `RC_i = w_i * MRC_i`。

### 4. 层次风险平价 (HRP)

Marcos Lopez de Prado (2016) 提出，通过层次聚类和递归二分法分配权重，无需估计协方差矩阵的逆。

## 输出说明

运行后将在 `output/` 目录生成 HTML 报告，包含：

1. **数据摘要**：各资产的年化收益率、波动率、夏普比率
2. **优化结果对比**：各方法的预期收益、波动率、夏普比率
3. **权重分配表**：各方法在各资产上的权重分配
4. **绩效指标对比**：年化收益、波动率、夏普、最大回撤、Calmar、Sortino、VaR、CVaR
5. **有效前沿图**：标注最优组合点
6. **权重饼图**：直观展示各方法的权重分配
7. **风险贡献条形图**：各资产的风险贡献占比
8. **相关性热力图**：资产间相关系数矩阵
9. **累积收益曲线**：各组合的累积收益对比

## 常见问题

**Q: 获取数据失败怎么办？**
A: 检查网络连接，确认股票代码正确（6位数字，如 600519），确认日期范围合理。

**Q: 优化求解失败怎么办？**
A: 尝试放宽权重约束，或使用 Ledoit-Wolf 协方差估计（`--cov-method ledoit_wolf`）。

**Q: 数据区间选择多长合适？**
A: 建议至少1年（244个交易日）以上，以保证统计可靠性。

## 免责声明

本工具仅供学术研究和学习参考，不构成任何投资建议。投资有风险，入市需谨慎。历史数据不能保证未来表现，优化结果存在过拟合风险。

## License

MIT
