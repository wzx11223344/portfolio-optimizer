"""
投资组合优化大师 (Portfolio Optimizer Master)
=============================================
多策略投资组合优化工具包，集成四种高级优化算法：
  - 均值-方差优化 (Markowitz)
  - Black-Litterman 模型
  - 风险平价优化 (Risk Parity)
  - 层次风险平价 (Hierarchical Risk Parity, HRP)

所有数据均来自 akshare 真实A股行情数据，基于凸优化理论构建。
"""

__version__ = "1.0.0"
__author__ = "Portfolio Optimizer Master"

# 导入核心模块，方便外部调用
from .data import DataFetcher
from .markowitz import MarkowitzOptimizer
from .black_litterman import BlackLittermanOptimizer
from .risk_parity import RiskParityOptimizer
from .hrp import HRPOptimizer
from .metrics import PortfolioMetrics
from .report import ReportGenerator

__all__ = [
    "DataFetcher",
    "MarkowitzOptimizer",
    "BlackLittermanOptimizer",
    "RiskParityOptimizer",
    "HRPOptimizer",
    "PortfolioMetrics",
    "ReportGenerator",
]
