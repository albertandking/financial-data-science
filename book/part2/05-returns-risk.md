# 第5章 收益率与风险度量

!!! info "配套代码"
    `notebooks/ch05_returns_risk.ipynb`

## 5.1 学习目标

- 区分简单收益率与对数收益率及其可加性
- 计算波动率、夏普比率、最大回撤
- 理解 VaR 与 ES（期望损失）

## 5.2 内容大纲

1. 简单收益 vs 对数收益：时间可加 vs 截面可加
2. 年化：$r_{ann} = (1+\bar r)^{252}-1$，$\sigma_{ann}=\sigma\sqrt{252}$
3. 风险调整收益：夏普比率、索提诺比率
4. 回撤与最大回撤
5. 风险价值 VaR 与期望损失 ES（历史法、参数法）

本章大量复用 `fds.metrics` 中的函数：

```python
from fds import daily_returns, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
```

## 5.3 练习

1. 对内置四只股票计算夏普比率并排序，解释结果。
2. 用历史模拟法计算某股票 95% 单日 VaR。
