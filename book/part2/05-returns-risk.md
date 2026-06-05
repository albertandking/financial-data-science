# 第5章 收益率与风险度量

!!! info "配套代码"
    本章代码见 `notebooks/ch05_returns_risk.ipynb`，依赖内置数据，离线可跑。
    大量复用 `fds.metrics` 中的函数。

## 5.1 学习目标

- 区分简单收益率与对数收益率，理解各自的可加性
- 计算年化收益、年化波动率、夏普比率、最大回撤
- 理解并计算风险价值 VaR 与期望损失 ES

## 5.2 两种收益率

设 $P_t$ 为第 $t$ 期价格。

**简单收益率**：

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

**对数收益率**：

$$\ell_t = \ln\frac{P_t}{P_{t-1}} = \ln P_t - \ln P_{t-1}$$

| 性质 | 简单收益率 | 对数收益率 |
|---|---|---|
| 时间可加（多期复合） | ✗（需连乘） | ✓（可直接相加） |
| 截面可加（组合加权） | ✓（$r_p=\sum w_i r_i$） | ✗ |
| 适用 | 组合收益、与实务一致 | 计量建模、统计性质好 |

!!! tip "怎么选"
    算**组合收益**用简单收益率；做**时间序列建模**用对数收益率。两者在小幅变动时近似相等。

```python
from fds import daily_returns
simple = daily_returns(prices)            # 简单收益率
log_ret = daily_returns(prices, log=True) # 对数收益率
```

## 5.3 年化

把日度统计量换算到年度（A 股约定一年 252 个交易日）：

$$r_{\text{年}} = (1+\bar r)^{252} - 1, \qquad \sigma_{\text{年}} = \sigma_{\text{日}}\sqrt{252}$$

波动率按 $\sqrt{T}$ 缩放，源于独立同分布下方差随时间线性增长。

```python
from fds import annualized_return, annualized_volatility
annualized_return(rets["BANK"]), annualized_volatility(rets["BANK"])
```

## 5.4 风险调整收益：夏普比率

单看收益不够，要看"每承担一单位风险换来多少超额收益"：

$$\text{Sharpe} = \frac{\bar r - r_f}{\sigma} \times \sqrt{252}$$

其中 $r_f$ 为无风险利率。夏普越高，性价比越好。

```python
from fds import sharpe_ratio
sharpe_ratio(rets["LIQUOR"], risk_free=0.02)   # 年化无风险利率 2%
```

!!! note "索提诺比率"
    夏普用总波动作分母，但投资者其实只厌恶**下行**波动。索提诺比率把分母换成下行标准差，
    对"只在赚钱方向波动大"的资产更友好。

## 5.5 回撤与最大回撤

**回撤**是净值从历史高点回落的幅度；**最大回撤（MDD）**是最惨的一次：

$$\text{MDD} = \min_t \left(\frac{V_t}{\max_{s\le t} V_s} - 1\right)$$

它衡量"最坏情况下要忍受多大亏损"，是实务中比波动率更贴近体感的风险指标。

```python
from fds import max_drawdown
max_drawdown(rets["TECH"])     # 返回负值，如 -0.35 表示最大回撤 35%
```

## 5.6 风险价值 VaR 与期望损失 ES

**VaR（Value at Risk）**：在给定置信水平 $\alpha$ 下，未来一期可能的最大损失。
例如"95% 单日 VaR 为 3%"指：有 95% 的把握单日亏损不超过 3%。

- **历史模拟法**：取历史收益分布的 $(1-\alpha)$ 分位数。
- **参数法**：假设正态，$\text{VaR}_\alpha = -(\mu + z_\alpha \sigma)$。

```python
import numpy as np
var95 = -np.quantile(rets["TECH"], 0.05)      # 历史模拟法 95% 单日 VaR
```

**ES（Expected Shortfall，期望损失）**：超过 VaR 那部分尾部损失的**平均值**，
回答"一旦突破 VaR，平均会亏多少"。

$$\text{ES}_\alpha = -\mathbb{E}[\,r \mid r \le -\text{VaR}_\alpha\,]$$

!!! warning "VaR 的局限"
    VaR 只给出"门槛"，不反映突破后有多惨；且不满足次可加性（不利于分散化论证）。
    巴塞尔协议已转向用 ES。**厚尾**下，正态参数法会严重低估 VaR/ES。

## 5.7 本章小结

- 组合收益用简单收益率，建模用对数收益率
- 年化：收益几何复合、波动率乘 $\sqrt{252}$
- 夏普 = 超额收益 / 波动；只看下行用索提诺
- 最大回撤刻画"最坏体感"；VaR 给门槛、ES 给尾部均值，厚尾下都要小心低估

## 5.8 练习

1. 对四只股票计算年化收益、年化波动、夏普比率，做成一张对比表并排序。
2. 用历史模拟法计算 `TECH` 的 95% 与 99% 单日 VaR 和对应 ES。
3. 比较正态参数法与历史模拟法的 VaR 差异，结合上一章的厚尾结论解释。
