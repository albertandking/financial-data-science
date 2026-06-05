# 第15章 量化策略与回测

!!! info "配套代码"
    `notebooks/ch15_backtesting.ipynb`

## 15.1 本章导读

“策略在纸面上赚了 300%”——这句话在量化圈早已是笑谈，因为每一位入门者都会经历的第一课，就是发现精心构建的回测结果在实盘中化为乌有。回测（Backtesting）是量化投资的核心环节：用历史数据“模拟”交易，检验策略是否真的有效。然而，正是这种“模拟”的性质，埋藏了大量可以让任何策略看起来完美的陷阱。

本章从信号生成开始，构建完整的向量化回测框架，引入A股真实交易成本，计算一套完整的绩效指标，并系统剖析前视偏差、幸存者偏差、过拟合等常见陷阱。最后通过参数稳健性分析帮助读者区分“有效策略”与“数据拟合”。

## 15.2 学习目标

读完本章，读者应能够：

1. 理解回测的意义与根本局限，区分向量化回测与事件驱动回测；
2. 用 `position = signal.shift(1)` 正确实现无前视偏差的向量化回测；
3. 计算A股双边交易成本，分析换手率与成本的关系；
4. 计算年化收益、夏普比率、最大回撤、卡尔玛比率等完整绩效指标；
5. 识别并避免前视偏差、幸存者偏差、数据窥探等回测陷阱；
6. 通过参数扫描和样本内/样本外检验评估策略稳健性。

---

## 15.3 回测的意义与局限

### 15.3.1 为什么需要回测

量化策略在上线前，无法进行大规模真实测试。回测提供了一个近似的“历史模拟器”，帮助回答以下问题：

- 策略在历史上是否具有统计显著的超额收益？
- 最坏情形（最大回撤）有多深？投资者心理上能否承受？
- 在哪些市场状态（牛市/熊市/震荡市）下策略失效？
- 加入交易成本后，策略是否依然盈利？

### 15.3.2 回测的根本局限

!!! warning "回测不是预测未来"
    回测只能告诉你策略**在历史上**表现如何，无法保证未来有效。以下几点尤为关键：

    1. **历史不重演**：市场结构、监管规则、参与者构成随时间变化；
    2. **自我消解**：一旦策略广为人知，超额收益往往消失；
    3. **模拟成本≠真实成本**：冲击成本、流动性限制在回测中难以精确建模；
    4. **数据陷阱**：任何历史数据都包含幸存者偏差、前视偏差等系统性问题。

### 15.3.3 向量化回测 vs 事件驱动回测

| 维度 | 向量化回测 | 事件驱动回测 |
|------|-----------|------------|
| 实现方式 | Pandas/NumPy矩阵运算 | 模拟逐笔报单/成交事件 |
| 速度 | 极快（秒级） | 慢（分钟~小时） |
| 建模精度 | 较低（假设理想成交） | 较高（可模拟部分成交/滑点） |
| 适用场景 | 日度策略快速筛选 | 高频/精细化策略验证 |
| 典型库 | pandas / vectorbt | zipline / backtrader |

本章重点介绍**向量化回测**。它运算效率高，适合快速筛选信号和参数扫描，是量化研究员的日常主力工具。

---

## 15.4 信号→持仓→收益：回测流程

### 15.4.1 三步核心流程

```
原始数据（价格/因子）
    ↓
信号生成（Signal）    ← t 日收盘后根据公开信息计算
    ↓
持仓决策（Position）  ← t+1 日开盘按信号执行（T+1规则）
    ↓
策略收益（Return）    ← t+1 日的收益 × t 日的持仓
```

### 15.4.2 T+1 规则与前视偏差

在A股，**当日买入的股票不能当日卖出**（T+1交割规则）。因此，即使信号在收盘后立即生成，也只能在**次日**成交。这在向量化回测中通过一行代码体现：

```python
position = signal.shift(1)   # 今日持仓 = 昨日信号
```

这行代码是整个回测框架中**最重要的一行**。若遗漏 `shift(1)`，则相当于当日收盘价信号当日成交，产生严重的**前视偏差（Look-Ahead Bias）**。

!!! warning "最常见的前视偏差"
    以下代码片段包含前视偏差：
    ```python
    # 错误：用当日收盘价信号当日持仓
    signal = (prices > prices.rolling(20).mean()).astype(int)
    strategy_return = signal * returns   # ← 错误！
    
    # 正确：信号在t日生成，t+1日才能成交
    position = signal.shift(1)
    strategy_return = position * returns   # ← 正确
    ```
    前视偏差会让回测净值“凭空”好看 20%~50%，是初学者最常犯的错误。

### 15.4.3 持仓信号的设计

常见信号类型：

| 信号值 | 含义 |
|--------|------|
| `+1`   | 做多（买入并持有） |
| `0`    | 空仓（不持有） |
| `-1`   | 做空（A股受限，一般设为0） |

对于A股普通股票，因为融券受限，通常只有多头（+1）和空仓（0）两种状态。

---

## 15.5 向量化回测实现

### 15.5.1 动量信号

**动量效应**（Momentum）是金融学中最持久的实证发现之一：近期涨幅较大的资产，未来短期内往往继续上涨。以20日动量为例：

$$\text{Signal}_t = \begin{cases} 1 & \text{若 } R_{t-20,t} > 0 \\ 0 & \text{其他} \end{cases}$$

其中 $R_{t-20,t}$ 为过去20个交易日的累计收益率。

在代码中：

```python
# 计算20日动量（过去20日收益）
momentum = prices.pct_change(20)

# 动量为正则做多，否则空仓
signal = (momentum > 0).astype(int)

# T+1执行：今日持仓 = 昨日信号
position = signal.shift(1)

# 策略日收益 = 持仓 × 标的日收益
returns = prices.pct_change()
strategy_return = (position * returns).dropna()
```

### 15.5.2 净值计算

从日度收益序列到净值曲线：

```python
# 起始净值为 1
nav = (1 + strategy_return).cumprod()

# 也可用对数累加（等价，误差极小）
# nav = np.exp(np.log1p(strategy_return).cumsum())
```

净值曲线（NAV, Net Asset Value）是策略表现最直观的展示，反映了从初始 1 元出发的累计复合增长。

---

## 15.6 交易成本与滑点

### 15.6.1 A股双边成本构成

A股真实交易中，每笔交易都会产生以下费用：

| 费用类型 | 方向 | 典型费率 | 说明 |
|---------|------|---------|------|
| 佣金（券商手续费） | 买卖双向 | ~0.025% | 互联网券商已低至 0.015%，部分有最低5元 |
| 印花税 | **仅卖出** | 0.05%（2023年调降） | 2023年8月降至0.05%，此前为0.1% |
| 过户费 | 买卖双向 | 0.001% | 上交所收取，深交所近年已取消 |
| 买入冲击成本/滑点 | 买卖双向 | 0.02%~0.1% | 大单成交时实际价格偏离报价 |

!!! note "简化计算"
    教学与研究中常用的简化假设：
    
    - 佣金：买卖各 0.025%
    - 印花税（仅卖出）：0.05%
    - 滑点（买卖各半）：0.05%
    
    综合**双边成本**（每完成一次完整的买入+卖出）约为：
    $$C_{双边} \approx 2 \times 0.025\% + 0.05\% + 2 \times 0.05\% = 0.2\%$$

### 15.6.2 换手率与成本的关系

**换手率**衡量持仓的变动频率：

$$\text{换手率}_{日均} = \frac{1}{T}\sum_{t=1}^{T}|\text{position}_t - \text{position}_{t-1}|$$

$$\text{年化换手率} = \text{换手率}_{日均} \times 252$$

交易成本对策略的侵蚀：

$$\text{年化成本} = \text{年化换手率} \times C_{单边}$$

以一个每月调仓一次（年化换手率约 24×）的策略为例：

$$\text{年化成本} \approx 24 \times 0.1\% = 2.4\%$$

这意味着策略每年必须在基准之上超额 2.4% 才能弥补成本。对于年化收益 8% 的策略来说，这并非小数。

!!! warning "高频换手率是成本杀手"
    日频调仓的策略年化换手率可能高达 200~500×，对应年化成本 20%~50%。
    绝大多数“看似盈利”的高频信号在加入成本后变为亏损。

### 15.6.3 在回测中加入成本

```python
# 计算每日换手率
turnover = position.diff().abs()

# 每日成本（单边成本率）
cost_per_trade = 0.001   # 0.1% 单边
daily_cost = turnover * cost_per_trade

# 扣除成本后的净收益
net_return = strategy_return - daily_cost
```

---

## 15.7 绩效评估体系

完整的绩效评估需要多维度指标，单一指标（如收益率）容易被操纵或误解。

### 15.7.1 收益类指标

**年化收益率**（几何复合）：

$$\bar{r}_{年化} = \left(\prod_{t=1}^{T}(1+r_t)\right)^{252/T} - 1$$

**超额收益（Alpha）**：相对于基准（如沪深300）的超额年化收益。

### 15.7.2 风险调整类指标

**夏普比率**（Sharpe Ratio）：每单位波动所获得的超额收益：

$$\text{Sharpe} = \frac{\bar{r} - r_f}{\sigma} \times \sqrt{252}$$

其中 $r_f$ 为年化无风险利率（常用货币市场利率，约 1.5%~2%）。

**卡尔玛比率**（Calmar Ratio）：年化收益与最大回撤之比：

$$\text{Calmar} = \frac{\bar{r}_{年化}}{|\text{MDD}|}$$

卡尔玛比率特别适合A股，因为A股市场波动剧烈，回撤管理至关重要。

### 15.7.3 回撤类指标

**最大回撤**（Maximum Drawdown, MDD）：

$$\text{MDD} = \min_{t \in [0,T]} \left(\frac{\text{NAV}_t}{\max_{\tau \leq t} \text{NAV}_\tau} - 1\right)$$

最大回撤反映了策略在最坏情形下的亏损幅度，是投资者心理承受力的关键参考。

**水下曲线**（Underwater Chart）：

$$\text{Drawdown}_t = \frac{\text{NAV}_t}{\max_{\tau \leq t} \text{NAV}_\tau} - 1$$

### 15.7.4 交易活跃度指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 胜率 | 正收益日 / 总交易日 | 日胜率通常约 50%~55%，均值回复策略可能更高 |
| 盈亏比 | 平均盈利 / 平均亏损绝对值 | 趋势跟踪通常盈亏比高，胜率低 |
| 年化换手率 | 日均换手 × 252 | 反映成本消耗 |
| 成本占收益比 | 年化成本 / 年化收益 | 反映成本对策略的侵蚀程度 |

### 15.7.5 完整绩效报告示例

```python
def performance_report(returns, risk_free=0.02, cost_per_trade=0.001,
                        position=None):
    """生成完整绩效报告。"""
    from fds import annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
    
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    sharpe  = sharpe_ratio(returns, risk_free=risk_free)
    mdd     = max_drawdown(returns)
    calmar  = ann_ret / abs(mdd) if mdd != 0 else np.nan
    win_rate = (returns > 0).mean()
    
    report = {
        '年化收益': f'{ann_ret:.2%}',
        '年化波动': f'{ann_vol:.2%}',
        '夏普比率': f'{sharpe:.3f}',
        '最大回撤': f'{mdd:.2%}',
        '卡尔玛比率': f'{calmar:.3f}',
        '胜率': f'{win_rate:.2%}',
    }
    
    if position is not None:
        turnover_daily = position.diff().abs().mean()
        turnover_annual = turnover_daily * 252
        annual_cost = turnover_annual * cost_per_trade
        report['年化换手率'] = f'{turnover_annual:.1f}×'
        report['年化成本'] = f'{annual_cost:.2%}'
        report['成本占收益比'] = f'{annual_cost / ann_ret:.1%}'
    
    return pd.Series(report)
```

---

## 15.8 回测陷阱详解

!!! danger "回测陷阱是系统性错误，不是小疏漏"
    以下任一陷阱都可能将真实无效的策略包装成“完美”的回测结果。
    在发表结果或实盘前，必须逐一排查。

### 15.8.1 前视偏差（Look-Ahead Bias）

**定义**：在信号计算中使用了信号生成时刻尚不可知的未来信息。

**常见来源**：
1. 忘记 `shift(1)`：用 t 日信号决定 t 日持仓；
2. 用 `rolling().mean()` 的默认参数时，窗口结束点就是当前时间点，本身没问题，但若用 `centered=True` 则包含未来数据；
3. 使用“已知”的财务数据，但实际上该数据在披露日之前不可获得；
4. 用未来的复权因子对历史价格复权（前复权数据中存在这一问题）。

!!! warning "前复权数据的陷阱"
    前复权（Forward-adjusted）价格在每次除权时都会追溯调整历史所有价格。
    这意味着你在2015年“看到”的价格实际上是用2023年的复权因子算出来的——
    这是一种微妙但严重的前视偏差。
    **建议**：用后复权价格计算收益率，避免前复权数据。

**量化影响**：前视偏差通常让年化超额收益虚高 10%~30%。

### 15.8.2 幸存者偏差（Survivorship Bias）

**定义**：数据集中只包含“幸存”到今天的股票，剔除了退市、ST、破产的公司。

**影响**：幸存的股票天然是“成功”的股票，在它们上测试的策略会系统性高估收益。

研究表明，忽略幸存者偏差会让年化收益高估约 2%~5%（取决于市场和时间段）。

!!! note "A股特殊背景"
    A股退市制度历来较宽松，但近年趋严。量化研究时应使用包含退市股的完整数据集。
    Wind、聚宽等数据供应商提供“全量股票”数据，包含历史退市标的。

### 15.8.3 过拟合（Overfitting）与数据窥探（Data Snooping）

**过拟合**：模型参数在样本内拟合了数据的噪声，样本外表现大幅下降。

**数据窥探**：研究者反复尝试不同参数/信号/组合，直到找到“显著”结果，但这些结果只是多重检验的统计产物。

!!! warning "p-Hacking 与多重检验"
    假设一个参数有 100 种选择，每种在随机数据上有 5% 概率“显著”（p<0.05）。
    如果不进行多重检验校正，预期有 5 种参数会“显著”——即使真实信号为零。
    
    **Bonferroni 校正**：若进行 $n$ 次检验，单次检验的显著性水平应调整为 $\alpha/n$。
    
    Bailey & López de Prado（2016）提出的**回测过拟合概率**（Probability of Backtest Overfitting, PBO）
    更为精细，见本章拓展阅读。

**检验标准**：
- 最低夏普比率（Minimum Track Record Length）：样本外夏普需达到某一阈值才可信；
- 参数敏感性：策略表现应对参数选择不敏感（见 15.9 节）。

### 15.8.4 未来函数（Future Function）

**定义**：技术指标的实现方式含有当期或未来数据。

典型案例：

```python
# 错误：TA-Lib 的某些函数默认用 "close" 价格，
# 但有时会自动往前对齐，产生未来函数效果。

# 警惕：使用 pandas rolling 时 min_periods 设置不当
signal = returns.rolling(20, min_periods=1).mean()
# min_periods=1 意味着前19天用不足20个样本计算均值，
# 这在某些情况下会引入偏差。建议设为 min_periods=20，
# 让前19天返回 NaN。
```

### 15.8.5 回测的其他常见错误

| 错误 | 描述 | 修正方法 |
|------|------|---------|
| 忽略停牌 | 停牌期间持仓无法调整 | 停牌日持仓强制延续 |
| 忽略涨跌停 | 涨停无法买入，跌停无法卖出 | 信号满足时检查是否涨跌停 |
| 假设无限流动性 | 大额订单实际会冲击市场 | 按流通市值比例限制仓位 |
| 历史数据质量 | 分红、复权、数据错误 | 严格数据清洗，验证价格连续性 |

---

## 15.9 参数稳健性分析

### 15.9.1 参数高原 vs 参数孤岛

在参数扫描中，我们希望找到“参数高原”（Plateau），而非“参数孤岛”（Island）。

- **参数孤岛**：只有某个特定参数值（如恰好第 21 天）表现好，稍微偏移立刻崩溃。这通常意味着过拟合；
- **参数高原**：在一个宽泛的参数范围内（如 15~30 天）策略都有合理表现。这说明信号本身有效，参数选择是次要的。

```
夏普比率
  ↑
1.2|         ████
1.0|       ████████     ← 参数高原（真实信号）
0.8|     ████████████
0.6|   ████████████████
0.4|
0.2|                      █  ← 参数孤岛（过拟合）
  ├──────────────────────────→ 动量窗口（天）
    5  10  15  20  25  30  35
```

### 15.9.2 样本内/样本外检验

**正确的研究流程**：

1. 用前 70% 数据（样本内）探索参数、优化策略；
2. 参数确定后**锁定**，用后 30% 数据（样本外）验证；
3. 样本外结果不佳时，**不应**返回样本内调整参数（否则样本外等于变成了样本内）。

```python
# 按时间切分，不打乱顺序！
T = len(prices)
split = int(T * 0.7)

prices_in  = prices.iloc[:split]   # 样本内：参数调优
prices_out = prices.iloc[split:]   # 样本外：验证

# 在样本内确定最优参数
best_window = 20  # 假设在样本内得到此参数

# 在样本外用固定参数验证
# ...（不再修改 best_window）
```

!!! warning "禁止时间旅行"
    样本外验证的核心前提是：参数选择不依赖样本外数据。
    一旦因为“样本外表现不好”而回头修改参数，样本外就失去了独立性。

### 15.9.3 Walk-Forward Analysis

更严格的方法是滚动向前检验（Walk-Forward Analysis）：将整个历史切成多个“训练窗口+验证窗口”的滑动区间，统计在所有验证窗口上的平均表现。这能更客观地评估策略的泛化能力。

---

## 15.10 A股实战：20日动量策略完整实现

<figure markdown>
  ![图 15-1　LIQUOR 动量策略回测净值 vs 买入持有](../assets/figures/ch15_backtest.png){ width="680" }
  <figcaption>图 15-1　LIQUOR 动量策略回测净值 vs 买入持有</figcaption>
</figure>


以内置的四只A股风格资产（BANK/LIQUOR/TECH/UTILITY）为例，实现完整的回测流程：

### 15.10.1 策略框架

```python
import numpy as np
import pandas as pd
from fds import (
    load_sample_prices, load_market, daily_returns, set_chinese_font,
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
)

set_chinese_font()

# 加载数据
prices = load_sample_prices()
returns = daily_returns(prices)

# === 策略参数 ===
WINDOW   = 20          # 动量窗口（交易日）
COST     = 0.001       # 单边成本（0.1%）
RF       = 0.02        # 年化无风险利率

# === 信号生成 ===
momentum = prices.pct_change(WINDOW)         # t日动量
signal   = (momentum > 0).astype(float)      # 1=做多, 0=空仓

# === 持仓（T+1执行）===
position = signal.shift(1)                   # 核心：昨日信号=今日持仓

# === 策略收益 ===
gross_ret = (position * returns).dropna()    # 税前收益
turnover  = position.diff().abs().dropna()   # 每日换手
cost      = turnover * COST                  # 每日成本
net_ret   = gross_ret - cost                 # 税后净收益
```

### 15.10.2 绩效指标计算

取 TECH 资产为例：

```python
ticker = 'TECH'
r_net  = net_ret[ticker]

metrics = {
    '年化收益（税后）': f'{annualized_return(r_net):.2%}',
    '年化波动':         f'{annualized_volatility(r_net):.2%}',
    '夏普比率':         f'{sharpe_ratio(r_net, rf):.3f}',
    '最大回撤':         f'{max_drawdown(r_net):.2%}',
    '卡尔玛比率':       f'{annualized_return(r_net) / abs(max_drawdown(r_net)):.3f}',
    '胜率':             f'{(r_net > 0).mean():.2%}',
    '年化换手率':       f'{turnover[ticker].mean() * 252:.1f}×',
}
```

### 15.10.3 前视偏差演示

```python
# 有前视偏差的"错误"版本（忘记shift）
signal_bias  = (momentum > 0).astype(float)
gross_biased = (signal_bias * returns).dropna()   # 直接用signal，不shift

# 净值对比
nav_correct = (1 + net_ret[ticker]).cumprod()
nav_biased  = (1 + gross_biased[ticker]).cumprod()

# 通常 nav_biased 会显著高于 nav_correct，揭示前视偏差的"虚假收益"
```

---

## 15.11 用 akquant 框架做事件驱动回测

前面几节我们手写了**向量化回测**：用 `position = signal.shift(1)` 一次性算出全部持仓与收益，
快、适合参数扫描与信号筛选。但它对撮合细节做了理想化假设（满仓成交、无逐笔风控）。
**生产级**研究往往需要**事件驱动回测**——逐根 K 线（bar）推进，模拟下单、成交、持仓与风控。

[akquant](https://github.com/akfamily/akquant) 是 akshare 生态的开源回测框架：**Rust 内核 + Python 接口**，
内置 walk-forward 滚动验证、TA-Lib 指标、因子表达式引擎、参数网格搜索与丰富的绩效报告，
并能与 akshare 无缝衔接取数。

### 15.11.1 安装

```bash
uv sync --extra quant          # 或：pip install akquant
```

### 15.11.2 核心概念

| 概念 | 说明 |
|---|---|
| `Strategy.on_bar(bar)` | 每根 K 线回调一次；`bar` 提供 `open/close/high/low/symbol/timestamp_iso` |
| `self.buy(symbol=, quantity=)` | 下买单 |
| `self.close_position(symbol=)` | 平仓 |
| `self.get_position(symbol)` | 查询持仓 |
| `aq.run_backtest(data, strategy, initial_cash, symbols)` | 运行回测，返回 `BacktestResult` |
| `result` / `result.metrics` / `result.equity_curve` | 绩效指标表、指标包装、净值曲线 |
| `result.report(filename=, benchmark=)` | 生成 HTML 绩效报告（含基准对比） |

与本章前面手写回测的对照：**向量化**重在快速研究、**事件驱动**重在贴近真实撮合，两者互补。

### 15.11.3 最小示例

下面用**内置数据**离线演示（实盘只需把数据换成 akshare 的真实行情）。
akquant 需要 OHLC 列，内置数据只有收盘价，这里据此构造示意的开高低：

```python
import akquant as aq
from akquant import Strategy
from fds import load_sample_prices

close = load_sample_prices()["LIQUOR"]
df = close.rename("close").reset_index().rename(columns={"index": "date"})
df["open"] = close.shift(1).bfill().to_numpy()
df["high"], df["low"], df["volume"] = close * 1.01, close * 0.99, 1000

class MaCross(Strategy):
    def on_bar(self, bar):
        pos = self.get_position(bar.symbol)
        if pos == 0 and bar.close > bar.open:        # 阳线买入
            self.buy(symbol=bar.symbol, quantity=100)
        elif pos > 0 and bar.close < bar.open:        # 阴线平仓
            self.close_position(symbol=bar.symbol)

result = aq.run_backtest(data=df, strategy=MaCross, initial_cash=100000.0, symbols="LIQUOR")
print(result)                  # 总收益、夏普、最大回撤、胜率等一应俱全
```

!!! note "事件驱动 vs 向量化"
    事件驱动回测更贴近真实交易（逐 bar 撮合、可加风控），但更慢；向量化回测更快、适合海量参数/信号筛选。
    教学这里用内置数据离线演示；真实研究把 `df` 换成 akshare 的真实行情即可（需 `uv sync --extra data`）。

## 15.12 本章小结

本章构建了向量化回测的完整框架，核心知识点如下：

1. **`position = signal.shift(1)` 是回测正确性的基石**，对应A股T+1制度，任何遗漏都导致前视偏差；
2. **A股双边交易成本约 0.2%**，高换手率策略的年化成本可能超过 20%，吞噬大部分甚至全部收益；
3. **绩效评估需要多指标**：夏普评价风险调整收益，卡尔玛关注回撤，换手率反映成本消耗；
4. **回测陷阱是系统性错误**：前视偏差、幸存者偏差、过拟合、数据窥探，每一个都能让虚假策略看起来完美；
5. **参数高原优于参数孤岛**：策略应对参数选择有一定的稳健性，“恰好”某参数最优往往是过拟合信号；
6. **样本外检验是策略有效性的唯一可信证据**，且参数不能在看到样本外结果后再调整。

---

## 15.13 习题

**习题15.1（基础）** 实现一个5日均线与20日均线的金叉/死叉策略：均线上方多头、均线下方空仓。加入0.1%单边成本后，在TECH资产上计算完整绩效指标，并与20日动量策略对比夏普比率。

*参考思路*：计算 `ma5 = prices.rolling(5).mean()`，`ma20 = prices.rolling(20).mean()`，当 `ma5 > ma20` 时信号为1，否则为0。注意 `signal.shift(1)`。

**习题15.2（进阶）** 演示前视偏差的量化影响：分别计算有前视偏差和无前视偏差两版20日动量策略的净值，比较它们的年化收益率差异。将差值（前视溢价）绘图。

*参考思路*：有前视版直接 `gross = signal * returns`；无前视版 `gross = signal.shift(1) * returns`。年化收益差 = 有前视年化 - 无前视年化，即为“虚假超额收益”。

**习题15.3（成本分析）** 以TECH为例，分别设定单边成本 0.0%、0.05%、0.10%、0.15%、0.20%，绘制“成本—夏普比率”关系图。找出使夏普比率降为0的临界成本率，并计算此时年化成本占无成本年化收益的比例。

*参考思路*：用循环遍历成本参数，每次计算净收益后算夏普比率，画折线图，用线性插值找零点。

**习题15.4（参数扫描）** 对TECH资产扫描动量窗口 `[5, 10, 15, 20, 25, 30, 40, 60]`，绘制“窗口—夏普比率”热图。讨论：最优参数附近是否呈现参数高原？是否有过拟合嫌疑？

*参考思路*：循环窗口参数，记录每个窗口的夏普比率，用 `plt.bar` 绘图；同时用时间前60%为样本内，后40%为样本外，对比两段的夏普比率走势。

**习题15.5（综合）** 构造一个“幸运策略”：使用`np.random.seed(42)`生成随机信号，对四只资产分别回测（无成本），选出夏普比率最高的资产。然后：(a) 计算若有10000种随机策略，预期最高夏普比率约为多少（用 `max(np.random.normal(0, 1, 10000) / np.sqrt(252)...)` 估算）；(b) 讨论这与“数据窥探”的关系。

*参考思路*：随机信号 + 大量尝试的组合，在随机数据上总能找到“显著”策略。Bonferroni 校正后，最高夏普的显著性水平应调整为 $\alpha / 10000 = 0.000005$，对应 $z \approx 4.4$ 的标准差水平，说明要求极高。

---

## 15.14 拓展阅读

1. **Bailey, D. H., & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management. —— 提出回测过拟合概率（PBO）的经典论文。

2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley. —— 第11章详细介绍多种回测陷阱与检验方法，是量化研究员必读书目。

3. **Prado, M. L. de (2018)**. *The 10 Reasons Most Machine Learning Funds Fail*. Journal of Portfolio Management. —— 系统总结了量化策略失败的10大原因，与本章内容高度相关。

4. **Aronson, D. (2006)**. *Evidence-Based Technical Analysis*. Wiley. —— 用统计学严格检验技术分析有效性，第7章专门处理数据窥探问题。

5. **Harvey, C. R., Liu, Y., & Zhu, H. (2016)**. *…and the Cross-Section of Expected Returns*. Review of Financial Studies. —— 指出大量发表的因子在严格多重检验校正后可能都不显著（t值应至少达到3.0）。
