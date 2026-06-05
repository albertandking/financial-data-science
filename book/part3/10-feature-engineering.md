# 第10章 金融特征工程

[![在 Colab 打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertandking/financial-data-science/blob/main/notebooks/ch10_feature_engineering.ipynb) [![在 Binder 打开](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/albertandking/financial-data-science/main?labpath=notebooks/ch10_feature_engineering.ipynb)

!!! info "配套代码"
    `notebooks/ch10_feature_engineering.ipynb`

## 10.1 本章导读

特征工程是机器学习流程的核心环节，尤其在金融领域，原始的价格和成交量数据极少直接作为模型输入。一方面，金融时序数据的**信噪比极低**——真实的预测信号可能淹没在大量随机噪声之中；另一方面，金融市场存在**时序依赖**结构，某一时刻的信息只能来自过去，任何涉及未来信息的特征都会引入“前视偏差”（look-ahead bias），导致回测严重虚高。

本章围绕 A 股量化实战，系统介绍价量特征、技术指标、滚动窗口与防前视技术、截面标准化，以及特征筛选方法，帮助读者建立一套可直接投入实践的特征工程体系。

## 10.2 学习目标

读完本章后，你应当能够：

1. 理解金融特征工程与通用机器学习特征工程的区别，尤其是**前视偏差**的来源与危害。
2. 手动构造多周期动量、波动率、换手率代理、量价背离等价量特征。
3. 用 `pandas` 正确实现 MA、EMA、RSI、MACD、布林带等技术指标，确保**只用历史信息**。
4. 理解 `rolling`/`ewm` + `shift(1)` 的标准写法，能够用对照实验量化前视偏差的影响。
5. 区分**时序标准化**与**截面标准化**，解释金融中通常采用截面标准化的原因。
6. 掌握基于相关性、方差阈值、IC（信息系数）的特征筛选方法。

---

## 10.3 金融特征工程的特殊性

### 10.3.1 低信噪比下的特征构造

在计算机视觉或 NLP 中，模型通常能从高维输入中提取显著信号。但股票收益率的日度预测 $R^2$ 很少超过 5%——真实信号微弱，噪声主导。这意味着：

- **特征的预测力（IC/IR）通常很低**，0.02~0.05 的截面 IC 在 A 股已属可用信号。
- 单一特征难以胜任，往往需要**组合多维特征**形成特征矩阵。
- 对数据质量（停牌填充、复权、异常值处理）极度敏感。

### 10.3.2 时序特征的三大原则

!!! warning "防前视偏差是金融特征工程的第一原则"
    **前视偏差**（look-ahead bias）指特征构造时意外用了未来信息。例如，“当天收盘价 / 过去 20 日最高价” 中的“最高价”如果包含了当天，就用了当天才知道的信息，在实际交易中无法复现。一旦引入前视偏差，回测业绩将严重虚高。

**原则一：时间对齐**。特征 $x_{i,t}$ 必须仅由 $t$ 时刻及之前的数据计算。通用做法是在构造特征后立即 `shift(1)`，确保预测 $t+1$ 时用的特征全部来自 $t$ 及之前。

**原则二：滚动窗口**。使用 `rolling(window).func()` 时，pandas 默认窗口包含当前点，即计算 $t$ 日特征时用到了 $t$ 日本身的数据。若随后对结果做 `shift(1)`，就能保证信息不泄漏。

**原则三：跨资产一致性**。同一截面（同一交易日）不同股票的特征要基于可比的口径，否则截面信号失去意义。

### 10.3.3 数据类型概览

| 特征类型 | 原始来源 | 典型例子 |
|---|---|---|
| 价格衍生特征 | 日度 OHLCV | 动量、反转、波动率 |
| 技术指标 | 收盘价 / 成交量 | MA、RSI、MACD、布林带 |
| 基本面特征 | 财报、公告 | PE、ROE、净利润增速 |
| 情绪特征 | 新闻、研报 | 情感得分、超预期 |
| 市场微结构 | L2 行情 | 买卖价差、订单不平衡 |

本章聚焦**前两类**，它们来自公开行情数据，是量化选股的基础。

---

## 10.4 价量特征

### 10.4.1 动量特征（Momentum）

动量来源于“过去强者继续强”的经验观察。经典的 $n$ 日动量定义为：

$$\text{MOM}_n(t) = \frac{P_t}{P_{t-n}} - 1$$

其中 $P_t$ 为 $t$ 日收盘价。在实践中，Jegadeesh & Titman（1993）发现跳过最近 1 个月（避免反转效应），以 $[t-252, t-21]$ 区间的累计收益作为动量信号效果更稳定。A 股常用 5 日、20 日、60 日多周期动量：

$$\text{MOM}_5 = \frac{P_t}{P_{t-5}} - 1, \quad
\text{MOM}_{20} = \frac{P_t}{P_{t-20}} - 1, \quad
\text{MOM}_{60} = \frac{P_t}{P_{t-60}} - 1$$

!!! tip "多周期动量的直觉"
    5 日动量捕捉短期反转前的动能；20 日动量对应月度趋势；60 日动量对应季度级别趋势。
    在 A 股中，短期反转（5 日）和中期动量（20~60 日）均有文献记录，但效果随时期变化。

### 10.4.2 反转特征（Reversal）

短期反转（short-term reversal）是动量的反面：上周涨幅最大的股票，下周倾向于回调。其信号通常用 1 周（5 日）收益率的负值来构造：

$$\text{REV}_5(t) = -\left(\frac{P_t}{P_{t-5}} - 1\right)$$

在 A 股日度截面，5 日收益率往往呈现明显负自相关，即短期反转效应。

### 10.4.3 波动率特征（Volatility）

已实现波动率（Realized Volatility）通常用过去 $n$ 日对数收益率的标准差衡量：

$$\text{VOL}_n(t) = \text{std}\left(\ln\frac{P_s}{P_{s-1}},\ s = t-n+1, \ldots, t\right) \times \sqrt{252}$$

波动率本身可作为**风险特征**（高波动率往往意味着更高的非系统性风险），也可用于识别**波动率突破**等信号。

### 10.4.4 换手率代理

换手率（Turnover）= 成交量 / 流通股本，是流动性的代理。内置数据无流通股本，可用**相对换手率**作为代理：

$$\text{TVOL\_rel}(t) = \frac{V_t}{\text{mean}(V_{t-20}, \ldots, V_{t-1})}$$

其中 $V_t$ 为 $t$ 日成交量。相对换手率 > 1 表示成交量放大，常伴随价格突破或主力行为。

### 10.4.5 量价背离

量价背离（Volume-Price Divergence）衡量价格上涨时成交量是否配合：

$$\text{VPD}(t) = \text{sign}(r_t) \times \left(1 - \frac{V_t}{\text{VOL\_ma}_{20}(t)}\right)$$

当价格上涨而成交量萎缩时，VPD 为负，提示上涨动能不足。

---

## 10.5 技术指标

<figure markdown>
  ![图 10-1　LIQUOR 均线与布林带](../assets/figures/ch10_bollinger.png){ width="680" }
  <figcaption>图 10-1　LIQUOR 均线与布林带</figcaption>
</figure>


!!! warning "所有技术指标只能用历史数据"
    以下指标的实现均确保在计算 $t$ 日特征时**只用 $t$ 日及之前的数据**。使用 `rolling` 后配合 `shift(1)` 确保下一期预测不会泄漏当期信息。

### 10.5.1 移动均线（MA / EMA）

**简单移动均线（SMA）**：

$$\text{MA}_n(t) = \frac{1}{n}\sum_{s=t-n+1}^{t} P_s$$

**指数移动均线（EMA）**，赋予近期数据更高权重：

$$\text{EMA}_n(t) = \alpha \cdot P_t + (1-\alpha) \cdot \text{EMA}_n(t-1), \quad \alpha = \frac{2}{n+1}$$

均线衍生特征常用**价格偏离均线的比率**（Price-MA Ratio）：

$$\text{PMR}_n(t) = \frac{P_t}{\text{MA}_n(t)} - 1$$

$\text{PMR} > 0$ 表示价格在均线之上（偏强），$\text{PMR} < 0$ 表示偏弱。

```python
# 正确写法：rolling 计算均线，shift(1) 确保预测下一期时不泄漏当期
ma20 = prices.rolling(20).mean().shift(1)
pmr20 = prices / ma20 - 1   # 当期价格相对昨日均线的偏离
```

### 10.5.2 相对强弱指数（RSI）

RSI 由 Wilder（1978）提出，衡量 $n$ 日内上涨日平均涨幅与下跌日平均跌幅之比：

$$\text{RSI}_n(t) = 100 - \frac{100}{1 + RS_n(t)}, \quad RS_n = \frac{\text{平均上涨}}{\text{平均下跌}}$$

RSI 的范围为 [0, 100]。传统解读：RSI > 70 为超买区域，RSI < 30 为超卖区域。

**注意**：在量化策略中，RSI 的超买/超卖阈值可能随市场结构变化。应对 RSI 进行**截面排名**而非使用绝对阈值。

```python
def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算 RSI，只用历史信息（shift 后再用于预测）。"""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float('inf'))
    return 100 - 100 / (1 + rs)
```

### 10.5.3 MACD（指数平滑异同移动平均线）

MACD 由 Appel（1979）提出，是两条 EMA 之差：

$$\text{DIF}(t) = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)$$

$$\text{DEA}(t) = \text{EMA}_9(\text{DIF})$$

$$\text{MACD}(t) = 2 \times (\text{DIF}(t) - \text{DEA}(t))$$

常用特征：DIF 上穿 DEA（金叉）、MACD 柱状图由负转正。

```python
def compute_macd(prices: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (DIF, DEA, MACD 柱) 三列 DataFrame。"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    dif  = ema_fast - ema_slow
    dea  = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    return dif, dea, macd
```

### 10.5.4 布林带（Bollinger Bands）

布林带在均线基础上加减标准差：

$$\text{Upper}(t) = \text{MA}_n(t) + k \cdot \sigma_n(t)$$
$$\text{Lower}(t) = \text{MA}_n(t) - k \cdot \sigma_n(t)$$

常用 $n=20, k=2$。常用的截面特征是**%B 指标**（价格在布林带中的相对位置）：

$$\%B(t) = \frac{P_t - \text{Lower}(t)}{\text{Upper}(t) - \text{Lower}(t)}$$

$\%B \approx 1$ 时价格接近上轨，$\%B \approx 0$ 时接近下轨。

---

## 10.6 滚动窗口特征与前视偏差

### 10.6.1 `rolling` 的工作方式

`pandas.DataFrame.rolling(window=n)` 在每个时间点 $t$ 用 $[t-n+1, t]$ 的 $n$ 个数据点计算统计量。**关键点**：计算 $t$ 日特征时，包含了 $t$ 日本身的数据。若特征直接用于预测次日收益率 $r_{t+1}$，则没有前视偏差（因为 $t$ 日数据在 $t$ 日收盘后已知）。

然而，在**截面策略**中，通常在 $t$ 日收盘**之前**（如开盘前）构造特征，此时还不知道 $t$ 日收盘价。为安全起见，规范做法是 `shift(1)`：

```python
# 推荐写法：rolling 计算后 shift(1)
# 在预测 t 日收益时，特征只用到 t-1 日及之前的数据
feature = prices.rolling(20).std().shift(1)
```

### 10.6.2 EWM 与前视偏差

`ewm` 默认 `adjust=True`，在序列开头使用了边界修正，但不会引入未来信息。`ewm(span=n, adjust=False)` 与 `rolling(n).mean()` 的主要区别在于权重形式，二者均不引入前视偏差。

但 `ewm` 没有严格的“窗口长度”，每一步都对全部历史加权，因此`shift(1)` 同样适用。

### 10.6.3 前视偏差的量化对比

!!! warning "用实验验证前视偏差的危害"
    下面的对比实验清晰展示前视偏差的代价：含前视的特征与未来收益的相关性**虚高**，因为特征构造时直接或间接用到了未来信息。

以 20 日动量为例：

```python
# 有前视偏差的写法（错误）：
# prices.pct_change(20) 在 t 日计算时包含 t 日本身，
# 特征与 t+1 日收益的相关性会虚高
mom_biased = prices.pct_change(20)          # 含 t 日信息

# 正确写法：shift(1) 确保预测用的是昨日及更早的信息
mom_clean  = prices.pct_change(20).shift(1) # 只含 t-1 日及更早信息
```

两种写法的特征与次日收益的 IC（Pearson 相关系数）往往相差 2~5 倍，直接影响策略评估的可靠性。

### 10.6.4 `min_periods` 与缺失值

`rolling(n, min_periods=m)` 允许窗口不足 $n$ 时以至少 $m$ 个数据点计算。建议：

- 对于样本量足够的情况，设 `min_periods=n`（严格要求窗口满）；
- 序列开头会产生 NaN，在构建特征矩阵时需要 `dropna()` 或填充处理。

---

## 10.7 截面特征标准化

### 10.7.1 为何金融用截面标准化

在监督学习中，时序标准化（用整个历史均值和标准差）最为常见。但金融截面策略有其特殊性：

- **策略是“同一天选哪些股”**，关注的是截面排名，而非时序水平。
- 市场整体水平随时间漂移——牛市时所有股票的动量特征都很高，时序标准化后的信号会失真。
- **截面标准化**（对同一天所有股票的特征做 z-score）消除了时间趋势，保留了截面信息。

数学定义：设 $t$ 日有 $N$ 只股票，特征值为 $x_{i,t}$，则截面 z-score 为：

$$z_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}, \quad
\mu_t = \frac{1}{N}\sum_i x_{i,t}, \quad
\sigma_t = \text{std}\left(\{x_{i,t}\}_{i=1}^N\right)$$

### 10.7.2 截面分位数与排序

截面分位数比 z-score 更稳健（对异常值不敏感）：

```python
# 截面 z-score：对每一天，跨股票标准化
def cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 每行（每个时间截面）做 z-score 标准化。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

# 截面排序（0~1 分位数）
def cross_section_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面排名，映射到 [0, 1] 区间。"""
    return df.rank(axis=1, pct=True)
```

!!! info "截面标准化的边界情况"
    当截面只有少数股票时，截面标准差可能极小，z-score 方差放大。实践中常在截面标准化前先做**Winsorize**（截断极端值），通常取 1%~99% 分位数。

---

## 10.8 缺失值与异常值处理

### 10.8.1 金融数据的缺失来源

| 缺失类型 | A 股常见场景 | 处理建议 |
|---|---|---|
| 新股上市初期 | 上市 20 天内无历史均线 | 丢弃或标记 |
| 停牌 | 价格不变，成交量为零 | 前向填充或剔除 |
| 指标窗口不足 | RSI(14) 前 13 天 NaN | 丢弃 |
| 数据源缺漏 | 部分数据库跳行 | 检查并填补 |

### 10.8.2 异常值的处理

金融数据中异常大的特征值（如 ST 股票的异常波动率）会干扰模型。处理步骤：

1. **Winsorize**（截尾）：将超过 $[\mu - 3\sigma, \mu + 3\sigma]$ 范围的值截断。
2. **截面百分位截断**：更常用，对每个截面取 1%~99% 分位数截断。
3. **对数变换**：对右偏严重的特征（如成交量、PE）先取对数。

```python
def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """截面 Winsorize：对每个截面（行）分别截断极端值。"""
    q_low  = df.quantile(lower, axis=1)
    q_high = df.quantile(upper, axis=1)
    return df.clip(lower=q_low, upper=q_high, axis=0)
```

---

## 10.9 特征选择

### 10.9.1 相关性筛除

高度相关的特征携带冗余信息，增加多重共线性风险。常见做法：

1. 计算特征间的 Pearson/Spearman 相关矩阵。
2. 若两特征相关系数 $|r| > 0.8$，保留与目标变量 IC 更高的一个。

```python
def remove_correlated(X: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """返回去除高相关特征后保留的列名列表。"""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in X.columns if c not in to_drop]
```

### 10.9.2 方差阈值

方差接近零的特征几乎不含信息，应剔除：

```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.001)
sel.fit(X)
kept = X.columns[sel.get_support()]
```

### 10.9.3 基于 IC 的特征筛选

**IC（Information Coefficient）**是特征与下期收益的 Spearman/Pearson 相关系数，是量化金融中最核心的特征评价指标：

$$IC_t = \text{corr}(x_{\cdot,t},\ r_{\cdot,t+1})$$

**ICIR（IC Information Ratio）**= IC 的均值/标准差，衡量 IC 的稳定性：

$$ICIR = \frac{\overline{IC}}{\text{std}(IC)}$$

筛选准则（A 股经验）：
- $|\overline{IC}| > 0.02$：有一定预测能力
- $ICIR > 0.3$：信号较稳定
- **IC 衰减曲线**：随滞后期延长，IC 应逐渐衰减（验证信号的时效性）

### 10.9.4 多重共线性与方差膨胀因子

**VIF（方差膨胀因子）**量化特征间的线性相关程度：

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

其中 $R_j^2$ 是用其余特征回归第 $j$ 个特征的 $R^2$。VIF > 10 提示严重共线性，应考虑：

- PCA 降维
- 删除冗余特征
- 使用 L1/L2 正则化的模型

---

## 10.10 A 股实战：构造特征矩阵

本节以内置的 4 只 A 股风格资产（BANK / LIQUOR / TECH / UTILITY）为例，构造一套完整的特征矩阵，并系统验证前视偏差。

### 10.10.1 特征构造流程

完整流程如下：

```
原始价格数据
    → 计算各类特征（动量、波动率、技术指标等）
    → shift(1) 时间对齐（防前视）
    → 截面 Winsorize（截断极端值）
    → 截面 z-score 标准化
    → 拼接为特征矩阵
    → 对齐目标变量（下期收益率）
    → 删除含 NaN 的行
```

### 10.10.2 前视偏差的对照实验

对照实验设计：

| 版本 | 写法 | 预期结果 |
|---|---|---|
| 含前视 | `pct_change(20)` 直接使用 | 与下期收益相关性**虚高** |
| 无前视 | `pct_change(20).shift(1)` | 与下期收益相关性**真实** |

通过计算两版本特征与下期收益的相关性（IC），可以定量展示前视偏差的危害。

---

## 10.11 本章小结

本章系统介绍了金融特征工程的核心方法论与实践技巧：

1. **前视偏差**是金融特征工程最重要的风险，`rolling`/`ewm` + `shift(1)` 是防止前视的标准范式。
2. **价量特征**（动量、反转、波动率、换手率代理、量价背离）是量化选股的基础信号。
3. **技术指标**（MA、RSI、MACD、布林带）均可用 pandas 手工实现，不依赖外部库。
4. **截面标准化**（z-score、排序）消除时间趋势，是截面策略的标准预处理步骤。
5. **特征筛选**（相关性、方差、IC/ICIR）帮助从候选特征集中挑选有效信号。

### 10.11.1 关键公式汇总

| 特征 | 公式 | 防前视写法 |
|---|---|---|
| $n$ 日动量 | $P_t/P_{t-n} - 1$ | `pct_change(n).shift(1)` |
| 已实现波动率 | $\text{std}(r_{t-n+1..t}) \times \sqrt{252}$ | `rolling(n).std().shift(1)` |
| RSI | $100 - 100/(1+RS)$ | 计算后 `shift(1)` |
| EMA | $\alpha P_t + (1-\alpha)\text{EMA}_{t-1}$ | `ewm(span).mean().shift(1)` |
| 截面 z-score | $(x_{i,t} - \mu_t)/\sigma_t$ | 截面操作，不涉及时序前视 |
| IC | $\text{corr}(x_t, r_{t+1})$ | 已含 shift |

---

## 10.12 习题

**习题 10.1**（前视偏差诊断）

给定如下代码片段，判断每行是否存在前视偏差，并说明原因：

```python
# (a) 以下三种特征构造方式，哪些有问题？
feat_a = prices.rolling(20).mean()             # 直接用滚动均值
feat_b = prices.rolling(20).mean().shift(1)    # 均值再 shift
feat_c = prices.pct_change(20).shift(1)        # 收益率再 shift

# (b) 如果用 feat_a 作为特征预测 t+1 期收益，有无前视偏差？
# (c) 如果策略在 t 日开盘前选股（不知道 t 日收盘价），feat_b 和 feat_c 哪个更安全？
```

参考思路：(a) feat_a 包含 $t$ 日信息，若预测 $t+1$ 日收益，$t$ 日收盘后才知道，不算偏差；但若在开盘前选股则有偏差。feat_b 和 feat_c 的 `shift(1)` 确保特征值是 $t-1$ 日及之前的数据，两者均安全。(b) 取决于策略信号触发时间。(c) feat_b 和 feat_c 均安全。

**习题 10.2**（RSI 实现）

手动实现 RSI(14)，并用内置 `load_sample_prices()` 数据计算 TECH 股票的 RSI 序列。绘制价格与 RSI 的双轴图，在图上标出 RSI = 30 和 RSI = 70 的水平线。注意构造 RSI 时不要引入前视偏差。

参考思路：参照 10.5.2 节的公式，`delta = prices.diff(); gain = delta.clip(lower=0).rolling(14).mean()` 等，最后 `shift(1)` 对齐。

**习题 10.3**（截面 vs 时序标准化）

以 TECH 的 20 日动量特征为例：
(a) 分别做**时序 z-score**（对整个时间序列标准化）和**截面 z-score**（对每日 4 只股票截面标准化）；
(b) 绘制两种标准化结果的时序对比图；
(c) 解释为何截面 z-score 更适合横截面选股策略。

参考思路：时序标准化用全样本均值/标准差；截面标准化对 DataFrame 的每行（日期）标准化（`sub(mean(axis=1), axis=0).div(std(axis=1), axis=0)`）。截面标准化消除了市场整体水平随时间的漂移，使信号在不同时期具有可比性。

**习题 10.4**（IC 计算与特征筛选）

构造以下 4 个特征：5 日动量、20 日动量、20 日已实现波动率、14 日 RSI（均经过 `shift(1)` 防前视）。
(a) 计算每个特征与次日收益率的时序平均 IC 和 ICIR；
(b) 根据 IC > 0.02 的标准筛选有效特征；
(c) 计算 4 个特征之间的相关系数矩阵，识别高度相关的特征对。

参考思路：对每个日期计算截面 Spearman 相关系数，取时序均值即为 $\overline{IC}$；$\overline{IC}/\text{std}(IC)$ 为 ICIR。

**习题 10.5**（布林带与 %B 特征）

(a) 实现布林带（MA(20) ± 2σ），计算 TECH 的 %B 特征序列；
(b) 分析 %B 与下期收益的相关性；
(c) 解释为何布林带的 %B 既可以用作动量信号（%B 高代表强势），也可以用作反转信号（%B 高代表超买）。在什么市场环境下各自更有效？

参考思路：%B = (P - Lower) / (Upper - Lower)。趋势市场中 %B 高（突破上轨）往往续涨（动量解读）；震荡市中 %B 高后均值回归（反转解读）。判断用哪种解读需结合更长周期的趋势过滤器。

---

## 10.13 拓展阅读

1. **Jegadeesh, N., & Titman, S. (1993)**. “Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.” *Journal of Finance*, 48(1), 65–91. — 动量效应的奠基论文。

2. **Fama, E. F., & French, K. R. (1992)**. “The Cross-Section of Expected Stock Returns.” *Journal of Finance*, 47(2), 427–465. — 规模与价值特征的实证来源。

3. **Gu, S., Kelly, B., & Xiu, D. (2020)**. “Empirical Asset Pricing via Machine Learning.” *Review of Financial Studies*, 33(5), 2223–2273. — 机器学习在特征选择与资产定价中的系统性应用，含超过 900 个特征的大规模实验。

4. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley. — 第 4~6 章专门讲述金融特征工程，强调前视偏差与标准化。

5. **华泰证券研究所 A 股多因子系列报告**（2016~2023）——对 A 股动量、反转、波动率因子有深入的实证分析与改进方案。
