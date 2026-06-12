# 第4章 探索性分析与可视化

[![在 Colab 打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertandking/financial-data-science/blob/main/notebooks/ch04_eda_visualization.ipynb) [![在 Binder 打开](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/albertandking/financial-data-science/main?labpath=notebooks/ch04_eda_visualization.ipynb)

!!! info "配套代码"
    本章示例可在配套示例 中运行。主体实验依赖内置数据，离线即可完成。

## 4.1 本章导读

在建立任何量化模型之前，必须先与数据“交朋友”——了解它的形状、脾气和怪癖。**探索性数据分析（Exploratory Data Analysis, EDA）**正是这种“交朋友”的过程：不预设模型，而是让数据自己说话。

金融数据有其鲜明的个性。与工业质量数据、医疗数据不同，股票收益率既不服从正态分布，也没有稳定的均值和方差——它有**厚尾**、有**波动率聚集**、有**杠杆效应**。这些被称为“风格化事实”（stylized facts）的特征，是几十年全球实证研究反复验证的规律，也是 GARCH、随机波动率等模型诞生的根本原因。

本章以四只合成A股风格股票（银行、白酒、科技、公用事业）为主线，系统介绍金融 EDA 的完整工具箱。

## 4.2 学习目标

学完本章后，你应能够：

1. 计算并解读收益率的四个矩（均值、方差、偏度、峰度）及其金融含义；
2. 用直方图、KDE、经验CDF、QQ图诊断分布形态与厚尾；
3. 识别并验证金融数据的五大风格化事实；
4. 对收益率进行正态性检验（Jarque-Bera）和自相关检验（Ljung-Box）；
5. 计算相关矩阵与滚动相关，理解相关性的时变性；
6. 绘制符合中国金融行业规范的专业图表。

---

## 4.3 描述性统计：四个矩与分位数

### 4.3.1 加载数据

<figure markdown>
  ![图4-1四只示例股票价格走势（起点归一为100）](../assets/figures/ch01_prices.png){ width="680" }
  <figcaption>图4-1四只示例股票价格走势（起点归一为100）</figcaption>
</figure>


```python
from fds import load_sample_prices, daily_returns, set_chinese_font
import pandas as pd

prices = load_sample_prices()   # 约750个交易日，列：BANK LIQUOR TECH UTILITY
rets = daily_returns(prices)    # 日度简单收益率
print(rets.shape, rets.index[0], "~", rets.index[-1])
```

### 4.3.2 四个矩的金融含义

**矩（moment）**是描述概率分布形状的统计量。前四阶矩在金融中各有实际意义：

| 矩 | 统计量 | 公式 | 金融含义 |
|:---:|:---|:---|:---|
| 一阶 | 均值 $\mu$ | $\frac{1}{T}\sum_{t}r_t$ | 平均收益，日度约为0，年化后才有意义 |
| 二阶 | 方差 $\sigma^2$ | $\frac{1}{T-1}\sum_t(r_t-\mu)^2$ | 风险的核心度量；$\sigma$（标准差）即波动率 |
| 三阶 | 偏度 $S$ | $E\!\left[\left(\frac{r-\mu}{\sigma}\right)^3\right]$ | 分布不对称性；负偏 $\Rightarrow$ 暴跌比暴涨更极端 |
| 四阶 | 超额峰度 $K$ | $E\!\left[\left(\frac{r-\mu}{\sigma}\right)^4\right]-3$ | 尾部厚度；$K>0$ 即比正态更厚尾（leptokurtic） |

!!! note "超额峰度 vs 峰度"
    Pandas 的 `.kurtosis()` 返回**超额峰度**（excess kurtosis），即已减去正态分布的参考值3。正态分布超额峰度 $= 0$；A股日度收益率超额峰度通常在3～10之间，远高于0。

表中偏度与峰度均写成**标准矩**（standardized moment）的形式，即先把收益率化为无量纲的标准化变量 $z=\frac{r-\mu}{\sigma}$，再取其三次方、四次方的期望。这样做有两个好处：一是去掉量纲，使不同资产、不同频率的偏度与峰度可直接横向比较；二是把“位置”“离散程度”从“形状”中剥离出来——均值刻画位置，方差刻画离散程度，而三阶、四阶标准矩才纯粹刻画分布的**形状**（不对称性与尾部厚度）。

在实际计算中，总体期望要用样本平均来估计。最朴素的样本估计量是

$$\hat{S}=\frac{1}{n}\sum_{i=1}^{n}\left(\frac{r_i-\bar r}{\hat\sigma}\right)^{3},\qquad \hat{K}=\frac{1}{n}\sum_{i=1}^{n}\left(\frac{r_i-\bar r}{\hat\sigma}\right)^{4}-3,$$

其中 $\bar r$ 是样本均值，$\hat\sigma$ 是样本标准差。需要提醒的是，$\hat\sigma$ 用 $n$ 还是 $n-1$ 作分母、是否再乘小样本无偏校正因子，各软件约定不同：Pandas 的 `.skew()`、`.kurtosis()` 用带校正的 Fisher 定义，`scipy.stats` 默认不校正。样本量在数百以上时两者差别可忽略，但比对教科书公式与代码输出时要心里有数。

!!! example "例4.1：手算一个小收益序列的偏度与峰度"
    取一段经过放大的“日收益率”（单位：%）样本，共 $n=5$ 个观测：

    $r = (-3,\ -1,\ 0,\ 1,\ 8).$

    **第一步：均值。** $\bar r=\frac{-3-1+0+1+8}{5}=\frac{5}{5}=1$。

    **第二步：离差与各阶幂。** 令 $d_i=r_i-\bar r$，依次为 $(-4,\ -2,\ -1,\ 0,\ 7)$。

    | $r_i$ | $d_i$ | $d_i^2$ | $d_i^3$ | $d_i^4$ |
    |:---:|:---:|:---:|:---:|:---:|
    | $-3$ | $-4$ | $16$ | $-64$ | $256$ |
    | $-1$ | $-2$ | $4$ | $-8$ | $16$ |
    | $0$ | $-1$ | $1$ | $-1$ | $1$ |
    | $1$ | $0$ | $0$ | $0$ | $0$ |
    | $8$ | $7$ | $49$ | $343$ | $2401$ |
    | **合计** | $0$ | $70$ | $270$ | $2674$ |

    **第三步：用总体矩定义（分母取 $n$）。** 二阶中心矩 $m_2=\frac{70}{5}=14$，故 $\hat\sigma=\sqrt{14}\approx 3.742$。三阶中心矩 $m_3=\frac{270}{5}=54$，四阶中心矩 $m_4=\frac{2674}{5}=534.8$。

    **第四步：标准化求偏度与峰度。**

    $\hat S=\frac{m_3}{m_2^{3/2}}=\frac{54}{14^{1.5}}=\frac{54}{52.38}\approx 1.03,$

    $\hat K=\frac{m_4}{m_2^{2}}-3=\frac{534.8}{196}-3\approx 2.728-3=-0.272.$

    **判读：** 偏度约 $+1.03$ 为**右偏**——这正是那个 $+8$ 的极端正值把右尾拉长的结果。超额峰度约 $-0.27$ 接近0、甚至略小于0，说明在“掐头去尾”只有5个点的小样本里，单个离群值虽然制造了不对称，却不足以让四阶矩明显超过正态参考值；真实金融日频数据有成百上千个观测，极端值反复出现，峰度才会被显著推高。这个手算例也提醒我们：偏度、峰度对**极端单点极其敏感**，小样本下数值波动很大，不能仅凭一两个数字就下结论。

### 4.3.3 分位数与极差

除四个矩之外，分位数（quantile）也是风险管理的重要工具：

- **VaR（风险价值）**：本质上是收益率的某个低分位数（如5% 分位数）。
- **四分位距（IQR）**：$Q_{75\%} - Q_{25\%}$，比标准差更稳健的离散度指标，不受极端值干扰。
- **极差（range）**：最大值 $-$ 最小值，衡量历史最极端涨跌幅。

```python
desc = rets.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(desc.round(4))
```

### 4.3.4 四只股票汇总统计表

| 股票 | 均值(%) | 年化波动率(%) | 偏度 | 超额峰度 | 历史最大日跌幅(%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| BANK | — | — | — | — | — |
| LIQUOR | — | — | — | — | — |
| TECH | — | — | — | — | — |
| UTILITY | — | — | — | — | — |

> 表中“—”在配套示例运行后会填入实际数值。银行与公用事业的波动率通常低于科技与白酒，但四只股票均呈现**负偏、高峰度**特征。

### 4.3.5 稳健统计量：当极端值搅局时

例4.1已经展示了偏度、峰度对单个极端点的敏感。事实上，**均值与标准差本身就不稳健**：一笔“乌龙指”或一根涨跌停就能把它们带偏。风险实务中常用一组**稳健统计量（robust statistics）**作为补充与交叉验证：

- **中位数（median）**替代均值刻画“典型水平”，其崩溃点（breakdown point）高达50%，即便一半数据被污染仍稳如泰山；
- **四分位距 IQR $=Q_{75\%}-Q_{25\%}$** 替代标准差刻画离散程度，完全不看两端的极端尾部；
- **中位数绝对偏差 MAD $=\operatorname{median}_i\,|r_i-\operatorname{median}(r)|$**，乘以 $1.4826$ 后可作为正态情形下标准差的一致估计。

!!! example "例4.2：一个异常值如何同时“骗过”均值和标准差"
    设某只股票连续10日的收益率（%）为

    $r=(0.2,\ -0.1,\ 0.3,\ 0.0,\ -0.2,\ 0.1,\ -0.3,\ 0.2,\ 0.0,\ 0.1),$

    这是一段平淡的窄幅震荡。容易算得均值 $\approx 0.03\%$、标准差 $\approx 0.19\%$、中位数 $=0.05\%$、IQR $=0.4\%$。

    现在把最后一天换成一根跌停 $-10$（模拟一次黑天鹅或数据错误）：

    $r'=(0.2,\ -0.1,\ 0.3,\ 0.0,\ -0.2,\ 0.1,\ -0.3,\ 0.2,\ 0.0,\ -10).$

    重新计算：均值骤降到 $\approx -0.98\%$（**被一个点拉低了30多倍**），标准差暴涨到 $\approx 3.2\%$（**膨胀约17倍**）。而中位数仍为 $0.0\%$、IQR 仅从 $0.4$ 微动到约 $0.45$——**几乎不动**。

    **判读：** 这正是“稳健”二字的含义。当某资产的均值/标准差与中位数/IQR 严重背离时，第一反应不应是“这资产收益高/风险大”，而应是“数据里是不是混进了极端值或脏数据”，先核查再下结论。这也是 EDA 第一步往往先画箱线图、先看分位数表的原因。

---

## 4.4 分布分析

### 4.4.1 直方图与核密度估计（KDE）

**直方图**（histogram）将数据分成若干“桶”，计数后可视化频率。但直方图依赖组距（binwidth）的选取，容易遮蔽真实形状。**核密度估计（KDE）**用连续的核函数（通常为高斯核）在数据点处叠加，得到光滑密度曲线：

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{x - x_i}{h}\right)$$

其中 $h$ 为带宽（bandwidth），越小越陡峭，越大越平滑。Pandas/Seaborn 默认用 Scott 法则自动选带宽。

```python
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

fig, ax = plt.subplots()
col = "TECH"
rets[col].plot.hist(bins=60, density=True, alpha=0.4, color="steelblue", label="直方图")
rets[col].plot.kde(ax=ax, color="steelblue", label="KDE")
x = np.linspace(rets[col].min(), rets[col].max(), 300)
ax.plot(x, norm.pdf(x, rets[col].mean(), rets[col].std()), "r--", label="正态分布")
ax.legend(); ax.set_title(f"{col} 收益率分布")
```

### 4.4.2 经验累积分布函数（ECDF）

**ECDF** 是样本分位数的直接可视化：$\hat{F}(x) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[x_i \le x]$。与理论 CDF 对比，可以直观看到尾部是否偏重——如果样本 ECDF 在极端负值处下降得比正态 CDF 慢，说明左尾更厚。

### 4.4.3 QQ 图：最直观的厚尾诊断

<figure markdown>
  ![图4-2四只股票正态 QQ 图：尾部点偏离对角线即厚尾](../assets/figures/ch04_qq.png){ width="680" }
  <figcaption>图4-2四只股票正态 QQ 图：尾部点偏离对角线即厚尾</figcaption>
</figure>


**QQ 图（Quantile-Quantile Plot）**将样本的经验分位数（纵轴）与理论正态分位数（横轴）配对作图。若数据真的是正态的，点应落在对角线上。厚尾的典型特征是：

- 左下角点向左下方偏离（左尾比正态更重）
- 右上角点向右上方偏离（右尾比正态更重）
- 形成“S 形向外翻翘”的 sigmoid 状

!!! warning "QQ 图的常见误读"
    QQ 图只看形状，不看纵轴绝对值大小。四只股票的 QQ 图应并排展示，方便横向比较哪只股票尾部更厚。

---

## 4.5 金融数据的风格化事实

实证金融文献（Cont 2001; Mandelbrot 1963）反复验证了以下五大规律，称为**风格化事实（stylized facts）**。它们是建模的出发点，也是模型好坏的评判标准。

### 4.5.1 厚尾（Fat Tails）

<figure markdown>
  ![图4-3　TECH 日收益分布 vs 正态分布：两端更厚](../assets/figures/ch04_return_hist.png){ width="680" }
  <figcaption>图4-3　TECH 日收益分布 vs 正态分布：两端更厚</figcaption>
</figure>


日收益率的超额峰度 $K \gg 0$，意味着极端涨跌发生的概率远高于正态分布。

**数字检验**：

- 正态分布：超出 $\pm 3\sigma$ 的概率约0.27%，对应750个交易日约2次。
- 若样本中超过3倍标准差的天数远多于2次，则厚尾有统计意义。

**实务含义**：用正态假设计算的 VaR 会系统性地**低估**极端风险（“尾风险”）。2008年金融危机中，许多基于正态分布的风控模型失效，正是因为忽视了厚尾。

### 4.5.2 波动率聚集（Volatility Clustering）

<figure markdown>
  ![图4-4收益近似不相关，但其绝对值有持续自相关](../assets/figures/ch04_vol_clustering.png){ width="680" }
  <figcaption>图4-4收益近似不相关，但其绝对值有持续自相关</figcaption>
</figure>


Mandelbrot（1963）最早观察到：“大变动往往跟着大变动，小变动往往跟着小变动，但方向不定。”

**自相关验证**：

- $r_t$ 本身的自相关系数（ACF）接近0，说明收益方向难以预测；
- $|r_t|$（或 $r_t^2$）的自相关系数显著且缓慢衰减，说明**波动率的幅度**有持续性。

```python
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

r = rets["TECH"]
lb_ret = acorr_ljungbox(r, lags=[5, 10, 20], return_df=True)
lb_abs = acorr_ljungbox(r.abs(), lags=[5, 10, 20], return_df=True)
print("收益率 Ljung-Box:\n", lb_ret)
print("\n绝对值 Ljung-Box:\n", lb_abs)
```

**Ljung-Box 检验**的原假设是“序列无自相关”，p 值越小越有证据拒绝。通常收益率的 p 值较大（无法拒绝不相关），而绝对值的 p 值接近0（强烈拒绝不相关，即有波动率聚集）。

### 4.5.3 杠杆效应（Leverage Effect）

股票价格下跌会提高公司的财务杠杆比率（债务/权益上升），从而增加公司风险，导致后续波动率上升。**简单验证**：把历史区间分为“大跌日后”和“大涨日后”，比较接下来 $N$ 日的已实现波动率：

```python
threshold = rets["TECH"].quantile(0.2)  # 前20%最差收益
mask_down = rets["TECH"] < threshold
mask_up   = rets["TECH"] > -threshold

vol_after_down = rets["TECH"].shift(-1)[mask_down].abs().mean()
vol_after_up   = rets["TECH"].shift(-1)[mask_up].abs().mean()
print(f"大跌后次日平均波动：{vol_after_down:.4f}")
print(f"大涨后次日平均波动：{vol_after_up:.4f}")
```

### 4.5.4 聚合高斯性（Aggregational Gaussianity）

**单日**收益率明显偏离正态（厚尾、负偏），但随着时间聚合（周、月），分布逐渐向正态靠拢。这是中心极限定理的直觉延伸，但要注意聚合并不能完全消除尾部风险——月度收益仍比正态厚。

```python
# 不同频率的超额峰度比较
for freq, label in [("D", "日度"), ("W", "周度"), ("ME", "月度")]:
    agg = prices.resample(freq).last().pct_change().dropna()
    kurt = agg.mean(axis=1).kurtosis()
    print(f"{label}超额峰度: {kurt:.3f}")
```

### 4.5.5 收益近似不相关但非独立

收益率序列的自相关通常极弱（接近市场有效性），但这**不等于**独立。即使 $\text{Corr}(r_t, r_{t-k})\approx 0$，我们仍可观察到 $\text{Corr}(r_t^2, r_{t-k}^2) \gg 0$（波动率聚集）。

这一区别在建模中极为重要：ARMA 模型针对线性相关，而 GARCH 模型专门刻画高阶（非线性）相依。

### 4.5.6 风格化事实的检验方法速查

前面五条风格化事实并非空泛的定性描述，每一条都对应着可落地的量化检验。把它们整理成一张对照表，方便在拿到一份新数据时按图索骥、逐条体检：

| 风格化事实 | 检验对象 | 常用方法 | 典型判读 |
|:---|:---|:---|:---|
| 厚尾 | $r_t$ 的四阶矩 / 尾部 | 超额峰度、QQ 图、Hill 估计、$\pm3\sigma$ 越界计数 | 超额峰度 $\gg 0$；QQ 图两端外翘 |
| 波动率聚集 | $|r_t|$、$r_t^2$ 的自相关 | $|r_t|$ 的 ACF、Ljung-Box | $r_t$ 自相关弱而 $|r_t|$ 自相关强且缓慢衰减 |
| 杠杆效应 | 收益符号与后续波动 | 收益—波动的不对称相关、EGARCH 系数 | 大跌后波动上升幅度大于大涨后 |
| 聚合高斯性 | 不同频率收益的形状 | 日/周/月超额峰度对比、JB 检验 | 频率越低峰度越小，向正态靠拢 |
| 近似不相关但非独立 | $r_t$ 与 $r_t^2$ 的自相关对比 | $r_t$ 的 ACF/Ljung-Box vs $r_t^2$ 的 ACF | $\rho(r_t)\approx 0$ 但 $\rho(r_t^2)\gg 0$ |

!!! example "例4.3：A股2015年股灾——厚尾不是教科书的修辞"
    以沪深300为背景考察2015年6月至9月的“去杠杆股灾”。在指数常态波动下，日波动率大致在 $1.5\%$ 量级；若硬套正态分布，单日跌幅超过 $3\sigma\approx 4.5\%$ 的概率约 $0.27\%$，意味着**一年（约244个交易日）平均还不到一次**。

    然而在那轮急跌中，沪深300在两三个月内出现了多个单日跌幅超过 $5\%$、甚至逼近或触及 $-8\%$ 的交易日——若按正态分布，单日跌幅超过 $5\sigma$（约 $-7.5\%$）的理论概率约为 $3\times10^{-7}$，即**平均上万年才出现一次**的“奇迹”，却在短短几周内反复上演。

    **判读：** 这正是厚尾的现实代价。用偏度约 $-1$、超额峰度高达 $7\sim10$ 的真实分布去看，这些极端日不过是“尾部常客”；而一旦风控模型误用正态假设，对尾部风险的低估就会以数量级计——这也是本章反复强调**“对金融收益率，正态假设是危险的默认值”**的原因。

---

## 4.6 正态性检验

### 4.6.1 Jarque-Bera 检验

Jarque-Bera（JB）检验利用偏度 $S$ 和超额峰度 $K$ 构造统计量：

$$JB = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right) \xrightarrow{H_0} \chi^2(2)$$

在正态原假设下，$JB \sim \chi^2(2)$。若 p 值 $< 0.05$，则在5% 显著性水平下拒绝正态性。

这条公式并非凭空而来，它的结构恰好把“偏度”和“峰度”两条偏离正态的证据**加权合并**成一个数。其来历可以这样理解：在正态分布下，样本偏度 $\hat S$ 与样本超额峰度 $\hat K$ 在大样本下都渐近服从正态，且

$$\hat S \xrightarrow{d} N\!\left(0,\ \tfrac{6}{n}\right),\qquad \hat K \xrightarrow{d} N\!\left(0,\ \tfrac{24}{n}\right).$$

把每一项除以各自的标准差再平方，就得到两个渐近服从 $\chi^2(1)$ 的标准化平方项：

$$\frac{\hat S^2}{6/n}=\frac{n}{6}\hat S^2,\qquad \frac{\hat K^2}{24/n}=\frac{n}{24}\hat K^2=\frac{n}{6}\cdot\frac{\hat K^2}{4}.$$

由于在正态下 $\hat S$ 与 $\hat K$ 渐近独立，两个独立的 $\chi^2(1)$ 相加即得自由度为2的卡方分布，于是

$$JB=\frac{n}{6}\hat S^2+\frac{n}{6}\cdot\frac{\hat K^2}{4}=\frac{n}{6}\!\left(\hat S^2+\frac{\hat K^2}{4}\right)\xrightarrow{H_0}\chi^2(2).$$

由此可见 JB 检验的本质是：**偏度不为0、或峰度不为3，任何一条都会推高统计量**，两条同时偏离时证据叠加。也正因为建立在偏度、峰度的渐近正态性上，JB 是一个**大样本检验**——样本太小时近似很差，这也是它在金融日频场景下大行其道、却不适合 $n<30$ 小样本的原因。$\chi^2(2)$ 在 $0.05$ 水平的临界值约 $5.99$，于是有个心算口诀：**JB 超过6就该警惕正态性，超过10基本可以拒绝。**

!!! example "例4.4：手算 Jarque-Bera 统计量并判读"
    沿用例4.1的小样本 $r=(-3,-1,0,1,8)$，我们已经算出（分母取 $n=5$）：偏度 $\hat S\approx 1.03$、超额峰度 $\hat K\approx -0.272$。代入 JB 公式：

    $JB=\frac{n}{6}\!\left(\hat S^2+\frac{\hat K^2}{4}\right)=\frac{5}{6}\!\left(1.03^2+\frac{(-0.272)^2}{4}\right).$

    分步算：$1.03^2\approx 1.061$；$\frac{0.272^2}{4}=\frac{0.0740}{4}\approx 0.0185$；括号内合计 $\approx 1.079$；再乘 $\frac{5}{6}\approx 0.833$，得

    $JB\approx 0.833\times 1.079\approx 0.90.$

    **查表判读：** 与 $\chi^2(2)$ 的 $0.05$ 临界值 $5.99$ 相比，$0.90\ll 5.99$，对应 $p\approx 0.64$，**远不能拒绝正态性**。

    这个结论看似与“金融数据非正态”矛盾，其实恰恰说明了 JB 的局限：**$n=5$ 太小，检验功效（power）极低**，即便数据真有不对称也察觉不出来。把同样形状的数据放大到 $n=2000$（即每个观测重复400次），统计量会近似放大400倍，达到 $JB\approx 360$，$p$ 值小到 $10^{-78}$ 量级——同一个“形状”，在大样本下立刻被铁证如山地拒绝。这正是“样本量决定检验功效”的生动注脚。

```python
from scipy.stats import jarque_bera

for col in rets.columns:
    stat, p = jarque_bera(rets[col])
    print(f"{col}: JB={stat:.1f}, p={p:.2e}")
```

!!! tip "如何读 p 值"
    - $p < 0.001$：极强证据反对原假设（正态性）
    - $0.001 \le p < 0.05$：有证据反对
    - $p \ge 0.05$：没有足够证据反对（注意：不等于“接受正态”）

    金融日度收益率的 JB 检验几乎总是给出 $p \ll 0.001$，即强烈拒绝正态性。

### 4.6.2 Shapiro-Wilk 检验（了解）

Shapiro-Wilk 检验对**小样本**更敏感，适合 $n < 50$ 的场合，但当样本量较大（如 $n > 5000$）时，任何微小偏离正态都会导致拒绝，实用意义降低。金融日频数据样本量通常在百到千级别，两种检验结论一致，JB 更常用。

---

## 4.7 自相关分析

### 4.7.1 自相关函数（ACF）

**自相关函数（ACF）**衡量序列与其自身滞后版本之间的线性相关程度：

$$\rho_k = \frac{\text{Cov}(r_t, r_{t-k})}{\text{Var}(r_t)} = \frac{\sum_{t=k+1}^{T}(r_t - \bar{r})(r_{t-k}-\bar{r})}{\sum_{t=1}^{T}(r_t-\bar{r})^2}$$

在大样本零自相关的原假设下，$\hat{\rho}_k \approx N(0, 1/T)$，因此 $\pm 1.96/\sqrt{T}$ 是95% 置信区间的参考线（虚线）。

### 4.7.2 Ljung-Box 检验

Ljung-Box 检验同时检验前 $m$ 阶自相关是否**联合**为零：

$$Q_{LB}(m) = T(T+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{T-k} \xrightarrow{H_0} \chi^2(m)$$

使用 statsmodels 实现：

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(rets["TECH"].abs(), lags=range(1, 21), return_df=True)
lb[["lb_stat", "lb_pvalue"]].head(10)
```

### 4.7.3 怎样读一张 ACF 图

ACF 图（也叫“相关图”，correlogram）的横轴是滞后阶数 $k$，纵轴是自相关系数 $\hat\rho_k$，每个滞后画一根竖线（stem），并叠加一对对称的虚线作为**显著性边界**。读图的关键就三件事：哪些竖线**穿出**虚线、穿出的竖线**衰减得快不快**、以及它们出现在**哪些滞后**上。

那对虚线的高度来自4.7.1的结论：在“无自相关”原假设下 $\hat\rho_k\approx N(0,1/T)$，故 $95\%$ 置信带为 $\pm 1.96/\sqrt{T}$。**样本越大，带越窄**：$T=750$ 时带宽约 $\pm 1.96/\sqrt{750}\approx \pm 0.072$；$T=250$ 时放宽到约 $\pm 0.124$。所以同一根竖线“显不显著”，与样本量直接相关——这也是为什么不能脱离 $T$ 空谈“这个相关高不高”。

!!! example "例4.5：解读一张 ACF 图——哪些滞后显著"
    设某收益绝对值序列 $|r_t|$ 共 $T=400$ 个观测，前8阶样本自相关系数如下：

    | 滞后 $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | $\hat\rho_k$ | $0.21$ | $0.15$ | $0.12$ | $0.09$ | $0.04$ | $0.11$ | $0.02$ | $-0.01$ |

    **第一步：算显著性边界。** $T=400\Rightarrow \pm 1.96/\sqrt{400}=\pm 1.96/20=\pm 0.098$。凡 $|\hat\rho_k|>0.098$ 的滞后即在 $5\%$ 水平下显著。

    **第二步：逐根比对。** 滞后1（$0.21$）、2（$0.15$）、3（$0.12$）、6（$0.11$）四根穿出了边界，**显著**；滞后4（$0.09$）刚好压在边界内侧，**临界、视作不显著**；滞后5、7、8都在带内，**不显著**。

    **第三步：判读形态。** 低阶（1、2、3）自相关为正且**随滞后单调衰减**——这正是波动率聚集的标准签名：今天波动大，明天、后天大概率仍偏大，但记忆随时间慢慢淡去。滞后6那根孤零零穿出的竖线值得多看一眼：单个高阶滞后偶然超界，在多重比较下很可能是**假阳性**（按 $5\%$ 水平，每20根里平均就有1根会偶然越界）；除非它落在有业务含义的周期上（如周度数据的滞后5、月度数据的滞后21），否则不必当真。

    **结论：** 该序列存在显著且缓慢衰减的低阶正自相关，符合“$|r_t|$ 有波动率聚集”的预期；若把同一读法用在 $r_t$ 本身上，通常各阶都落在带内（近似不相关），二者对照正是4.5.5那条风格化事实的可视化证据。

---

## 4.8 相关性分析

### 4.8.1 Pearson vs Spearman 相关系数

| 指标 | 衡量 | 适用场景 | 局限 |
|:---|:---|:---|:---|
| Pearson $\rho$ | 线性相关 | 两个变量呈线性关系时最优 | 对极端值敏感；无法捕捉非线性关系 |
| Spearman $r_s$ | 秩相关（单调关系） | 分布厚尾或存在异常值时更稳健 | 丢失量的信息，只保留排名 |

对于金融收益率（厚尾、偶有极端值），**Spearman 相关系数**通常比 Pearson 更稳健。两者差异大时，说明极端值对线性相关有较大影响。

```python
pearson = rets.corr(method="pearson")
spearman = rets.corr(method="spearman")
```

### 4.8.2 相关矩阵与热力图

<figure markdown>
  ![图4-5四只股票日收益相关性热力图](../assets/figures/ch04_corr_heatmap.png){ width="680" }
  <figcaption>图4-5四只股票日收益相关性热力图</figcaption>
</figure>


多资产投资组合的核心输入之一是**相关矩阵**：

- 对角线恒为1；
- 矩阵对称；
- 所有特征值 $\ge 0$（半正定）；
- 相关系数越接近1，分散化效果越差。

热力图（heatmap）是相关矩阵的标准可视化方式，用颜色深浅表示相关强弱：

```python
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = rets.corr(method=method)
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1,
                ax=ax, fmt=".2f")
    ax.set_title(f"{method.capitalize()} 相关系数")
plt.tight_layout()
```

### 4.8.3 滚动相关系数（相关性时变）

!!! tip "危机同跌与分散化失效"
    在市场平稳期，不同行业股票相关性可能较低（分散化有效）。但在市场危机（大跌、流动性危机）时，各资产相关性往往骤升，接近1——分散化几乎失效。这被称为**相关性时变（time-varying correlation）**，是多因子和动态风险模型的研究热点。

```python
roll_corr = rets["BANK"].rolling(60).corr(rets["TECH"])
roll_corr.plot(figsize=(10, 4), title="BANK 与 TECH 的 60 日滚动相关系数")
```

### 4.8.4 算例深化：异常值、秩相关与时变相关

4.8.1的对照表说 Spearman 比 Pearson 稳健，4.8.3说危机期相关性会骤升。这两点都值得用具体数字坐实，否则只是口号。下面三个小算例分别回答：异常值能把 Pearson 扭曲到什么程度、Spearman 为何不为所动、以及“危机同跌”在数字上长什么样。

!!! example "例4.6：一个异常值如何颠倒 Pearson 与 Spearman 的结论"
    设两只股票配对的收益率（%）有6个观测，前5对呈清晰的正向同步：

    $X=(1,\ 2,\ 3,\ 4,\ 5,\ 6),\qquad Y=(1,\ 2,\ 3,\ 4,\ 5,\ -20).$

    前5对 $X$ 与 $Y$ 完全相等（完美正相关），只有第6对里 $Y$ 突然变成一根 $-20$ 的暴跌（比如该股当天单独爆雷）。

    **Spearman（秩相关）怎么看？** 把数值换成排名：$X$ 的秩是 $(1,2,3,4,5,6)$；$Y$ 的秩里 $-20$ 最小排第1，其余依次后移，得 $(2,3,4,5,6,1)$。秩之差 $d=(−1,−1,−1,−1,−1,5)$，$\sum d^2=5\times1+25=30$。代入 Spearman 公式

    $r_s=1-\frac{6\sum d^2}{n(n^2-1)}=1-\frac{6\times 30}{6\times 35}=1-\frac{180}{210}\approx 0.143.$

    **Pearson 怎么看？** 那个 $-20$ 把 $Y$ 的均值、离差全部带偏，逐项算下来 $\rho\approx -0.62$。

    **判读：** 同一份数据，Pearson 报出一个**中等强度的负相关**（$-0.62$），Spearman 却只给出一个**很弱的正相关**（$+0.14$）。哪个更可信？这两只股票6天里有5天亦步亦趋，只是末日一只单独爆雷——把它叫“负相关资产、可用来对冲”显然是危险的误判。**当 Pearson 与 Spearman 符号相反或差距悬殊时，几乎一定是少数极端值在作祟**，此时应优先信任秩相关，并回头核查那几个离群点。

!!! example "例4.7：滚动相关——平静期与危机期的数字对照"
    设两只行业 ETF 在“平静期”和“危机期”各取5个交易日的日收益率（%）。

    **平静期**（各自随行业逻辑独立波动）：

    $A=(0.3,\ -0.2,\ 0.5,\ -0.1,\ 0.0),\qquad B=(-0.1,\ 0.4,\ -0.2,\ 0.3,\ -0.4).$

    粗略计算其 Pearson 相关约为 $-0.6$ 量级——两者方向时常相反，分散化“看起来”很有效。

    **危机期**（系统性恐慌，泥沙俱下）：

    $A'=(-2.1,\ -1.8,\ -3.0,\ -0.9,\ -2.4),\qquad B'=(-1.9,\ -2.2,\ -2.7,\ -1.1,\ -2.0).$

    此时两列同向、同步、且都为大负值，Pearson 相关跃升到约 $+0.9$ 以上。

    **判读：** 把窗口沿时间滑过这两段，滚动相关系数会从平静期的负值/低值，在进入危机段后**快速抬升并逼近1**。这就是4.8.3所说“相关性时变”的微观机制：恐慌之下，**驱动各资产的不再是行业基本面，而是同一个流动性/情绪因子**，于是原本各行其是的资产被“焊”在了一起。

!!! example "例4.8：A股案例——危机同跌如何让分散化失效"
    考虑一个朴素的“分散化”组合：白酒 + 银行 + 科技 + 公用事业，等权各 $25\%$。在2017—2019的多数时段，这几个板块相关性中等（约 $0.3\sim0.5$），组合波动率明显低于任一单板块，分散化红利实实在在。

    可一旦进入系统性风险事件——例如2018年中美贸易摩擦升级、2020年2月新冠冲击下的开盘“千股跌停”、或2015年去杠杆股灾——板块间相关性会在数日内集体抬升到 $0.8\sim0.95$。等权组合的方差近似为

    $\sigma_p^2=\frac{1}{n}\bar\sigma^2+\frac{n-1}{n}\,\overline{\rho}\,\bar\sigma^2,$

    其中第二项随平均相关 $\overline{\rho}$ 线性放大。当 $\overline{\rho}$ 从 $0.4$ 跳到 $0.9$，组合方差里“无法被分散掉”的系统性部分几乎翻倍，**组合在最需要保护的时刻反而和单一资产一起跳水**。

    **判读：** 这就是“分散化在危机中失效”的定量根源——它失效得恰好不是时候。实务上的应对是用**动态相关（DCC-GARCH）**而非静态历史相关度量风险，并对“尾部相关性”单独建模，而不能只看平静期那张漂亮的相关热力图。

---

## 4.9 金融专用可视化

### 4.9.1 累计净值曲线

<figure markdown>
  ![累计净值](../assets/figures/ch01_nav.png){ width="680" }
  <figcaption>图4-6四只示例股票累计净值（初始 = 1）</figcaption>
</figure>


将初始投资额设为1，每日复利增长得到**累计净值（cumulative NAV）**：

$$\text{NAV}_t = \prod_{\tau=1}^{t}(1 + r_\tau)$$

这是比较不同策略/股票长期表现的最直观方式。

```python
nav = (1 + rets).cumprod()
nav.plot(figsize=(10, 5), title="四只股票累计净值（初始=1）")
```

### 4.9.2 回撤曲线

**最大回撤（Maximum Drawdown, MDD）**衡量从历史最高点到随后最低点的最大跌幅：

$$\text{DD}_t = \frac{\text{NAV}_t}{\max_{\tau \le t}\text{NAV}_\tau} - 1$$

$$\text{MDD} = \min_t \text{DD}_t$$

回撤永远 $\le 0$，越接近0表示回撤越小，策略越稳健。

```python
from fds import max_drawdown

for col in rets.columns:
    mdd = max_drawdown(rets[col])
    print(f"{col} 最大回撤：{mdd:.2%}")
```

### 4.9.3 按年收益分布（箱线图）

箱线图（boxplot）以年为单位展示每年日度收益率的分布，直观反映**年际间波动规律的差异**：

```python
rets_with_year = rets.copy()
rets_with_year["year"] = rets.index.year
rets_with_year.boxplot(column="TECH", by="year", figsize=(10, 5))
```

### 4.9.4 K 线图（蜡烛图，简介）

K 线图（Candlestick Chart）是最传统的金融图表，每根蜡烛包含四个价格信息：开盘价、收盘价、最高价、最低价。在 Python 中可使用 **mplfinance** 库绘制：

```python
# 需要 OHLCV 格式的数据
# import mplfinance as mpf
# mpf.plot(ohlcv_df, type="candle", style="charles", title="K 线图示例")
```

!!! info "mplfinance 非必装依赖"
    本书的内置数据为收盘价格，不含日内开高低数据，故 K 线图不在本章强制演示范围内。有兴趣的读者可通过 `uv add mplfinance` 安装后自行探索。

---

## 4.10 中文绘图规范

制作面向中国金融业的图表，有以下规范需要特别注意：

### 4.10.1 中文字体

```python
from fds import set_chinese_font
set_chinese_font()   # 每章 示例开头调用一次
```

`set_chinese_font()` 会自动按系统检测合适的 CJK 字体（Windows 下优先使用微软雅黑或黑体），并修正负号显示问题（`axes.unicode_minus = False`）。

### 4.10.2 涨跌配色

**中国 A 股**：红色代表上涨，绿色代表下跌（“红涨绿跌”）。

**国际市场（欧美）**：绿色代表上涨，红色代表下跌（“绿涨红跌”）。

在面向中国读者的图表中，务必遵循 A 股习惯，否则容易引起混淆：

```python
RISE_COLOR = "#e03c31"   # 红色（A股涨）
FALL_COLOR = "#00a651"   # 绿色（A股跌）
```

### 4.10.3 多图共享坐标轴

对比多只股票时，应统一纵轴范围，否则视觉欺骗读者：

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, col in zip(axes.ravel(), rets.columns):
    rets[col].plot.hist(bins=50, ax=ax, density=True)
    ax.set_title(col)
plt.tight_layout()
```

### 4.10.4 标注规范

每张图必须具备：

1. **标题**（`ax.set_title()`）：简洁说明图表内容；
2. **轴标签**（`ax.set_xlabel()`, `ax.set_ylabel()`）：含单位；
3. **图例**（`ax.legend()`）：多条线时必须有图例；
4. **数据来源**（必要时）：注释在图下方。

---

## 4.11 本章小结

本章的核心目标，是把“看数据”从直觉提升为有规范的探索过程。学完后，建议按以下三层来掌握：

**必须掌握**

1. **分布特征**：日收益均值通常接近0，但波动、负偏与高峰度共同决定真实风险。
2. **风格化事实**：厚尾、波动率聚集、杠杆效应和相关性上升，是金融数据 EDA 中最常见的四个结论。
3. **诊断工具**：直方图、QQ 图、ACF、滚动相关和 Ljung-Box / JB 等检验，是把“看起来像”变成“有证据支持”的关键工具。

**理解即可**

4. **聚合高斯性**：采样频率降低后，收益分布会更接近正态，但不会自动完全变成正态。
5. **相关性差异**：Pearson 更敏感于极端值，Spearman 在金融数据中往往更稳健。

**实践提醒**

EDA 不是画几张图就结束，而是为了给后续建模提供约束：如果你已经看到厚尾和波动率聚集，就不该再默认正态和同方差。

---

## 4.12 习题

!!! note "使用建议"
    建议按“分布诊断 → 时序特征 → 相关结构”顺序完成本章习题。若是本科主线课程，可重点完成 1、2、3、5 题；第 4 题更适合小组讨论或行业解释训练。

### 分布诊断

**习题4.1**（描述统计）：对四只股票分别计算日度均值、年化波动率、偏度、超额峰度，整合成一张汇总表格。哪只股票的超额峰度最大？这对投资者意味着什么？

> 参考思路：使用 `rets.describe()`、`rets.skew()`、`rets.kurtosis()`，年化波动率 = 日度标准差 × $\sqrt{252}$。

**习题4.2**（厚尾诊断）：对四只股票分别画正态 QQ 图，并在同一图中叠加对角线参考线。根据图形判断哪只股票厚尾最显著，并用超额峰度数值验证。

> 参考思路：用 `scipy.stats.probplot`；QQ 图越“向外翘”，峰度越大。

### 时序特征

**习题4.3**（自相关分析）：选取 `LIQUOR`，分别画收益率与收益率绝对值的 ACF 图（滞后1~30阶），并对两者做 Ljung-Box 检验（取 $m=10$）。解释为何 p 值差异如此之大。

> 参考思路：`acf` 函数来自 `statsmodels.tsa.stattools`；ACF 图中超过虚线（置信区间）的竖线越多，序列相关越显著。

### 相关结构

**习题4.4**（滚动相关）：计算所有六对股票两两之间的90日滚动相关系数，并用4×1或3×2子图展示。找出相关性最稳定的一对和波动最大的一对，尝试从行业属性解释。

> 参考思路：`rets.rolling(90).corr()` 返回一个 MultiIndex DataFrame，需要提取特定列对。

**习题4.5**（聚合高斯性）：对 `TECH` 分别计算日度、周度、月度收益率，分别绘制直方图+KDE+正态曲线的对比图，并计算各频率的超额峰度。验证“聚合频率越低，分布越接近正态”这一规律是否在你的数据中成立。

> 参考思路：使用 `prices.resample("W").last().pct_change().dropna()` 获得周度收益；注意样本量会相应减少。

---

## 4.13 拓展阅读

- **Cont, R. (2001).** “Empirical properties of asset returns: stylized facts and statistical issues.” *Quantitative Finance*, 1(2), 223–236. — 风格化事实的经典综述，必读。
- **Mandelbrot, B. (1963).** “The variation of certain speculative prices.” *Journal of Business*, 36(4), 394–419. — 首次系统描述厚尾与波动率聚集的里程碑论文。
- **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997).** *The Econometrics of Financial Markets*. — 金融计量经典教材，第1章有完整的风格化事实讨论。
- **Tsay, R. S. (2010).** *Analysis of Financial Time Series* (3rd ed.). — 中文金融时间序列分析最常用的参考书之一。
- **statsmodels 文档**：[https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html) — ACF、Ljung-Box、单位根检验等。


