# 第6章 金融时间序列分析

[![在 Colab 打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertandking/financial-data-science/blob/main/notebooks/ch06_time_series.ipynb) [![在 Binder 打开](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/albertandking/financial-data-science/main?labpath=notebooks/ch06_time_series.ipynb)

!!! info "配套代码"
    `notebooks/ch06_time_series.ipynb`（使用 statsmodels 与 arch 库）。运行前请先执行：
    ```bash
    uv run python scripts/make_sample_data.py
    ```

---

## 6.1 本章导读

**“预测波动率比预测收益更可行。”**

这句金融从业者之间流传的行话，深刻揭示了时间序列分析在金融领域的核心矛盾：股票价格（和收益率）的均值几乎无法预测，但其**波动率**却呈现出明显的持续性与聚集性——大幅波动之后往往跟随大幅波动，平静之后往往跟随平静。

以A股市场为例：2015年股灾期间，沪深300指数在短短两个月内暴跌超过40%，期间日波动率从平均1.5%飙升至5%以上，并在此后数月维持高位。如果我们的模型能提前识别这一“波动率切换”，就能在风险暴露、期权定价和止损策略上做出更好的决策。

本章将系统介绍金融时间序列分析的核心方法，从理论出发，结合A股数据实战，帮助读者建立从**数据探索→平稳性检验→均值方程建模→波动率建模→样本外预测**的完整分析框架。

### 6.1.1 学习目标

完成本章学习后，你将能够：

1. 理解严平稳与宽平稳的概念，解释为什么金融建模需要平稳序列；
2. 使用 ADF 检验和 KPSS 检验判断序列是否存在单位根；
3. 识别并解读 ACF/PACF 图，用于 ARMA 模型定阶；
4. 构建 ARIMA(p,d,q) 模型，使用 AIC/BIC 选择最优阶数并进行残差诊断；
5. 理解 ARCH 效应和波动率聚集，构建 GARCH(1,1) 模型；
6. 实施滚动窗口预测，正确评估样本外预测精度，避免前视偏差。

---

## 6.2 平稳性：金融建模的基础前提

### 6.2.1 严平稳与宽平稳

**严平稳（Strict Stationarity）**要求序列的联合概率分布对任意时间平移不变：

$$
(X_{t_1}, X_{t_2}, \ldots, X_{t_k}) \overset{d}{=} (X_{t_1+h}, X_{t_2+h}, \ldots, X_{t_k+h}), \quad \forall k, h
$$

这一条件在实际中难以验证，通常退而求其次，使用**宽平稳（Wide-Sense Stationarity）**，要求：

| 条件 | 数学表达 |
|------|---------|
| 均值不随时间变化 | $E[X_t] = \mu$，常数 |
| 方差有限且不随时间变化 | $\text{Var}(X_t) = \sigma^2 < \infty$ |
| 自协方差只依赖时间间隔 | $\text{Cov}(X_t, X_{t-h}) = \gamma(h)$，仅是 $h$ 的函数 |

!!! note "严平稳 vs 宽平稳"
    - 若序列服从正态分布，则严平稳 ↔ 宽平稳（正态由均值和方差完全刻画）。
    - 对于非正态分布（如金融收益的厚尾分布），宽平稳不能推出严平稳，但实践中宽平稳已足够。
    - 本章提及的“平稳”，除非特别说明，均指**宽平稳**。

### 6.2.2 为何金融建模需要平稳序列

| 序列特征 | 问题后果 | 举例 |
|---------|---------|------|
| 非平稳均值（趋势） | OLS 估计有偏，$t$统计量失效 | 股价随时间上涨 |
| 非平稳方差（异方差） | 标准误估计不可靠 | 波动率聚集 |
| 两序列均有趋势 | **伪回归**（$R^2$ 高但实际无关） | 用GDP回归沪指 |

!!! warning "伪回归陷阱"
    对两个独立随机游走做回归，即使二者毫无关联，样本 $R^2$ 可高达0.9，$t$值显著。这是非平稳数据最常见的陷阱，Granger & Newbold (1974) 最早系统揭示这一问题。

### 6.2.3 价格 vs 收益率：哪个更接近平稳？

以A股TECH板块为例：

- **股价**：具有明显的上升趋势，均值随时间递增 → **非平稳**
- **对数收益率** $r_t = \ln(P_t/P_{t-1})$：均值接近零，方差相对稳定 → **近似平稳**（但方差有时聚集）

$$
P_t = P_{t-1} \cdot e^{r_t} \implies \ln P_t = \ln P_{t-1} + r_t
$$

对数价格是收益率的随机游走（单位根过程），对其做一阶差分即得到（对数）收益率。

---

## 6.3 单位根与平稳性检验

### 6.3.1 随机游走与单位根

最简单的非平稳过程是**随机游走**（Random Walk）：

$$
P_t = P_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)
$$

写成 AR(1) 形式 $P_t = \phi P_{t-1} + \varepsilon_t$，随机游走对应 $\phi = 1$，即“单位根”。

- 若 $|\phi| < 1$：序列均值回复，宽平稳；
- 若 $\phi = 1$：单位根，非平稳，方差随时间线性增长；
- 若 $\phi > 1$：爆炸性，极少在金融中出现。

### 6.3.2 ADF 检验（Augmented Dickey-Fuller）

**原假设 $H_0$**：序列存在单位根（非平稳）  
**备择假设 $H_1$**：序列平稳

ADF 检验通过以下辅助回归实现：

$$
\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{j=1}^{p} \delta_j \Delta X_{t-j} + \varepsilon_t
$$

检验统计量：$\tau = \hat{\gamma} / \text{se}(\hat{\gamma})$，临界值来自 Dickey-Fuller 分布（非正态）。

| 结论 | 条件 |
|------|------|
| 拒绝 $H_0$（平稳） | $p$ 值 $< 0.05$（通常取5%显著性） |
| 不拒绝 $H_0$（非平稳） | $p$ 值 $\geq 0.05$ |

!!! tip "ADF 检验的局限"
    - ADF 检验**低功效**（容易漏检平稳序列），短样本尤为明显。
    - 检验结果对滞后阶数 $p$ 的选择敏感，通常用 AIC/BIC 自动选阶。
    - 建议**配合 KPSS 检验**一起使用，形成“双向夹击”：
        - ADF 不拒绝 + KPSS 拒绝 → 强证据支持单位根
        - ADF 拒绝 + KPSS 不拒绝 → 强证据支持平稳
        - 两者均拒绝 or 均不拒绝 → 结论模糊，需结合经济理论判断

### 6.3.3 KPSS 检验

**原假设 $H_0$**：序列平稳（与 ADF 相反！）  
**备择假设 $H_1$**：存在单位根

```python
from statsmodels.tsa.stattools import kpss
stat, p_value, lags, crit = kpss(series, regression='c')
# p_value < 0.05 → 拒绝平稳假设 → 非平稳
```

### 6.3.4 差分与过差分

**差分**（Differencing）是消除单位根的标准处理方式：

$$
\Delta X_t = X_t - X_{t-1} \quad (\text{一阶差分})
$$

对股价做一阶差分近似得到收益率。若差分后仍非平稳，可做**二阶差分**。

!!! warning "过差分的危害"
    对本已平稳的序列进行差分，称为**过差分**（Overdifferencing）。过差分会：
    - 引入额外的移动平均成分（MA(1) 不可逆结构）；
    - 损失信息，参数估计效率下降；
    - 使预测区间不必要地扩大。
    
    **规则**：先用 ADF/KPSS 检验判断，确认需要差分再差分，切勿盲目。

---

## 6.4 自相关与偏自相关

### 6.4.1 自相关函数（ACF）

<figure markdown>
  ![图 6-1　TECH 收益率的 ACF 与 PACF](../assets/figures/ch06_acf_pacf.png){ width="680" }
  <figcaption>图 6-1　TECH 收益率的 ACF 与 PACF</figcaption>
</figure>


**自相关系数**（Autocorrelation Function, ACF）衡量序列与其自身滞后值的线性相关性：

$$
\rho(h) = \frac{\text{Cov}(X_t, X_{t-h})}{\text{Var}(X_t)} = \frac{\gamma(h)}{\gamma(0)}
$$

样本估计：
$$
\hat{\rho}(h) = \frac{\sum_{t=h+1}^{T}(X_t - \bar{X})(X_{t-h} - \bar{X})}{\sum_{t=1}^{T}(X_t - \bar{X})^2}
$$

在零假设（白噪声）下，$\hat{\rho}(h) \approx N(0, 1/T)$，**95%置信区间**为 $\pm 1.96/\sqrt{T}$（图中蓝色阴影带）。

### 6.4.2 偏自相关函数（PACF）

**偏自相关系数**（Partial ACF, PACF）衡量在**控制中间滞后**后，$X_t$ 与 $X_{t-h}$ 的直接线性相关：

$$
\phi_{hh} = \text{Corr}(X_t, X_{t-h} \mid X_{t-1}, \ldots, X_{t-h+1})
$$

PACF 通过 Yule-Walker 方程逐步计算，截断特征比 ACF 更清晰。

### 6.4.3 用 ACF/PACF 图定阶

| 模型 | ACF 特征 | PACF 特征 |
|------|---------|----------|
| AR(p) | 拖尾（指数衰减/震荡衰减） | **截尾**（$h > p$ 后急速归零） |
| MA(q) | **截尾**（$h > q$ 后急速归零） | 拖尾 |
| ARMA(p,q) | 拖尾 | 拖尾 |
| 白噪声 | 全部在置信带内 | 全部在置信带内 |

!!! tip "实操经验"
    ACF/PACF 定阶是“工艺”而非“科学”——实践中建议：
    1. 画出 ACF/PACF（滞后到20-40期）；
    2. 识别截尾点和拖尾模式；
    3. 用 AIC/BIC 在候选模型集合中自动选择最优。

---

## 6.5 白噪声与 Ljung-Box 检验

**白噪声**（White Noise, WN）是时间序列分析的基准：均值为零、方差恒定、任意滞后自相关均为零：

$$
\varepsilon_t \sim \text{WN}(0, \sigma^2) \iff E[\varepsilon_t] = 0, \quad \text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \; (t \neq s)
$$

ARMA 模型的残差应当是白噪声。验证方法使用 **Ljung-Box 检验**：

$$
Q(m) = T(T+2) \sum_{h=1}^{m} \frac{\hat{\rho}^2(h)}{T-h} \sim \chi^2(m - p - q)
$$

- $H_0$：前 $m$ 阶自相关均为零（白噪声）
- $p$ 值 $< 0.05$ → 残差仍有序列相关，模型未充分拟合

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print(lb)
```

---

## 6.6 ARIMA 模型族

### 6.6.1 AR(p)：自回归模型

$$
X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t
$$

**直觉**：今天的值依赖于过去 $p$ 天的值。AR(1) 模型中，若 $|\phi_1|$ 接近1，序列表现出强持续性。

**平稳条件**：特征多项式 $1 - \phi_1 z - \cdots - \phi_p z^p = 0$ 的所有根在单位圆外。

### 6.6.2 MA(q)：移动平均模型

$$
X_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
$$

**直觉**：今天的值依赖于过去 $q$ 个随机冲击（残差）的线性组合。MA 模型**恒平稳**，但需要满足**可逆条件**（类似 AR 平稳条件）。

### 6.6.3 ARMA(p,q)

$$
X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}
$$

ARMA 模型结合了 AR 和 MA 两种机制，常用比纯 AR 或纯 MA 更少的参数达到相近的拟合效果。

### 6.6.4 ARIMA(p,d,q)

对于非平稳序列（如股价），先做 $d$ 阶差分使其平稳，再拟合 ARMA(p,q)：

$$
\Delta^d X_t \sim \text{ARMA}(p, q)
$$

其中 $\Delta^d$ 表示 $d$ 次差分算子。完整记法：**ARIMA(p,d,q)**，其中：

- $p$ = 自回归阶数
- $d$ = 差分次数（通常0或1）
- $q$ = 移动平均阶数

!!! info "建模流程（Box-Jenkins 方法论）"
    1. **识别**：画 ACF/PACF，判断差分次数，初步定阶 $(p, q)$；
    2. **估计**：极大似然估计（MLE）参数；
    3. **诊断**：残差应为白噪声（Ljung-Box 检验），QQ图检查正态性；
    4. **预测**：用拟合模型做区间预测。

### 6.6.5 信息准则定阶：AIC 与 BIC

$$
\text{AIC} = -2\ln L + 2k, \quad \text{BIC} = -2\ln L + k\ln T
$$

其中 $L$ 为最大似然值，$k$ 为参数个数，$T$ 为样本量。

| 准则 | 偏好 | 适用场景 |
|------|------|---------|
| AIC | 相对宽松，偏向复杂模型 | 预测为主（允许轻微过拟合） |
| BIC | 对参数惩罚更重，偏向简单模型 | 结构识别为主 |

**最优模型**：在候选阶数范围内，AIC/BIC 最小者。

```python
from statsmodels.tsa.arima.model import ARIMA
best_aic, best_order = np.inf, None
for p in range(4):
    for q in range(4):
        try:
            res = ARIMA(returns, order=(p, 0, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, 0, q)
        except Exception:
            pass
print(f"最优 ARIMA 阶数（按AIC）：{best_order}")
```

---

## 6.7 波动率建模：ARCH 与 GARCH

### 6.7.1 波动率聚集与 ARCH 效应

金融收益率最显著的“程式化事实”之一是**波动率聚集**（Volatility Clustering）：

> 大幅波动之后往往跟随大幅波动，平静之后往往跟随平静。

这意味着收益率的**条件方差**并非常数，而是时变的，这违反了普通 ARMA 模型的同方差假设。Engle (1982) 提出 **ARCH 模型**来刻画这一现象。

**ARCH-LM 检验**（检验是否存在ARCH效应）：

$$
H_0: \text{残差平方序列不存在自相关（无ARCH效应）}
$$

```python
from statsmodels.stats.diagnostic import het_arch
lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=10)
# p_value < 0.05 → 存在ARCH效应
```

### 6.7.2 GARCH(1,1) 模型

<figure markdown>
  ![图 6-2　GARCH(1,1) 条件波动率捕捉波动率聚集](../assets/figures/ch06_garch.png){ width="680" }
  <figcaption>图 6-2　GARCH(1,1) 条件波动率捕捉波动率聚集</figcaption>
</figure>


Bollerslev (1986) 在 ARCH 基础上提出**广义ARCH（GARCH）**模型，最常用的是 GARCH(1,1)：

**均值方程**：
$$
r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \overset{iid}{\sim} N(0,1)
$$

**方差方程**：
$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

参数含义：

| 参数 | 含义 | 约束 |
|------|------|------|
| $\omega > 0$ | 基准波动率（长期方差的贡献） | 正数 |
| $\alpha \geq 0$ | ARCH系数：对上期冲击的反应强度 | 非负 |
| $\beta \geq 0$ | GARCH系数：波动率持续性 | 非负 |
| $\alpha + \beta < 1$ | 平稳性条件 | 严格小于1 |

**长期均衡波动率**（无条件方差）：

$$
\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}
$$

$\alpha + \beta$ 越接近1，波动率冲击持续越久（均值回复越慢）。典型的A股日度数据 $\alpha + \beta \approx 0.93 \sim 0.98$。

!!! info "GARCH(1,1) 经济含义解读"
    - $\alpha$ 大：模型对新信息（昨日冲击）反应灵敏；
    - $\beta$ 大：波动率记忆长，持续性强；
    - $\omega$ 小、$\alpha + \beta$ 接近1：接近单位根（IGARCH），长期预测退化为常数。

### 6.7.3 杠杆效应：EGARCH 与 GJR-GARCH

股票市场中，**下跌比上涨引发更大波动**，称为**杠杆效应（Leverage Effect）**。标准 GARCH 对正负冲击对称，无法捕捉杠杆效应。

**GJR-GARCH（Glosten-Jagannathan-Runkle, 1993）**：

$$
\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{\varepsilon_{t-1}<0}) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

$\gamma > 0$ 表示负冲击引发更大波动（杠杆效应显著）。

**EGARCH（Nelson, 1991）**：

$$
\ln \sigma_t^2 = \omega + \alpha \left(\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}} - \sqrt{2/\pi}\right) + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}} + \beta \ln \sigma_{t-1}^2
$$

$\gamma < 0$ 对应杠杆效应。EGARCH 自然保证 $\sigma_t^2 > 0$（对数变换）。

```python
from arch import arch_model
# GJR-GARCH
am_gjr = arch_model(returns * 100, vol="GARCH", p=1, o=1, q=1)
# EGARCH
am_egarch = arch_model(returns * 100, vol="EGARCH", p=1, q=1)
```

---

## 6.8 样本外预测与评估

### 6.8.1 为何要样本外预测

**样本内（In-sample）**拟合度好不等于**样本外（Out-of-sample）**预测能力强。过度复杂的模型会**过拟合**历史数据，在新数据上表现差。

金融时间序列建模的终极目标是预测，因此必须在严格的样本外评估框架下验证模型。

### 6.8.2 前视偏差（Look-ahead Bias）

!!! warning "前视偏差：最常见的回测错误"
    **前视偏差**（Look-ahead Bias）指在预测时使用了预测时刻**未来才会知道的信息**。常见形式：
    
    - 用整个样本的均值/标准差做归一化，再拆分训练/测试集；
    - 用 $t$ 时刻之后的数据估计模型参数，用于预测 $t$ 时刻；
    - 忽略数据发布延迟（如GDP数据公布比期末晚45天）。
    
    **规则**：在预测时点 $t$，只允许使用 $t$ 及 $t$ 之前的信息。

### 6.8.3 滚动窗口 vs 扩展窗口

| 方法 | 训练集 | 优点 | 缺点 |
|------|--------|------|------|
| **滚动窗口（Rolling）** | 固定长度（如250天），向前滑动 | 捕捉结构变化（参数时变） | 忽略早期信息 |
| **扩展窗口（Expanding）** | 从起点到预测时刻（逐步增大） | 充分利用历史 | 早期估计不稳定，计算量大 |

```python
# 滚动窗口示例（以250天为训练窗口）
window = 250
predictions = []
for i in range(window, len(returns)):
    train = returns.iloc[i - window : i]
    model = ARIMA(train, order=(1, 0, 1)).fit()
    pred = model.forecast(steps=1).iloc[0]
    predictions.append(pred)
```

### 6.8.4 预测评估指标

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}, \quad
\text{MAE} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|
$$

| 指标 | 对异常值敏感度 | 量纲 | 常用场景 |
|------|--------------|------|---------|
| RMSE | 高（平方惩罚大误差） | 与 $y$ 相同 | 一般预测评估 |
| MAE | 低（绝对值惩罚均等） | 与 $y$ 相同 | 鲁棒评估 |
| MAPE | 中（相对误差） | 百分比 | 跨尺度比较 |

---

## 6.9 A股实战：TECH 股收益率全流程分析

本节以内置数据集中 **TECH 股**的日度对数收益率为例，完整演示：ADF 检验 → ARIMA 定阶与拟合 → ARCH-LM 检验 → GARCH(1,1) 建模 → 条件波动率可视化。

### 6.9.1 数据准备与探索性分析

```python
from fds import load_sample_prices, daily_returns, set_chinese_font
prices = load_sample_prices()
returns = daily_returns(prices, log=True)['TECH']
```

**关键统计量**（TECH日度对数收益）：

- 均值约为 $+0.05\%$/天（年化约$+12\%$）
- 标准差约为 $2.1\%$/天（年化约$+33\%$）
- 超额峰度 $\approx 2.1$：明显厚尾
- 偏度 $\approx -0.3$：轻微左偏

### 6.9.2 平稳性检验小结

对 TECH 价格和收益率分别进行 ADF 检验：

| 序列 | ADF 统计量 | p值 | 结论 |
|------|----------|-----|------|
| TECH 价格（对数） | $\approx -1.5$ | $\approx 0.52$ | 不拒绝 $H_0$，非平稳 |
| TECH 收益率 | $\approx -22.5$ | $< 0.001$ | 拒绝 $H_0$，平稳 |

→ 对收益率做ARIMA时，$d=0$（无需差分）。

### 6.9.3 ARIMA 定阶与诊断

通过 AIC 网格搜索（$p \in \{0,1,2,3\}$，$q \in \{0,1,2,3\}$）选出最优模型，常见结果为 **ARIMA(1,0,1)** 或 **ARIMA(1,0,0)**。残差须通过 Ljung-Box 检验（$p>0.05$）。

### 6.9.4 GARCH(1,1) 建模

用 `arch` 库对 ARIMA 残差（或直接对收益率）拟合 GARCH(1,1)：

- 典型参数：$\alpha \approx 0.08$，$\beta \approx 0.88$，$\alpha+\beta \approx 0.96$
- 含义：波动率冲击持续约 $1/(1-0.96) = 25$ 天才半衰减

---

## 6.10 本章小结

| 步骤 | 工具 | 关键判断 |
|------|------|---------|
| 探索性分析 | 时序图、收益分布 | 是否有趋势、聚集、厚尾 |
| 平稳性检验 | ADF、KPSS | $d$ 的选择 |
| 自相关分析 | ACF/PACF | 初步定阶 $p$、$q$ |
| 均值方程建模 | ARIMA + AIC/BIC | 最优阶数；残差 Ljung-Box |
| 波动率检验 | ARCH-LM | 是否需要 GARCH |
| 波动率建模 | GARCH/EGARCH/GJR | 捕捉波动率聚集/杠杆 |
| 样本外评估 | 滚动预测 + RMSE/MAE | 避免前视偏差 |

!!! tip "实践建议"
    1. 对金融收益率序列，**通常不需要差分**（$d=0$）；对价格，$d=1$。
    2. GARCH(1,1) 在绝大多数情况下已足够，更复杂的 EGARCH/GJR 仅在杠杆效应显著时才有改善。
    3. 波动率预测比收益率预测更可靠，实际价值更大（用于期权定价、VaR 估计、仓位管理）。
    4. 始终做样本外验证，小心前视偏差。

---

## 6.11 习题

**习题 6.1** 平稳性判断  
对 `BANK`、`LIQUOR`、`UTILITY` 三只股票的**价格**和**对数收益率**分别做 ADF 检验和 KPSS 检验。整理结果表格，说明哪些序列存在单位根，哪些平稳。根据两种检验的交叉验证结论，判断差分次数 $d$。

> **参考思路**：价格 → ADF 不显著+KPSS 显著 → 单位根；对数收益率 → ADF 显著+KPSS 不显著 → 平稳，$d=0$。

**习题 6.2** ARIMA 定阶  
对 `LIQUOR` 日度对数收益率做 AIC/BIC 网格搜索（$p, q \in \{0,1,2,3\}$），画出 AIC 热力图，选出最优模型并拟合。对最优模型的残差做 Ljung-Box 检验（前10阶），判断残差是否为白噪声。

> **参考思路**：用嵌套循环 + try/except 实现网格搜索，用 `seaborn.heatmap` 画热力图；Ljung-Box $p>0.05$ 为通过。

**习题 6.3** GARCH 建模与杠杆效应  
对 `TECH` 日度对数收益率（×100）分别拟合 GARCH(1,1)、GJR-GARCH(1,1,1) 和 EGARCH(1,1)，比较三种模型的 AIC/BIC。GJR-GARCH 中 $\gamma$ 的符号和显著性说明了什么？

> **参考思路**：`arch_model(..., vol="GARCH")` / `vol="GARCH", o=1` / `vol="EGARCH"`；$\gamma>0$ 且显著 → 杠杆效应存在，下跌引发更大波动。

**习题 6.4** 滚动窗口预测  
对 `BANK` 对数收益率做滚动窗口（窗口250天）的一步 ARIMA(1,0,1) 预测（预测最后150天），计算 RMSE 和 MAE，与**随机游走基准**（预测值=前一日收益率）比较。ARIMA 是否优于随机游走？

> **参考思路**：随机游走预测 $\hat{r}_{t+1} = r_t$，ARIMA 通常很难稳定战胜随机游走（有效市场），这本身就是重要发现。

**习题 6.5** 波动率预测评估  
用 GARCH(1,1) 对 `TECH` 做 5 步条件波动率预测（样本外最后 60 天），以**实际绝对收益率**（$|r_t|$）作为真实波动率的代理，计算 RMSE 并与**历史波动率基准**（前20日收益率标准差）比较。

> **参考思路**：`res.forecast(horizon=5)` 返回 5 步预测；GARCH 在波动率高企时通常显著优于历史波动率。

---

## 6.12 拓展阅读

| 文献 / 资源 | 简介 |
|------------|------|
| Tsay, R.S. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley. | 金融时间序列分析经典教材，ARCH/GARCH、高频数据、非线性模型均有深入覆盖 |
| Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. | 计量经济学时间序列权威著作，理论严密，适合进阶学习 |
| Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. | Box-Jenkins 方法论原典，ARIMA 建模的奠基之作 |
| Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007. | ARCH 模型原始论文，2003年诺贝尔经济学奖工作 |
| Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327. | GARCH 模型奠基论文 |
| Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. [免费在线](https://otexts.com/fpp3/) | 面向实践的预测方法书，含 ARIMA 和 ETS 详细讲解，配套 R/Python 代码 |
| `arch` 库文档：[https://arch.readthedocs.io](https://arch.readthedocs.io) | Python ARCH/GARCH 库官方文档，含丰富示例 |
| `statsmodels` 文档：[https://www.statsmodels.org](https://www.statsmodels.org) | Python 统计建模库，ARIMA、ADF、ACF/PACF 等均在此 |
