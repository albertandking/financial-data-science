# 第8章 面板数据与回归

!!! info "配套代码"
    `notebooks/ch08_panel_regression.ipynb`（使用 linearmodels / statsmodels）  
    本章实战部分采用**内置财务面板数据集 `fundamentals`（见附录C数据字典）**，  
    数据生成过程内置已知系数（leverage→ROA = −0.12）与公司固定效应，  
    故可检验 FE 能否还原真实系数。所有代码离线可运行。

---

## 8.1 本章导读与学习目标

金融研究中最常见的数据结构，既不是纯截面（某年所有公司的一张快照），也不是纯时间序列（某只股票逐日收益），而是**面板数据**（Panel Data）——同时跨越多个个体（公司、股票、国家……）和多个时间点。典型例子：

- **公司-年度面板**：2010—2023年 A 股上市公司的财务指标（资产负债率、ROA、营收增长率……）
- **股票-月度面板**：沪深300成分股每月的收益率、市值、账面市值比（用于 Fama-French 因子模型）
- **基金-季度面板**：公募基金季报披露的持仓、规模与业绩

面板数据的核心优势在于它同时具备**横截面差异**与**时间维度变化**两个维度，这让研究者能够"控制住那些随个体变化但不随时间变化的不可观测因素"——这正是 OLS 最头疼的遗漏变量问题。

**学习目标**

完成本章学习后，读者应能：

1. 识别面板数据的结构（MultiIndex、平衡 vs 非平衡），并会用 `pandas` 构造标准格式；
2. 解释混合 OLS 忽略个体异质性所导致的偏误来源；
3. 推导固定效应（FE）"组内变换"的代数逻辑，理解为什么它能消除不可观测的个体效应；
4. 区分随机效应（RE）与 FE 的假设差异，会运行 Hausman 检验做模型选择；
5. 理解金融面板为何几乎总要使用聚类稳健标准误，会在 `linearmodels` 中指定 `cov_type="clustered"`；
6. 使用 `linearmodels` 跑 Pooled OLS / FE / RE 并横向对比系数与标准误；
7. 认识内生性问题的两种来源（遗漏变量、双向因果），理解 FE 能解决哪一类、不能解决哪一类。

---

## 8.2 面板数据结构

### 8.2.1 个体 × 时间的二维表格

设我们观测 $N$ 个个体（entity，如公司），每个个体在 $T$ 个时间点均有记录。观测值用下标 $(i, t)$ 表示：$i = 1, \ldots, N$；$t = 1, \ldots, T$。

| 变量 | 含义 |
|------|------|
| $y_{it}$ | 因变量（如 ROA） |
| $\mathbf{x}_{it}$ | 自变量向量（如杠杆率、规模） |
| $\alpha_i$ | **个体固定效应**（不随时间变化，不可直接观测） |
| $\lambda_t$ | **时间固定效应**（不随个体变化，如宏观冲击） |
| $u_{it}$ | 纯随机误差 |

一个完整的双向固定效应模型写为：

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \alpha_i + \lambda_t + u_{it}$$

其中 $\alpha_i$ 和 $\lambda_t$ 都是未知参数，由数据识别。

### 8.2.2 平衡面板 vs 非平衡面板

**平衡面板（Balanced Panel）**：每个个体在所有 $T$ 个时间点均有记录，共 $N \times T$ 个观测。  
**非平衡面板（Unbalanced Panel）**：部分个体在某些时间点缺失（如新上市公司、退市公司）。

!!! warning "非平衡面板的处理"
    非平衡面板在金融中极为常见（公司上市/退市、股票停牌等）。`linearmodels` 能自动处理非平衡面板；注意**生存偏差**问题——如果只保留全程存活的公司，估计结果会偏向"好公司"。

### 8.2.3 与截面/时序数据的比较

| 维度 | 截面数据 | 时间序列 | 面板数据 |
|------|----------|----------|----------|
| 观测轴 | 个体 $i$ | 时间 $t$ | 个体 × 时间 $(i,t)$ |
| 样本量 | $N$ | $T$ | $N \times T$（或更少） |
| 个体异质性 | 无法控制 | 单一个体 | **可通过 FE 控制** |
| 时间趋势 | 无 | 直接观测 | 可通过时间 FE 控制 |
| 典型问题 | 遗漏变量偏误 | 自相关/单位根 | 异方差 + 序列相关 + 截面相关 |

### 8.2.4 pandas MultiIndex 表示

在 Python 中，面板数据以 `(entity, time)` 的 **MultiIndex** DataFrame 表示：

```python
# 构造标准面板 DataFrame
panel = pd.DataFrame({...}, index=pd.MultiIndex.from_tuples(
    [(entity_id, year), ...], names=['entity', 'time']
))
```

`linearmodels` 要求数据必须有两层 MultiIndex，且第一层为个体、第二层为时间。

---

## 8.3 混合 OLS（Pooled OLS）及其问题

### 8.3.1 模型设定

最朴素的做法是**忽略面板结构**，把所有 $N \times T$ 个观测当作独立的截面数据来做 OLS：

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \varepsilon_{it}$$

其中 $\varepsilon_{it} = \alpha_i + u_{it}$——把个体效应 $\alpha_i$ 丢进了误差项。

### 8.3.2 忽略个体异质性的危害

**问题 1：遗漏变量偏误（Omitted Variable Bias）**

若 $\alpha_i$（不可观测的个体特征，如公司管理层能力、行业地位）与 $\mathbf{x}_{it}$ 相关，则 $\text{Cov}(\mathbf{x}_{it}, \varepsilon_{it}) \neq 0$，OLS 估计量 $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ **不一致**，存在系统性偏误。

以杠杆率（leverage）对 ROA 的影响为例：高质量公司（$\alpha_i$ 大）可能同时有更高的 ROA 和更保守的杠杆，导致 Pooled OLS 高估杠杆率的负效应（或低估正效应）。

**问题 2：标准误严重低估**

同一公司的多个观测（$t = 1, 2, \ldots$）高度相关，误差项存在序列相关。若用 OLS 假定误差独立，标准误会被**严重低估**，导致 t 统计量虚高，大量假显著。

### 8.3.3 Pooled OLS 的适用前提

只有在 $\alpha_i$ **不与** $\mathbf{x}_{it}$ 相关，且 $\alpha_i$ 仅是随机噪声时，Pooled OLS 才是一致的（这正是随机效应模型的前提）。

---

## 8.4 固定效应模型（Fixed Effects）

### 8.4.1 个体固定效应

**模型：**

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \alpha_i + u_{it}$$

其中 $\alpha_i$ 为**个体特定常数**，可与 $\mathbf{x}_{it}$ 任意相关——这是 FE 的关键优势。

**估计思路——"组内变换"（Within Transformation）**

对个体 $i$ 取时间均值：

$$\bar{y}_{i\cdot} = \bar{\mathbf{x}}_{i\cdot}^{\prime} \boldsymbol{\beta} + \alpha_i + \bar{u}_{i\cdot}$$

用原式减去均值式，消去 $\alpha_i$：

$$\underbrace{y_{it} - \bar{y}_{i\cdot}}_{\tilde{y}_{it}} = \underbrace{(\mathbf{x}_{it} - \bar{\mathbf{x}}_{i\cdot})^{\prime}}_{\tilde{\mathbf{x}}_{it}^{\prime}} \boldsymbol{\beta} + \underbrace{u_{it} - \bar{u}_{i\cdot}}_{\tilde{u}_{it}}$$

对变换后的数据跑 OLS，即得 **组内（Within）估计量** $\hat{\boldsymbol{\beta}}_{\text{FE}}$。

!!! tip "直觉理解"
    组内变换把所有变量"中心化"到各公司自身的均值上，相当于"用公司自己和自己比"，从而排除了公司间不可观测差异（$\alpha_i$）的干扰。FE 只利用**组内（within）变异**，忽略**组间（between）变异**。

**自由度损失**：FE 估计 $N$ 个个体虚拟变量，消耗 $N$ 个自由度，在 $T$ 很小、$N$ 很大时效率损失较大。

### 8.4.2 时间固定效应

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \lambda_t + u_{it}$$

控制宏观冲击（利率变化、金融危机）等**共同时间趋势**。对 $t$ 做时间中心化后估计。

### 8.4.3 双向固定效应（Two-Way FE）

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \alpha_i + \lambda_t + u_{it}$$

同时控制个体效应和时间效应，是公司金融实证研究的标准配置。在 `linearmodels` 中：

```python
PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
    cov_type="clustered", cluster_entity=True
)
```

!!! note "FE 能识别什么？"
    FE 只能估计**随时间变化**的自变量对因变量的影响。**不随时间变化的变量**（如行业代码、公司注册地）会被 $\alpha_i$ 完全吸收，系数无法识别。这是 FE 的根本局限。

---

## 8.5 随机效应模型（Random Effects）

### 8.5.1 模型假设

RE 模型与 FE 同形：

$$y_{it} = \mathbf{x}_{it}^{\prime} \boldsymbol{\beta} + \alpha_i + u_{it}$$

但核心假设不同：

$$\alpha_i \overset{\text{iid}}{\sim} (0, \sigma_\alpha^2), \quad u_{it} \overset{\text{iid}}{\sim} (0, \sigma_u^2), \quad \text{且} \quad \text{Cov}(\mathbf{x}_{it}, \alpha_i) = 0$$

$\alpha_i$ 被视为**随机干扰项的一部分**，而非固定参数，因此个体效应可以与截面数据的协变量"分享"信息。

### 8.5.2 GLS 估计

合并误差 $v_{it} = \alpha_i + u_{it}$ 不是 iid 的：

$$\text{Var}(v_{it}) = \sigma_\alpha^2 + \sigma_u^2, \quad \text{Cov}(v_{it}, v_{is}) = \sigma_\alpha^2 \quad (t \neq s)$$

RE 使用**广义最小二乘（GLS）**，对"组内相关"做加权处理。等价于在变换后的数据上做 OLS，其中变换参数为：

$$\theta = 1 - \sqrt{\frac{\sigma_u^2}{\sigma_u^2 + T\sigma_\alpha^2}}$$

RE 的变换介于**不做变换（Pooled OLS，$\theta=0$）**和**完全组内变换（FE，$\theta=1$）**之间，因此 RE 利用了组间和组内两部分变异，在 $\text{Cov}(\mathbf{x}_{it}, \alpha_i) = 0$ 成立时**比 FE 更有效率**。

### 8.5.3 FE vs RE：核心权衡

| 维度 | 固定效应（FE） | 随机效应（RE） |
|------|----------------|----------------|
| $\alpha_i$ 与 $\mathbf{x}_{it}$ 的关系 | 可以相关 | **必须不相关** |
| 一致性 | 无论相关与否均一致 | 若相关则**不一致** |
| 效率 | 若 RE 假设成立，效率较低 | 若假设成立，效率更高 |
| 不随时间变化的变量 | **不可估计** | 可估计 |
| 典型适用场景 | 公司 FE（管理层质量与杠杆相关） | 若有强理论保证不相关 |

---

## 8.6 Hausman 检验

### 8.6.1 原理

Hausman（1978）提出了一个系统性检验：

- **原假设 $H_0$**：$\text{Cov}(\mathbf{x}_{it}, \alpha_i) = 0$，RE 假设成立；FE 与 RE 均一致，RE 更有效率。
- **备择假设 $H_1$**：$\text{Cov}(\mathbf{x}_{it}, \alpha_i) \neq 0$；仅 FE 一致，RE 不一致，应选 FE。

检验统计量：

$$H = (\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}})^{\prime} \left[\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{\text{FE}}) - \widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{\text{RE}})\right]^{-1} (\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}}) \overset{H_0}{\sim} \chi^2(k)$$

其中 $k$ 为时变自变量的个数。**拒绝 $H_0$**（p 值小）→ 选 FE；**不能拒绝 $H_0$** → FE 与 RE 一致，可选更有效率的 RE。

### 8.6.2 实践中的注意事项

!!! warning "Hausman 检验的局限"
    1. Hausman 检验对**聚类标准误**不敏感，传统版本假定同方差。在金融面板（有序列相关）中，检验结论有时不稳健。  
    2. 即使 Hausman 检验不显著，公司金融研究者通常**仍倾向于使用 FE**，因为"管理层能力与财务决策相关"几乎是行业共识。  
    3. Hausman 检验只是辅助工具，最终选择应结合**经济直觉**和**鲁棒性检验**。

---

## 8.7 标准误问题

### 8.7.1 金融面板的误差结构

金融面板的误差项 $u_{it}$ 通常同时存在：

- **序列相关（Serial Correlation）**：同一公司的误差跨期相关——宏观冲击、公司特定事件会持续多期。
- **截面相关（Cross-Sectional Correlation）**：同一时期不同公司的误差相关——系统性风险、行业冲击。
- **异方差（Heteroscedasticity）**：不同公司误差方差不同——大公司和小公司的波动率差异。

OLS 标准误在上述情形下均**严重低估**，导致 t 统计量虚高。

### 8.7.2 聚类稳健标准误

**按个体（公司）聚类**是金融面板最常用的修正方式，它允许同一公司内部误差任意相关（序列相关），但假定不同公司间独立。

!!! tip "Petersen (2009) 的关键发现"
    Petersen（2009, *Review of Financial Studies*）研究发现：  
    - 对于**公司-年度面板**，最主要的误差相关是**公司内序列相关**，应按公司聚类。  
    - 若时间维度也有显著相关（如金融危机期间），应考虑**双向聚类**（按公司 + 按年度）。  
    - Fama-MacBeth 两步法在处理截面相关时有优势（将在第9章介绍）。

**`linearmodels` 中的聚类标准误：**

```python
# 按个体（公司）聚类
res_fe = PanelOLS(y, X, entity_effects=True).fit(
    cov_type="clustered", cluster_entity=True
)

# 双向聚类（按公司 + 按时间）
res_fe2 = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
    cov_type="clustered", cluster_entity=True, cluster_time=True
)
```

### 8.7.3 普通标准误 vs 聚类标准误：对比

下表展示了两种标准误的典型差异（数值来自内置 `fundamentals` 数据集实验）：

| 估计方法 | 系数 $\hat{\beta}$ | 普通 SE | 聚类 SE | 聚类/普通比 |
|----------|-------------------|---------|---------|-------------|
| Pooled OLS | × | 极小 | — | — |
| FE（个体） | ✓ | 偏小 | 更大 | ≈ 1.5–3× |
| FE（双向） | ✓ | 偏小 | 更大 | ≈ 1.5–3× |

!!! danger "不用聚类标准误的后果"
    忽略序列相关时，标准误可能被低估 50%—200%，相当于把 t = 1.3 的不显著结果报告为 t = 2.6 的显著结果。这是金融实证中最常见的统计谬误之一。

---

## 8.8 内生性简介

### 8.8.1 遗漏变量

当自变量与误差项相关（$\text{Cov}(\mathbf{x}_{it}, u_{it}) \neq 0$）时，OLS 估计量不一致。FE 通过消除 $\alpha_i$，能解决**不随时间变化的遗漏变量**问题，但对**随时间变化的遗漏变量**（如公司层面的年度冲击）无能为力。

### 8.8.2 双向因果（同期内生性）

在公司金融中，杠杆率↔ROA 可能同期互为因果：

- 高 ROA → 内部资金充裕 → 降低杠杆（啄序理论）
- 高杠杆 → 利息负担 → 降低 ROA（财务困境成本）

FE 无法解决这类**同期内生性**。需要工具变量（IV）或准自然实验（DID、RD 等），将在后续章节介绍。

!!! info "本章目标：识别问题，而非完全解决"
    本章聚焦于用 FE/RE 处理个体异质性导致的遗漏变量问题，并认识其局限。因果推断方法详见第10—11章。

---

## 8.9 实战案例：杠杆率对盈利能力的影响

!!! info "内置财务面板数据集 `fundamentals`"
    本节使用 `fds.load_fundamentals()` 加载内置公司-年度财务面板（见附录C数据字典）：  
    **200 家公司 × 8 年（2018—2025），共 1600 个观测**（平衡面板）。  
    数据生成过程内置已知系数（$\beta_{\text{leverage}} = -0.12$）与公司固定效应（$\alpha_i$ 与 leverage **负相关**），  
    故可检验 FE 能否还原真实系数，并直观展示 Pooled OLS 偏误来源。

    ```python
    from fds import load_fundamentals
    df_raw = load_fundamentals()
    df = df_raw.set_index(['firm', 'year'])
    ```

### 8.9.1 面板设计

内置数据集包含以下变量：

| 变量 | 含义 | 备注 |
|------|------|------|
| `firm` | 公司代码（F000—F199） | 个体索引 |
| `year` | 年份（2018—2025） | 时间索引 |
| `roa` | 资产收益率（ROA，**因变量**） | 内置真实系数 |
| `leverage` | 资产负债率 | 与 $\alpha_i$ 负相关（故意设计） |
| `size` | log 总资产 | 规模控制变量 |
| `revenue_growth` | 营收增长率 | 备用控制变量 |
| `industry` | 行业分类 | 不随时间变化，FE 无法识别 |

数据生成过程内置**个体固定效应** $\alpha_i$：高盈利能力公司（$\alpha_i$ 大）倾向于保守融资（低杠杆），即 $\text{Cov}(\alpha_i, \text{leverage}_{it}) < 0$——这正是 Pooled OLS 产生偏误的根源。

### 8.9.2 Pooled OLS vs FE vs RE 系数对比

配套 notebook 运行 `load_fundamentals()` 数据的实验结果：

| 估计方法 | $\hat{\beta}_{\text{leverage}}$ | 偏误 | 备注 |
|----------|--------------------------------|------|------|
| Pooled OLS | ≈ $-0.40$ | **−0.28（严重夸大）** | $\alpha_i$ 混入误差项，偏误约 3 倍 |
| FE（个体） | ≈ $-0.13$ | ≈ −0.01（近似无偏）| FE 消除偏误 ✓ |
| FE（双向） | ≈ $-0.13$ | ≈ −0.01 | 额外控制宏观年度冲击 |
| RE | 介于两者之间 | 有偏 | $\alpha_i \perp X$ 假设被违反 |

**真实系数（内置）**：$\beta_{\text{leverage}} = -0.12$

Pooled OLS 偏误来源：$\alpha_i$（盈利能力）与 leverage 负相关（高能力公司 → $\alpha_i$ 大，低杠杆），Pooled OLS 把 $\alpha_i$ 的正影响错误归因到低杠杆上，从而大幅夸大了杠杆的负效应（$-0.40$ vs 真实 $-0.12$）。

### 8.9.3 Hausman 检验结果解读

`fundamentals` 数据中，$\alpha_i$ 刻意设计为与杠杆率负相关，因此 Hausman 检验**拒绝 $H_0$**（p 值 < 0.05），支持选择 FE 而非 RE。

### 8.9.4 聚类标准误的重要性

`fundamentals` 数据含有序列相关成分（同公司跨期误差相关）。比较结果：

- 普通 FE 标准误：较小，t 统计量偏大（假性显著）
- 聚类 FE 标准误（按公司）：更大，t 统计量更保守（反映真实不确定性）

**结论**：在任何金融面板研究中，FE + 聚类标准误是标配组合。

---

## 8.10 本章小结

| 知识点 | 要点 |
|--------|------|
| 面板数据结构 | $(i, t)$ MultiIndex，平衡/非平衡 |
| Pooled OLS | 忽略 $\alpha_i$，若 $\alpha_i$ 与 $X$ 相关则不一致 |
| 固定效应 FE | 组内变换消去 $\alpha_i$，一致但损失组间信息；无法估计时不变变量 |
| 随机效应 RE | GLS 估计，效率高；要求 $\alpha_i \perp X$，若违反则不一致 |
| Hausman 检验 | $H_0$: RE 假设成立；拒绝则选 FE |
| 聚类标准误 | 金融面板必备，按公司聚类处理序列相关 |
| 内生性 | FE 解决不随时间变化的遗漏变量；同期内生性需 IV/DID |

**选择流程**（实践建议）：

```
数据是面板结构?
  ├─ 否 → 截面/时序方法
  └─ 是 → 个体效应与X相关?
           ├─ 是（经济上合理）→ FE + 聚类SE
           ├─ 否（有理论保证）→ Hausman检验 → p<0.05: FE | p≥0.05: RE
           └─ 不确定 → 汇报FE为主，RE为鲁棒性检验
```

---

## 8.11 习题

**习题 8.1**（面板结构识别）  
给定一个包含 50 家公司、8 年数据的 DataFrame（存在部分缺失），请：  
(a) 判断是平衡还是非平衡面板；  
(b) 构造正确的 `(entity, time)` MultiIndex；  
(c) 计算每家公司的观测年数分布。  
*参考思路：用 `groupby('entity').size()` 检查各公司观测数；用 `pd.MultiIndex.from_frame()` 构造索引。*

**习题 8.2**（FE 组内变换推导）  
设简单单变量面板模型 $y_{it} = \beta x_{it} + \alpha_i + u_{it}$，$N=3$，$T=4$。  
(a) 写出组内变换后的方程；  
(b) 证明组内变换后 $\alpha_i$ 完全消失；  
(c) 若 $x_{it}$ 不随时间变化（$x_{it} = x_i$），变换后发生什么？  
*参考思路：(c) 变换后 $\tilde{x}_{it} = x_i - x_i = 0$，完全消失，系数无法识别。*

**习题 8.3**（Pooled OLS 偏误方向）  
在内置数据集 `fundamentals` 中，个体效应 $\alpha_i$ 与杠杆率 $\text{leverage}_{it}$ 负相关（高盈利能力公司低杠杆）。  
(a) 预测 Pooled OLS 对 $\beta_{\text{leverage}}$ 的偏误方向（高估还是低估负效应绝对值）；  
(b) 用 `load_fundamentals()` 跑 Pooled OLS，验证预测；  
(c) 计算偏误大小：$\hat{\beta}_{\text{Pooled}} - \beta_{\text{true}}$，其中内置真实值 $\beta_{\text{true}} = -0.12$。  
*参考思路：$\alpha_i$ 与 leverage 负相关，且 $\alpha_i$ 对 ROA 正影响 → Pooled OLS 把 $\alpha_i$ 的正影响归因到低杠杆上，导致高估负效应绝对值（$|\hat{\beta}| > |\beta_{\text{true}}|$，约为 −0.40）。*

**习题 8.4**（Hausman 检验解读）  
使用 `load_fundamentals()` 数据跑 FE 和 RE，手动计算 Hausman 统计量（只对 `leverage` 这一个系数）：  
$$H_{\text{手动}} = \frac{(\hat{\beta}_{\text{FE}} - \hat{\beta}_{\text{RE}})^2}{\widehat{\text{Var}}(\hat{\beta}_{\text{FE}}) - \widehat{\text{Var}}(\hat{\beta}_{\text{RE}})}$$  
与 $\chi^2(1)$ 的临界值 3.84（5%水平）比较，得出结论。  
*参考思路：分子 = 两模型系数差的平方；分母 = FE 方差 - RE 方差（应为正数）。*

**习题 8.5**（双向 FE 与时间效应）  
在内置数据集 `fundamentals` 中，数据生成过程包含宏观年度冲击（时间固定效应）。  
(a) 比较单向 FE（仅个体效应）和双向 FE（个体+时间）的 `leverage` 系数差异；  
(b) 解释为什么时间效应会影响 $\hat{\beta}_{\text{leverage}}$ 的估计；  
(c) 用 F 检验（`entity_effects=True, time_effects=True` 的 $F$ 统计量）验证时间效应是否联合显著。  
*参考思路：若宏观因素同时影响 ROA 和杠杆决策（如经济下行期公司既增加杠杆又降低 ROA），不控制时间效应会产生混淆。*

---

## 8.12 拓展阅读

1. **Wooldridge, J. M. (2010)**. *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.  
   — 面板数据计量经济学圣经，第10—14章系统讲述 FE/RE/工具变量，数学推导完整。

2. **Baltagi, B. H. (2021)**. *Econometric Analysis of Panel Data* (6th ed.). Springer.  
   — 覆盖非平衡面板、动态面板（GMM）、空间面板等高级专题，计量经济系研究生必读。

3. **Petersen, M. A. (2009)**. Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies*, 22(1), 435—480.  
   — 金融实证必读。通过 Monte Carlo 模拟对比 OLS SE、Fama-MacBeth、聚类 SE 在不同误差结构下的表现，结论：公司-年度面板应按公司聚类。

4. **Hausman, J. A. (1978)**. Specification tests in econometrics. *Econometrica*, 46(6), 1251—1271.  
   — 原始 Hausman 检验论文，奠定 FE/RE 选择的统计基础。

5. **Arellano, M., & Bond, S. (1991)**. Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *Review of Economic Studies*, 58(2), 277—297.  
   — 动态面板 GMM（Arellano-Bond 估计），适用于含滞后因变量的模型。

6. **Cameron, A. C., & Miller, D. L. (2015)**. A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317—372.  
   — 聚类标准误的实践指南，涵盖何时聚类、如何选择聚类变量、小样本修正等实际操作问题。
