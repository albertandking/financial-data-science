# 第14章 投资组合优化

!!! info "配套代码"
    `notebooks/ch14_portfolio_optimization.ipynb`（使用 scipy.optimize）

---

## 14.0 本章导读

> “不要把鸡蛋放在同一个篮子里。”——这句朴素的谚语背后，隐藏着严谨的数学框架。

1952 年，哈里·马科维茨（Harry Markowitz）在其博士论文中首次用均值-方差框架把“分散投资”量化成一个可求解的优化问题，开创了现代组合理论（Modern Portfolio Theory，MPT）。本章以 A 股示例数据为载体，系统讲授：

1. **组合的期望收益与风险**如何由权重向量决定；
2. **均值-方差有效前沿**的推导与数值求解；
3. **引入无风险资产**后资本市场线（CML）与切点组合的出现；
4. **约束条件**（禁止做空、权重上限）对前沿形态的影响；
5. **协方差矩阵的估计问题**与 Ledoit-Wolf 收缩方案；
6. **等权与风险平价**策略的直觉与实现。

!!! warning "数学准备"
    本章需要线性代数基础（矩阵乘法、二次型）和初步的凸优化概念。
    若尚不熟悉，建议先回顾附录A的矩阵符号约定。

---

## 14.1 学习目标

完成本章后，读者应能：

- 用矩阵公式计算组合期望收益 $\mu_p$ 与组合方差 $\sigma_p^2$；
- 使用 `scipy.optimize.minimize` 数值求解全局最小方差（GMV）组合与最大夏普组合；
- 绘制有效前沿、标注 GMV 与切点组合，添加资本市场线（CML）；
- 对比无约束、禁止做空两种设定下有效前沿的变化；
- 用 Ledoit-Wolf 收缩估计器替换样本协方差，观察对优化结果的影响；
- 理解等权基准与风险平价策略的核心差异。

---

## 14.2 组合收益与风险

### 14.2.1 符号约定

设市场有 $N$ 只资产，我们用以下符号：

| 符号 | 含义 |
|------|------|
| $w \in \mathbb{R}^N$ | 权重向量，$w_i$ 表示第 $i$ 只资产的投资比例 |
| $\mathbf{1}$ | 全1列向量，满足约束 $w^\top \mathbf{1} = 1$ |
| $\mu \in \mathbb{R}^N$ | 各资产期望（年化）收益率向量 |
| $\Sigma \in \mathbb{R}^{N\times N}$ | 资产收益率协方差矩阵（正定对称） |
| $r_f$ | 无风险利率（年化） |

### 14.2.2 组合期望收益

设各资产日收益率为随机向量 $\mathbf{r} = (r_1,\ldots,r_N)^\top$，组合收益率为：

$$
r_p = w^\top \mathbf{r} = \sum_{i=1}^N w_i r_i
$$

对期望取线性性质：

$$
\boxed{\mu_p = \mathbb{E}[r_p] = w^\top \mu = \sum_{i=1}^N w_i \mu_i}
$$

直觉：组合的期望收益是各资产期望收益的**加权平均**，权重就是资金比例。

### 14.2.3 组合方差（风险）

$$
\sigma_p^2 = \mathrm{Var}(w^\top \mathbf{r}) = w^\top \Sigma w
$$

展开为双重求和：

$$
\boxed{\sigma_p^2 = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij}}
$$

其中 $\sigma_{ij} = \mathrm{Cov}(r_i, r_j)$，当 $i=j$ 时即各资产方差 $\sigma_i^2$。

!!! note "分散化效应"
    当资产之间相关系数 $\rho_{ij} < 1$ 时，组合方差 $\sigma_p^2 < \sum_i w_i^2\sigma_i^2$，
    即**分散化降低风险**。相关性越低，降险效果越显著。

**数值示例**（两资产情形，$w_1=0.6,\ w_2=0.4$）：

$$
\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho_{12}\sigma_1\sigma_2
$$

若 $\sigma_1=0.20,\ \sigma_2=0.25,\ \rho_{12}=0.3$：

$$
\sigma_p^2 = 0.36\times0.04 + 0.16\times0.0625 + 2\times0.6\times0.4\times0.3\times0.05
= 0.0144 + 0.01 + 0.0072 = 0.0316 \implies \sigma_p \approx 17.8\%
$$

单独持有两只资产的波动率分别为 20% 和 25%，组合后降至 17.8%，体现分散化效益。

---

## 14.3 均值-方差优化与有效前沿

<figure markdown>
  ![图 14-1　均值-方差有效前沿、GMV、最大夏普与 CML](../assets/figures/ch14_frontier.png){ width="680" }
  <figcaption>图 14-1　均值-方差有效前沿、GMV、最大夏普与 CML</figcaption>
</figure>


### 14.3.1 优化问题的形式化

马科维茨框架的核心思想：**在给定目标收益水平 $\mu_0$ 下，最小化组合方差**。

$$
\begin{aligned}
\min_{w} \quad & \frac{1}{2} w^\top \Sigma w \\
\text{s.t.} \quad & w^\top \mu = \mu_0 \\
           & w^\top \mathbf{1} = 1
\end{aligned}
$$

!!! tip "等价形式"
    也可以**在给定方差上限下最大化期望收益**，两者在凸集上的解完全等价，
    描绘出同一条有效前沿。

扫描一系列目标收益 $\mu_0$，把最优方差 $\sigma^*(\mu_0)$ 描绘在 $(\sigma, \mu)$ 平面上，
即得到**有效前沿（Efficient Frontier）**。

### 14.3.2 全局最小方差组合（GMV）解析解

**只**加权重和为1的约束（不限定收益），最小化方差即得 GMV：

$$
\min_w \quad w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mathbf{1} = 1
$$

用拉格朗日乘子法，构造 $\mathcal{L} = w^\top\Sigma w - \lambda(w^\top\mathbf{1}-1)$，
对 $w$ 求偏导并令其为零：

$$
2\Sigma w = \lambda \mathbf{1} \implies w = \frac{\lambda}{2}\Sigma^{-1}\mathbf{1}
$$

代入约束 $w^\top\mathbf{1}=1$，得 $\lambda/2 = 1/(\mathbf{1}^\top\Sigma^{-1}\mathbf{1})$，因此：

$$
\boxed{w_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}}
$$

这是一个纯粹由协方差矩阵决定的**解析解**，与期望收益无关。

### 14.3.3 有效前沿的扫描算法

实践中，$N > 3$ 时解析求解复杂，通常**数值扫描**：

```
对每个目标收益 μ₀ ∈ [μ_min, μ_max]：
    使用 scipy.optimize.minimize 求解最小方差组合
    记录 (σ*, μ₀)
连接所有点，绘制有效前沿
```

!!! info "scipy.optimize 约束写法"
    ```python
    from scipy.optimize import minimize

    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},            # 权重和=1
        {'type': 'eq', 'fun': lambda w: w @ mu - target_mu},     # 目标收益
    ]
    bounds = None  # 无约束（可做空）
    # 或 bounds = [(0, 1)] * N  # 禁止做空
    ```

---

## 14.4 引入无风险资产：资本市场线与切点组合

### 14.4.1 超额夏普比率与切点组合

引入无风险资产后，投资者可以把资金在**无风险资产**（银行存款、国债）和**风险组合**之间分配。
混合组合的期望收益与方差为：

$$
\mu_{\text{mix}} = (1-\alpha)r_f + \alpha \mu_p, \quad
\sigma_{\text{mix}} = \alpha \sigma_p \quad (\alpha \in [0,1])
$$

消去 $\alpha$，得到直线方程（**资本配置线 CAL**）：

$$
\mu_{\text{mix}} = r_f + \frac{\mu_p - r_f}{\sigma_p}\cdot\sigma_{\text{mix}}
$$

斜率 $\frac{\mu_p - r_f}{\sigma_p}$ 即为**夏普比率**。所有可行 CAL 中，斜率最大的那条
就是**资本市场线（CML）**，其对应的纯风险组合即**切点组合（Tangency Portfolio）**：

$$
\max_w \quad S(w) = \frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma w}}
\quad \text{s.t.} \quad w^\top\mathbf{1} = 1
$$

解析上，切点组合权重为：

$$
\boxed{w_{\text{tan}} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu - r_f\mathbf{1})}}
$$

!!! note "CAPM 含义"
    在 CAPM 均衡中，切点组合恰好是**市场组合**（Market Portfolio）。
    每个理性投资者都持有市场组合与无风险资产的混合，差别只是混合比例。

### 14.4.2 CML 的斜率与截距

$$
\text{CML}: \quad \mu = r_f + \text{SR}_{\text{max}} \cdot \sigma
$$

其中 $\text{SR}_{\text{max}} = \max_w S(w)$ 是有效前沿上所有组合中最大的夏普比率。
CML 从 $(0, r_f)$ 出发，与有效前沿相切于切点组合，向右延伸（通过借贷杠杆可超越切点组合）。

---

## 14.5 约束优化：禁止做空与权重上限

### 14.5.1 无约束 vs 禁止做空

| 设定 | 约束 | 说明 |
|------|------|------|
| 无约束（允许做空） | $w^\top\mathbf{1}=1$ | 理论有效前沿，$w_i$ 可为负 |
| 禁止做空 | $w^\top\mathbf{1}=1$，$w_i \geq 0$ | 实际基金常用，前沿位于右方 |
| 权重上限 | $w_i \leq u_i$ | 防止过度集中 |
| 行业约束 | $\sum_{i\in S_k} w_i \geq lb_k$ | 满足投资政策约束 |

**添加禁止做空约束后**，有效前沿通常向右移动（同收益水平下方差更大），
因为限制了通过做空高风险资产来对冲的能力。

在 `scipy.optimize.minimize` 中：

```python
bounds = [(0, 1)] * N  # 每只资产权重在 [0, 1] 之间
```

### 14.5.2 约束的经济含义

!!! warning "禁止做空的影响"
    在禁止做空约束下，求解最大夏普组合时，极端情况可能出现**角点解**：
    只投资少数几只资产（其余权重压缩至0边界），组合高度集中。
    实践中需配合权重上限约束来防止过度集中风险。

---

## 14.6 协方差估计的稳定性问题

### 14.6.1 样本协方差的缺陷

设样本量为 $T$，资产数为 $N$。样本协方差矩阵 $\hat\Sigma = \frac{1}{T-1}\sum_{t=1}^T(r_t-\bar r)(r_t-\bar r)^\top$ 的问题：

1. **条件数过大**：$N$ 接近 $T$ 时，$\hat\Sigma$ 接近奇异，$\hat\Sigma^{-1}$ 不稳定；
2. **估计误差放大**：优化器会**放大**样本协方差的估计误差，形成“误差最大化”；
3. **极端权重**：导致优化结果中权重分配极端，在样本外表现差。

!!! example "经验规律"
    A 股月度数据，500 只股票、3 年历史，$N/T \approx 500/36 \approx 14 \gg 1$，
    样本协方差矩阵几乎不可用，**必须正则化**。

### 14.6.2 Ledoit-Wolf 收缩估计

Ledoit & Wolf（2004）提出将样本协方差矩阵向**结构化目标矩阵**收缩：

$$
\hat\Sigma_{\text{LW}} = (1-\alpha)\hat\Sigma + \alpha F
$$

其中 $F$ 是目标矩阵（如对角矩阵、单位矩阵倍数等），$\alpha \in [0,1]$ 是**收缩系数**。

直觉：
- $\alpha=0$：完全使用样本协方差（高估计方差，低偏差）；
- $\alpha=1$：完全使用目标矩阵（低估计方差，高偏差）；
- 最优 $\alpha^*$：通过解析公式最小化估计误差的期望值。

`scikit-learn` 已内置最优解析收缩：

```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf().fit(returns_matrix)
cov_lw = lw.covariance_  # 收缩后协方差矩阵
alpha = lw.shrinkage_    # 收缩系数
```

| 估计方法 | 优点 | 缺点 |
|---------|------|------|
| 样本协方差 | 无偏、简单 | 小样本极不稳定，容易奇异 |
| Ledoit-Wolf | 最优收缩系数有解析解，稳定 | 轻微有偏（但MSE更低） |
| 因子模型协方差 | 参数少，可解释 | 需要合理指定因子结构 |

---

## 14.7 等权与风险平价策略

### 14.7.1 等权组合（Equal Weight）

最简单的基准：$w_i = 1/N$，即平均分配资金。

优势：
- 无需估计 $\mu$ 和 $\Sigma$，规避估计误差；
- 相比于均值-方差优化，在样本外往往具有竞争力（DeMiguel et al., 2009）。

### 14.7.2 风险平价（Risk Parity）

等权分配的是**资金**，而风险平价分配的是**风险贡献**：

$$
RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p} \quad \text{（第 $i$ 只资产的风险贡献）}
$$

风险平价要求所有资产的风险贡献相等：$RC_1 = RC_2 = \cdots = RC_N = \sigma_p / N$。

这等价于求解：

$$
\min_w \sum_{i=1}^N \left(w_i(\Sigma w)_i - \frac{\sigma_p^2}{N}\right)^2
$$

!!! note "与等权的区别"
    风险平价中，**低波动资产**获得更高权重（因为它贡献的风险更少），
    典型例子是债券获得比股票更多资金分配，以实现等风险贡献。

### 14.7.3 策略对比

| 策略 | 需要估计 $\mu$？ | 需要估计 $\Sigma$？ | 直觉 |
|------|---------------|------------------|------|
| 等权 | 否 | 否 | 完全分散，绕开估计误差 |
| GMV | 否 | 是 | 只看风险，不看收益 |
| 最大夏普 | 是 | 是 | 风险调整后收益最优 |
| 风险平价 | 否 | 是 | 等额分配风险贡献 |
| 均值-方差 | 是 | 是 | 完整马科维茨框架 |

---

## 14.8 A股实战：四只资产的有效前沿

本节用内置的四只 A 股风格资产（BANK、LIQUOR、TECH、UTILITY）演示完整流程。

### 14.8.1 数据准备与参数估计

```python
from fds import load_sample_prices, load_market, daily_returns, set_chinese_font
import numpy as np

prices = load_sample_prices()
rets = daily_returns(prices)

TRADING_DAYS = 252
mu = rets.mean() * TRADING_DAYS          # 年化期望收益（向量）
cov = rets.cov() * TRADING_DAYS          # 年化协方差矩阵
rf = load_market()['rf_annual'].mean()   # 年化无风险利率
```

### 14.8.2 蒙特卡洛撒点验证

直觉可视化：随机生成 5000 个组合，在 $(\sigma_p, \mu_p)$ 平面上描绘散点图。
散点的“左边界”即为有效前沿的近似轮廓。

```python
np.random.seed(42)
n_sim = 5000
sim_results = []
for _ in range(n_sim):
    w = np.random.dirichlet(np.ones(N))  # 随机正权重，和为1
    mu_p = w @ mu
    sigma_p = np.sqrt(w @ cov @ w)
    sim_results.append((sigma_p, mu_p))
```

### 14.8.3 有效前沿对比图

最终图形应包含：

1. **灰色散点**：蒙特卡洛随机组合（直觉边界）；
2. **蓝色曲线**：无约束有效前沿（含负权重）；
3. **橙色曲线**：禁止做空有效前沿；
4. **红色星号**：GMV 组合；
5. **绿色星号**：最大夏普（切点）组合；
6. **黑色虚线**：CML；
7. **菱形标记**：等权基准。

---

## 14.9 本章小结

| 知识点 | 核心公式 / 结论 |
|--------|----------------|
| 组合期望收益 | $\mu_p = w^\top\mu$ |
| 组合方差 | $\sigma_p^2 = w^\top\Sigma w$ |
| GMV 解析解 | $w_{\text{GMV}} = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^\top\Sigma^{-1}\mathbf{1})$ |
| 切点组合 | $w_{\text{tan}} \propto \Sigma^{-1}(\mu - r_f\mathbf{1})$ |
| 资本市场线 | $\mu = r_f + \text{SR}_{\max}\cdot\sigma$ |
| 约束的影响 | 禁止做空使前沿右移，限制分散化潜力 |
| Ledoit-Wolf | 收缩系数 $\alpha^*$ 有解析解，显著降低 MSE |
| 等权 vs 风险平价 | 前者分资金，后者分风险贡献 |

!!! summary "关键结论"
    马科维茨框架在理论上优雅，但实践中**协方差估计误差**是最大的挑战。
    Ledoit-Wolf 收缩、因子模型约束和样本外验证是缓解这一问题的主要手段。
    简单的等权基准在样本外往往难以被超越，是检验任何组合策略的重要基准。

---

## 14.10 习题

**习题 14.1**（基础）已知三只股票的年化收益率 $\mu=(0.12,\ 0.18,\ 0.08)^\top$，
年化协方差矩阵为：

$$
\Sigma = \begin{pmatrix} 0.04 & 0.012 & 0.008 \\ 0.012 & 0.09 & 0.015 \\ 0.008 & 0.015 & 0.0225 \end{pmatrix}
$$

权重为 $w=(0.4,\ 0.4,\ 0.2)^\top$。
（a）计算组合期望收益和年化波动率；
（b）验证 $w^\top\mathbf{1}=1$。

??? hint "参考思路"
    直接套用公式：$\mu_p = w^\top\mu = 0.4\times0.12 + 0.4\times0.18 + 0.2\times0.08 = 0.136$；
    $\sigma_p = \sqrt{w^\top\Sigma w}$，用 `numpy` 矩阵运算即可。

---

**习题 14.2**（编程）用内置四只 A 股资产，
（a）用解析公式计算 GMV 权重，与 `scipy.optimize` 数值解对比；
（b）绘制 100 条目标收益扫描得到的有效前沿。

??? hint "参考思路"
    解析解：`w_gmv = inv(cov) @ ones / (ones @ inv(cov) @ ones)`。
    数值解：固定收益约束，`minimize` 最小化 `w @ cov @ w`，对比两者权重差异应 $<10^{-6}$。

---

**习题 14.3**（思考）加入禁止做空约束后，有效前沿为何向右偏移（同等收益下方差增大）？
请从**可行集缩小**的角度给出直觉解释，并结合本章 A 股数据举例说明哪只资产在无约束情形下权重为负。

??? hint "参考思路"
    禁止做空相当于在可行集上加了若干半空间约束，原来的最优解若包含负权重则变得不可行，
    必须寻找次优解，因此前沿右移。运行无约束最大夏普组合，输出各资产权重，
    负权重的资产即为被“做空”的资产。

---

**习题 14.4**（进阶）比较样本协方差与 Ledoit-Wolf 收缩协方差对 GMV 权重的影响：
（a）分别使用两种协方差矩阵计算 GMV 权重；
（b）对比权重向量的 L2 距离 $\|w_{\text{sample}} - w_{\text{LW}}\|_2$；
（c）讨论样本量不足时（如仅取前 60 天数据），差异如何变化。

??? hint "参考思路"
    使用 `sklearn.covariance.LedoitWolf` 拟合收益率矩阵，`lw.covariance_` 即为收缩矩阵。
    样本量越少，收缩系数 `lw.shrinkage_` 越大，GMV 权重差异也越明显。

---

**习题 14.5**（综合）实现一个简单的**风险平价**组合：
（a）写出等风险贡献的优化目标，用 `scipy.optimize.minimize` 求解；
（b）与等权基准、GMV 组合在年化波动率上进行对比；
（c）解释为何风险平价在低相关资产集合中更接近等权？

??? hint "参考思路"
    目标函数：最小化 $\sum_i(RC_i - \sigma_p/N)^2$，其中 $RC_i = w_i(\Sigma w)_i / \sigma_p$。
    使用对数障碍技巧：$-\sum_i\ln w_i$ 保证 $w>0$，然后再归一化。
    当所有资产波动率相同且相关性为零时，等权 $=$ 风险平价。

---

## 14.11 拓展阅读

1. **Markowitz, H. M. (1952)**. “Portfolio Selection.” *Journal of Finance*, 7(1), 77–91.
   — 原始论文，奠定现代组合理论基础，值得精读。

2. **Ledoit, O., & Wolf, M. (2004)**. “Honey, I Shrunk the Sample Covariance Matrix.”
   *Journal of Portfolio Management*, 30(4), 110–119.
   — Ledoit-Wolf 收缩估计的直觉版解释，配合原始数学论文（2003, *Journal of Multivariate Analysis*）一同阅读。

3. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**. “Optimal Versus Naive Diversification.”
   *Review of Financial Studies*, 22(5), 1915–1953.
   — 实证证明等权 $1/N$ 策略在样本外难以被均值-方差策略超越，引发广泛讨论。

4. **Maillard, S., Roncalli, T., & Teïletche, J. (2010)**. “The Properties of Equally Weighted
   Risk Contribution Portfolios.” *Journal of Portfolio Management*, 36(4), 60–70.
   — 风险平价的理论性质与实证分析。

5. **Roncalli, T. (2013)**. *Introduction to Risk Parity and Budgeting*. CRC Press.
   — 系统性教材，涵盖因子风险平价与资产配置实践。
