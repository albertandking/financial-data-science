# 附录B 数学与概率回顾

[![在 Colab 打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertandking/financial-data-science/blob/main/notebooks/appendix_b_math.ipynb) [![在 Binder 打开](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/albertandking/financial-data-science/main?labpath=notebooks/appendix_b_math.ipynb)

!!! info "配套代码"
    `notebooks/appendix_b_math.ipynb`（用 numpy/scipy 演示本附录的关键计算，离线可跑）。

本附录为金融数据科学所需的数学基础做系统回顾，**面向已学过但需要唤醒记忆的读者**。
每节末尾标注「→ 在本书中的应用」，把抽象概念与正文章节对应起来。
若某一处推导暂时跳过也无妨，遇到正文用到时再回查即可。

---

## B.1 线性代数

线性代数是组合优化、因子模型、机器学习的通用语言。金融数据天然是矩阵——
「时间 × 资产」「样本 × 特征」。

### B.1.1 向量与矩阵

**向量** $\mathbf{x} \in \mathbb{R}^n$ 是 $n$ 个有序实数。本书用列向量约定：

$$\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top$$

**矩阵** $A \in \mathbb{R}^{m\times n}$ 有 $m$ 行 $n$ 列，$A_{ij}$ 为第 $i$ 行第 $j$ 列元素。
**转置** $A^\top$ 把行列互换：$(A^\top)_{ij} = A_{ji}$。

### B.1.2 内积、范数与夹角

两向量的**内积**（点积）：

$$\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^n a_i b_i$$

**欧氏范数**（长度）$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^\top \mathbf{x}}$。**余弦相似度**衡量方向接近度：

$$\cos\theta = \frac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$$

> → 在本书中的应用：第13章用余弦相似度做文本检索；第19章 RAG 检索同理。

### B.1.3 矩阵乘法

$A \in \mathbb{R}^{m\times k}$、$B \in \mathbb{R}^{k\times n}$，则 $C = AB \in \mathbb{R}^{m\times n}$：

$$C_{ij} = \sum_{l=1}^k A_{il} B_{lj}$$

注意矩阵乘法**不可交换**（$AB \neq BA$）。Python 中用 `@` 运算符（`A @ B`）。

**组合收益**是最常见的应用。设权重 $\mathbf{w}$、各资产期望收益 $\boldsymbol{\mu}$：

$$\mu_p = \mathbf{w}^\top \boldsymbol{\mu} = \sum_i w_i \mu_i$$

> → 第2章（NumPy 向量化）、第16章（组合优化）。

### B.1.4 特殊矩阵

| 名称 | 定义 | 用途 |
|---|---|---|
| 单位矩阵 $I$ | 对角线为1、其余为0 | $AI = A$ |
| 对称矩阵 | $A = A^\top$ | 协方差矩阵 |
| 对角矩阵 | 非对角元为0 | 独立资产的协方差 |
| 逆矩阵 $A^{-1}$ | $AA^{-1}=I$ | 解线性方程、GMV 组合 |

### B.1.5 二次型与正定性

**二次型**是形如 $\mathbf{x}^\top A \mathbf{x}$ 的标量。**组合方差**就是一个二次型：

$$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w} = \sum_i\sum_j w_i w_j \sigma_{ij} \ge 0$$

若对任意 $\mathbf{x}\neq\mathbf{0}$ 都有 $\mathbf{x}^\top A\mathbf{x} > 0$，称 $A$ **正定**。
协方差矩阵 $\Sigma$ 半正定（方差非负），这保证了均值-方差优化是凸问题。

> → 第5章（风险度量）、第16章（组合优化的凸性）。

### B.1.6 特征值、特征向量与 PCA

若 $A\mathbf{v} = \lambda\mathbf{v}$（$\mathbf{v}\neq\mathbf{0}$），则 $\lambda$ 是**特征值**、$\mathbf{v}$ 是**特征向量**。
对称矩阵可做**谱分解** $A = Q\Lambda Q^\top$，$Q$ 各列为正交特征向量。

**主成分分析（PCA）**对协方差矩阵做谱分解，最大特征值方向是数据方差最大的方向。
在金融中，第一主成分常对应「市场因子」（所有股票同涨同跌）。

> → 第7章（因子的统计基础）、降维与多重共线性诊断。

---

## B.2 微积分与最优化

### B.2.1 导数与梯度

一元函数 $f(x)$ 的导数 $f'(x)$ 是瞬时变化率。多元函数 $f(\mathbf{x})$ 的**梯度**是偏导数构成的向量：

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top$$

梯度指向函数**上升最快**的方向；负梯度指向下降最快方向——这是梯度下降的依据。

**海森矩阵**（Hessian）是二阶偏导构成的矩阵，刻画曲率：$H_{ij} = \dfrac{\partial^2 f}{\partial x_i \partial x_j}$。

### B.2.2 泰勒展开

在 $\mathbf{x}_0$ 附近，$f$ 可二阶近似为：

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top(\mathbf{x}-\mathbf{x}_0)
+ \tfrac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top H (\mathbf{x}-\mathbf{x}_0)$$

> → 第11章（XGBoost 用损失函数的二阶泰勒展开推导分裂增益）。

### B.2.3 凸函数与凸优化

若函数图像上任意两点的连线不低于函数本身，则为**凸函数**；等价地，二阶导数非负
（多元情形海森半正定）。凸优化的**局部最优即全局最优**，这是均值-方差优化、
逻辑回归、岭回归等能可靠求解的根本原因。

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}), \quad \theta\in[0,1]$$

### B.2.4 带约束优化与拉格朗日乘子

求解「在约束 $g(\mathbf{x})=0$ 下最小化 $f(\mathbf{x})$」，构造**拉格朗日函数**：

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda\, g(\mathbf{x})$$

令 $\nabla_{\mathbf{x}}\mathcal{L}=0$、$\partial\mathcal{L}/\partial\lambda=0$ 得最优解。

**例（全局最小方差组合）**：$\min \mathbf{w}^\top\Sigma\mathbf{w}$ s.t. $\mathbf{w}^\top\mathbf{1}=1$，解得

$$\mathbf{w}_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}$$

> → 第16章（组合优化的解析解推导）。

### B.2.5 梯度下降

无解析解时，用迭代法。**梯度下降**沿负梯度方向更新参数：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\nabla f(\boldsymbol{\theta}_t)$$

$\eta$ 为学习率。过大震荡发散，过小收敛慢。变体（SGD、Adam）是深度学习的训练核心。

> → 第9章（正则化回归）、第12章（神经网络训练）。

---

## B.3 概率论

### B.3.1 随机变量、期望与方差

**随机变量** $X$ 把随机结果映射为数值。**期望**（均值）是长期平均：

$$\mathbb{E}[X] = \sum_x x\,p(x) \quad\text{或}\quad \int x f(x)\,dx$$

**方差**衡量离散程度，**标准差**是其平方根：

$$\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mu^2, \qquad \sigma = \sqrt{\mathrm{Var}(X)}$$

期望是线性的：$\mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y]$；方差不是：

$$\mathrm{Var}(aX+bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$$

> → 第5章（收益与风险）、第16章（组合方差正是这条公式的多维推广）。

### B.3.2 常见分布

| 分布 | 记号 | 金融用途 |
|---|---|---|
| 正态 | $N(\mu,\sigma^2)$ | 收益率的一阶近似、参数法 VaR |
| 学生 t | $t_\nu$ | 厚尾收益率建模 |
| 对数正态 | $\ln X \sim N$ | 价格（恒正）、几何布朗运动 |
| 伯努利/二项 | $\text{Bern}(p)$ | 涨跌方向、违约（0/1） |
| 泊松 | $\text{Pois}(\lambda)$ | 单位时间事件数（如违约次数） |

**正态分布**密度：$f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$。
但真实收益率比正态**更厚尾**（见第4章），故风险度量常改用 t 分布或历史模拟法。

### B.3.3 偏度与峰度

**偏度**（三阶标准矩）衡量不对称：负偏表示左尾长（暴跌比暴涨剧烈）。
**峰度**（四阶标准矩）衡量尾部厚度，正态峰度为 3，**超额峰度** = 峰度 − 3：

$$\text{Skew} = \mathbb{E}\!\left[\Big(\tfrac{X-\mu}{\sigma}\Big)^3\right], \qquad
\text{Kurt} = \mathbb{E}\!\left[\Big(\tfrac{X-\mu}{\sigma}\Big)^4\right]$$

> → 第4章（风格化事实）、第5章（厚尾对 VaR 的影响）。

### B.3.4 联合分布、协方差与相关

**协方差**衡量两变量同向变动程度，**相关系数**是其标准化版本（$\in[-1,1]$）：

$$\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)], \qquad
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}$$

多资产时，所有两两协方差构成**协方差矩阵** $\Sigma$（对称半正定）。

> → 第4章（相关性热力图）、第7章（贝塔 = $\mathrm{Cov}(r_i,r_m)/\mathrm{Var}(r_m)$）、第16章。

### B.3.5 条件期望与独立

**条件期望** $\mathbb{E}[Y\mid X]$ 是「已知 $X$ 后对 $Y$ 的最优预测」，是一切回归与监督学习的理论目标。
**独立**意味着 $X$ 不携带关于 $Y$ 的信息：$p(x,y)=p(x)p(y)$。注意金融收益率「近似不相关」
**不等于**独立——波动率聚集就是高阶相依（见第4、6章）。

### B.3.6 大数定律与中心极限定理

**大数定律（LLN）**：样本均值随样本量增大收敛到真实期望——这是「用历史均值估计期望收益」的依据。

**中心极限定理（CLT）**：大量独立随机变量之和近似正态，与单个分布形态无关：

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)$$

> → 解释了为何低频（月、年）收益比日收益更接近正态（聚合高斯性，第4章）。

---

## B.4 统计推断

### B.4.1 估计量的性质

用样本估计总体参数。好的估计量应：

- **无偏**：$\mathbb{E}[\hat\theta] = \theta$（平均而言不偏）
- **一致**：$\hat\theta \xrightarrow{p} \theta$（样本越大越准）
- **有效**：在无偏估计中方差最小

!!! warning "金融中的估计误差"
    期望收益 $\mu$ 极难估准（信噪比低，见第1章），其估计误差会被组合优化**放大**。
    这正是第16章引入 Ledoit-Wolf 协方差收缩、以及等权组合常胜的原因。

### B.4.2 极大似然估计（MLE）

选择使观测数据出现概率最大的参数：

$$\hat{\boldsymbol\theta}_{\text{MLE}} = \arg\max_{\boldsymbol\theta} \prod_i p(x_i\mid\boldsymbol\theta)
= \arg\max_{\boldsymbol\theta} \sum_i \ln p(x_i\mid\boldsymbol\theta)$$

> → 第6章（ARIMA/GARCH 用 MLE 估计）、第9章（逻辑回归的交叉熵即负对数似然）。

### B.4.3 置信区间与假设检验

**假设检验**用数据判断某命题是否成立：

1. 设原假设 $H_0$（如「该因子无效，系数=0」）与备择 $H_1$；
2. 计算检验统计量（如 $t = \hat\beta/\mathrm{se}(\hat\beta)$）；
3. 由 **p 值** 判断：$p<\alpha$（如 0.05）则拒绝 $H_0$。

**p 值**是「若 $H_0$ 为真，观测到当前或更极端结果的概率」，**不是**「$H_0$ 为真的概率」。

| | $H_0$ 真 | $H_0$ 假 |
|---|---|---|
| 拒绝 $H_0$ | 第一类错误（$\alpha$，假阳） | 正确 |
| 不拒绝 | 正确 | 第二类错误（$\beta$，假阴） |

> → 第6章（ADF/Ljung-Box 检验）、第7章（因子显著性）、第8章（系数 t 检验）。

### B.4.4 多重检验问题

!!! danger "金融研究的头号陷阱"
    若同时检验 100 个无效因子，在 5% 显著性下平均会有约 5 个「碰巧显著」。
    反复试不同策略/参数直到「跑出」漂亮结果，即**数据窥探（data snooping）**。
    缓解：Bonferroni 校正、样本外验证、控制错误发现率（FDR）。这是第9、17章
    反复强调样本外检验与时序交叉验证的统计学根源。

---

## B.5 回归基础

**线性回归**用矩阵形式写为 $\mathbf{y} = X\boldsymbol\beta + \boldsymbol\varepsilon$，
**最小二乘（OLS）**解为：

$$\hat{\boldsymbol\beta} = (X^\top X)^{-1} X^\top \mathbf{y}$$

经典假设：误差零均值、同方差、无自相关、与自变量不相关。金融数据常违反同方差与
无自相关假设，故需**稳健/聚类标准误**（第8章）。$R^2$ 衡量被解释的方差比例。

> → 第7章（因子回归）、第8章（面板回归）、第9章（监督学习基础）。

---

## B.6 进一步阅读

- **线性代数**：Gilbert Strang,《Introduction to Linear Algebra》
- **概率统计**：Wasserman,《All of Statistics》；陈希孺,《概率论与数理统计》
- **凸优化**：Boyd & Vandenberghe,《Convex Optimization》（免费电子版）
- **计量经济学**：Wooldridge,《计量经济学导论》
