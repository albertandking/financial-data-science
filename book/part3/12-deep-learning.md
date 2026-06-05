# 第12章 深度学习与序列模型

!!! info "配套代码"
    `notebooks/ch12_deep_learning.ipynb`（使用 PyTorch，需 `--extra advanced`）

## 本章导读

过去十年，深度学习在图像识别、自然语言处理和语音合成等领域取得了革命性进展。金融界对此也充满期待：价格序列看似比图像更"规则"，是否可以用深度网络直接从历史数据中"学出"市场规律？

本章将系统回答这个问题。我们从神经元的基本运算出发，逐步构建多层感知机（MLP）与循环神经网络（RNN/LSTM/GRU），最终用 PyTorch 搭建一个真实的 LSTM 波动率预测器，并与传统基线模型做严格的样本外对比。更重要的是，我们将深入讨论金融序列的特殊困境——**低信噪比与小样本**——以及深度模型在这种环境中的局限性。

## 学习目标

完成本章学习后，读者应能够：

1. 理解神经网络的基本构成：神经元、激活函数、前向传播与反向传播
2. 掌握多层感知机的训练流程：epoch、batch、优化器与学习率
3. 理解 RNN 的结构缺陷（梯度消失/爆炸），以及 LSTM 如何通过门控机制解决该问题
4. 正确构造金融序列的滑动窗口样本，避免前视偏差
5. 用 PyTorch 实现并训练一个 LSTM 序列预测模型
6. 运用 Dropout、早停、权重衰减等方法控制过拟合
7. 客观评估深度学习在金融预测中的优势与局限

---

## 12.1 神经网络基础

### 12.1.1 神经元与激活函数

神经网络的基本计算单元是**人工神经元**，模仿生物神经元的"积累-激发"行为：

$$
z = \mathbf{w}^{\top} \mathbf{x} + b, \quad a = f(z)
$$

其中 $\mathbf{x} \in \mathbb{R}^d$ 是输入特征，$\mathbf{w}$ 是权重，$b$ 是偏置，$f(\cdot)$ 是**激活函数**，$a$ 是激活值（输出）。激活函数引入非线性，使网络能够拟合复杂函数。常用激活函数见下表：

| 激活函数 | 表达式 | 输出范围 | 特点 |
|---|---|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $(0, 1)$ | 光滑，但梯度消失严重；常用于输出层（二分类） |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | 零中心，仍有梯度消失问题 |
| ReLU | $\max(0, z)$ | $[0, +\infty)$ | 训练快，无梯度消失（正区间）；存在"死亡神经元" |
| Leaky ReLU | $\max(\alpha z, z)$ | $\mathbb{R}$ | 解决死亡神经元，$\alpha=0.01$ |

**在金融序列模型中**，隐藏层通常选择 ReLU；LSTM 内部门控使用 Sigmoid 与 Tanh。

### 12.1.2 前向传播

一个 $L$ 层全连接网络的前向计算：

$$
\mathbf{a}^{(0)} = \mathbf{x}, \quad
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad
\mathbf{a}^{(l)} = f^{(l)}\!\left(\mathbf{z}^{(l)}\right)
$$

最终输出 $\hat{y} = \mathbf{a}^{(L)}$，例如对回归任务取线性输出，对分类任务经过 Softmax 得到概率。

### 12.1.3 损失函数

模型训练的目标是最小化**损失函数** $\mathcal{L}$：

- **均方误差（MSE）**，用于回归：
  $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **二元交叉熵（BCE）**，用于二分类：
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]$$

### 12.1.4 反向传播与梯度下降

训练过程本质上是用**梯度下降**最小化损失：

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

其中 $\eta$ 是**学习率**（learning rate）。梯度通过**反向传播**算法高效计算——本质上是链式法则从输出层逐层向输入层传播梯度。

**Adam 优化器**是目前最常用的变种，它对每个参数维护一阶矩（梯度均值）和二阶矩（梯度方差），自适应调整有效学习率：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

典型超参数：$\beta_1 = 0.9,\; \beta_2 = 0.999,\; \epsilon = 10^{-8}$。

---

## 12.2 多层感知机（MLP）

**多层感知机**（Multi-Layer Perceptron, MLP）由一个输入层、若干隐藏层和一个输出层组成。以下是用 PyTorch 定义一个两层 MLP 的典型写法：

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

### 12.2.1 训练流程核心概念

| 概念 | 含义 |
|------|------|
| **epoch** | 对训练集完整遍历一次 |
| **batch（mini-batch）** | 每次参数更新所使用的样本数，典型值 32/64/128 |
| **iteration** | 每处理一个 batch 完成一次前向+后向+参数更新 |
| **学习率（lr）** | 控制每步参数更新幅度，过大震荡，过小收敛慢 |
| **学习率调度** | 随训练进行动态降低学习率，如 `CosineAnnealingLR` |

!!! tip "金融序列的 batch 构造"
    对时间序列，每个 batch 中的样本应来自**不同时间窗口**，而非随机打乱（打乱会破坏 batch normalization 的时序语义，但对 LSTM 的最终训练通常无害）。更谨慎的做法是保持时序顺序训练。

---

## 12.3 循环神经网络（RNN）与 LSTM

### 12.3.1 标准 RNN 的结构

金融价格序列、收益率序列等是典型的**时间序列**，具有"历史信息影响当前"的特性。标准循环神经网络（RNN）通过隐藏状态 $\mathbf{h}_t$ 传递历史信息：

$$
\mathbf{h}_t = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})
$$

其中 $\mathbf{x}_t$ 是 $t$ 时刻的输入（如当日特征），$\mathbf{h}_{t-1}$ 是前一时刻的隐藏状态。

**梯度消失/爆炸问题**：在反向传播时，梯度需要沿时间步链式相乘。若 $\|\mathbf{W}_h\| < 1$，梯度指数衰减（**梯度消失**）；若 $\|\mathbf{W}_h\| > 1$，梯度指数爆炸（**梯度爆炸**）。对于金融序列中常见的长期相关性（如月度动量），标准 RNN 几乎无法捕捉。

### 12.3.2 LSTM：长短期记忆网络

LSTM（Long Short-Term Memory）由 Hochreiter & Schmidhuber（1997）提出，通过**门控机制**精细控制信息的记忆与遗忘：

**遗忘门**（Forget Gate）：决定从上一时刻记忆中"忘掉"多少：

$$
\mathbf{f}_t = \sigma\!\left(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)
$$

**输入门**（Input Gate）：决定向记忆中"写入"多少新信息：

$$
\mathbf{i}_t = \sigma\!\left(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right),\quad
\tilde{\mathbf{c}}_t = \tanh\!\left(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right)
$$

**记忆更新**（Cell State）：

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**输出门**（Output Gate）：决定从记忆中"读出"多少作为当前输出：

$$
\mathbf{o}_t = \sigma\!\left(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right),\quad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

其中 $\odot$ 表示逐元素相乘，$\sigma$ 为 Sigmoid 函数。

!!! note "LSTM 的关键直觉"
    记忆单元 $\mathbf{c}_t$ 就像一条"信息高速公路"，梯度能沿此通道无衰减地长程流动。遗忘门接近 1 时保留历史（如季节性），接近 0 时忘记历史（如异常事件后重置）。这种自适应记忆机制正是 LSTM 优于标准 RNN 的根本原因。

### 12.3.3 GRU：门控循环单元

GRU（Gated Recurrent Unit, Cho et al. 2014）是 LSTM 的简化版，将遗忘门与输入门合并为**更新门** $\mathbf{z}_t$，用**重置门** $\mathbf{r}_t$ 控制历史隐藏状态的使用：

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]),\quad
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]),\quad
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

GRU 参数量更少，在小样本金融数据集上往往不亚于 LSTM，且训练更快。

---

## 12.4 金融序列建模的工程要点

深度学习用于金融序列预测时，有几个容易出错的工程细节，直接决定结果是否可信。

### 12.4.1 滑动窗口构造样本

金融序列预测通常转化为**监督学习**问题：给定过去 $T$ 步的特征，预测未来某个目标。

```
输入序列：x_{t-T+1}, x_{t-T+2}, ..., x_t   →   输出：y_{t+1}
```

滑动窗口（look-back window）大小 $T$ 是关键超参数：
- $T$ 太小：捕捉不到中期趋势（如月度动量）
- $T$ 太大：增加模型复杂度，样本减少

**代码示意**：

```python
def make_sequences(X, y, T):
    """构造 (n_samples, T, n_features) 的序列样本，严格防前视。"""
    Xs, ys = [], []
    for i in range(T, len(X)):
        Xs.append(X[i-T:i])   # 第 i-T 到 i-1 步作为输入
        ys.append(y[i])       # 第 i 步作为目标
    return np.array(Xs), np.array(ys)
```

!!! warning "防前视偏差"
    `X[i-T:i]` 使用的是第 $t-T$ 到 $t-1$ 时刻的数据，预测 $t$ 时刻的目标 `y[i]`。构造特征时，**所有特征必须已经使用了 `.shift(1)`**，即用前一日数据预测当日结果。任何一步疏漏都会造成前视偏差，使测试集表现虚高。

### 12.4.2 按时间切分训练/验证集

金融数据有强烈的时序依赖，**绝对不能随机切分**训练与测试集：

```python
# 正确做法：按时间前后切分
split = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]
```

这样可以保证验证集完全在训练集之后，模拟真实的样本外预测场景。

### 12.4.3 标准化只用训练集统计量

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_2d)           # 只用训练集的均值和标准差
X_train_sc = scaler.transform(X_train_2d)
X_val_sc   = scaler.transform(X_val_2d)   # 不能在验证集上重新 fit！
```

!!! warning "常见错误：信息泄露"
    若在整个数据集上做标准化（`scaler.fit(X_all)`），验证集的均值和方差信息就"泄露"进了训练过程，造成乐观的评估结果。在金融场景中这会被视为严重的方法论错误。

---

## 12.5 过拟合控制

金融序列的信噪比极低（可预测的信号远少于噪声），深度模型极容易把噪声"记住"而非学习真实规律。以下是最常用的正则化手段。

### 12.5.1 Dropout

Dropout（Srivastava et al. 2014）在训练时以概率 $p$ 随机将神经元置零：

$$
\tilde{a}_j = \begin{cases} a_j / (1-p) & \text{以概率 } 1-p \\ 0 & \text{以概率 } p \end{cases}
$$

测试时关闭 Dropout（`model.eval()`），等效于对所有 Dropout 子网络的平均。

**对 LSTM 的 Dropout**：一般加在 LSTM 输出到全连接层之间；也可以通过 `nn.LSTM(dropout=p)` 对 LSTM 的层间输出施加 Dropout（序列内部的 Dropout 需用 VariationalDropout）。

### 12.5.2 早停（Early Stopping）

监控验证集损失，当连续若干 epoch 不再改善时停止训练，并保存最优模型权重：

```python
best_val_loss = float('inf')
patience, wait = 10, 0

for epoch in range(max_epochs):
    # ... 训练 ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'早停于 epoch {epoch}')
            break
model.load_state_dict(torch.load('best_model.pt'))
```

### 12.5.3 权重衰减（L2 正则化）

在优化器中通过 `weight_decay` 参数实现：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

等效于在损失中添加 $\lambda \|\mathbf{W}\|_2^2$，惩罚过大的权重。

!!! warning "过拟合在金融预测中尤为危险"
    在图像分类中，验证集准确率 95% 通常意味着模型真的学到了模式。但在金融收益率预测中，"高训练精度、低测试精度"极为常见——模型记住了训练期的随机噪声。

    **症状**：训练 loss 持续下降，验证 loss 在早期就开始上升（或持平），二者之间出现明显"剪刀差"。

    **应对**：减小网络规模、增大 Dropout、增强 L2 正则、增加数据量、缩短训练 epoch。

---

## 12.6 深度学习 vs 传统模型：理性评估

### 12.6.1 深度学习的潜在优势

- **非线性**：自动学习特征之间的高阶交互，无需人工构造交叉特征
- **序列建模**：LSTM/GRU 能自然处理变长时间依赖，无需手动设计滞后阶数
- **端到端**：从原始价格/文本直接到预测输出，减少特征工程工作量

### 12.6.2 金融场景的核心困境

!!! warning "深度学习在金融预测中未必更好"
    **低信噪比**：股票日收益率中可预测的部分可能不足 1%，深度网络的大量参数几乎全部用于拟合噪声。

    **小样本**：A 股历史数据通常只有 10-20 年，即 2500-5000 个交易日。而 LSTM 通常需要数万样本才能可靠训练。

    **非平稳性**：市场制度（监管政策、投资者结构）频繁变化，训练期学到的规律可能在测试期完全失效。

    **数据挖掘偏差**：在有限数据上反复调参，极易找到"看起来好"但不具备泛化能力的超参数组合。

| 对比维度 | LSTM | ARIMA/GARCH | 随机森林/XGBoost |
|---------|------|------------|----------------|
| 特征工程 | 较少 | 需手动指定阶数 | 需构造特征 |
| 样本需求 | 大（数万+） | 小（数百即可） | 中等 |
| 可解释性 | 差 | 强（有统计检验） | 中等（特征重要性）|
| 过拟合风险 | 高 | 低 | 中 |
| 捕捉非线性 | 强 | 弱 | 强 |
| 实践表现 | 不稳定 | 波动率建模稳健 | 涨跌分类尚可 |

**实践建议**：将 LSTM 作为**集成成员之一**，而非独立押注；将传统时间序列模型（ARIMA、GARCH）作为强基线；用**严格的滚动样本外测试**（walk-forward validation）评估所有模型。

---

## 12.7 实战：LSTM 预测波动率

本节搭建一个完整的 LSTM 预测流程，预测目标为**实现波动率**（rolling standard deviation of returns）。波动率序列比收益率方向更具可预测性，是深度学习在金融中最有希望的应用之一。

### 12.7.1 数据准备与特征

```python
import torch
import torch.nn as nn
from fds import load_sample_prices, daily_returns
import numpy as np, pandas as pd

torch.manual_seed(0)

prices = load_sample_prices()
rets   = daily_returns(prices)
stock  = rets['TECH']

# 目标：未来5日实现波动率（已知，用前一日构造）
vol5 = stock.rolling(5).std().shift(-5).dropna()
```

### 12.7.2 定义 LSTM 模型

```python
class VolatilityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 取最后时间步
        return self.fc(out).squeeze(-1)
```

### 12.7.3 训练与评估

```python
model    = VolatilityLSTM(input_dim=3, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
```

完整可运行代码详见配套 notebook。

---

## 12.8 本章小结

本章系统介绍了深度学习在金融序列建模中的理论与实践：

1. **基础架构**：神经元 → MLP → RNN → LSTM/GRU，每一层抽象都在解决前一层的缺陷
2. **门控机制**：LSTM 通过遗忘/输入/输出三门控制信息流，解决了 RNN 的梯度消失问题
3. **工程规范**：滑动窗口防前视、时序切分、训练集专属标准化——这三点是金融时序建模的"铁律"
4. **过拟合控制**：Dropout + 早停 + 权重衰减是对抗噪声的三重防线
5. **理性评估**：深度学习在低信噪比、小样本的金融场景中优势有限，严格的样本外测试是唯一可信的评估标准

!!! tip "本章核心外卖"
    在金融预测中，**方法论正确性**（防前视、样本外测试）比模型选择更重要。一个简单的基线模型配上严格的评估流程，往往比一个复杂但方法论有漏洞的深度模型更可靠。

---

## 习题

**1. LSTM 门控推导**

写出 LSTM 遗忘门 $\mathbf{f}_t = 1$（全保留）、$\mathbf{f}_t = 0$（全遗忘）时记忆单元 $\mathbf{c}_t$ 的更新方程。这两种极端情形分别对应什么金融场景？

??? note "参考思路"
    全保留（$\mathbf{f}_t=1$）：$\mathbf{c}_t = \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$，历史记忆完全继承，类比于趋势市场中的动量效应。
    全遗忘（$\mathbf{f}_t=0$）：$\mathbf{c}_t = \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$，完全依赖当前输入，类比于重大政策冲击后的市场重置。

**2. 滑动窗口的前视检查**

给定如下代码，指出是否存在前视偏差：

```python
feat_df['vol'] = rets.rolling(5).std()          # 特征
feat_df['target'] = (rets.shift(-1) > 0).astype(int)  # 目标
X = feat_df[['vol']].values
y = feat_df['target'].values
```

??? note "参考思路"
    `vol` 使用的是当日收益率（`rets` 未 shift），而 `target` 是次日涨跌。当日收益率本身不构成前视（已发生），但若 `rets` 中包含当日收盘价计算的特征，则需确认用的是开盘前可用数据还是收盘后数据。若是收盘后，用收盘价特征预测当日收盘涨跌才是前视；预测次日涨跌则无问题。总体而言该代码无明显前视，但需确认 rolling 窗口的 min_periods 设置。

**3. GRU vs LSTM 选择**

在数据量仅有 2 年（约 500 个交易日）的小样本场景下，你会优先选择 GRU 还是 LSTM？说明理由并给出两种模型的参数量计算公式（输入维度 $d$，隐藏维度 $h$）。

??? note "参考思路"
    优先选 GRU。LSTM 参数量为 $4h(h+d+1)$，GRU 为 $3h(h+d+1)$，约少 25%。小样本下 GRU 过拟合风险更低，训练更稳定。两模型性能通常差异不大，GRU 是小样本金融场景的默认选择。

**4. 早停与信息泄露**

某同学使用整个数据集的均值/方差做标准化，再按 8:2 切分训练/验证，发现验证集损失明显低于"先切分后标准化"的做法。请解释原因并说明正确流程。

??? note "参考思路"
    "先标准化后切分"将验证集的统计信息（均值、方差）引入了训练过程，等同于"偷看"了未来数据——这是信息泄露。正确流程：①先按时序切分；②在训练集上 fit scaler；③用训练集 scaler transform 训练集和验证集。该错误在实践中极为常见，是金融回测失真的主要来源之一。

**5. 深度学习在波动率预测中的优势**

相比于 GARCH(1,1) 模型，LSTM 在波动率预测中可能有哪些优势？又有哪些劣势？在什么条件下你会选择 LSTM 而非 GARCH？

??? note "参考思路"
    **优势**：LSTM 可利用多特征（成交量、跨市场信息等），捕捉非线性的波动率聚集模式；在数据量充足时可能更灵活。
    **劣势**：需要大量数据；GARCH 有严格的统计推断框架（参数显著性检验、似然比检验）；GARCH 的平稳性和矩条件可验证；LSTM 是黑箱。
    **选择条件**：当有高频或多源特征、数据量充足（5年以上日频或更高频）、任务是集成预测而非单独预测时，可考虑引入 LSTM 作为集成成员。

---

## 拓展阅读

- **Goodfellow, Bengio & Courville（2016）**《Deep Learning》，MIT Press。第10章专门讲序列建模与 RNN，第11章讲正则化，均有深度数学推导。在线免费：[deeplearningbook.org](https://www.deeplearningbook.org)

- **Hochreiter & Schmidhuber（1997）**"Long Short-Term Memory"，*Neural Computation* 9(8)。LSTM 原始论文，仍是理解门控机制的最佳一手资料。

- **Sezer, Gudelek & Ozbayoglu（2020）**"Financial time series forecasting with deep learning: A systematic literature review"，*Applied Soft Computing*。系统综述了 2005-2019 年深度学习用于金融预测的 140+ 篇论文，结论是"没有一致性胜者"。

- **Gu, Kelly & Xiu（2020）**"Empirical Asset Pricing via Machine Learning"，*Review of Financial Studies*。用机器学习（含神经网络）预测美国股票横截面收益率，是金融 ML 领域的重要实证研究。

- **Lim & Zohren（2021）**"Time-series forecasting with deep learning: a survey"，*Philosophical Transactions of the Royal Society A*。覆盖 Transformer 在时序预测中的应用，对第 17 章（大模型）有铺垫作用。
