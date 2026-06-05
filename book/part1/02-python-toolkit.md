# 第2章 Python 数据科学工具栈

!!! info "配套代码"
    本章代码见 `notebooks/ch02_python_toolkit.ipynb`，可逐格运行（依赖内置数据，离线可跑）。运行前请先执行 `uv run python scripts/make_sample_data.py` 生成内置数据集。

## 2.1 本章导读与学习目标

金融数据科学离不开一套经过实战验证的工具栈。本章从项目管理（uv）开始，
系统介绍 NumPy 的向量化思维、Pandas 的时间序列能力，以及如何高效存储与可视化金融数据。
这些工具将贯穿全书——后续章节的每一个模型、每一张图，都建立在本章基础之上。

**学习目标**

- 理解本书的项目结构，掌握 uv 环境管理的核心命令与依赖组机制
- 掌握 NumPy `ndarray` 的 dtype、轴(axis)概念、向量化运算与广播规则
- 能比较向量化与 for 循环的性能差距，理解其背后原因
- 掌握 Pandas `Series`/`DataFrame` 的创建、索引（`loc`/`iloc`/布尔索引）
- 能处理 `DatetimeIndex`、缺失值、`groupby`、`merge`、`resample`、`rolling`/`ewm`
- 了解 CSV 与 Parquet 格式的差异，选择合适的存储方案
- 能绘制含中文标注的时间序列图

---

## 2.2 为什么是 Python

金融数据科学的事实标准语言是 Python，原因主要有三：

1. **生态完整**：从数据获取（akshare、tushare）、数值计算（NumPy）、数据处理（Pandas）、
   统计建模（statsmodels）到机器学习（scikit-learn）、深度学习（PyTorch），
   所有工具均可无缝协作。
2. **可读性强**：代码贴近数学伪代码，如 $\mathbf{r} = \mathbf{P}_{t}/\mathbf{P}_{t-1} - 1$
   可直接写成 `r = P[1:] / P[:-1] - 1`，便于教学与学术协作。
3. **可复现性**：配合 uv 等现代工具，环境与依赖可一键复现，是"可复现研究"的重要保障。

### 2.2.1 uv 环境管理

本书采用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境与依赖。
uv 由 Rust 编写，速度极快，能在几秒内完成环境创建与依赖解析。

**核心命令速查表**

| 命令 | 说明 |
|------|------|
| `uv sync` | 按 `pyproject.toml` 安装全部依赖（含 `uv.lock` 锁定版本） |
| `uv sync --extra data` | 额外安装数据获取组（akshare、tushare 等） |
| `uv sync --extra advanced` | 额外安装进阶组（XGBoost、PyTorch 等） |
| `uv run python xxx.py` | 在项目虚拟环境中运行脚本 |
| `uv run jupyter lab` | 在项目环境中启动 JupyterLab |
| `uv add numpy` | 添加新依赖并更新 `uv.lock` |
| `uv remove numpy` | 移除依赖 |

**pyproject.toml 结构**

```toml
[project]
name = "financial-data-science"
requires-python = ">=3.11"

dependencies = [          # 核心依赖：所有章节必须
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=16.0",
]

[project.optional-dependencies]
data = ["akshare>=1.14"]   # 可选：数据获取
advanced = ["torch>=2.2"]  # 可选：深度学习

[dependency-groups]
docs = ["mkdocs>=1.6", "jupyter>=1.0", "nbconvert>=7.16"]
dev  = ["pytest>=8.0"]
```

!!! tip "可复现研究的关键"
    把 `pyproject.toml` 和 `uv.lock` 一同提交到 Git 仓库。
    合作者只需 `uv sync` 一条命令，就能得到**完全一致**的环境，
    包括 Python 版本和所有第三方包的精确版本号。

### 2.2.2 项目目录结构

```
financial-data-science/
├── book/           # Markdown 正文
│   └── part1/
├── notebooks/      # 配套 Jupyter Notebook
├── scripts/        # 数据生成与工具脚本
├── src/fds/        # 本书复用工具包
├── data/
│   ├── raw/        # 原始数据（不提交 Git）
│   └── processed/  # 处理后数据
├── pyproject.toml
└── uv.lock
```

---

## 2.3 NumPy：向量化思维

### 2.3.1 ndarray 与 dtype

NumPy 的核心是 `ndarray`（N 维数组）。与 Python 列表不同，`ndarray` 要求
所有元素具有**相同的数据类型（dtype）**，从而能存储在连续内存块中，
大幅提升计算效率。

```python
import numpy as np

# 整数数组：dtype 自动推断为 int64
a = np.array([1, 2, 3, 4])
print(a.dtype)   # int64

# 浮点数组：金融计算默认用 float64
prices = np.array([10.0, 10.5, 10.3, 10.8])
print(prices.dtype)   # float64

# 显式指定 dtype
b = np.array([1, 2, 3], dtype=np.float32)   # 节省内存，牺牲精度
```

常用 dtype：

| dtype | 描述 | 金融场景 |
|-------|------|---------|
| `float64` | 双精度浮点 | **默认，价格/收益率** |
| `float32` | 单精度浮点 | 大规模深度学习特征 |
| `int64` | 64 位整数 | 成交量、股票代码 |
| `bool` | 布尔 | 掩码、条件筛选 |

### 2.3.2 轴（axis）概念

`ndarray` 的每个维度称为一个**轴**。金融中最常见的二维数组是"时间 × 标的"：

```
axis=0 ↓（沿行，对时间维度聚合）
      BANK  LIQUOR  TECH  UTILITY
t=1 [  10      50    100       20 ]
t=2 [  11      52     98       21 ]
t=3 [  10      51    102       19 ]
          → axis=1（沿列，对标的维度聚合）
```

```python
mat = np.array([[10, 50, 100, 20],
                [11, 52,  98, 21],
                [10, 51, 102, 19]], dtype=float)

mat.mean(axis=0)   # 每只股票的均价，shape (4,)
mat.mean(axis=1)   # 每天的截面均价，shape (3,)
mat.std(axis=0)    # 每只股票的价格波动
```

### 2.3.3 向量化 vs for 循环：性能对比

向量化的速度优势来自两个方面：
1. NumPy 底层用 C/Fortran 实现，避免 Python 解释器开销
2. 数据连续存储在内存，CPU 缓存命中率高

```python
import time

n = 1_000_000
prices = np.random.rand(n) * 100 + 50

# 方式一：for 循环（慢）
t0 = time.perf_counter()
returns_loop = []
for i in range(1, len(prices)):
    returns_loop.append(prices[i] / prices[i-1] - 1)
t_loop = time.perf_counter() - t0

# 方式二：向量化（快）
t0 = time.perf_counter()
returns_vec = prices[1:] / prices[:-1] - 1
t_vec = time.perf_counter() - t0

print(f"循环耗时：{t_loop*1000:.1f} ms")
print(f"向量化：  {t_vec*1000:.1f} ms")
print(f"加速比：  {t_loop/t_vec:.0f}x")
# 典型输出：循环 ~400 ms，向量化 ~3 ms，加速约 100x
```

!!! warning "避免在金融计算中写 for 循环"
    当数据量超过万条（A 股日度数据很容易达到几十万行），for 循环会成为明显瓶颈。
    本书所有示例均采用向量化写法，请养成"先想有没有数组运算"的思维习惯。

### 2.3.4 广播规则（Broadcasting）

当两个形状不同的数组运算时，NumPy 按以下规则自动"广播"对齐：

1. 从最后一维开始对齐，维度数不足的在左侧补 1
2. 每个维度大小必须相同，**或**其中一个为 1（则自动扩展）
3. 任何维度不满足上述条件则报错

**实例：给多只股票去均值（标准化截面）**

```python
rng = np.random.default_rng(42)
mat = rng.standard_normal((250, 4))   # 250 天 × 4 只股票

# mat.mean(axis=0) 形状为 (4,)
# mat 形状为 (250, 4)
# 广播：(4,) → (1, 4) → (250, 4)，再做减法
demeaned = mat - mat.mean(axis=0)
print(demeaned.mean(axis=0).round(10))   # 每列均值 ≈ 0
```

**实例：组合权重 × 收益矩阵**

设 $\mathbf{w}$ 为 4 只股票的权重向量，$\mathbf{R}$ 为 $T \times 4$ 收益矩阵，
则每日组合收益为：

$$r_p = \mathbf{R} \cdot \mathbf{w} \quad \Leftrightarrow \quad \mathbf{R} @ \mathbf{w}$$

```python
w = np.array([0.4, 0.3, 0.2, 0.1])    # 权重，shape (4,)
R = rng.standard_normal((250, 4))      # 收益矩阵，shape (250, 4)

# 矩阵乘（@ 运算符，等价于 np.dot）
portfolio_ret = R @ w                   # shape (250,)

# 广播版：每列乘以对应权重后求和
portfolio_ret2 = (R * w).sum(axis=1)   # 结果相同
print(np.allclose(portfolio_ret, portfolio_ret2))   # True
```

!!! info "为什么用 `@`"
    `@` 是 Python 3.5+ 引入的矩阵乘运算符（PEP 465）。
    `np.dot` 与 `@` 对二维数组等价，但 `@` 更易读、与数学符号直接对应，
    本书后续讨论均值-方差优化（$w^\top \mu$、$w^\top \Sigma w$）时将大量使用。

### 2.3.5 随机数：default_rng

NumPy 1.17 起推荐用 `default_rng`，避免全局随机状态：

```python
rng = np.random.default_rng(seed=42)   # 固定种子，保证可复现

# 正态分布（均值, 标准差, 形状）
noise = rng.normal(loc=0.0, scale=0.01, size=(252, 4))

# 均匀分布
u = rng.uniform(0, 1, size=1000)

# 标准正态
z = rng.standard_normal((100, 4))
```

!!! tip "可复现性"
    固定 `seed` 是科研和教学的好习惯：结果可被他人精确复现，
    方便排查 bug 和撰写报告。

### 2.3.6 基础线性代数：点积与矩阵乘

| 操作 | 代码 | 说明 |
|------|------|------|
| 向量点积 $\mathbf{a}^\top \mathbf{b}$ | `a @ b` | 标量 |
| 矩阵向量乘 $A\mathbf{x}$ | `A @ x` | 向量 |
| 矩阵乘 $AB$ | `A @ B` | 矩阵 |
| 转置 $A^\top$ | `A.T` | 矩阵 |

```python
mu = np.array([0.10, 0.12, 0.15, 0.08])   # 预期年化收益
w  = np.array([0.25, 0.25, 0.25, 0.25])   # 等权组合

# 组合预期收益：w^T μ
port_mu = w @ mu
print(f"组合预期收益：{port_mu:.2%}")   # 11.25%

# 协方差矩阵（后续第5章详细推导）
Sigma = np.diag([0.04, 0.09, 0.16, 0.01])   # 对角（无相关）示例
port_var = w @ Sigma @ w                       # w^T Σ w
print(f"组合方差：{port_var:.4f}")
print(f"组合波动：{port_var**0.5:.2%}")
```

---

## 2.4 Pandas：金融数据的主力

### 2.4.1 Series 与 DataFrame

```python
import pandas as pd

# Series：带索引的一维数组
price_series = pd.Series(
    [10.5, 11.0, 10.8, 11.3, 11.1],
    index=pd.date_range("2025-01-02", periods=5, freq="B"),
    name="BANK"
)

# DataFrame：带行列标签的二维表
data = {
    "BANK":    [10.5, 11.0, 10.8],
    "LIQUOR":  [50.2, 51.5, 50.8],
    "TECH":    [100.3, 98.7, 101.2],
    "UTILITY": [20.1, 20.5, 20.3],
}
df = pd.DataFrame(data, index=pd.date_range("2025-01-02", periods=3, freq="B"))
```

### 2.4.2 索引：loc / iloc / 布尔索引

这是 Pandas 初学者最容易混淆的地方：

| 方法 | 基于 | 示例 |
|------|------|------|
| `loc[行标签, 列标签]` | **标签** | `df.loc["2025-01-02", "BANK"]` |
| `iloc[行整数, 列整数]` | **位置** | `df.iloc[0, 0]` |
| 布尔索引 | 条件 | `df[df["BANK"] > 11]` |

```python
from fds import load_sample_prices

prices = load_sample_prices()

# loc：按日期标签切片（包含两端）
q1_2025 = prices.loc["2025-01-01":"2025-03-31"]

# loc：选行 + 选列
bank_2025 = prices.loc["2025", "BANK"]

# iloc：第 0 到 4 行，全部列
first5 = prices.iloc[:5, :]

# 布尔索引：找 TECH 超过历史均值的交易日
above_mean = prices[prices["TECH"] > prices["TECH"].mean()]

# 赋值时务必用 loc 避免 SettingWithCopyWarning
prices_copy = prices.copy()
prices_copy.loc["2025-01-02", "BANK"] = 11.0
```

!!! warning "链式赋值陷阱"
    `df[df["A"] > 0]["B"] = 1` 可能不会修改原始 DataFrame（取决于 Pandas 是否复制）。
    正确做法是用 `df.loc[df["A"] > 0, "B"] = 1`。
    Pandas 2.0 会对链式赋值发出 `FutureWarning`，Pandas 3.0 将彻底禁止。

### 2.4.3 DatetimeIndex 与时间切片

```python
prices = load_sample_prices()
print(type(prices.index))          # <class 'pandas.DatetimeIndex'>
print(prices.index.freq)           # <BusinessDay>

# 字符串切片（仅 DatetimeIndex 支持）
prices.loc["2025"]                  # 2025 全年
prices.loc["2025-01":"2025-03"]     # 一季度
prices.loc["2024-Q4"]               # 第四季度（Pandas ≥1.4）

# 常用时间属性
prices.index.year
prices.index.month
prices.index.day_of_week            # 0=周一，4=周五
```

**频率别名速查**

| 别名 | 含义 | 说明 |
|------|------|------|
| `B` | 工作日 | 跳过周末 |
| `D` | 日历日 | 含节假日 |
| `W` | 每周日 | 可 `W-FRI` 指定截止日 |
| `ME` | 月末 | Pandas 2.x 新别名，旧版用 `M` |
| `QE` | 季末 | 旧版用 `Q` |
| `YE` | 年末 | 旧版用 `A`/`Y` |

!!! tip "Pandas 2.x 频率别名变更"
    Pandas 2.2 起，`M`/`Q`/`Y`/`A` 等别名已弃用并在 Pandas 3.0 中移除。
    请统一使用 `ME`/`QE`/`YE`，以免代码在未来版本报 `FutureWarning`。

### 2.4.4 缺失值处理

金融数据经常遇到缺失值（停牌、节假日、数据源问题）。

```python
# 检测缺失值
prices.isna().sum()           # 每列缺失数量
prices.isna().any()           # 是否存在缺失

# 前向填充（停牌用前一日价格）
prices_filled = prices.ffill()

# 后向填充（少数场景：用后续已知值填补历史缺口）
prices_bfill = prices.bfill()

# 删除含缺失值的行
prices_clean = prices.dropna()

# 用特定值填充（如成交量缺失填 0）
# volume.fillna(0)
```

### 2.4.5 groupby：按年/月统计

```python
# 计算日度收益率
from fds import daily_returns
ret = daily_returns(prices)   # log=False 默认，算术收益率

# 添加年份列，按年分组
ret["year"] = ret.index.year
annual_stats = ret.groupby("year")[["BANK", "LIQUOR", "TECH", "UTILITY"]].agg(
    ["mean", "std"]
)
print(annual_stats)

# 更简洁：用 Grouper 按年分组
annual_ret = (
    ret.drop(columns="year")
    .groupby(pd.Grouper(freq="YE"))
    .apply(lambda x: (1 + x).prod() - 1)   # 年化复利
)
```

### 2.4.6 merge 与 concat：多标的对齐

实务中经常需要把来自不同数据源的数据拼合。

```python
# concat：列方向拼合（外连接，日期不匹配时填 NaN）
bank = prices[["BANK"]].copy()
liquor = prices[["LIQUOR"]].iloc[5:]   # 故意错开 5 行

merged = pd.concat([bank, liquor], axis=1)
print(merged.isna().sum())   # LIQUOR 前 5 行为 NaN

# merge：按某个键列连接（适合有明确关联键的场景）
# 例如：把宏观利率数据和股价数据按月份对齐
stock_monthly = prices.resample("ME").last()
stock_monthly["year_month"] = stock_monthly.index.to_period("M")

rate_data = pd.DataFrame({
    "year_month": pd.period_range("2023-01", periods=24, freq="M"),
    "rate": [3.5 + i*0.01 for i in range(24)]
})
combined = stock_monthly.merge(rate_data, on="year_month", how="left")
```

### 2.4.7 resample：时间频率转换

`resample` 是处理时间序列的利器，可在任意频率之间转换：

```python
# 日 → 月
month_end  = prices.resample("ME").last()     # 月末价格
month_high = prices.resample("ME").max()      # 月内最高价
month_ret  = month_end.pct_change().dropna()  # 月度算术收益率

# 日 → 周（每周五）
week_ret = prices.resample("W-FRI").last().pct_change().dropna()

# 日 → 年
annual_price = prices.resample("YE").last()

# OHLC（开高低收）聚合
ohlc = prices["BANK"].resample("ME").ohlc()
```

### 2.4.8 rolling 与 ewm：移动窗口

移动窗口统计是技术分析与风险监控的基础：

```python
# 20 日简单移动平均（SMA）
sma20 = prices["TECH"].rolling(window=20).mean()

# 20 日移动波动率（年化）
vol20 = prices["TECH"].rolling(window=20).std() * (252**0.5)

# 指数加权移动平均（EWM / EWMA）：近期权重更高
# span=20 ≈ 对应 20 日半衰期，com/halflife 参数也可用
ewma20 = prices["TECH"].ewm(span=20, adjust=False).mean()

# EWM 波动率（RiskMetrics 模型常用）
ewm_vol = prices["TECH"].pct_change().ewm(span=60).std() * (252**0.5)
```

| 方法 | 特点 | 典型用途 |
|------|------|---------|
| `rolling(n).mean()` | 等权，窗口内数据 | SMA、MACD |
| `rolling(n).std()` | 等权，历史波动率 | 布林带、VaR |
| `ewm(span=n).mean()` | 指数权重，更敏感 | EWMA 均线 |
| `ewm(span=n).std()` | 指数权重 | RiskMetrics 波动率 |

### 2.4.9 apply vs 向量化

`apply` 允许对 DataFrame 的每行/每列应用任意 Python 函数，但速度较慢。
能向量化就不要用 `apply`：

```python
# 慢（apply + lambda）
sharpe_apply = ret.apply(lambda col: col.mean() / col.std() * 252**0.5)

# 快（向量化）
sharpe_vec = ret.mean() / ret.std() * (252**0.5)

# 两者结果相同，但后者快 10–50 倍
# apply 保留给真正复杂、无法向量化的自定义逻辑
```

### 2.4.10 MultiIndex 简介

MultiIndex（多级索引）用于面板数据（多标的 × 时间），将在第 8 章详细介绍。

```python
# 构造 MultiIndex DataFrame（标的 × 日期）
arrays = [
    ["BANK", "BANK", "TECH", "TECH"],
    pd.to_datetime(["2025-01-02", "2025-01-03"] * 2),
]
idx = pd.MultiIndex.from_arrays(arrays, names=["ticker", "date"])
panel = pd.Series([10.5, 11.0, 100.3, 98.7], index=idx)

# 按外层索引切片
panel.loc["BANK"]

# 交叉截面
panel.unstack("ticker")   # 转为宽表（行=日期，列=标的）
```

---

## 2.5 时间处理

### 2.5.1 pd.to_datetime

```python
# 字符串转日期
dates = pd.to_datetime(["2025-01-02", "20250103", "Jan 4, 2025"])

# 指定格式（提速）
dates = pd.to_datetime(["20250102", "20250103"], format="%Y%m%d")

# Unix 时间戳
ts = pd.to_datetime(1735689600, unit="s")
```

### 2.5.2 日期运算与偏移量

```python
import pandas as pd

today = pd.Timestamp("2025-12-31")

# 日期加减
yesterday = today - pd.Timedelta(days=1)
next_month = today + pd.DateOffset(months=1)

# 工作日偏移
one_bday_later = today + pd.offsets.BDay(1)

# 生成日期范围
trading_days = pd.date_range("2025-01-01", "2025-12-31", freq="B")
print(len(trading_days))   # ~261 个工作日（非精确交易日）
```

---

## 2.6 数据存储：CSV vs Parquet

金融数据处理中，选对存储格式事半功倍。

| 维度 | CSV | Parquet |
|------|-----|---------|
| 可读性 | 文本，可直接打开 | 二进制，需工具 |
| 体积 | 大（无压缩） | **小（列式+压缩，通常 5–10x）** |
| 读写速度 | 慢（逐行解析） | **快（列式批量读取）** |
| 类型保留 | 无（全部字符串） | **有（日期/浮点/类别等）** |
| 跨语言 | 所有工具 | R/Python/Spark 等均支持 |
| 建议场景 | 对外交换、小文件 | **日常存储、中大型数据** |

```python
import pandas as pd

prices = pd.read_parquet("data/processed/prices.parquet")   # 推荐入库格式

# 写 Parquet（需要 pyarrow 或 fastparquet）
prices.to_parquet("output/prices.parquet", compression="snappy")

# 写 CSV（对外交换、给 Excel 用户）
prices.to_csv("output/prices.csv", encoding="utf-8-sig")    # utf-8-sig 防 Excel 乱码

# 往返一致性验证
back = pd.read_parquet("output/prices.parquet")
assert prices.equals(back), "读写前后数据不一致！"
```

!!! info "为什么优先 Parquet"
    以本书内置数据（750 交易日 × 4 只股票）为例：
    CSV 约 45 KB，Parquet 约 8 KB，体积节省 80%；
    读取速度 Parquet 约快 3–5 倍（数据量越大优势越明显）。
    更重要的是，Parquet 保留 `DatetimeIndex` 的类型信息，
    避免读回后需要再次 `pd.to_datetime` 转换。

---

## 2.7 matplotlib 绘图基础

### 2.7.1 Figure 与 Axes

matplotlib 的绘图对象分两层：
- `Figure`：整张图，相当于"画布"
- `Axes`：坐标系，一张 Figure 可包含多个 Axes

```python
import matplotlib.pyplot as plt
from fds import load_sample_prices, set_chinese_font

set_chinese_font()   # 设置中文字体，避免方块字

prices = load_sample_prices()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 上子图：股价走势
axes[0].plot(prices.index, prices["TECH"], label="科技", color="steelblue")
axes[0].plot(prices.index, prices["BANK"], label="银行", color="salmon")
axes[0].set_ylabel("价格（元）")
axes[0].set_title("A股风格模拟数据：科技 vs 银行")
axes[0].legend()

# 下子图：科技股 20 日移动平均
tech_sma = prices["TECH"].rolling(20).mean()
axes[1].plot(prices.index, prices["TECH"], alpha=0.4, label="科技原始")
axes[1].plot(prices.index, tech_sma, linewidth=2, label="20日均线")
axes[1].set_ylabel("价格（元）")
axes[1].set_xlabel("日期")
axes[1].legend()

fig.tight_layout()
plt.savefig("output/ch02_price_trend.png", dpi=150, bbox_inches="tight")
plt.show()
```

!!! tip "中文字体"
    `set_chinese_font()` 会自动检测系统可用的中文字体（Windows 上优先 SimHei，
    macOS 上优先 PingFang SC，Linux 上优先 Noto Sans CJK），确保标题与标签正常显示。
    在 notebook 第一格调用一次即可全局生效。

---

## 2.8 性能与可读性建议

| 场景 | 推荐写法 | 避免 |
|------|---------|------|
| 逐元素计算 | 向量化 `arr * factor` | `for` 循环 |
| 条件赋值 | `df.loc[cond, col] = val` | 链式赋值 `df[cond][col] = val` |
| 分组统计 | `groupby().agg()` | `for group in groupby` 手动累加 |
| 自定义函数 | 先试向量化，再用 `apply` | 过度使用 `apply` |
| 多次读同一文件 | 读入后缓存变量 | 每次都从磁盘读 |
| 大型 DataFrame 预览 | `.head()` / `.sample(5)` | 直接 print 整个 DataFrame |

---

## 2.9 本章小结

本章建立了贯穿全书的工具基础：

- **uv**：通过 `pyproject.toml` + `uv.lock` 实现一键复现的 Python 环境
- **NumPy**：`ndarray` 的向量化运算（快 100x 以上）、广播规则、矩阵乘 `@`
  为后续组合收益 $w^\top \mu$ 和组合方差 $w^\top \Sigma w$ 铺垫
- **Pandas**：`DatetimeIndex` + `loc/iloc/布尔索引` + `resample/rolling/ewm`
  构成金融时间序列处理的完整工具链
- **存储**：日常数据优先 Parquet（快、小、保类型），对外交换用 CSV
- **可视化**：`set_chinese_font()` 一行解决中文字体问题

---

## 2.10 习题

1. **向量化练习**：用 NumPy 向量化（不使用循环）计算以下指标：
   加载内置数据 `load_sample_prices()`，计算 LIQUOR 的 5 日简单移动平均
   （提示：可用 `np.convolve(prices, np.ones(5)/5, mode='valid')`），
   并与 Pandas `rolling(5).mean()` 结果对比，验证两者一致。

2. **广播应用**：对所有 4 只股票，用 NumPy 广播计算"超额收益"
   （每日各股票收益率减去当日等权平均收益率），结果转为 DataFrame，
   打印前 5 行。（提示：先用 `daily_returns` 算收益率矩阵，再减去行均值。）

3. **时间序列分析**：对内置数据按以下步骤操作：
   a. `resample("W-FRI")` 得到周频价格；
   b. 计算周度收益率；
   c. 用 `groupby(pd.Grouper(freq="YE"))` 统计每年的周度收益率均值与标准差；
   d. 打印结果并说明哪一年波动最大。

4. **Parquet 往返**：将波动率最高的一只股票（用 `std()` 衡量）的日度收益率
   单独保存为 Parquet 文件，读回后验证 `equals()` 返回 `True`，
   并比较该文件与对应 CSV 文件的字节大小。

5. **可视化**：绘制一张图，同时展示 TECH 股的日度收盘价（蓝色细线，alpha=0.4）、
   20 日 SMA（橙色实线）、20 日 EWMA（绿色虚线），
   加上中文标题"科技股移动平均对比"和图例，保存为 PNG 文件。
   思路：先调用 `set_chinese_font()`，再用 `fig, ax = plt.subplots()` 绘制。

---

## 2.11 拓展阅读

1. **McKinney, W.** (2022). *Python for Data Analysis*, 3rd ed. O'Reilly.
   Pandas 作者亲著，第 10–11 章对时间序列有深入讲解。
2. **NumPy 官方文档** — Broadcasting：<https://numpy.org/doc/stable/user/basics.broadcasting.html>
3. **Pandas 官方文档** — Time Series：<https://pandas.pydata.org/docs/user_guide/timeseries.html>
4. **Pandas 官方文档** — Enhancing Performance：<https://pandas.pydata.org/docs/user_guide/enhancingperf.html>
5. **VanderPlas, J.** (2023). *Python Data Science Handbook*, 2nd ed. O'Reilly.
   开源全文：<https://jakevdp.github.io/PythonDataScienceHandbook/>
6. **uv 文档**：<https://docs.astral.sh/uv/>，工作区与依赖组的完整说明。
