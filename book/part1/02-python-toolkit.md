# 第2章 Python 数据科学工具栈

!!! info "配套代码"
    本章代码见 `notebooks/ch02_python_toolkit.ipynb`，可逐格运行（依赖内置数据，离线可跑）。

## 2.1 学习目标

- 理解本书的项目结构与 uv 环境管理方式
- 掌握 NumPy 数组的向量化运算
- 掌握 Pandas 的 `Series` / `DataFrame`、时间索引与重采样
- 能用 Pandas 读写金融数据（CSV / Parquet）

## 2.2 为什么是 Python

金融数据科学的事实标准是 Python，原因有三：

1. **生态完整**：从数据获取（akshare）、数值计算（NumPy）、数据处理（Pandas）、
   统计建模（statsmodels）到机器学习（scikit-learn）一应俱全。
2. **可读性强**：代码接近伪代码，便于教学与协作。
3. **可复现**：配合 uv 等工具，环境与依赖可一键复现。

### uv 与项目结构

本书用 [uv](https://docs.astral.sh/uv/) 管理环境。核心命令只有几个：

```bash
uv sync                 # 按 pyproject.toml 安装依赖（含锁定版本）
uv run python xxx.py    # 在项目环境里运行脚本
uv run jupyter lab      # 启动 Jupyter
```

依赖声明在 `pyproject.toml`，精确版本锁定在 `uv.lock`。把这两个文件交给同学，
对方 `uv sync` 就能得到**完全一致**的环境——这正是"可复现研究"的基础。

## 2.3 NumPy：向量化思维

NumPy 的核心是 `ndarray`（N 维数组）。金融计算中，**用数组运算代替 for 循环**
既简洁又快得多。

```python
import numpy as np

prices = np.array([10.0, 10.5, 10.3, 10.8])
returns = prices[1:] / prices[:-1] - 1   # 一次算出所有日收益，无需循环
```

这种"对整个数组一次性运算"的方式叫**向量化（vectorization）**，是高效数值计算的关键。

### 广播（broadcasting）

形状不同的数组运算时，NumPy 会自动"广播"对齐。例如给每只股票减去各自的均值：

```python
mat = np.random.randn(100, 4)        # 100 天 × 4 只股票
demeaned = mat - mat.mean(axis=0)    # mean 形状 (4,)，自动广播到 (100, 4)
```

## 2.4 Pandas：金融数据的主力

Pandas 在 NumPy 之上提供了**带标签的**数据结构：

- `Series`：带索引的一维数组（如单只股票的价格序列）
- `DataFrame`：带行列标签的二维表（如多只股票的价格表）

金融数据几乎都是"以日期为索引"的表，Pandas 的 `DatetimeIndex` 专为此而生。

### 时间索引与切片

```python
from fds import load_sample_prices

prices = load_sample_prices()        # 索引是 DatetimeIndex
prices.loc["2025"]                    # 取 2025 年全部
prices.loc["2025-01":"2025-03"]       # 取一季度（区间切片）
```

### 重采样：日 → 月

`resample` 按时间频率聚合，是计算月度/周度指标的利器：

```python
month_end = prices.resample("ME").last()     # 每月最后一个交易日收盘价
monthly_ret = month_end.pct_change()         # 月度收益率
```

!!! tip "频率别名"
    `D` 日、`W` 周、`ME` 月末、`QE` 季末、`YE` 年末。注意新版 Pandas 用 `ME`/`YE`
    取代了旧的 `M`/`Y`。

## 2.5 读写数据：CSV vs Parquet

| 格式 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| CSV | 通用、可读 | 体积大、无类型 | 对外交换、小数据 |
| Parquet | 列式、压缩、带类型、快 | 需二进制查看 | **入库存储、大数据** |

```python
prices.to_parquet("prices.parquet")          # 推荐
df = pd.read_parquet("prices.parquet")
prices.to_csv("prices.csv", encoding="utf-8") # 对外交换
```

本书内置数据集同时提供两种格式（见 `data/processed/`）。

## 2.6 本章小结

- uv 管理环境，`pyproject.toml` + `uv.lock` 保证复现
- NumPy 用向量化与广播代替循环
- Pandas 的 `DataFrame` + `DatetimeIndex` 是金融数据的主力，`resample` 做频率转换
- 存储优先 Parquet，交换用 CSV

## 2.7 练习

1. 用 `resample` 把内置日度价格转为**周度**收益率，并打印前 5 行。
2. 不用循环，用 NumPy 向量化计算 `LIQUOR` 的 20 日移动平均（提示：`np.convolve` 或 Pandas `rolling`）。
3. 把内置数据中波动最大的一只股票单独存成一个 Parquet 文件再读回，验证数据一致。
