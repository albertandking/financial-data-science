# 第2章 Python 数据科学工具栈

!!! info "配套代码"
    `notebooks/ch02_python_toolkit.ipynb`

## 2.1 学习目标

- 掌握 NumPy 数组、Pandas 的 Series/DataFrame
- 熟悉本书的环境管理方式（uv）与项目结构
- 能用 Pandas 读写金融数据、做时间索引切片

## 2.2 内容大纲

1. uv 环境与 `pyproject.toml`：依赖如何声明与复现
2. NumPy：向量化运算、广播
3. Pandas：DataFrame、`DatetimeIndex`、重采样 `resample`
4. 读写数据：CSV / Parquet / Excel
5. `from fds import ...`：复用本书工具包

## 2.3 练习

1. 用 `resample` 把日度价格转为月度，并计算月收益率。
2. 用 NumPy 向量化重写一个 for 循环计算移动平均。
