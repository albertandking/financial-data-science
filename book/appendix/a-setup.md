# 附录A 本地环境配置

本书用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境，一条命令即可复现。

## A.1 安装 uv

=== "Windows (PowerShell)"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

uv 会在需要时**自动下载 Python**，无需单独安装。本书推荐 **Python 3.14**
（仓库 `.python-version` 已固定为 3.14，`uv sync` 会自动使用它）。代码同样兼容 3.11+。

## A.2 安装依赖

在项目根目录执行：

```bash
# 核心依赖 + 成书工具链
uv sync

# 联网抓取中国市场数据（可选）
uv sync --extra data

# 进阶章节：深度学习 / NLP（可选，较重）
uv sync --extra advanced
```

## A.3 生成内置数据

```bash
uv run python scripts/make_sample_data.py
```

## A.4 常用命令

| 任务 | 命令 |
|---|---|
| 启动 Jupyter | `uv run jupyter lab` |
| 运行测试 | `uv run pytest` |
| 执行某 notebook | `uv run jupyter nbconvert --to notebook --execute notebooks/ch01_introduction.ipynb` |
| 导出 notebook 到 md | `uv run python scripts/export_notebooks.py` |
| 本地预览书 | `uv run mkdocs serve` |
| 构建静态网站 | `uv run mkdocs build` |

## A.5 复现环境：库版本表

下表为本书写作与全部 notebook 验证所用的精确版本（**Python 3.14.3**）。
`uv.lock` 已锁定全部依赖的精确版本与哈希，`uv sync` 即可一字不差地复现；
此表供查阅与离线参考。

### 核心依赖（`uv sync`）

| 库 | 版本 | 用途 |
|---|---|---|
| `numpy` | 2.4.6 | 数值计算 |
| `pandas` | 3.0.3 | 数据处理（注意 3.x 用 `ME`/`YE` 频率别名） |
| `scipy` | 1.17.1 | 科学计算、统计 |
| `matplotlib` | 3.10.9 | 绘图 |
| `seaborn` | 0.13.2 | 统计可视化 |
| `statsmodels` | 0.14.6 | 计量/时间序列（第6–9章） |
| `arch` | 8.0.0 | GARCH 波动率模型（第6章） |
| `linearmodels` | 7.0 | 面板数据回归（第8章） |
| `scikit-learn` | 1.9.0 | 机器学习（第9–11、16章） |
| `pyarrow` | 24.0.0 | Parquet 读写 |
| `tqdm` | 4.67.3 | 进度条 |
| `openpyxl` | 3.1.5 | 读写 Excel |

### 中国市场数据（`uv sync --extra data`）

| 库 | 版本 | 用途 |
|---|---|---|
| `akshare` | 1.18.64 | A股/宏观数据（免注册） |
| `tushare` | 1.4.29 | A股数据（需 token） |
| `requests` | 2.34.2 | HTTP 请求 |

### 进阶/深度（`uv sync --extra advanced`）

| 库 | 版本 | 用途 |
|---|---|---|
| `xgboost` | 3.2.0 | 梯度提升树（第11章） |
| `torch` | 2.12.0 | 深度学习（第12章） |
| `transformers` | 5.10.1 | 预训练模型/FinBERT（第13章） |
| `jieba` | 0.42.1 | 中文分词（第13章） |

### 成书与工具（默认随 `uv sync` 安装）

| 库 | 版本 | 用途 |
|---|---|---|
| `mkdocs` | 1.6.1 | 静态站点生成 |
| `mkdocs-material` | 9.7.6 | 文档主题 |
| `jupyter` | 1.1.1 | Notebook 内核 |
| `jupyterlab` | 4.5.7 | 交互式开发 |
| `nbconvert` | 7.17.1 | notebook 执行/导出 |
| `jupytext` | 1.19.3 | notebook ↔ md 互转 |
| `pytest` | 9.0.3 | 测试 |

!!! tip "版本以 uv.lock 为准"
    上表为可读参考；真正保证复现的是仓库内的 `uv.lock`（锁定精确版本与哈希）。
    若需升级某个库：`uv lock --upgrade-package <名称>`，再 `uv sync`。

## A.6 常见问题

- **中文乱码**：调用 `from fds import set_chinese_font; set_chinese_font()`；Windows 通常已有"微软雅黑"。
- **akshare 抓取失败**：多为网络或接口变更，重试或更新 `uv lock --upgrade-package akshare`。
- **PyTorch 安装慢**：仅进阶章节需要，可单独 `uv sync --extra advanced`。
- **Python 版本**：推荐 3.14；若所在平台某重依赖暂无 3.14 wheel，可在 `.python-version` 改回 3.11/3.12 后 `uv sync`。
