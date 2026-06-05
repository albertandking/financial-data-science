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

uv 会在需要时**自动下载 Python 3.11**，无需单独安装 Python。

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

## A.5 常见问题

- **中文乱码**：调用 `from fds import set_chinese_font; set_chinese_font()`；Windows 通常已有"微软雅黑"。
- **akshare 抓取失败**：多为网络或接口变更，重试或更新 `uv lock --upgrade-package akshare`。
- **PyTorch 安装慢**：仅进阶章节需要，可单独 `uv sync --extra advanced`。
