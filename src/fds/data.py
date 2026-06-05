"""数据加载工具。

内置示例数据集存放在仓库的 data/processed/ 下，**离线即可读取**，
保证书中所有基础章节断网也能运行。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 定位到仓库根目录下的 data/，无论从 notebook 还是脚本调用都能找到。
# 本文件位于 src/fds/data.py，向上三级即仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data" / "processed"


def list_datasets() -> list[str]:
    """列出 data/processed/ 下所有可用的内置数据集文件名。"""
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.parquet"))


def load_sample_prices() -> pd.DataFrame:
    """加载内置的示例股票日度价格数据集。

    返回一个以日期为索引、各列为不同股票收盘价的 DataFrame。
    若数据不存在，提示先运行生成脚本。
    """
    path = DATA_DIR / "sample_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到内置数据集 {path}。\n"
            f"请先运行：uv run python scripts/make_sample_data.py"
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df
