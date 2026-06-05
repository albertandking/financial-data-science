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


def _load(name: str) -> pd.DataFrame:
    """读取 data/processed/<name>.parquet，缺失时给出友好提示。"""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到内置数据集 {path}。\n"
            f"请先运行：uv run python scripts/make_sample_data.py"
        )
    return pd.read_parquet(path)


def load_sample_prices() -> pd.DataFrame:
    """加载内置的示例股票日度价格数据集。

    返回以日期为索引、各列为不同股票收盘价的 DataFrame
    （列：BANK / LIQUOR / TECH / UTILITY）。
    """
    df = _load("sample_prices")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_market() -> pd.DataFrame:
    """加载内置的市场指数与无风险利率日度序列。

    列：index_close（指数收盘）、index_return（指数日收益）、
    rf_annual（年化无风险利率）、rf_daily（日度无风险利率）。
    内置 4 只股票对该指数有真实的 beta，可直接用于第7章 CAPM。
    """
    df = _load("market")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_factors() -> pd.DataFrame:
    """加载内置的教学因子日度序列（供第7章多因子回归）。

    列：MKT（市场超额收益，真实）、HML（价值−成长，由股票多空构造）、
    SMB、MOM（标注的合成示意因子）。索引为交易日。
    """
    df = _load("factors")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_fundamentals() -> pd.DataFrame:
    """加载内置的公司-年度财务面板（平衡面板，200家×8年）。

    列：firm、year、industry、roa、leverage、size（log总资产）、revenue_growth。
    数据生成过程内置已知系数与公司固定效应，可用于第8章面板回归并验证还原。
    """
    return _load("fundamentals")


def load_credit() -> pd.DataFrame:
    """加载内置的信用违约样本（5000个借款人，约12%违约，含类别不平衡）。

    特征列含 age、income、debt_to_income、credit_history_months、
    num_open_accounts、num_delinquencies、utilization；标签列 default（0/1）。
    用于第16章信用风险评分卡与不平衡处理。
    """
    return _load("credit")
