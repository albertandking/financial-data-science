"""冒烟测试：保证 fds 工具包与内置数据可用。

运行：
    uv run python scripts/make_sample_data.py   # 先生成数据
    uv run pytest
"""

import numpy as np
import pandas as pd

from fds import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    load_sample_prices,
    max_drawdown,
    sharpe_ratio,
)


def test_load_sample_prices():
    prices = load_sample_prices()
    assert isinstance(prices, pd.DataFrame)
    assert len(prices) > 100
    assert {"BANK", "LIQUOR", "TECH", "UTILITY"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_metrics_run():
    prices = load_sample_prices()
    rets = daily_returns(prices)
    assert (rets.abs() < 1).all().all()  # 日收益率应远小于 100%

    ann_ret = annualized_return(rets["BANK"])
    ann_vol = annualized_volatility(rets["BANK"])
    sr = sharpe_ratio(rets["BANK"], risk_free=0.02)
    mdd = max_drawdown(rets["BANK"])

    assert np.isfinite(ann_ret)
    assert ann_vol > 0
    assert np.isfinite(sr)
    assert -1 <= mdd <= 0
