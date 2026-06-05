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
    load_credit,
    load_factors,
    load_fundamentals,
    load_market,
    load_sample_prices,
    max_drawdown,
    sharpe_ratio,
)


def test_load_sample_prices() -> None:
    prices = load_sample_prices()
    assert isinstance(prices, pd.DataFrame)
    assert len(prices) > 100
    assert {"BANK", "LIQUOR", "TECH", "UTILITY"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_metrics_run() -> None:
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


def test_load_market() -> None:
    mkt = load_market()
    assert {"index_close", "index_return", "rf_annual", "rf_daily"}.issubset(mkt.columns)
    assert isinstance(mkt.index, pd.DatetimeIndex)
    assert (mkt["index_close"] > 0).all()


def test_load_factors() -> None:
    fac = load_factors()
    assert {"MKT", "SMB", "HML", "MOM"}.issubset(fac.columns)
    assert isinstance(fac.index, pd.DatetimeIndex)
    assert len(fac) > 100


def test_load_fundamentals_panel() -> None:
    f = load_fundamentals()
    assert {"firm", "year", "roa", "leverage", "size", "revenue_growth"}.issubset(f.columns)
    assert f["firm"].nunique() == 200
    assert f["year"].nunique() == 8
    assert len(f) == 1600  # 平衡面板


def test_load_credit() -> None:
    c = load_credit()
    assert "default" in c.columns
    assert set(c["default"].unique()) <= {0, 1}
    assert 0 < c["default"].mean() < 1  # 存在两类样本（不平衡）
