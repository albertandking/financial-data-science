"""常用金融指标计算。

这些函数贯穿全书（收益率、波动率、夏普比率、最大回撤），
统一实现一次，正文与各章 notebook 共用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # A股一年约定的交易日数，年化时使用


def daily_returns(prices: pd.DataFrame | pd.Series, log: bool = False):
    """由价格序列计算日度收益率。

    Parameters
    ----------
    prices : 价格序列或多列价格表
    log : True 则计算对数收益率，否则简单收益率
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def annualized_return(returns: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS):
    """几何年化收益率。"""
    n = len(returns)
    cumulative = (1 + returns).prod()
    return cumulative ** (periods / n) - 1


def annualized_volatility(returns: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS):
    """年化波动率。"""
    return returns.std(ddof=1) * np.sqrt(periods)


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS,
):
    """年化夏普比率。risk_free 为年化无风险利率。"""
    excess = returns - risk_free / periods
    return (excess.mean() / returns.std(ddof=1)) * np.sqrt(periods)


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤（负值，越接近 0 越好）。"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())
