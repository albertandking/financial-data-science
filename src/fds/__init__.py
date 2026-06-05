"""fds —— 《金融数据科学》全书复用工具包。

正文与各章 notebook 统一从这里导入，避免逻辑在多处重复维护：

    from fds import load_sample_prices, daily_returns, set_chinese_font
"""

from fds.data import (
    DATA_DIR,
    list_datasets,
    load_credit,
    load_fundamentals,
    load_market,
    load_sample_prices,
)
from fds.metrics import (
    annualized_return,
    annualized_volatility,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
)
from fds.plotting import set_chinese_font

__all__ = [
    "DATA_DIR",
    "list_datasets",
    "load_sample_prices",
    "load_market",
    "load_fundamentals",
    "load_credit",
    "daily_returns",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "set_chinese_font",
]

__version__ = "0.1.0"
