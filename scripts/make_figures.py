"""生成嵌入正文的图示（PNG），保存到 book/assets/figures/。

最佳实践：正文图示由本脚本用与各章 notebook **相同的 fds 工具、内置数据与统一
主题**（set_chinese_font）生成，因此与 notebook 风格一致、可复现；正文用 Markdown
的 ``<figure>`` 语法引用并配图注。图片入库，使书无需运行代码即可查看。

重新生成全部图示：
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from fds import daily_returns, load_sample_prices, max_drawdown, set_chinese_font

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "book" / "assets" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    """保存图到 FIG_DIR/name.png 并关闭，避免内存累积。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  - {name}.png")


def _acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """样本自相关函数（滞后 1..nlags）。"""
    x = np.asarray(x) - np.mean(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def fig_ch01_prices(prices: pd.DataFrame) -> None:
    """第1章：四只示例股票价格走势（起点归一为 100）。"""
    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    normalized.plot(ax=ax)
    ax.set_title("四只示例股票价格走势（起点归一为 100）")
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化价格")
    ax.legend(title="股票")
    _save(fig, "ch01_prices")


def fig_ch01_nav(rets: pd.DataFrame) -> None:
    """第1章：累计净值曲线。"""
    nav = (1 + rets).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    nav.plot(ax=ax)
    ax.set_title("四只示例股票累计净值（初始 = 1）")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    _save(fig, "ch01_nav")


def fig_ch04_return_hist(rets: pd.DataFrame) -> None:
    """第4章：TECH 日收益分布 vs 正态（厚尾）。"""
    x = rets["TECH"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=60, density=True, alpha=0.6, label="TECH 实际分布")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, norm.pdf(grid, x.mean(), x.std()), "--", color="#C0504D", label="正态分布")
    ax.set_title("日收益分布 vs 正态：注意两端的厚尾")
    ax.set_xlabel("日收益率")
    ax.set_ylabel("概率密度")
    ax.legend()
    _save(fig, "ch04_return_hist")


def fig_ch04_qq(rets: pd.DataFrame) -> None:
    """第4章：四只股票正态 QQ 图（诊断厚尾）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), rets.columns, strict=False):
        stats.probplot(rets[col], dist="norm", plot=ax)
        ax.set_title(f"{col} 正态 QQ 图")
    fig.tight_layout()
    _save(fig, "ch04_qq")


def fig_ch04_corr_heatmap(rets: pd.DataFrame) -> None:
    """第4章：四只股票日收益相关性热力图。"""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(rets.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("四只股票日收益相关性")
    _save(fig, "ch04_corr_heatmap")


def fig_ch04_vol_clustering(rets: pd.DataFrame) -> None:
    """第4章：波动率聚集——收益 vs 收益绝对值的自相关。"""
    lags = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(lags - 0.2, _acf(rets["TECH"].to_numpy()), width=0.4, label="收益率")
    ax.bar(lags + 0.2, _acf(rets["TECH"].abs().to_numpy()), width=0.4, label="收益率绝对值")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_title("波动率聚集：收益近似不相关，但其绝对值有持续自相关")
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("自相关系数")
    ax.legend()
    _save(fig, "ch04_vol_clustering")


def fig_ch05_drawdown(rets: pd.DataFrame) -> None:
    """第5章：TECH 累计净值与回撤（水下曲线）。"""
    nav = (1 + rets["TECH"]).cumprod()
    drawdown = nav / nav.cummax() - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    nav.plot(ax=ax1, color="#2E5A88")
    ax1.set_title("TECH 累计净值")
    ax1.set_ylabel("净值")
    drawdown.plot(ax=ax2, color="#C0504D")
    ax2.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#C0504D", alpha=0.3)
    ax2.set_title(f"TECH 回撤（最大回撤 {max_drawdown(rets['TECH']):.1%}）")
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    fig.tight_layout()
    _save(fig, "ch05_drawdown")


def fig_ch05_var(rets: pd.DataFrame) -> None:
    """第5章：TECH 日收益分布与 95% VaR。"""
    x = rets["TECH"]
    var95 = -np.quantile(x, 0.05)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=50, alpha=0.7)
    ax.axvline(-var95, color="#C0504D", lw=2, label=f"95% VaR = {var95:.3f}")
    ax.set_title("日收益分布与 95% 单日 VaR")
    ax.set_xlabel("日收益率")
    ax.set_ylabel("频数")
    ax.legend()
    _save(fig, "ch05_var")


def main() -> None:
    """生成全部正文图示。"""
    set_chinese_font()
    prices = load_sample_prices()
    rets = daily_returns(prices)

    print("生成正文图示到 book/assets/figures/：")
    fig_ch01_prices(prices)
    fig_ch01_nav(rets)
    fig_ch04_return_hist(rets)
    fig_ch04_qq(rets)
    fig_ch04_corr_heatmap(rets)
    fig_ch04_vol_clustering(rets)
    fig_ch05_drawdown(rets)
    fig_ch05_var(rets)
    print("完成。")


if __name__ == "__main__":
    main()
