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
from scipy.optimize import minimize
from scipy.stats import norm

from fds import (
    daily_returns,
    load_credit,
    load_market,
    load_sample_prices,
    max_drawdown,
    set_chinese_font,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "book" / "assets" / "figures"
RF = 0.02  # 年化无风险利率（图示用）


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


def _make_clf_dataset(rets: pd.DataFrame, col: str = "TECH") -> tuple[pd.DataFrame, pd.Series]:
    """用滞后收益构造分类特征与「次日上涨」标签（防前视）。"""
    r = rets[col]
    feat = pd.DataFrame(
        {f"lag{k}": r.shift(k) for k in range(1, 6)}
        | {"ma5": r.rolling(5).mean().shift(1), "vol5": r.rolling(5).std().shift(1)}
    )
    label = (r.shift(-1) > 0).astype(int)
    data = feat.join(label.rename("y")).dropna()
    return data.drop(columns="y"), data["y"]


# ── 第1章 ──────────────────────────────────────────────────────────────────
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


# ── 第2章 ──────────────────────────────────────────────────────────────────
def fig_ch02_ma(prices: pd.DataFrame) -> None:
    """第2章：价格与移动平均（rolling 的典型应用）。"""
    s = prices["LIQUOR"]
    fig, ax = plt.subplots(figsize=(9, 5))
    s.plot(ax=ax, label="收盘价", alpha=0.7)
    s.rolling(20).mean().plot(ax=ax, label="20 日均线")
    s.rolling(60).mean().plot(ax=ax, label="60 日均线")
    ax.set_title("LIQUOR 收盘价与移动平均线")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.legend()
    _save(fig, "ch02_ma")


# ── 第3章 ──────────────────────────────────────────────────────────────────
def fig_ch03_fillna(prices: pd.DataFrame) -> None:
    """第3章：停牌缺失与前向填充。"""
    s = prices["TECH"].iloc[:120].copy()
    rng = np.random.default_rng(7)
    halt = rng.choice(len(s), size=8, replace=False)
    dirty = s.copy()
    dirty.iloc[halt] = np.nan
    fig, ax = plt.subplots(figsize=(9, 5))
    dirty.ffill().plot(ax=ax, label="前向填充后", color="#2E5A88")
    dirty.plot(ax=ax, label="含停牌缺失（断点）", color="#C0504D", marker="o", ms=3, lw=0)
    ax.set_title("停牌缺失与前向填充（ffill）")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.legend()
    _save(fig, "ch03_fillna")


# ── 第4章 ──────────────────────────────────────────────────────────────────
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


# ── 第5章 ──────────────────────────────────────────────────────────────────
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


# ── 第6章 ──────────────────────────────────────────────────────────────────
def fig_ch06_acf_pacf(rets: pd.DataFrame) -> None:
    """第6章：TECH 收益率的 ACF 与 PACF。"""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    r = rets["TECH"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(r, lags=20, ax=axes[0])
    axes[0].set_title("收益率 ACF")
    plot_pacf(r, lags=20, ax=axes[1], method="ywm")
    axes[1].set_title("收益率 PACF")
    fig.tight_layout()
    _save(fig, "ch06_acf_pacf")


def fig_ch06_garch(rets: pd.DataFrame) -> None:
    """第6章：GARCH(1,1) 估计的条件波动率。"""
    from arch import arch_model

    r = rets["TECH"] * 100
    res = arch_model(r, vol="Garch", p=1, q=1).fit(disp="off")
    cond_vol = res.conditional_volatility
    fig, ax = plt.subplots(figsize=(9, 5))
    r.abs().plot(ax=ax, color="#B0B0B0", alpha=0.6, label="|日收益|（×100）")
    pd.Series(cond_vol, index=r.index).plot(ax=ax, color="#C0504D", label="GARCH 条件波动率")
    ax.set_title("GARCH(1,1) 条件波动率捕捉波动率聚集")
    ax.set_xlabel("日期")
    ax.set_ylabel("波动率（%）")
    ax.legend()
    _save(fig, "ch06_garch")


# ── 第7章 ──────────────────────────────────────────────────────────────────
def _capm_betas(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """估计各股票对真实市场的 CAPM beta，返回 (beta, 年化超额收益, 市场超额)。"""
    rets = daily_returns(prices)
    mkt = load_market()
    rf = mkt["rf_daily"].reindex(rets.index)
    mexc = (mkt["index_return"] - mkt["rf_daily"]).reindex(rets.index).dropna()
    betas, exret = {}, {}
    for c in prices.columns:
        y = (rets[c] - rf).reindex(mexc.index)
        betas[c] = float(np.polyfit(mexc, y, 1)[0])
        exret[c] = float(y.mean() * 252)
    return pd.Series(betas), pd.Series(exret), mexc


def fig_ch07_sml(prices: pd.DataFrame) -> None:
    """第7章：证券市场线（beta 与年化超额收益）。"""
    beta, exret, _ = _capm_betas(prices)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(beta, exret, s=80, color="#2E5A88", zorder=3)
    for c in beta.index:
        ax.annotate(c, (beta[c], exret[c]), textcoords="offset points", xytext=(6, 6))
    xs = np.linspace(0, beta.max() * 1.1, 50)
    slope = (exret.mean() - RF) / beta.mean()
    ax.plot(xs, RF + xs * slope, "--", color="#C0504D", label="SML（示意）")
    ax.set_title("证券市场线：beta 越高，要求的超额收益越高")
    ax.set_xlabel("CAPM Beta")
    ax.set_ylabel("年化超额收益")
    ax.legend()
    _save(fig, "ch07_sml")


def fig_ch07_beta_bar(prices: pd.DataFrame) -> None:
    """第7章：四只股票的 CAPM beta 对比。"""
    beta, _, _ = _capm_betas(prices)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    beta.sort_values().plot.bar(ax=ax, color="#2E5A88")
    ax.axhline(1.0, color="#C0504D", ls="--", label="市场 beta = 1")
    ax.set_title("四只股票对真实市场指数的 Beta")
    ax.set_ylabel("Beta")
    ax.legend()
    _save(fig, "ch07_beta_bar")


# ── 第8章 ──────────────────────────────────────────────────────────────────
def fig_ch08_coef() -> None:
    """第8章：Pooled OLS vs 固定效应的杠杆系数（FE 还原真实值）。"""
    import statsmodels.api as sm

    from fds import load_fundamentals

    f = load_fundamentals()
    pooled = (
        sm.OLS(f["roa"], sm.add_constant(f[["leverage", "size", "revenue_growth"]]))
        .fit()
        .params["leverage"]
    )
    d = f.copy()
    for col in ["roa", "leverage", "size", "revenue_growth"]:
        d[col] = d[col] - d.groupby(f["firm"])[col].transform("mean")
    fe = sm.OLS(d["roa"], d[["leverage", "size", "revenue_growth"]]).fit().params["leverage"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(
        ["Pooled OLS", "固定效应 FE", "真实值"],
        [pooled, fe, -0.12],
        color=["#C0504D", "#2E5A88", "#4E9A65"],
    )
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_title("杠杆对 ROA 的系数：Pooled 有偏，FE 还原真实值 −0.12")
    ax.set_ylabel("leverage 系数")
    _save(fig, "ch08_coef")


# ── 第9章 ──────────────────────────────────────────────────────────────────
def fig_ch09_roc(rets: pd.DataFrame) -> None:
    """第9章：逻辑回归预测涨跌的 ROC 曲线（时序划分）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = _make_clf_dataset(rets)
    split = int(len(x) * 0.7)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x.iloc[:split], y.iloc[:split])
    proba = model.predict_proba(x.iloc[split:])[:, 1]
    fpr, tpr, _ = roc_curve(y.iloc[split:], proba)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color="#2E5A88", label=f"ROC（AUC = {auc(fpr, tpr):.2f}）")
    ax.plot([0, 1], [0, 1], "--", color="#B0B0B0", label="随机猜测")
    ax.set_title("涨跌预测 ROC 曲线（样本外）")
    ax.set_xlabel("假阳率 FPR")
    ax.set_ylabel("真阳率 TPR")
    ax.legend()
    _save(fig, "ch09_roc")


# ── 第10章 ─────────────────────────────────────────────────────────────────
def fig_ch10_bollinger(prices: pd.DataFrame) -> None:
    """第10章：均线与布林带（技术指标特征）。"""
    s = prices["LIQUOR"]
    ma = s.rolling(20).mean()
    sd = s.rolling(20).std()
    fig, ax = plt.subplots(figsize=(9, 5))
    s.plot(ax=ax, label="收盘价", color="#2E5A88")
    ma.plot(ax=ax, label="20 日均线", color="#E08E45")
    ax.fill_between(
        s.index, ma - 2 * sd, ma + 2 * sd, alpha=0.2, color="#8064A2", label="布林带 ±2σ"
    )
    ax.set_title("LIQUOR 均线与布林带")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.legend()
    _save(fig, "ch10_bollinger")


# ── 第11章 ─────────────────────────────────────────────────────────────────
def fig_ch11_importance(rets: pd.DataFrame) -> None:
    """第11章：随机森林的特征重要性。"""
    from sklearn.ensemble import RandomForestClassifier

    x, y = _make_clf_dataset(rets)
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=1)
    model.fit(x, y)
    imp = pd.Series(model.feature_importances_, index=x.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot.barh(ax=ax, color="#4E9A65")
    ax.set_title("随机森林特征重要性（预测次日涨跌）")
    ax.set_xlabel("重要性")
    _save(fig, "ch11_importance")


# ── 第12章 ─────────────────────────────────────────────────────────────────
def fig_ch12_activations() -> None:
    """第12章：常见激活函数。"""
    z = np.linspace(-5, 5, 200)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z, np.maximum(0, z), label="ReLU")
    ax.plot(z, 1 / (1 + np.exp(-z)), label="Sigmoid")
    ax.plot(z, np.tanh(z), label="Tanh")
    ax.axhline(0, color="#B0B0B0", lw=0.8)
    ax.axvline(0, color="#B0B0B0", lw=0.8)
    ax.set_title("常见激活函数")
    ax.set_xlabel("输入 z")
    ax.set_ylabel("激活值")
    ax.legend()
    _save(fig, "ch12_activations")


# ── 第13章 ─────────────────────────────────────────────────────────────────
def fig_ch13_words() -> None:
    """第13章：正负面财经文本的高频情感词（示意词频）。"""
    pos = pd.Series(
        {"增长": 5, "大增": 4, "超预期": 3, "利好": 3, "盈利": 3, "创新高": 2, "分红": 2, "强劲": 2}
    ).sort_values()
    neg = pd.Series(
        {"下滑": 5, "亏损": 4, "暴雷": 3, "问询": 3, "违约": 2, "减持": 2, "造假": 2, "处罚": 2}
    ).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pos.plot.barh(ax=axes[0], color="#C0504D")
    axes[0].set_title("正面文本高频词")
    axes[0].set_xlabel("词频")
    neg.plot.barh(ax=axes[1], color="#4E9A65")
    axes[1].set_title("负面文本高频词")
    axes[1].set_xlabel("词频")
    fig.tight_layout()
    _save(fig, "ch13_words")


# ── 第14章 ─────────────────────────────────────────────────────────────────
def fig_ch14_frontier(prices: pd.DataFrame) -> None:
    """第14章：有效前沿、GMV、最大夏普组合与资本市场线。"""
    rets = daily_returns(prices)
    mu = rets.mean().to_numpy() * 252
    cov = rets.cov().to_numpy() * 252
    n = len(mu)
    ones = np.ones(n)
    bnds = [(0.0, 1.0)] * n
    eq = {"type": "eq", "fun": lambda w: w.sum() - 1}

    rng = np.random.default_rng(42)
    weights = rng.dirichlet(ones, 4000)
    pr = weights @ mu
    pv = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpe = (pr - RF) / pv

    gmv = minimize(lambda w: w @ cov @ w, ones / n, bounds=bnds, constraints=[eq]).x
    tan = minimize(
        lambda w: -(w @ mu - RF) / np.sqrt(w @ cov @ w), ones / n, bounds=bnds, constraints=[eq]
    ).x

    targets = np.linspace(pr.min(), pr.max(), 40)
    fvol = []
    for t in targets:
        cons = [eq, {"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w, ones / n, bounds=bnds, constraints=cons)
        fvol.append(np.sqrt(res.fun) if res.success else np.nan)

    def stat(w: np.ndarray) -> tuple[float, float]:
        return float(np.sqrt(w @ cov @ w)), float(w @ mu)

    gv, gr = stat(gmv)
    tv, tr = stat(tan)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(pv, pr, c=sharpe, cmap="viridis", s=8, alpha=0.5)
    fig.colorbar(sc, label="夏普比率")
    ax.plot(fvol, targets, color="#C0504D", lw=2, label="有效前沿")
    ax.scatter([gv], [gr], color="#2E5A88", s=120, marker="*", zorder=5, label="最小方差 GMV")
    ax.scatter([tv], [tr], color="#E08E45", s=120, marker="*", zorder=5, label="最大夏普")
    xs = np.linspace(0, pv.max(), 50)
    ax.plot(xs, RF + xs * (tr - RF) / tv, "--", color="#444444", label="资本市场线 CML")
    ax.set_title("均值-方差有效前沿")
    ax.set_xlabel("年化波动率")
    ax.set_ylabel("年化收益")
    ax.legend()
    _save(fig, "ch14_frontier")


# ── 第15章 ─────────────────────────────────────────────────────────────────
def fig_ch15_backtest(prices: pd.DataFrame) -> None:
    """第15章：20 日动量策略回测净值（含成本）vs 买入持有。"""
    s = prices["LIQUOR"]
    ret = s.pct_change()
    signal = (s.pct_change(20) > 0).astype(int)
    position = signal.shift(1)
    strat = (position * ret).fillna(0)
    cost = position.diff().abs().fillna(0) * 0.001
    nav_strat = (1 + strat - cost).cumprod()
    nav_hold = (1 + ret.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    nav_hold.plot(ax=ax, label="买入持有", color="#B0B0B0")
    nav_strat.plot(ax=ax, label="20 日动量（扣双边成本）", color="#2E5A88")
    ax.set_title("LIQUOR 动量策略回测净值")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend()
    _save(fig, "ch15_backtest")


# ── 第16章 ─────────────────────────────────────────────────────────────────
def fig_ch16_roc_ks() -> None:
    """第16章：信用评分卡 ROC 与 KS 曲线。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    credit = load_credit()
    x = credit.drop(columns="default")
    y = credit["default"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    fpr, tpr, thr = roc_curve(y_te, proba)
    ks = np.max(tpr - fpr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.plot(fpr, tpr, color="#2E5A88", label=f"AUC = {auc(fpr, tpr):.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="#B0B0B0")
    ax1.set_title("ROC 曲线")
    ax1.set_xlabel("假阳率 FPR")
    ax1.set_ylabel("真阳率 TPR")
    ax1.legend()
    ax2.plot(tpr, color="#C0504D", label="累计坏客户 TPR")
    ax2.plot(fpr, color="#4E9A65", label="累计好客户 FPR")
    ax2.plot(tpr - fpr, color="#8064A2", label=f"KS = {ks:.3f}")
    ax2.set_title("KS 曲线")
    ax2.set_xlabel("按评分排序的样本")
    ax2.set_ylabel("累计占比")
    ax2.legend()
    fig.tight_layout()
    _save(fig, "ch16_roc_ks")


# ── 第17章 ─────────────────────────────────────────────────────────────────
def fig_ch17_rag() -> None:
    """第17章：TF-IDF 检索（模拟 RAG）对各知识片段的相似度。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    kb = [
        "贵州茅台 2023 年营业收入同比增长约 18%",
        "央行下调存款准备金率以释放流动性",
        "沪深 300 指数成分股按市值加权编制",
        "新能源汽车销量持续增长带动锂电池需求",
        "某科技公司发布季度财报净利润超预期",
    ]
    query = "茅台的营业收入增速是多少"
    # 字符级分词（中文按字切分），无需额外分词器即可演示检索
    vec = TfidfVectorizer(tokenizer=list, token_pattern=None)
    mat = vec.fit_transform(kb)
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = np.argsort(sims)
    ax.barh([f"片段{i + 1}" for i in order], sims[order], color="#2E5A88")
    ax.set_title(f"RAG 检索相似度（查询：{query}）")
    ax.set_xlabel("余弦相似度")
    _save(fig, "ch17_rag")


def main() -> None:
    """生成全部正文图示。"""
    set_chinese_font()
    prices = load_sample_prices()
    rets = daily_returns(prices)

    print("生成正文图示到 book/assets/figures/：")
    fig_ch01_prices(prices)
    fig_ch01_nav(rets)
    fig_ch02_ma(prices)
    fig_ch03_fillna(prices)
    fig_ch04_return_hist(rets)
    fig_ch04_qq(rets)
    fig_ch04_corr_heatmap(rets)
    fig_ch04_vol_clustering(rets)
    fig_ch05_drawdown(rets)
    fig_ch05_var(rets)
    fig_ch06_acf_pacf(rets)
    fig_ch06_garch(rets)
    fig_ch07_sml(prices)
    fig_ch07_beta_bar(prices)
    fig_ch08_coef()
    fig_ch09_roc(rets)
    fig_ch10_bollinger(prices)
    fig_ch11_importance(rets)
    fig_ch12_activations()
    fig_ch13_words()
    fig_ch14_frontier(prices)
    fig_ch15_backtest(prices)
    fig_ch16_roc_ks()
    fig_ch17_rag()
    print("完成。")


if __name__ == "__main__":
    main()
