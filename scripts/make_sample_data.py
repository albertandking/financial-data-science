"""生成内置示例数据集（离线、可复现）。

为了让全书章节**断网也能运行**，这里用固定随机种子合成多份贴近真实形态的
教学数据，写入 data/processed/。所有数据均为合成，**不代表真实行情/公司/借款人**。
需要真实数据时见 scripts/fetch_data.py。

生成的数据集：
1. sample_prices.parquet —— 4 只 A 股风格股票日度价格（约750交易日）
2. market.parquet        —— 市场指数与无风险利率日度序列（与上面股票相关）
3. fundamentals.parquet  —— 公司-年度财务面板（200家×8年，内置已知固定效应与系数）
4. credit.parquet        —— 信用违约样本（5000个借款人，二分类标签，含类别不平衡）

运行：
    uv run python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed"

# ── 股票与市场 ────────────────────────────────────────────────────────────
# 用拼音/英文列名模拟若干只股票（避免编码问题），中文释义见 data/README.md
TICKERS = {
    "BANK": dict(start=12.0, mu=0.05, sigma=0.18),     # 某银行股，低波动
    "LIQUOR": dict(start=180.0, mu=0.15, sigma=0.30),  # 某白酒股，高成长高波动
    "TECH": dict(start=45.0, mu=0.12, sigma=0.40),     # 某科技股，最高波动
    "UTILITY": dict(start=8.0, mu=0.03, sigma=0.14),   # 某公用事业，最稳
}

N_DAYS = 750            # 约三年交易日
SEED = 20260604         # 股票/市场种子
SEED_FUND = 20260605    # 财务面板种子
SEED_CREDIT = 20260606  # 信用样本种子
SEED_FACTOR = 20260607  # 因子数据种子

MARKET_MU = 0.08        # 市场年化漂移
MARKET_SIGMA = 0.20     # 市场年化波动
RF_ANNUAL = 0.02        # 无风险利率（年化，约定值）


def simulate_prices():
    """合成 4 只股票日度价格，并返回驱动它们的共同市场冲击与日期。

    返回 (prices_df, market_shocks, dates)。注意：保持随机数抽取顺序不变，
    使股票价格与历史版本完全一致。
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(end="2025-12-31", periods=N_DAYS)

    # 共同的市场因子（标准正态冲击），制造股票间的正相关
    market = rng.normal(0, 1, size=N_DAYS)

    cols = {}
    for name, cfg in TICKERS.items():
        dt = 1 / 252
        idio = rng.normal(0, 1, size=N_DAYS)
        beta = cfg["sigma"] / 0.25  # 粗略的市场暴露
        shocks = 0.6 * beta * market + 0.4 * idio
        drift = (cfg["mu"] - 0.5 * cfg["sigma"] ** 2) * dt
        diffusion = cfg["sigma"] * np.sqrt(dt) * shocks
        log_ret = drift + diffusion
        price = cfg["start"] * np.exp(np.cumsum(log_ret))
        cols[name] = np.round(price, 2)

    df = pd.DataFrame(cols, index=dates)
    df.index.name = "date"
    return df, market, dates


def simulate_market(market_shocks: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """由与股票相同的市场冲击构造市场指数与无风险利率日度序列。

    这样股票对该指数有真实的 beta，第7章 CAPM 可直接使用内置市场组合。
    """
    dt = 1 / 252
    idx_logret = (MARKET_MU - 0.5 * MARKET_SIGMA ** 2) * dt + MARKET_SIGMA * np.sqrt(dt) * market_shocks
    index_close = 3000.0 * np.exp(np.cumsum(idx_logret))  # 类沪深300起点
    index_ret = np.concatenate([[np.nan], np.diff(index_close) / index_close[:-1]])

    # 无风险利率：在年化 2% 附近做极小幅缓慢波动
    rf_annual = RF_ANNUAL + 0.003 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))

    df = pd.DataFrame({
        "index_close": np.round(index_close, 2),
        "index_return": index_ret,
        "rf_annual": np.round(rf_annual, 5),
        "rf_daily": np.round(rf_annual / 252, 8),
    }, index=dates)
    df.index.name = "date"
    return df


def simulate_factors(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """构造内置教学因子日度序列（供第7章多因子回归直接使用）。

    与 Fama-French 方法一致——因子为资产的多空组合：
    - MKT：市场超额收益 = 指数日收益 − 无风险日利率（真实，来自 market）
    - HML：价值−成长 = (BANK+UTILITY)/2 − (TECH+LIQUOR)/2 的日收益多空
            （以低波动/高波动作价值/成长代理，与4只股票真实相关，回归可见显著载荷）
    - SMB、MOM：标注的合成示意因子（小市值、动量），便于演示多因子回归
            （本内置股票池仅4只、无市值，规模/动量无法真实构造，故为合成）
    """
    rng = np.random.default_rng(SEED_FACTOR)
    rets = prices.pct_change()

    mkt = (market["index_return"] - market["rf_daily"]).rename("MKT")
    value = 0.5 * (rets["BANK"] + rets["UTILITY"])
    growth = 0.5 * (rets["TECH"] + rets["LIQUOR"])
    hml = (value - growth).rename("HML")

    # 合成示意因子：小均值（年化溢价）、与 HML 量级相当的日波动
    scale = float(hml.std())
    n = len(prices)
    smb = pd.Series(0.02 / 252 + rng.normal(0, scale, n), index=prices.index, name="SMB")
    mom = pd.Series(0.05 / 252 + rng.normal(0, scale, n), index=prices.index, name="MOM")

    df = pd.concat([mkt, smb, hml, mom], axis=1).dropna()
    df.index.name = "date"
    return df.round(6)


def simulate_fundamentals() -> pd.DataFrame:
    """合成公司-年度财务面板（平衡面板，含已知个体固定效应与真实系数）。

    数据生成过程（教学用，可被面板回归还原）：
        roa_it = 0.08 + B_LEV*leverage_it + B_SIZE*size_c_it + B_GROWTH*growth_it
                 + alpha_i + u_it
    其中 alpha_i 为公司固定效应，且与 leverage 相关（故意制造，凸显 FE 的必要性）。
    """
    rng = np.random.default_rng(SEED_FUND)
    n_firms, n_years = 200, 8
    years = np.arange(2018, 2018 + n_years)
    industries = np.array(["金融", "消费", "科技", "工业", "公用"])

    # 真实系数（面板回归应能近似还原 B_LEV、B_SIZE、B_GROWTH）
    B_LEV, B_SIZE, B_GROWTH = -0.12, 0.010, 0.05

    # 公司固定效应，并让杠杆与固定效应相关：好公司（alpha 高）杠杆更低
    alpha = rng.normal(0, 0.04, size=n_firms)
    firm_industry = rng.choice(industries, size=n_firms)

    rows = []
    for i in range(n_firms):
        base_lev = 0.45 - 1.5 * alpha[i] + rng.normal(0, 0.05)  # 与 alpha 负相关
        size0 = rng.normal(22.0, 1.0)                            # log 总资产
        for y in years:
            leverage = np.clip(base_lev + rng.normal(0, 0.05), 0.05, 0.95)
            size = size0 + 0.05 * (y - years[0]) + rng.normal(0, 0.1)
            growth = rng.normal(0.10, 0.15)
            roa = (0.08 + B_LEV * leverage + B_SIZE * (size - 22.0)
                   + B_GROWTH * growth + alpha[i] + rng.normal(0, 0.03))
            rows.append((f"F{i:03d}", int(y), firm_industry[i],
                         round(roa, 5), round(leverage, 4),
                         round(size, 4), round(growth, 4)))

    df = pd.DataFrame(rows, columns=["firm", "year", "industry",
                                     "roa", "leverage", "size", "revenue_growth"])
    return df


def simulate_credit() -> pd.DataFrame:
    """合成信用违约样本（二分类，含类别不平衡，约 12% 违约）。

    通过 logistic 数据生成过程构造，特征对违约的方向符合金融直觉，
    可用于第16章评分卡、不平衡处理与 KS/AUC 评估。
    """
    rng = np.random.default_rng(SEED_CREDIT)
    n = 5000

    age = np.clip(rng.normal(38, 11, n), 21, 70).round(0)
    income = np.clip(rng.lognormal(mean=11.2, sigma=0.5, size=n), 3e4, 1e6).round(0)  # 年收入(元)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 1.2, 0.0, 1.5).round(4)
    credit_history_months = np.clip(rng.normal(80, 40, n), 0, 360).round(0)
    num_open_accounts = rng.poisson(4, n)
    num_delinquencies = rng.poisson(0.5, n)
    utilization = np.clip(rng.beta(2, 4, n), 0, 1).round(4)

    # 标准化若干连续变量用于线性预测
    z_income = (np.log(income) - np.log(income).mean()) / np.log(income).std()
    z_hist = (credit_history_months - credit_history_months.mean()) / credit_history_months.std()

    logit = (-2.4
             + 2.0 * debt_to_income
             - 0.8 * z_income
             - 0.5 * z_hist
             + 0.35 * num_delinquencies
             + 1.2 * utilization
             - 0.015 * (age - 38))
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age.astype(int),
        "income": income.astype(int),
        "debt_to_income": debt_to_income,
        "credit_history_months": credit_history_months.astype(int),
        "num_open_accounts": num_open_accounts,
        "num_delinquencies": num_delinquencies,
        "utilization": utilization,
        "default": default,
    })
    return df


def _save(df: pd.DataFrame, name: str, index: bool) -> None:
    df.to_parquet(OUT_DIR / f"{name}.parquet", index=index)
    df.to_csv(OUT_DIR / f"{name}.csv", index=index, encoding="utf-8")
    print(f"  - {name}.parquet / .csv  ({len(df)} 行 x {df.shape[1]} 列)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prices, market_shocks, dates = simulate_prices()
    market = simulate_market(market_shocks, dates)
    factors = simulate_factors(prices, market)
    fundamentals = simulate_fundamentals()
    credit = simulate_credit()

    print("已生成内置示例数据：")
    _save(prices, "sample_prices", index=True)
    _save(market, "market", index=True)
    _save(factors, "factors", index=True)
    _save(fundamentals, "fundamentals", index=False)
    _save(credit, "credit", index=False)

    print("\n违约率：{:.1%}（信用样本）".format(credit["default"].mean()))
    print("市场指数末值：", market["index_close"].iloc[-1])


if __name__ == "__main__":
    main()
