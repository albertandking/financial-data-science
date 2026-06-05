"""生成内置示例数据集（离线、可复现）。

为了让全书基础章节**断网也能运行**，这里用固定随机种子合成一份贴近真实
形态的 A 股风格日度价格数据（几何布朗运动 + 轻微相关性），写入
data/processed/sample_prices.parquet。

合成数据仅用于教学演示，不代表真实行情。需要真实数据时见 scripts/fetch_data.py。

运行：
    uv run python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed"

# 用拼音/英文列名模拟若干只股票（避免编码问题），中文释义见 data/README.md
TICKERS = {
    "BANK": dict(start=12.0, mu=0.05, sigma=0.18),   # 某银行股，低波动
    "LIQUOR": dict(start=180.0, mu=0.15, sigma=0.30),  # 某白酒股，高成长高波动
    "TECH": dict(start=45.0, mu=0.12, sigma=0.40),   # 某科技股，最高波动
    "UTILITY": dict(start=8.0, mu=0.03, sigma=0.14),  # 某公用事业，最稳
}

N_DAYS = 750  # 约三年交易日
SEED = 20260604


def simulate_prices() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(end="2025-12-31", periods=N_DAYS)

    # 共同的市场因子，制造股票间的正相关
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
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = simulate_prices()

    parquet_path = OUT_DIR / "sample_prices.parquet"
    csv_path = OUT_DIR / "sample_prices.csv"
    df.to_parquet(parquet_path)
    df.to_csv(csv_path, encoding="utf-8")

    print(f"已生成内置示例数据：{len(df)} 行 x {df.shape[1]} 只股票")
    print(f"  - {parquet_path}")
    print(f"  - {csv_path}")
    print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
