"""从中国市场数据接口抓取真实数据（联网，可选）。

依赖 data 组：
    uv sync --extra data

默认用 akshare（免注册）。tushare 需要在环境变量 TUSHARE_TOKEN 中提供 token。
抓取到的原始数据写入 data/raw/（该目录默认不入库，见 .gitignore）。

示例：
    uv run python scripts/fetch_data.py --symbol 600519 --start 20230101 --end 20251231
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查时需要，避免运行时强依赖 pandas
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def fetch_with_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """用 akshare 抓取 A 股日线（前复权）。"""
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    return df


def fetch_with_tushare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """用 tushare 抓取 A 股日线，需要 TUSHARE_TOKEN。"""
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN")
    pro = ts.pro_api(token)
    # tushare 代码形如 600519.SH / 000001.SZ
    ts_code = symbol if "." in symbol else f"{symbol}.SH"
    return pro.daily(ts_code=ts_code, start_date=start, end_date=end)


def main() -> None:
    """解析命令行参数，抓取行情并保存到 data/raw/。"""
    parser = argparse.ArgumentParser(description="抓取中国市场行情数据")
    parser.add_argument("--symbol", default="600519", help="股票代码，如 600519")
    parser.add_argument("--start", default="20230101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--source",
        choices=["akshare", "tushare"],
        default="akshare",
        help="数据源（默认 akshare，免注册）",
    )
    args = parser.parse_args()

    if args.source == "akshare":
        df = fetch_with_akshare(args.symbol, args.start, args.end)
    else:
        df = fetch_with_tushare(args.symbol, args.start, args.end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{args.source}_{args.symbol}_{args.start}_{args.end}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存 {len(df)} 行到 {out}")


if __name__ == "__main__":
    main()
