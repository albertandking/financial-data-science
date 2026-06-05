# 数据目录说明

| 子目录 | 是否入库 | 内容 |
|---|---|---|
| `processed/` | ✅ 入库 | 清洗好的**内置示例数据集**，离线可读，全书基础章节依赖它 |
| `raw/` | ❌ 不入库 | 从中国市场接口（akshare/tushare）抓取的**原始数据** |
| `external/` | ❌ 不入库 | 其他外部下载数据（如手动下载的 Excel/CSV） |

## 内置示例数据集

`processed/sample_prices.parquet`（及同名 `.csv`）由
`scripts/make_sample_data.py` 用固定随机种子合成，**可复现**。

| 列名 | 含义（教学虚构标的） | 形态 |
|---|---|---|
| `BANK` | 银行股 | 低波动 |
| `LIQUOR` | 白酒股 | 高成长高波动 |
| `TECH` | 科技股 | 最高波动 |
| `UTILITY` | 公用事业 | 最稳健 |

索引为交易日（约 750 个交易日）。合成数据仅用于教学演示，**不代表真实行情**。

## 获取真实数据

```bash
uv sync --extra data
uv run python scripts/fetch_data.py --symbol 600519 --start 20230101 --end 20251231
```
