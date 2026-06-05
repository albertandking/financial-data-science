# 附录C 数据字典

## C.1 内置示例数据 `sample_prices`

由 `scripts/make_sample_data.py` 生成（固定随机种子，可复现）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `date`（索引） | datetime | 交易日 |
| `BANK` | float | 银行股收盘价（虚构，低波动） |
| `LIQUOR` | float | 白酒股收盘价（虚构，高成长高波动） |
| `TECH` | float | 科技股收盘价（虚构，最高波动） |
| `UTILITY` | float | 公用事业收盘价（虚构，最稳） |

加载方式：

```python
from fds import load_sample_prices
prices = load_sample_prices()
```

## C.2 真实数据字段（akshare `stock_zh_a_hist`）

| 字段 | 含义 |
|---|---|
| 日期 | 交易日 |
| 开盘 / 收盘 / 最高 / 最低 | OHLC 价格 |
| 成交量 | 成交股数 |
| 成交额 | 成交金额 |
| 涨跌幅 | 当日涨跌百分比 |
| 换手率 | 当日换手率 |

> 接口字段可能随 akshare 版本变化，以实际返回为准。
