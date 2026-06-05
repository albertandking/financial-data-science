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

## C.2 内置市场数据 `market`

市场指数与无风险利率日度序列；4 只股票对该指数有真实 beta（第7章 CAPM 用）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `date`（索引） | datetime | 交易日 |
| `index_close` | float | 市场指数收盘（类沪深300，起点3000） |
| `index_return` | float | 指数日收益率 |
| `rf_annual` | float | 年化无风险利率（约2%） |
| `rf_daily` | float | 日度无风险利率 |

加载：`from fds import load_market`。

## C.3 内置教学因子 `factors`

日度因子序列（749行），供第7章多因子回归。

| 字段 | 类型 | 含义 |
|---|---|---|
| `date`（索引） | datetime | 交易日 |
| `MKT` | float | 市场超额收益（真实，= index_return − rf_daily） |
| `HML` | float | 价值−成长，由 (BANK+UTILITY)/2 − (TECH+LIQUOR)/2 多空构造（真实相关） |
| `SMB` | float | 小市值因子（合成示意，股票池无市值无法真实构造） |
| `MOM` | float | 动量因子（合成示意） |

`MKT/HML` 基于真实数据构造、回归结论可信；`SMB/MOM` 为标注的合成因子。
加载：`from fds import load_factors`。

## C.4 内置财务面板 `fundamentals`

公司-年度平衡面板（200家×8年=1600行），内置已知系数与公司固定效应（第8章用）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `firm` | str | 公司代码 F000–F199 |
| `year` | int | 年度 2018–2025 |
| `industry` | str | 行业 |
| `roa` | float | 资产收益率（因变量） |
| `leverage` | float | 资产负债率（真实系数 −0.12） |
| `size` | float | 公司规模（log 总资产） |
| `revenue_growth` | float | 营收增长率 |

加载：`from fds import load_fundamentals`。

## C.5 内置信用样本 `credit`

信用违约样本（5000个借款人，含类别不平衡，第18章用）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `age` | int | 年龄 |
| `income` | int | 年收入（元） |
| `debt_to_income` | float | 负债收入比 |
| `credit_history_months` | int | 信用历史（月） |
| `num_open_accounts` | int | 在用账户数 |
| `num_delinquencies` | int | 历史逾期次数 |
| `utilization` | float | 额度使用率 |
| `default` | int | 违约标签（0/1） |

加载：`from fds import load_credit`。

## C.6 真实数据字段（akshare `stock_zh_a_hist`）

| 字段 | 含义 |
|---|---|
| 日期 | 交易日 |
| 开盘 / 收盘 / 最高 / 最低 | OHLC 价格 |
| 成交量 | 成交股数 |
| 成交额 | 成交金额 |
| 涨跌幅 | 当日涨跌百分比 |
| 换手率 | 当日换手率 |

> 接口字段可能随 akshare 版本变化，以实际返回为准。
