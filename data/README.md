# 数据目录说明

| 子目录 | 是否入库 | 内容 |
|---|---|---|
| `processed/` | ✅ 入库 | 清洗好的**内置示例数据集**，离线可读，全书章节依赖它 |
| `raw/` | ❌ 不入库 | 从中国市场接口（akshare/tushare）抓取的**原始数据** |
| `external/` | ❌ 不入库 | 其他外部下载数据（如手动下载的 Excel/CSV） |

## 内置示例数据集

全部由 `scripts/make_sample_data.py` 用固定随机种子合成，**可复现**。
均为合成数据，仅用于教学演示，**不代表真实行情/公司/借款人**。

```bash
uv run python scripts/make_sample_data.py
```

### 1. `sample_prices` —— 股票日度价格（约750交易日）

| 列名 | 含义（教学虚构标的） | 形态 |
|---|---|---|
| `BANK` | 银行股 | 低波动 |
| `LIQUOR` | 白酒股 | 高成长高波动 |
| `TECH` | 科技股 | 最高波动 |
| `UTILITY` | 公用事业 | 最稳健 |

索引为交易日。加载：`from fds import load_sample_prices`。

### 2. `market` —— 市场指数与无风险利率（与上面股票相关）

| 列名 | 含义 |
|---|---|
| `index_close` | 市场指数收盘（类沪深300，起点3000） |
| `index_return` | 指数日收益率 |
| `rf_annual` | 年化无风险利率（约2%） |
| `rf_daily` | 日度无风险利率 |

4 只股票对该指数有**真实的 beta**（UTILITY/BANK 低、LIQUOR≈1、TECH 高），
第7章 CAPM 可直接使用。加载：`from fds import load_market`。

### 3. `factors` —— 教学因子日度序列（第7章多因子回归）

| 列名 | 含义 |
|---|---|
| `MKT` | 市场超额收益（真实，= index_return − rf_daily） |
| `HML` | 价值−成长，由 (BANK+UTILITY)/2 − (TECH+LIQUOR)/2 多空构造（与 FF 方法一致，真实相关） |
| `SMB` | 小市值因子（**合成示意**——本股票池仅4只、无市值，规模无法真实构造） |
| `MOM` | 动量因子（**合成示意**，同上） |

价值股（BANK/UTILITY）对 HML 有显著正载荷、成长股（TECH/LIQUOR）显著负载荷，
回归结论可信且可解释。`MKT/HML` 基于真实数据构造，`SMB/MOM` 为标注的合成因子。
加载：`from fds import load_factors`。

### 4. `fundamentals` —— 公司-年度财务面板（200家×8年=1600行）

| 列名 | 含义 |
|---|---|
| `firm` | 公司代码（F000–F199） |
| `year` | 年度（2018–2025） |
| `industry` | 行业 |
| `roa` | 资产收益率（因变量） |
| `leverage` | 资产负债率 |
| `size` | 公司规模（log 总资产） |
| `revenue_growth` | 营收增长率 |

数据生成过程内置**已知系数**（杠杆对 ROA 的真实效应 −0.12）与**公司固定效应**，
且固定效应与杠杆相关——故意制造，凸显固定效应回归相对 Pooled OLS 的必要性。
第8章面板回归可还原真实系数。加载：`from fds import load_fundamentals`。

### 5. `credit` —— 信用违约样本（5000个借款人）

| 列名 | 含义 |
|---|---|
| `age` | 年龄 |
| `income` | 年收入（元） |
| `debt_to_income` | 负债收入比 |
| `credit_history_months` | 信用历史（月） |
| `num_open_accounts` | 在用账户数 |
| `num_delinquencies` | 历史逾期次数 |
| `utilization` | 额度使用率 |
| `default` | **违约标签（0/1）** |

通过 logistic 数据生成过程构造，含**类别不平衡**，特征方向符合金融直觉。
第16章信用评分卡、不平衡处理与 KS/AUC 评估可直接使用。
加载：`from fds import load_credit`。

## 获取真实数据

```bash
uv sync --extra data
uv run python scripts/fetch_data.py --symbol 600519 --start 20230101 --end 20251231
```
