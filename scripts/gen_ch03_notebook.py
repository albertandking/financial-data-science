"""生成 ch03_data_acquisition.ipynb，避免直接写入 JSON 时的编码问题。"""
import json
from pathlib import Path

def code_cell(cell_id, source_lines):
    return {
        "id": cell_id,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def md_cell(cell_id, source_lines):
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


cells = []

# ── c03-000 intro ─────────────────────────────────────────────────────────────
cells.append(md_cell("c03-000", [
    "# 第3章 金融数据获取与清洗 —— 配套代码\n",
    "\n",
    "对应正文 `book/part1/03-data-acquisition.md`。\n",
    "\n",
    "**运行前请先生成内置数据：**\n",
    "```bash\n",
    "uv run python scripts/make_sample_data.py\n",
    "```\n",
    "\n",
    "本 notebook 分为两部分：\n",
    "- **离线部分**（全部可跑）：用内置数据模拟真实问题——停牌缺失、复权跳变、异常值、多标的对齐、数据质量检查；\n",
    "- **联网格**（标有 [网络] 的格）：调用 akshare/tushare 抓取真实数据，需先 `uv sync --extra data`；"
    "已用 `try/except` 包裹，未安装/无网时**自动跳过、不报错**。",
]))

# ── c03-001 全局导入 ──────────────────────────────────────────────────────────
cells.append(code_cell("c03-001", [
    "# 全局导入与配置\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from fds import load_sample_prices, daily_returns, set_chinese_font\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "set_chinese_font()  # 设置中文字体，避免乱码\n",
    "\n",
    "# 加载内置数据\n",
    "prices = load_sample_prices()\n",
    "print(f'内置数据形状：{prices.shape}')\n",
    "print(f'时间范围：{prices.index.min().date()} ~ {prices.index.max().date()}')\n",
    "print(f'股票列：{prices.columns.tolist()}')\n",
    "prices.head(3)",
]))

# ── c03-010 复权原理标题 ──────────────────────────────────────────────────────
cells.append(md_cell("c03-010", [
    "## 3.1 复权原理数值演示\n",
    "\n",
    "构造一个含除权跳变的简化价格序列，演示前复权如何消除跳变。\n",
    "\n",
    "场景：100 天价格序列，第 50 天发生「10 转 5」（股数变为 1.5 倍，价格变为 2/3）。\n",
    "\n",
    "**前复权思路**：将除权日之前的历史价格乘以除权比例（2/3），使之与除权后的价格在同一量纲下，序列自然连续。",
]))

# ── c03-011 构造除权数据 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-011", [
    "# 构造含除权跳变的未复权价格\n",
    "rng = np.random.default_rng(0)\n",
    "n = 100\n",
    "dates = pd.bdate_range('2024-01-02', periods=n)\n",
    "\n",
    "# 生成平滑价格（几何随机游走）\n",
    "log_ret = rng.normal(0.0005, 0.015, n)\n",
    "price_smooth = 100 * np.exp(np.cumsum(log_ret))\n",
    "\n",
    "# 第 50 天插入除权：价格乘以 2/3（10转5，股数x1.5，价格÷1.5）\n",
    "EX_DAY = 50\n",
    "EX_RATIO = 2 / 3\n",
    "\n",
    "price_raw = price_smooth.copy()\n",
    "price_raw[EX_DAY:] = price_smooth[EX_DAY:] * EX_RATIO\n",
    "\n",
    "print(f'未复权 第{EX_DAY-1}天（除权前）: {price_raw[EX_DAY-1]:.2f}')\n",
    "print(f'未复权 第{EX_DAY}天（除权后）:  {price_raw[EX_DAY]:.2f}')\n",
    "raw_ret = price_raw[EX_DAY] / price_raw[EX_DAY-1] - 1\n",
    "print(f'未复权 除权日虚假跌幅: {raw_ret:.2%}  <- 根本没有真实亏损！')",
]))

# ── c03-012 前复权计算 ────────────────────────────────────────────────────────
cells.append(code_cell("c03-012", [
    "# 构造前复权价格\n",
    "# 前复权：将除权日之前的历史价格按比例压低，使序列连续\n",
    "# adj_factor[:EX_DAY] = EX_RATIO  -> 除权前价格乘以 EX_RATIO，与除权后同一量纲\n",
    "# adj_factor[EX_DAY:] = 1.0       -> 除权后价格不变（最新价为基准）\n",
    "adj_factor = np.ones(n)\n",
    "adj_factor[:EX_DAY] = EX_RATIO   # 前复权：历史价向下调整\n",
    "\n",
    "price_qfq = price_raw * adj_factor   # adj_factor[-1]=1，无需额外归一化\n",
    "\n",
    "qfq_ret  = price_qfq[EX_DAY] / price_qfq[EX_DAY-1] - 1\n",
    "true_ret = price_smooth[EX_DAY] / price_smooth[EX_DAY-1] - 1\n",
    "\n",
    "print(f'前复权 除权日收益率: {qfq_ret:.6f}')\n",
    "print(f'真实收益率:         {true_ret:.6f}')\n",
    "print(f'差异:               {abs(qfq_ret - true_ret):.2e}  <- 接近 0，复权成功')",
]))

# ── c03-013 复权可视化 ────────────────────────────────────────────────────────
cells.append(code_cell("c03-013", [
    "fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n",
    "\n",
    "ax = axes[0]\n",
    "ax.plot(dates, price_raw, label='未复权（原始）', color='tomato', lw=1.5)\n",
    "ax.plot(dates, price_qfq, label='前复权 (qfq)', color='steelblue', lw=1.5, ls='--')\n",
    "ax.axvline(dates[EX_DAY], color='gray', ls=':', label=f'除权日 (Day {EX_DAY})')\n",
    "ax.set_ylabel('价格')\n",
    "ax.set_title('前复权消除价格跳变')\n",
    "ax.legend()\n",
    "\n",
    "ax2 = axes[1]\n",
    "ret_raw_s = pd.Series(price_raw, index=dates).pct_change()\n",
    "ret_qfq_s = pd.Series(price_qfq, index=dates).pct_change()\n",
    "ax2.plot(dates, ret_raw_s * 100, label='未复权收益率 (%)', color='tomato', lw=1)\n",
    "ax2.plot(dates, ret_qfq_s * 100, label='前复权收益率 (%)', color='steelblue', lw=1, ls='--')\n",
    "ax2.axvline(dates[EX_DAY], color='gray', ls=':')\n",
    "ax2.set_ylabel('日收益率 (%)')\n",
    "ax2.set_title('收益率对比：除权日虚假暴跌消失')\n",
    "ax2.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# ── c03-020 停牌标题 ──────────────────────────────────────────────────────────
cells.append(md_cell("c03-020", [
    "## 3.2 停牌缺失与多标的日历对齐\n",
    "\n",
    "用内置数据模拟停牌场景：\n",
    "- 对 `TECH` 随机置 NaN（模拟停牌）；\n",
    "- 让 `BANK` 前 60 天无数据（模拟较晚上市）；\n",
    "- 演示正确的对齐与填充流程，并比较不同策略对年化波动率的影响。",
]))

# ── c03-021 制造停牌 ──────────────────────────────────────────────────────────
cells.append(code_cell("c03-021", [
    "dirty = prices.copy().astype(float)\n",
    "\n",
    "rng2 = np.random.default_rng(42)\n",
    "halt_idx = rng2.choice(len(dirty), size=15, replace=False)\n",
    "dirty.iloc[halt_idx, dirty.columns.get_loc('TECH')] = np.nan\n",
    "dirty.iloc[:60, dirty.columns.get_loc('BANK')] = np.nan\n",
    "\n",
    "print('制造缺失后各列 NaN 数量：')\n",
    "print(dirty.isna().sum())",
]))

# ── c03-022 波动率对比 ────────────────────────────────────────────────────────
cells.append(code_cell("c03-022", [
    "def annvol(series):\n",
    "    return series.pct_change().std() * np.sqrt(252)\n",
    "\n",
    "clean_ffill   = dirty.ffill()\n",
    "clean_ffill5  = dirty.ffill(limit=5)\n",
    "clean_drop    = dirty.dropna()\n",
    "\n",
    "cols = ['TECH', 'BANK']\n",
    "results = pd.DataFrame({\n",
    "    '原始（无缺失）': [annvol(prices[c]) for c in cols],\n",
    "    'ffill 无限':     [annvol(clean_ffill[c]) for c in cols],\n",
    "    'ffill(limit=5)': [annvol(clean_ffill5[c]) for c in cols],\n",
    "    'dropna':         [annvol(clean_drop[c]) for c in cols],\n",
    "}, index=cols)\n",
    "\n",
    "print('年化波动率对比（各清洗策略）：')\n",
    "print(results.round(4))\n",
    "print('\\n结论：ffill 将停牌日收益置 0，压低波动率；dropna 最接近真实。')",
]))

# ── c03-023 停牌可视化 ────────────────────────────────────────────────────────
cells.append(code_cell("c03-023", [
    "fig, ax = plt.subplots(figsize=(11, 4))\n",
    "ax.plot(dirty.index, dirty['TECH'], label='含停牌（NaN）', color='gray', lw=1, alpha=0.6)\n",
    "ax.plot(clean_ffill.index, clean_ffill['TECH'], label='ffill 无限', color='steelblue', lw=1)\n",
    "ax.plot(clean_ffill5.index, clean_ffill5['TECH'], label='ffill(limit=5)', color='orange', lw=1, ls='--')\n",
    "\n",
    "halt_dates = dirty.index[dirty['TECH'].isna()]\n",
    "ax.scatter(halt_dates, [dirty['TECH'].dropna().mean()] * len(halt_dates),\n",
    "           color='red', s=20, zorder=5, label='停牌日', marker='x')\n",
    "\n",
    "ax.set_title('TECH 停牌处理：ffill 填充策略对比')\n",
    "ax.set_ylabel('价格')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# ── c03-030 异常值标题 ────────────────────────────────────────────────────────
cells.append(md_cell("c03-030", [
    "## 3.3 异常值识别：涨跌停约束 + MAD 法\n",
    "\n",
    "在内置数据中注入极端收益，对比两种检测方法：\n",
    "- **涨跌停约束**：|日收益| > 10%\n",
    "- **MAD 法**：修正 Z-score > 3.5（比 3sigma 法更稳健）",
]))

# ── c03-031 注入异常值 ────────────────────────────────────────────────────────
cells.append(code_cell("c03-031", [
    "liquor_prices = prices['LIQUOR'].copy()\n",
    "rng3 = np.random.default_rng(99)\n",
    "inject_idx = rng3.choice(range(5, len(liquor_prices)-1), size=5, replace=False)\n",
    "\n",
    "liquor_dirty = liquor_prices.copy()\n",
    "inject_mags = rng3.choice([1.28, 1.30, 0.72, 0.70, 1.29], size=5)\n",
    "for i, mag in zip(inject_idx, inject_mags):\n",
    "    liquor_dirty.iloc[i] = liquor_dirty.iloc[i] * mag\n",
    "\n",
    "ret_dirty = liquor_dirty.pct_change().dropna()\n",
    "print(f'注入异常后 最大日收益：{ret_dirty.max():.2%}')\n",
    "print(f'注入异常后 最小日收益：{ret_dirty.min():.2%}')",
]))

# ── c03-032 异常值检测函数 ────────────────────────────────────────────────────
cells.append(code_cell("c03-032", [
    "def detect_outliers_limit(series, limit=0.10):\n",
    "    \"\"\"涨跌停约束法：|收益率| > limit 标记为异常。\"\"\"\n",
    "    return series.abs() > limit\n",
    "\n",
    "\n",
    "def detect_outliers_mad(series, k=3.5):\n",
    "    \"\"\"MAD 法：修正 Z-score 绝对值 > k 标记为异常。\n",
    "    修正 Z-score = 0.6745 * (x - median) / MAD\n",
    "    \"\"\"\n",
    "    med = series.median()\n",
    "    mad = (series - med).abs().median()\n",
    "    if mad == 0:\n",
    "        return pd.Series(False, index=series.index)\n",
    "    modified_z = 0.6745 * (series - med) / mad\n",
    "    return modified_z.abs() > k\n",
    "\n",
    "\n",
    "def detect_outliers_3sigma(series, n=3.0):\n",
    "    \"\"\"3sigma 法：|x - mean| > n * std 标记为异常。\"\"\"\n",
    "    return (series - series.mean()).abs() > n * series.std()\n",
    "\n",
    "\n",
    "flag_limit  = detect_outliers_limit(ret_dirty, limit=0.10)\n",
    "flag_mad    = detect_outliers_mad(ret_dirty, k=3.5)\n",
    "flag_3sigma = detect_outliers_3sigma(ret_dirty, n=3.0)\n",
    "\n",
    "print(f'涨跌停约束（|r|>10%） 检出：{flag_limit.sum()} 天')\n",
    "print(f'MAD 法（k=3.5）       检出：{flag_mad.sum()} 天')\n",
    "print(f'3sigma 法（n=3）      检出：{flag_3sigma.sum()} 天')",
]))

# ── c03-033 异常值可视化 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-033", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "\n",
    "ax = axes[0]\n",
    "ax.plot(ret_dirty.index, ret_dirty * 100, color='steelblue', lw=0.8, alpha=0.7)\n",
    "ax.scatter(ret_dirty[flag_mad].index, ret_dirty[flag_mad] * 100,\n",
    "           color='red', s=50, zorder=5, label='MAD 异常')\n",
    "ax.scatter(ret_dirty[flag_limit].index, ret_dirty[flag_limit] * 100,\n",
    "           color='orange', s=30, marker='D', zorder=4, label='涨跌停约束异常')\n",
    "ax.axhline(10, color='orange', ls='--', lw=0.8, alpha=0.5)\n",
    "ax.axhline(-10, color='orange', ls='--', lw=0.8, alpha=0.5)\n",
    "ax.set_title('LIQUOR 日收益率（含注入异常值）')\n",
    "ax.set_ylabel('日收益率 (%)')\n",
    "ax.legend()\n",
    "\n",
    "ax2 = axes[1]\n",
    "ax2.hist(ret_dirty * 100, bins=60, color='steelblue', alpha=0.7, edgecolor='white')\n",
    "for v in ret_dirty[flag_mad]:\n",
    "    ax2.axvline(v * 100, color='red', lw=1.5, alpha=0.8)\n",
    "ax2.set_title('收益率直方图（红线=MAD 异常）')\n",
    "ax2.set_xlabel('日收益率 (%)')\n",
    "ax2.set_ylabel('频次')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# ── c03-040 多标的对齐标题 ────────────────────────────────────────────────────
cells.append(md_cell("c03-040", [
    "## 3.4 多标的交易日历对齐\n",
    "\n",
    "模拟三只上市时间不同的股票，演示正确的对齐流程：\n",
    "1. `pd.concat` 按日期索引自动对齐（并集）\n",
    "2. `ffill(limit=5)` 前向填充停牌日价格\n",
    "3. 在对齐后的价格序列上计算收益率",
]))

# ── c03-041 多标的对齐代码 ────────────────────────────────────────────────────
cells.append(code_cell("c03-041", [
    "s_bank   = prices['BANK']\n",
    "s_liquor = prices['LIQUOR'].iloc[80:]   # 第80天才上市\n",
    "s_tech   = prices['TECH'].iloc[30:].copy()\n",
    "\n",
    "# TECH 额外加 10 天停牌\n",
    "rng4 = np.random.default_rng(7)\n",
    "halt_tech = rng4.choice(range(len(s_tech)), size=10, replace=False)\n",
    "s_tech.iloc[halt_tech] = np.nan\n",
    "\n",
    "# 对齐\n",
    "panel_raw = pd.concat(\n",
    "    [s_bank.rename('BANK'), s_liquor.rename('LIQUOR'), s_tech.rename('TECH')],\n",
    "    axis=1\n",
    ").sort_index()\n",
    "\n",
    "panel_ffill = panel_raw.ffill(limit=5)\n",
    "panel_ret   = panel_ffill.pct_change()\n",
    "\n",
    "print(f'并集日历天数：{len(panel_raw)}')\n",
    "print(f'交集日历天数：{len(panel_raw.dropna())}')\n",
    "print('\\n相关系数矩阵（ffill对齐后）：')\n",
    "print(panel_ret.corr().round(3))",
]))

# ── c03-042 多标的可视化 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-042", [
    "cumret = (1 + panel_ret.dropna(how='all')).cumprod()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "for col in cumret.columns:\n",
    "    ax.plot(cumret.index, cumret[col], label=col, lw=1.5)\n",
    "\n",
    "ax.axvline(s_liquor.index[0], color='orange', ls=':', lw=1.2, label='LIQUOR 上市日')\n",
    "ax.axvline(s_tech.index[0], color='green', ls=':', lw=1.2, label='TECH 上市日')\n",
    "\n",
    "ax.set_title('多标的对齐后累积收益曲线')\n",
    "ax.set_ylabel('累积收益（净值）')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# ── c03-050 数据质量标题 ──────────────────────────────────────────────────────
cells.append(md_cell("c03-050", [
    "## 3.5 数据质量检查函数\n",
    "\n",
    "实现通用的 `data_quality_report`，覆盖四个维度：\n",
    "**完整性 / 一致性 / 时效性 / 准确性**",
]))

# ── c03-051 质量检查函数 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-051", [
    "def data_quality_report(df, limit_pct=0.10, verbose=True):\n",
    "    \"\"\"金融数据质量四维检查报告。\"\"\"\n",
    "    report = {}\n",
    "\n",
    "    # 1. 完整性\n",
    "    report['completeness'] = {\n",
    "        'nan_rate':  df.isna().mean().to_dict(),\n",
    "        'dup_index': int(df.index.duplicated().sum()),\n",
    "    }\n",
    "\n",
    "    # 2. 时效性\n",
    "    latest = df.index.max()\n",
    "    report['timeliness'] = {\n",
    "        'start': str(df.index.min().date()),\n",
    "        'end':   str(latest.date()),\n",
    "        'lag_days': int((pd.Timestamp.today() - latest).days),\n",
    "        'n_rows': len(df),\n",
    "    }\n",
    "\n",
    "    # 3. 准确性（日涨跌幅超限检测）\n",
    "    rets = df.pct_change()\n",
    "    extreme = (rets.abs() > limit_pct).sum()\n",
    "    report['accuracy'] = extreme.to_dict()\n",
    "\n",
    "    if verbose:\n",
    "        bar = '=' * 55\n",
    "        t = report['timeliness']\n",
    "        print(bar)\n",
    "        print('  数据质量检查报告')\n",
    "        print(bar)\n",
    "        print(f\"  形状：{df.shape}  时间：{t['start']} ~ {t['end']}\")\n",
    "        print(f\"  最新日期距今 {t['lag_days']} 天\")\n",
    "        print(f\"  重复索引：{report['completeness']['dup_index']} 个\")\n",
    "        print()\n",
    "        print('【完整性】')\n",
    "        for col, rate in report['completeness']['nan_rate'].items():\n",
    "            status = 'OK' if rate == 0 else f'WARNING {rate:.2%} NaN'\n",
    "            print(f'    {col:12s}: {status}')\n",
    "        print()\n",
    "        print(f'【准确性】超 +/-{limit_pct*100:.0f}% 的日收益记录数：')\n",
    "        for col, cnt in report['accuracy'].items():\n",
    "            print(f'    {col:12s}: {cnt} 条')\n",
    "        print(bar)\n",
    "\n",
    "    return report\n",
    "\n",
    "\n",
    "print('=== 内置数据（干净）===')\n",
    "_ = data_quality_report(prices)\n",
    "print()\n",
    "print('=== 含停牌NaN的数据 ===')\n",
    "_ = data_quality_report(dirty)",
]))

# ── c03-060 Parquet 缓存标题 ─────────────────────────────────────────────────
cells.append(md_cell("c03-060", [
    "## 3.6 数据存储：Parquet 缓存演示\n",
    "\n",
    "演示「优先读缓存，缓存不存在才生成」的模式，并对比 Parquet vs CSV 的存储效率。",
]))

# ── c03-061 Parquet 代码 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-061", [
    "import tempfile\n",
    "from pathlib import Path\n",
    "\n",
    "def load_or_generate(symbol, cache_dir):\n",
    "    cache_dir = Path(cache_dir)\n",
    "    cache_dir.mkdir(parents=True, exist_ok=True)\n",
    "    cache_path = cache_dir / f'{symbol}.parquet'\n",
    "\n",
    "    if cache_path.exists():\n",
    "        print(f'[缓存命中] 读取 {cache_path.name}')\n",
    "        return pd.read_parquet(cache_path)\n",
    "\n",
    "    print('[缓存未命中] 生成数据并落盘...')\n",
    "    df = prices[[symbol]]\n",
    "    df.to_parquet(cache_path)\n",
    "    print(f'  已写入 {cache_path.name}  大小：{cache_path.stat().st_size / 1024:.1f} KB')\n",
    "    return df\n",
    "\n",
    "\n",
    "tmp_dir = Path(tempfile.mkdtemp()) / 'fds_cache'\n",
    "\n",
    "df1 = load_or_generate('BANK', tmp_dir)  # 第一次：未命中\n",
    "df2 = load_or_generate('BANK', tmp_dir)  # 第二次：命中\n",
    "print(f'往返一致：{df1.equals(df2)}')\n",
    "\n",
    "# 存储效率对比\n",
    "csv_path = tmp_dir / 'BANK.csv'\n",
    "prices[['BANK']].to_csv(csv_path)\n",
    "parquet_path = tmp_dir / 'BANK.parquet'\n",
    "print(f'CSV 大小：     {csv_path.stat().st_size / 1024:.1f} KB')\n",
    "print(f'Parquet 大小： {parquet_path.stat().st_size / 1024:.1f} KB')\n",
    "print(f'压缩比：       {csv_path.stat().st_size / parquet_path.stat().st_size:.1f}x')",
]))

# ── c03-070 联网标题 ──────────────────────────────────────────────────────────
cells.append(md_cell("c03-070", [
    "## 3.7 [网络] 联网抓取（可选）\n",
    "\n",
    "需先运行 `uv sync --extra data`。以下格已用 `try/except` 包裹，未安装时安全跳过。",
]))

# ── c03-071 akshare 日线 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-071", [
    "try:\n",
    "    import akshare as ak\n",
    "    print(f'akshare 版本：{ak.__version__}')\n",
    "    df_moutai = ak.stock_zh_a_hist(\n",
    "        symbol='600519', period='daily',\n",
    "        start_date='20230101', end_date='20241231', adjust='qfq',\n",
    "    )\n",
    "    print(f'贵州茅台前复权日线行数：{len(df_moutai)}')\n",
    "    print(df_moutai.tail(3).to_string())\n",
    "except ImportError:\n",
    "    print('[跳过] akshare 未安装。请运行：uv sync --extra data')\n",
    "except Exception as e:\n",
    "    print(f'[跳过] 抓取失败（{type(e).__name__}: {e}）')",
]))

# ── c03-072 akshare 宏观 ──────────────────────────────────────────────────────
cells.append(code_cell("c03-072", [
    "try:\n",
    "    import akshare as ak\n",
    "    cpi = ak.macro_china_cpi_monthly()\n",
    "    print('中国 CPI 月度数据（最新5行）：')\n",
    "    print(cpi.tail())\n",
    "except ImportError:\n",
    "    print('[跳过] akshare 未安装')\n",
    "except Exception as e:\n",
    "    print(f'[跳过] 宏观数据抓取失败（{type(e).__name__}: {e}）')",
]))

# ── c03-073 tushare ───────────────────────────────────────────────────────────
cells.append(code_cell("c03-073", [
    "try:\n",
    "    import os, tushare as ts\n",
    "    token = os.environ.get('TUSHARE_TOKEN', '')\n",
    "    if not token:\n",
    "        print('[跳过] 未设置 TUSHARE_TOKEN 环境变量')\n",
    "    else:\n",
    "        ts.set_token(token)\n",
    "        pro = ts.pro_api()\n",
    "        df_ts = pro.daily(ts_code='600519.SH', start_date='20230101', end_date='20231231')\n",
    "        print('tushare 贵州茅台日线（前3行）：')\n",
    "        print(df_ts.head(3))\n",
    "except ImportError:\n",
    "    print('[跳过] tushare 未安装')\n",
    "except Exception as e:\n",
    "    print(f'[跳过] tushare 失败（{type(e).__name__}: {e}）')",
]))

# ── c03-080 习题标题 ──────────────────────────────────────────────────────────
cells.append(md_cell("c03-080", [
    "## 3.8 习题参考解答\n",
    "\n",
    "以下为第3章习题的参考答案（离线题给出可运行代码）。",
]))

# ── c03-081 习题1 ─────────────────────────────────────────────────────────────
cells.append(code_cell("c03-081", [
    "# 习题1：停牌缺失 + 三种策略 + 波动率比较\n",
    "rng_e1 = np.random.default_rng(2024)\n",
    "tech_ex = prices['TECH'].copy()\n",
    "halt_e1 = rng_e1.choice(range(5, len(tech_ex)-1), size=15, replace=False)\n",
    "tech_d = tech_ex.copy()\n",
    "tech_d.iloc[halt_e1] = np.nan\n",
    "\n",
    "vol_true   = tech_ex.pct_change().std() * np.sqrt(252)\n",
    "vol_ffill  = tech_d.ffill().pct_change().std() * np.sqrt(252)\n",
    "vol_ffill3 = tech_d.ffill(limit=3).pct_change().std() * np.sqrt(252)\n",
    "vol_drop   = tech_d.dropna().pct_change().std() * np.sqrt(252)\n",
    "\n",
    "print('习题1 TECH 年化波动率对比')\n",
    "print(f'  原始（无缺失）    : {vol_true:.4f}')\n",
    "print(f'  ffill 无限        : {vol_ffill:.4f}  偏低 {(vol_true-vol_ffill)/vol_true:.1%}')\n",
    "print(f'  ffill(limit=3)    : {vol_ffill3:.4f}  偏低 {(vol_true-vol_ffill3)/vol_true:.1%}')\n",
    "print(f'  dropna            : {vol_drop:.4f}  最接近真实')",
]))

# ── c03-082 习题2 ─────────────────────────────────────────────────────────────
cells.append(code_cell("c03-082", [
    "# 习题2：复权因子数值验证\n",
    "rng_e2 = np.random.default_rng(333)\n",
    "p2 = 50 * np.exp(np.cumsum(rng_e2.normal(0.0003, 0.012, 100)))\n",
    "EX2 = 50; RATIO2 = 0.7\n",
    "p2_raw = p2.copy()\n",
    "p2_raw[EX2:] *= RATIO2\n",
    "\n",
    "adj2 = np.ones(100)\n",
    "adj2[:EX2] = RATIO2\n",
    "p2_qfq = p2_raw * adj2\n",
    "\n",
    "ratio_qfq  = p2_qfq[EX2] / p2_qfq[EX2-1] - 1\n",
    "ratio_true = p2[EX2] / p2[EX2-1] - 1\n",
    "\n",
    "print('习题2：前复权验证')\n",
    "print(f'  前复权除权日收益率: {ratio_qfq:.8f}')\n",
    "print(f'  真实收益率:         {ratio_true:.8f}')\n",
    "print(f'  差异:               {abs(ratio_qfq-ratio_true):.2e}')\n",
    "assert abs(ratio_qfq - ratio_true) < 1e-10\n",
    "print('  验证通过')",
]))

# ── c03-083 习题3 ─────────────────────────────────────────────────────────────
cells.append(code_cell("c03-083", [
    "# 习题3：MAD vs 3sigma 检出率对比\n",
    "liquor_base = prices['LIQUOR'].pct_change().dropna()\n",
    "rng_e3 = np.random.default_rng(55)\n",
    "n_inj = 5\n",
    "pos_e3 = rng_e3.choice(range(10, len(liquor_base)-10), size=n_inj, replace=False)\n",
    "ret_e3 = liquor_base.copy()\n",
    "vals   = rng_e3.choice([0.30, -0.30, 0.28, -0.28, 0.32], size=n_inj)\n",
    "ret_e3.iloc[pos_e3] = vals\n",
    "true_set = set(pos_e3)\n",
    "\n",
    "f_mad = detect_outliers_mad(ret_e3, k=3.5)\n",
    "f_3s  = detect_outliers_3sigma(ret_e3, n=3.0)\n",
    "\n",
    "def recall(flag, true_set):\n",
    "    return len(set(np.where(flag)[0]) & true_set) / len(true_set)\n",
    "\n",
    "def fpr(flag, true_set, total):\n",
    "    fp = set(np.where(flag)[0]) - true_set\n",
    "    return len(fp) / (total - len(true_set))\n",
    "\n",
    "N = len(ret_e3)\n",
    "print('习题3：MAD vs 3sigma（注入 5 个 +-28%~32% 极端值）')\n",
    "print(f'  MAD(k=3.5)  召回率：{recall(f_mad, true_set):.0%}   误报率：{fpr(f_mad, true_set, N):.2%}')\n",
    "print(f'  3sigma(n=3) 召回率：{recall(f_3s, true_set):.0%}   误报率：{fpr(f_3s, true_set, N):.2%}')",
]))

# ── 组装 notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent.parent / "notebooks" / "ch03_data_acquisition.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written {out_path}  ({out_path.stat().st_size} bytes)")
