"""Generate ch07_factor_models.ipynb with proper JSON encoding."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "notebooks" / "ch07_factor_models.ipynb"


def md_cell(cell_id, source):
    return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(cell_id, source):
    return {
        "id": cell_id,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── c07-000: 导读 ──────────────────────────────────────────────────
cells.append(md_cell("c07-000", (
    "# 第7章 资产定价与因子模型 —— 配套代码\n"
    "\n"
    "对应正文 `book/part2/07-factor-models.md`。\n"
    "\n"
    "> **运行前准备**：请先在终端执行 `uv run python scripts/make_sample_data.py` 生成内置示例数据。\n"
    "\n"
    "## 数据说明\n"
    "\n"
    "本 notebook 使用两类数据：\n"
    "\n"
    "1. **内置价格数据**（`load_sample_prices()`）：合成的 4 只 A 股风格资产"
    "（BANK/LIQUOR/TECH/UTILITY），约 750 个交易日，用于 CAPM 时序回归演示，**完全离线可跑**。\n"
    "2. **教学示意因子数据**：由于内置数据无基本面信息，无法构造真实 SMB/HML，"
    "多因子部分使用固定随机种子（`np.random.default_rng(42)`）生成的**模拟因子序列**。"
    "**这些数据仅用于演示多因子回归的操作步骤与结果解读，不代表真实 A 股因子结论**。\n"
    "\n"
    "## 演示内容\n"
    "\n"
    "1. 环境初始化与数据加载\n"
    "2. 构造市场代理组合与超额收益\n"
    "3. 单因子 CAPM 回归（OLS 摘要解读）\n"
    "4. 四只股票 Beta 对比汇总\n"
    "5. SML 可视化 + 散点拟合图\n"
    "6. Beta 条形图 + R2 方差分解\n"
    "7. （示意数据）生成模拟因子序列\n"
    "8. （示意数据）FF 三因子回归\n"
    "9. （示意数据）各股票 FF3 汇总\n"
    "10. （示意数据）Carhart 四因子回归\n"
    "11. 因子相关性与 VIF 多重共线性诊断\n"
    "12. 业绩归因分解（CAPM，真实数据）\n"
    "13. 因子累计收益时序图\n"
    "14. 习题参考解答：习题 7.1\n"
    "15. 习题参考解答：习题 7.2\n"
    "16. 习题参考解答：习题 7.3\n"
    "17. 习题参考解答：习题 7.4"
)))

# ── c07-010: 初始化 ──────────────────────────────────────────────
cells.append(code_cell("c07-010", (
    "# Cell 1：环境初始化与数据加载\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mtick\n"
    "import statsmodels.api as sm\n"
    "from scipy import stats\n"
    "\n"
    "from fds import load_sample_prices, daily_returns, set_chinese_font\n"
    "\n"
    "set_chinese_font()\n"
    "\n"
    "prices = load_sample_prices()\n"
    "rets   = daily_returns(prices)       # 简单日收益率\n"
    "\n"
    "TICKERS      = list(rets.columns)    # ['BANK', 'LIQUOR', 'TECH', 'UTILITY']\n"
    "RF_ANNUAL    = 0.02                  # 年化无风险利率 2%\n"
    "RF_DAILY     = RF_ANNUAL / 252       # 折日无风险利率\n"
    "TRADING_DAYS = 252\n"
    "\n"
    "print(f'价格数据维度：{prices.shape}')\n"
    "print(f'收益率数据维度：{rets.shape}')\n"
    "print(f'交易日范围：{rets.index[0].date()} 至 {rets.index[-1].date()}')\n"
    "print(f'无风险利率（年化）：{RF_ANNUAL:.1%}，折日：{RF_DAILY:.6f}')\n"
    "prices.tail(3)"
)))

# ── c07-020: 市场代理 md ──────────────────────────────────────────
cells.append(md_cell("c07-020", (
    "## 7.2 构造市场代理组合与超额收益\n"
    "\n"
    "由于内置数据只有 4 只股票，我们用**等权组合**作为市场组合的代理。\n"
    "\n"
    "CAPM 时序回归方程：\n"
    "\n"
    "$$r_{i,t} - r_{f,t} = \\alpha_i + \\beta_i (r_{m,t} - r_{f,t}) + \\varepsilon_{i,t}$$\n"
    "\n"
    "其中 $r_{f,t}$ 为日度无风险利率（年化 2% ÷ 252），$r_{m,t}$ 为等权市场组合日收益率。"
)))

# ── c07-021: 市场代理 code ──────────────────────────────────────
cells.append(code_cell("c07-021", (
    "# Cell 2：构造市场代理与超额收益序列\n"
    "\n"
    "market_ret = rets.mean(axis=1)       # 等权市场组合\n"
    "market_ret.name = 'Market'\n"
    "mkt_excess  = market_ret - RF_DAILY  # 市场超额收益\n"
    "excess_rets = rets.sub(RF_DAILY)     # 各股票超额收益\n"
    "\n"
    "print('=== 市场代理组合统计 ===')\n"
    "print(f'日均收益：{market_ret.mean():.4f}')\n"
    "print(f'日波动率：{market_ret.std():.4f}')\n"
    "print(f'年化收益：{market_ret.mean() * TRADING_DAYS:.2%}')\n"
    "print(f'年化波动：{market_ret.std() * np.sqrt(TRADING_DAYS):.2%}')\n"
    "print(f'市场风险溢价（年化）：{mkt_excess.mean() * TRADING_DAYS:.2%}')\n"
    "print()\n"
    "print('各股票日超额收益（前 3 行）')\n"
    "excess_rets.head(3)"
)))

# ── c07-030: CAPM md ──────────────────────────────────────────────
cells.append(md_cell("c07-030", (
    "## 7.3 单因子 CAPM 回归：以 TECH 为例\n"
    "\n"
    "用 `statsmodels.OLS` 估计 CAPM，重点解读：\n"
    "- `const`（截距）= $\\alpha$，CAPM 下应为 0\n"
    "- 斜率 = $\\beta$，衡量市场敏感度\n"
    "- $R^2$ = 系统性风险占总风险比例\n"
    "- $t$-统计量和 $p$-值：检验系数是否显著异于 0"
)))

# ── c07-031: CAPM code ──────────────────────────────────────────
cells.append(code_cell("c07-031", (
    "# Cell 3：TECH 单因子 CAPM 回归（详细摘要）\n"
    "\n"
    "y = excess_rets['TECH']\n"
    "X = sm.add_constant(mkt_excess)\n"
    "X.columns = ['alpha_const', 'MKT']\n"
    "\n"
    "result_tech = sm.OLS(y, X).fit()\n"
    "print(result_tech.summary())"
)))

# ── c07-040: All 4 stocks CAPM ──────────────────────────────────
cells.append(code_cell("c07-040", (
    "# Cell 4：四只股票 CAPM 回归汇总\n"
    "\n"
    "capm_results = {}\n"
    "X_mkt = sm.add_constant(mkt_excess)\n"
    "X_mkt.columns = ['alpha_const', 'MKT']\n"
    "\n"
    "rows = []\n"
    "for ticker in TICKERS:\n"
    "    res = sm.OLS(excess_rets[ticker], X_mkt).fit()\n"
    "    capm_results[ticker] = res\n"
    "    rows.append({\n"
    "        '股票': ticker,\n"
    "        'Alpha年化': round(res.params['alpha_const'] * TRADING_DAYS, 4),\n"
    "        'Alpha_t': round(res.tvalues['alpha_const'], 3),\n"
    "        'Alpha_p': round(res.pvalues['alpha_const'], 4),\n"
    "        'Beta': round(res.params['MKT'], 4),\n"
    "        'Beta_t': round(res.tvalues['MKT'], 3),\n"
    "        'R2': round(res.rsquared, 4),\n"
    "        '特质风险': round(1 - res.rsquared, 4),\n"
    "    })\n"
    "\n"
    "df_capm = pd.DataFrame(rows).set_index('股票')\n"
    "print('=== 四只股票 CAPM 回归结果汇总 ===')\n"
    "print(df_capm.to_string())\n"
    "print()\n"
    "print('Alpha_p < 0.05：在 5% 水平下 alpha 显著异于 0（CAPM 预测为 0）')\n"
    "print('Beta_t 大：市场因子对该股票的解释力显著')"
)))

# ── c07-050: SML md ──────────────────────────────────────────────
cells.append(md_cell("c07-050", (
    "## 7.4 CAPM 可视化：SML 与散点图\n"
    "\n"
    "**证券市场线（SML）**：横轴为 $\\beta$，纵轴为年化超额收益。\n"
    "CAPM 预测所有资产都应落在 SML 上；实际位置偏离代表正/负 alpha。"
)))

# ── c07-051: SML code ──────────────────────────────────────────
cells.append(code_cell("c07-051", (
    "# Cell 5：SML 可视化 + TECH 散点回归图\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
    "colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']\n"
    "\n"
    "# -- 左图：SML --\n"
    "ax1 = axes[0]\n"
    "mkt_premium_ann   = mkt_excess.mean() * TRADING_DAYS\n"
    "betas             = {t: capm_results[t].params['MKT'] for t in TICKERS}\n"
    "actual_excess_ann = excess_rets.mean() * TRADING_DAYS\n"
    "\n"
    "beta_range = np.linspace(0, max(betas.values()) * 1.25, 100)\n"
    "ax1.plot(beta_range, beta_range * mkt_premium_ann, 'k--', lw=2,\n"
    "         label=f'SML（市场溢价={mkt_premium_ann:.1%}/年）')\n"
    "\n"
    "for i, ticker in enumerate(TICKERS):\n"
    "    beta_i    = betas[ticker]\n"
    "    capm_pred = beta_i * mkt_premium_ann\n"
    "    actual    = actual_excess_ann[ticker]\n"
    "    ax1.annotate('', xy=(beta_i, actual), xytext=(beta_i, capm_pred),\n"
    "                 arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))\n"
    "    ax1.scatter(beta_i, actual, color=colors[i], s=100, zorder=5, label=ticker)\n"
    "    alpha_ann = capm_results[ticker].params['alpha_const'] * TRADING_DAYS\n"
    "    ax1.annotate(f'{ticker}\\na={alpha_ann:.1%}',\n"
    "                 xy=(beta_i, actual), xytext=(8, 5),\n"
    "                 textcoords='offset points', color=colors[i], fontsize=9)\n"
    "\n"
    "ax1.axhline(0, color='gray', lw=0.8, linestyle=':')\n"
    "ax1.set_xlabel('Beta (β)')\n"
    "ax1.set_ylabel('年化超额收益')\n"
    "ax1.set_title('证券市场线（SML）与实际收益对比')\n"
    "ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "ax1.legend(fontsize=9)\n"
    "\n"
    "# -- 右图：TECH 散点 + 回归线 --\n"
    "ax2 = axes[1]\n"
    "res_t  = capm_results['TECH']\n"
    "x_vals = mkt_excess.values\n"
    "y_vals = excess_rets['TECH'].values\n"
    "ax2.scatter(x_vals, y_vals, alpha=0.3, s=15, color='#2ca02c', label='日度数据点')\n"
    "\n"
    "x_line = np.linspace(x_vals.min(), x_vals.max(), 100)\n"
    "y_line = res_t.params['alpha_const'] + res_t.params['MKT'] * x_line\n"
    "b_str  = f\"{res_t.params['MKT']:.2f}\"\n"
    "a_str  = f\"{res_t.params['alpha_const']*TRADING_DAYS:.1%}\"\n"
    "ax2.plot(x_line, y_line, 'r-', lw=2,\n"
    "         label=f'CAPM fit: a={a_str}/年, b={b_str}')\n"
    "\n"
    "ax2.axhline(0, color='gray', lw=0.8, linestyle=':')\n"
    "ax2.axvline(0, color='gray', lw=0.8, linestyle=':')\n"
    "ax2.set_xlabel('市场超额收益（日）')\n"
    "ax2.set_ylabel('TECH 超额收益（日）')\n"
    "ax2.set_title(f'TECH CAPM 散点图（R2={res_t.rsquared:.3f}）')\n"
    "ax2.legend(fontsize=9)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print('Beta 汇总：')\n"
    "for t in TICKERS:\n"
    "    print(f'  {t}: beta = {betas[t]:.4f}')"
)))

# ── c07-060: Beta bar + R2 ─────────────────────────────────────
cells.append(code_cell("c07-060", (
    "# Cell 6：Beta 条形图 + R2 方差分解\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
    "colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']\n"
    "\n"
    "# 左图：Beta 对比\n"
    "ax1 = axes[0]\n"
    "beta_vals = [capm_results[t].params['MKT'] for t in TICKERS]\n"
    "bars = ax1.bar(TICKERS, beta_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)\n"
    "ax1.axhline(1.0, color='black', lw=1.5, linestyle='--', label='beta=1（市场基准）')\n"
    "ax1.set_title('各股票 Beta（市场敏感度）对比')\n"
    "ax1.set_ylabel('Beta')\n"
    "ax1.set_ylim(0, max(beta_vals) * 1.25)\n"
    "ax1.legend()\n"
    "for bar, val in zip(bars, beta_vals):\n"
    "    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n"
    "             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
    "\n"
    "# 右图：R2 分解\n"
    "ax2 = axes[1]\n"
    "r2_vals   = [capm_results[t].rsquared for t in TICKERS]\n"
    "idio_vals = [1 - r for r in r2_vals]\n"
    "x = np.arange(len(TICKERS))\n"
    "width = 0.5\n"
    "bar1 = ax2.bar(x, r2_vals,   width, label='系统性风险 (R2)',  color='steelblue', alpha=0.85)\n"
    "       \n"
    "ax2.bar(x, idio_vals, width, bottom=r2_vals,\n"
    "        label='特质风险 (1-R2)', color='salmon', alpha=0.85)\n"
    "ax2.set_xticks(x)\n"
    "ax2.set_xticklabels(TICKERS)\n"
    "ax2.set_title('方差分解：系统性风险 vs 特质风险')\n"
    "ax2.set_ylabel('占总方差比例')\n"
    "ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "ax2.set_ylim(0, 1.0)\n"
    "ax2.legend()\n"
    "for bar, val in zip(bar1, r2_vals):\n"
    "    ax2.text(bar.get_x() + bar.get_width()/2, val/2,\n"
    "             f'{val:.1%}', ha='center', va='center',\n"
    "             fontsize=10, color='white', fontweight='bold')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print('R2 越高，CAPM 对该股票解释力越强；1-R2 越高，特质风险越大')"
)))

# ── c07-070: Multi-factor md ─────────────────────────────────────
cells.append(md_cell("c07-070", (
    "## 7.5 多因子回归：Fama-French 三因子（教学示意数据）\n"
    "\n"
    "> **重要提示**：以下使用 `np.random.default_rng(42)` 生成的**模拟因子数据**，\n"
    "> 仅用于演示多因子回归的操作步骤和结果解读，**不代表真实 A 股因子结论**。\n"
    "> 如需真实因子数据，请从 CSMAR、Wind 或 Tushare Pro 获取。\n"
    "\n"
    "FF 三因子回归方程：\n"
    "\n"
    "$$r_{i,t} - r_{f,t} = \\alpha_i + \\beta^{MKT}_i MKT_t\n"
    "+ \\beta^{SMB}_i SMB_t + \\beta^{HML}_i HML_t + \\varepsilon_{i,t}$$"
)))

# ── c07-071: Generate simulated factors ──────────────────────────
cells.append(code_cell("c07-071", (
    "# Cell 7：生成示意因子数据（固定随机种子，可复现）\n"
    "# 以下为教学示意数据，非真实 A 股因子\n"
    "\n"
    "rng = np.random.default_rng(42)  # 固定种子\n"
    "T   = len(rets)\n"
    "\n"
    "mkt_sim = rng.normal(0.0002, 0.012, T)\n"
    "smb_sim = rng.normal(0.0001, 0.008, T)\n"
    "hml_sim = rng.normal(0.0001, 0.007, T)\n"
    "umd_sim = rng.normal(0.0002, 0.009, T)\n"
    "\n"
    "factors_sim = pd.DataFrame(\n"
    "    {'MKT': mkt_sim, 'SMB': smb_sim, 'HML': hml_sim, 'UMD': umd_sim},\n"
    "    index=rets.index,\n"
    ")\n"
    "\n"
    "print('以下均为教学示意数据（随机生成），非真实 A 股因子')\n"
    "print()\n"
    "ann_f  = factors_sim.mean() * TRADING_DAYS\n"
    "ann_v  = factors_sim.std()  * np.sqrt(TRADING_DAYS)\n"
    "sharpe = ann_f / ann_v\n"
    "print(pd.DataFrame({'年化均值': ann_f, '年化波动': ann_v, '夏普': sharpe}).round(4))"
)))

# ── c07-080: FF3 regression ──────────────────────────────────────
cells.append(code_cell("c07-080", (
    "# Cell 8：FF 三因子回归（示意数据，等权组合为被解释变量）\n"
    "# 以下为教学示意数据\n"
    "\n"
    "y_port = excess_rets.mean(axis=1)   # 等权组合超额收益\n"
    "\n"
    "X_ff3 = sm.add_constant(factors_sim[['MKT', 'SMB', 'HML']])\n"
    "X_ff3.columns = ['alpha_const', 'MKT', 'SMB', 'HML']\n"
    "\n"
    "result_ff3 = sm.OLS(y_port, X_ff3).fit()\n"
    "print('=== FF 三因子回归结果（示意数据）===')\n"
    "print(result_ff3.summary())\n"
    "print()\n"
    "print('提醒：因子数据为随机生成，系数仅演示解读方法，无实际意义')"
)))

# ── c07-090: FF3 all stocks summary ──────────────────────────────
cells.append(code_cell("c07-090", (
    "# Cell 9：各股票 FF 三因子回归汇总（示意数据）\n"
    "\n"
    "ff3_rows = []\n"
    "for ticker in TICKERS:\n"
    "    res3 = sm.OLS(excess_rets[ticker], X_ff3).fit()\n"
    "    ff3_rows.append({\n"
    "        '股票': ticker,\n"
    "        'a_年化': round(res3.params['alpha_const'] * TRADING_DAYS, 4),\n"
    "        'a_t': round(res3.tvalues['alpha_const'], 3),\n"
    "        'b_MKT': round(res3.params['MKT'], 4),\n"
    "        'b_SMB': round(res3.params['SMB'], 4),\n"
    "        'b_HML': round(res3.params['HML'], 4),\n"
    "        'R2_FF3': round(res3.rsquared, 4),\n"
    "        'R2_CAPM': round(capm_results[ticker].rsquared, 4),\n"
    "        'R2提升': round(res3.rsquared - capm_results[ticker].rsquared, 4),\n"
    "    })\n"
    "\n"
    "print('=== 四只股票 FF 三因子回归汇总（示意数据）===')\n"
    "print(pd.DataFrame(ff3_rows).set_index('股票').to_string())\n"
    "print('注：实际 A 股数据中 FF3 通常比 CAPM 的 R2 高出 0.10~0.30')"
)))

# ── c07-100: Carhart md ──────────────────────────────────────────
cells.append(md_cell("c07-100", (
    "## 7.6 Carhart 四因子：加入动量（示意数据）\n"
    "\n"
    "> **以下仍为示意数据演示**，UMD 因子为随机生成，仅演示四因子回归的操作流程。\n"
    "\n"
    "$$r_{i,t} - r_{f,t} = \\alpha_i + \\beta^{MKT} MKT_t\n"
    "+ \\beta^{SMB} SMB_t + \\beta^{HML} HML_t + \\beta^{UMD} UMD_t + \\varepsilon_{i,t}$$"
)))

# ── c07-101: Carhart 4-factor ────────────────────────────────────
cells.append(code_cell("c07-101", (
    "# Cell 10：Carhart 四因子回归（示意数据）\n"
    "\n"
    "X_c4 = sm.add_constant(factors_sim[['MKT', 'SMB', 'HML', 'UMD']])\n"
    "X_c4.columns = ['alpha_const', 'MKT', 'SMB', 'HML', 'UMD']\n"
    "\n"
    "c4_rows = []\n"
    "for ticker in TICKERS:\n"
    "    res4 = sm.OLS(excess_rets[ticker], X_c4).fit()\n"
    "    c4_rows.append({\n"
    "        '股票': ticker,\n"
    "        'a_年化': round(res4.params['alpha_const'] * TRADING_DAYS, 4),\n"
    "        'a_t': round(res4.tvalues['alpha_const'], 3),\n"
    "        'a_p': round(res4.pvalues['alpha_const'], 4),\n"
    "        'b_MKT': round(res4.params['MKT'], 4),\n"
    "        'b_SMB': round(res4.params['SMB'], 4),\n"
    "        'b_HML': round(res4.params['HML'], 4),\n"
    "        'b_UMD': round(res4.params['UMD'], 4),\n"
    "        'R2_C4': round(res4.rsquared, 4),\n"
    "        'AIC': round(res4.aic, 2),\n"
    "    })\n"
    "\n"
    "print('=== 四只股票 Carhart 四因子回归汇总（示意数据）===')\n"
    "print(pd.DataFrame(c4_rows).set_index('股票').to_string())\n"
    "print()\n"
    "print('alpha 检验：a_p < 0.05 说明在 5% 水平下存在显著超额 alpha')\n"
    "print('因子数据为随机生成，以上数字仅演示操作，无实际意义')"
)))

# ── c07-110: Collinearity md ─────────────────────────────────────
cells.append(md_cell("c07-110", (
    "## 7.7 因子相关性与多重共线性诊断\n"
    "\n"
    "多因子回归的前提：各因子之间相关性不能过高，否则系数估计不稳定。\n"
    "\n"
    "**方差膨胀因子 VIF**：\n"
    "$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$\n"
    "其中 $R_j^2$ 是用其他所有因子回归第 $j$ 个因子的 $R^2$。\n"
    "VIF < 5 正常；> 10 视为严重多重共线性。"
)))

# ── c07-111: Collinearity code ───────────────────────────────────
cells.append(code_cell("c07-111", (
    "# Cell 11：因子相关性 + VIF 计算（示意数据）\n"
    "\n"
    "factor_names = ['MKT', 'SMB', 'HML', 'UMD']\n"
    "F = factors_sim[factor_names]\n"
    "\n"
    "corr_matrix = F.corr()\n"
    "print('=== 因子相关性矩阵（示意数据）===')\n"
    "print(corr_matrix.round(4).to_string())\n"
    "print()\n"
    "\n"
    "def compute_vif(X_df):\n"
    "    rows = []\n"
    "    for col in X_df.columns:\n"
    "        X_others = sm.add_constant(X_df.drop(columns=[col]))\n"
    "        r2  = sm.OLS(X_df[col], X_others).fit().rsquared\n"
    "        vif = 1 / (1 - r2) if r2 < 1 else float('inf')\n"
    "        rows.append({'因子': col, 'R2': round(r2, 4), 'VIF': round(vif, 4)})\n"
    "    return pd.DataFrame(rows).set_index('因子')\n"
    "\n"
    "print('=== VIF（示意数据）===')\n"
    "print(compute_vif(F).to_string())\n"
    "print()\n"
    "print('示意数据各因子独立生成，VIF 接近 1（低共线性）')\n"
    "print('真实 A 股中 SMB 与 HML 可能存在较高相关性，需特别检验')\n"
    "\n"
    "# 热图\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)\n"
    "plt.colorbar(im, ax=ax, label='Pearson 相关系数')\n"
    "ax.set_xticks(range(len(factor_names)))\n"
    "ax.set_yticks(range(len(factor_names)))\n"
    "ax.set_xticklabels(factor_names)\n"
    "ax.set_yticklabels(factor_names)\n"
    "ax.set_title('因子相关性热图（示意数据）')\n"
    "for i in range(len(factor_names)):\n"
    "    for j in range(len(factor_names)):\n"
    "        val = corr_matrix.iloc[i, j]\n"
    "        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=11,\n"
    "                color='white' if abs(val) > 0.5 else 'black')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)))

# ── c07-120: Attribution md ──────────────────────────────────────
cells.append(md_cell("c07-120", (
    "## 7.8 业绩归因分解（CAPM，真实内置数据）\n"
    "\n"
    "$$\\underbrace{r_i - r_f}_{\\text{总超额收益}}\n"
    "= \\underbrace{\\alpha_i}_{\\text{真实 alpha}}\n"
    "+ \\underbrace{\\beta_i \\cdot \\overline{MKT}}_{\\text{市场因子贡献}}$$\n"
    "\n"
    "多因子版本将各因子贡献逐项拆解，便于判断收益来源于选股能力还是因子暴露。"
)))

# ── c07-121: Attribution code ────────────────────────────────────
cells.append(code_cell("c07-121", (
    "# Cell 12：业绩归因（CAPM 单因子，真实内置数据）\n"
    "\n"
    "mkt_mean_ann = mkt_excess.mean() * TRADING_DAYS\n"
    "\n"
    "attr_rows = []\n"
    "for ticker in TICKERS:\n"
    "    res       = capm_results[ticker]\n"
    "    alpha_ann = res.params['alpha_const'] * TRADING_DAYS\n"
    "    beta_val  = res.params['MKT']\n"
    "    mkt_ctr   = beta_val * mkt_mean_ann\n"
    "    total     = excess_rets[ticker].mean() * TRADING_DAYS\n"
    "    attr_rows.append({\n"
    "        '股票': ticker,\n"
    "        '总超额收益年化': round(total, 4),\n"
    "        'Alpha年化': round(alpha_ann, 4),\n"
    "        '市场因子贡献': round(mkt_ctr, 4),\n"
    "        '误差': round(total - alpha_ann - mkt_ctr, 8),\n"
    "    })\n"
    "\n"
    "df_attr = pd.DataFrame(attr_rows).set_index('股票')\n"
    "print('=== CAPM 单因子业绩归因（年化，真实内置数据）===')\n"
    "print(df_attr.round(4).to_string())\n"
    "print(f'\\n样本内年化市场超额收益（代理）：{mkt_mean_ann:.2%}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "x = np.arange(len(TICKERS))\n"
    "alpha_vals = [r['Alpha年化'] for r in attr_rows]\n"
    "mkt_vals   = [r['市场因子贡献'] for r in attr_rows]\n"
    "ax.bar(x, alpha_vals, 0.5, label='Alpha（选股贡献）',\n"
    "       color='#2ca02c', alpha=0.85)\n"
    "ax.bar(x, mkt_vals, 0.5, bottom=alpha_vals,\n"
    "       label='市场因子贡献（Beta 暴露）', color='steelblue', alpha=0.85)\n"
    "for i, total in enumerate(df_attr['总超额收益年化']):\n"
    "    ax.text(i, total + 0.002, f'{total:.1%}', ha='center',\n"
    "            fontsize=10, fontweight='bold')\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels(TICKERS)\n"
    "ax.set_ylabel('年化超额收益')\n"
    "ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "ax.set_title('CAPM 业绩归因：Alpha vs 市场因子贡献')\n"
    "ax.axhline(0, color='black', lw=0.8)\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)))

# ── c07-130: Factor time series ──────────────────────────────────
cells.append(code_cell("c07-130", (
    "# Cell 13：因子累计收益时序图\n"
    "\n"
    "plot_data = factors_sim.copy()\n"
    "plot_data['MKT'] = mkt_excess.values   # 替换为真实市场超额收益\n"
    "\n"
    "factor_labels = [\n"
    "    ('MKT', '市场超额收益（真实内置数据）'),\n"
    "    ('SMB', 'SMB 规模因子（教学示意数据）'),\n"
    "    ('HML', 'HML 价值因子（教学示意数据）'),\n"
    "    ('UMD', 'UMD 动量因子（教学示意数据）'),\n"
    "]\n"
    "factor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']\n"
    "\n"
    "fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)\n"
    "for ax, (fname, flabel), color in zip(axes, factor_labels, factor_colors):\n"
    "    cumret  = (1 + plot_data[fname]).cumprod() - 1\n"
    "    ann_ret = plot_data[fname].mean() * TRADING_DAYS\n"
    "    ax.plot(plot_data.index, cumret, color=color, lw=1.2)\n"
    "    ax.fill_between(plot_data.index, cumret, 0, color=color, alpha=0.15)\n"
    "    ax.set_title(f'{flabel} | 年化 = {ann_ret:.2%}', fontsize=10)\n"
    "    ax.set_ylabel('累计收益')\n"
    "    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n"
    "    ax.axhline(0, color='gray', lw=0.8, linestyle=':')\n"
    "axes[-1].set_xlabel('日期')\n"
    "fig.suptitle('因子累计收益时序图', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)))

# ── c07-140: Summary md ──────────────────────────────────────────
cells.append(md_cell("c07-140", (
    "## 7.9 本章小结\n"
    "\n"
    "| 模型 | 因子数 | 用途 | 关键限制 |\n"
    "|---|---|---|---|\n"
    "| CAPM | 1（MKT） | beta 估计、基准收益 | 低 R2，忽略规模/价值/动量 |\n"
    "| FF 三因子 | 3（+SMB, HML） | 风险归因、alpha 检验 | 忽略动量 |\n"
    "| Carhart 四因子 | 4（+UMD） | 基金绩效评估 | 动量在 A 股有争议 |\n"
    "| FF 五因子 | 5（+RMW, CMA） | 全面风险分解 | HML 冗余，未纳入动量 |\n"
    "\n"
    "**A 股特殊注意**：壳价值、涨跌停、停牌、行业集中均会干扰因子构建，\n"
    "推荐因子数据库：CSMAR、Wind、Tushare Pro。"
)))

# ── c07-150: Exercise intro md ───────────────────────────────────
cells.append(md_cell("c07-150", (
    "## 习题参考解答\n"
    "\n"
    "以下代码对应正文第 7.11 节习题，可直接运行。"
)))

# ── c07-161: Ex 1 ────────────────────────────────────────────────
cells.append(code_cell("c07-161", (
    "# === 习题 7.1：CAPM 代数计算 ===\n"
    "print('习题 7.1：CAPM 预期收益与 alpha 计算')\n"
    "rf_q1 = 0.02; rm_q1 = 0.08; beta_q1 = 1.4; actual_q1 = 0.12\n"
    "er_q1    = rf_q1 + beta_q1 * (rm_q1 - rf_q1)\n"
    "alpha_q1 = actual_q1 - er_q1\n"
    "print(f'CAPM 预期收益 = {rf_q1:.1%} + {beta_q1} x {rm_q1-rf_q1:.1%} = {er_q1:.2%}')\n"
    "print(f'实际收益 = {actual_q1:.1%}')\n"
    "print(f'Alpha = {actual_q1:.1%} - {er_q1:.2%} = {alpha_q1:.2%}')\n"
    "print('alpha > 0：在风险调整后仍有超额表现')"
)))

# ── c07-162: Ex 2 ────────────────────────────────────────────────
cells.append(code_cell("c07-162", (
    "# === 习题 7.2：四只股票 Beta 与预期收益 ===\n"
    "print('习题 7.2：四只股票 Beta 与预期收益估计')\n"
    "mkt_prem_ann = mkt_excess.mean() * TRADING_DAYS\n"
    "\n"
    "ex72 = []\n"
    "for ticker in TICKERS:\n"
    "    res = capm_results[ticker]\n"
    "    bv  = res.params['MKT']\n"
    "    ex72.append({\n"
    "        '股票': ticker,\n"
    "        'Beta': round(bv, 4),\n"
    "        'Alpha年化': round(res.params['alpha_const'] * TRADING_DAYS, 4),\n"
    "        'CAPM预期年化': round(RF_ANNUAL + bv * mkt_prem_ann, 4),\n"
    "        '实际超额年化': round(excess_rets[ticker].mean() * TRADING_DAYS, 4),\n"
    "        'R2': round(res.rsquared, 4),\n"
    "    })\n"
    "df72 = pd.DataFrame(ex72).set_index('股票')\n"
    "print(df72.round(4).to_string())\n"
    "print(f'市场年化超额收益（代理）：{mkt_prem_ann:.2%}')\n"
    "print(f'Beta 最高：{df72[\"Beta\"].idxmax()}')\n"
    "print(f'CAPM 预期收益最高：{df72[\"CAPM预期年化\"].idxmax()}')"
)))

# ── c07-163: Ex 3 ────────────────────────────────────────────────
cells.append(code_cell("c07-163", (
    "# === 习题 7.3：R2 与系统性/特质风险 ===\n"
    "print('习题 7.3：R2 与系统性/特质风险分析')\n"
    "from fds import annualized_volatility\n"
    "vol_vals = annualized_volatility(rets)\n"
    "\n"
    "ex73 = []\n"
    "for ticker in TICKERS:\n"
    "    r2 = capm_results[ticker].rsquared\n"
    "    ex73.append({\n"
    "        '股票': ticker,\n"
    "        '年化总波动': round(vol_vals[ticker], 4),\n"
    "        'R2系统性占比': round(r2, 4),\n"
    "        '1-R2特质占比': round(1 - r2, 4),\n"
    "        '系统波动': round(vol_vals[ticker] * np.sqrt(r2), 4),\n"
    "        '特质波动': round(vol_vals[ticker] * np.sqrt(1 - r2), 4),\n"
    "    })\n"
    "df73 = pd.DataFrame(ex73).set_index('股票')\n"
    "print(df73.to_string())\n"
    "mv = vol_vals.idxmax()\n"
    "mr = df73['R2系统性占比'].idxmax()\n"
    "print(f'\\n波动率最高：{mv}，R2 最高：{mr}')\n"
    "if mv != mr:\n"
    "    print('结论：高波动率并不等于高 R2。高波动中若特质风险为主，则 R2 反而偏低。')"
)))

# ── c07-164: Ex 4 ────────────────────────────────────────────────
cells.append(code_cell("c07-164", (
    "# === 习题 7.4：多重共线性影响演示 ===\n"
    "print('习题 7.4：多重共线性影响（构造高相关因子对比）')\n"
    "\n"
    "rng2    = np.random.default_rng(99)\n"
    "T2      = len(rets)\n"
    "f_base  = rng2.normal(0, 0.01, T2)\n"
    "f_high1 = f_base + rng2.normal(0, 0.003, T2)\n"
    "f_high2 = f_base + rng2.normal(0, 0.003, T2)\n"
    "f_indep = rng2.normal(0, 0.01, T2)\n"
    "y_q4    = excess_rets['TECH'].values\n"
    "\n"
    "X_low  = sm.add_constant(np.column_stack([f_high1, f_indep]))\n"
    "X_high = sm.add_constant(np.column_stack([f_high1, f_high2]))\n"
    "res_low  = sm.OLS(y_q4, X_low).fit()\n"
    "res_high = sm.OLS(y_q4, X_high).fit()\n"
    "\n"
    "print(f'高相关对（f1 vs f2）：Corr = {np.corrcoef(f_high1, f_high2)[0,1]:.4f}')\n"
    "print(f'低相关对（f1 vs indep）：Corr = {np.corrcoef(f_high1, f_indep)[0,1]:.4f}')\n"
    "print()\n"
    "print('低共线性情况：')\n"
    "print(f'  f1 t-stat = {res_low.tvalues[1]:.3f},  独立因子 t-stat = {res_low.tvalues[2]:.3f}')\n"
    "print('高共线性情况：')\n"
    "print(f'  f1 t-stat = {res_high.tvalues[1]:.3f},  f2 t-stat = {res_high.tvalues[2]:.3f}')\n"
    "print()\n"
    "print('结论：高共线性时 t-stat 均下降，各因子独立贡献难以区分。')\n"
    "print('解决方案：(1) 正交化因子；(2) 删除冗余因子；(3) 岭回归')"
)))

# ── assemble and write ────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written {len(cells)} cells to {OUT}")

# Quick validation
with open(OUT, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
code_cells = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
md_cells   = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
print(f"JSON valid: {len(nb2['cells'])} cells ({code_cells} code, {md_cells} markdown)")
