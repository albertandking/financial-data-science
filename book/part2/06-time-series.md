# 第6章 金融时间序列分析

!!! info "配套代码"
    `notebooks/ch06_time_series.ipynb`（使用 statsmodels）

## 6.1 学习目标

- 平稳性检验与差分
- AR / MA / ARIMA 建模与定阶
- 波动率建模：ARCH / GARCH

## 6.2 内容大纲

1. 平稳性：ADF 检验、单位根
2. 自相关与偏自相关（ACF / PACF）
3. ARIMA 建模流程与信息准则（AIC / BIC）
4. 波动率聚集与 GARCH(1,1)
5. 样本外预测与评估

## 6.3 练习

1. 对某股票对数收益做 ADF 检验，判断是否平稳。
2. 拟合 GARCH(1,1) 并预测未来 5 日波动率。
