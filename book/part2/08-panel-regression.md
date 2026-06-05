# 第8章 面板数据与回归

!!! info "配套代码"
    `notebooks/ch08_panel_regression.ipynb`（使用 linearmodels / statsmodels）

## 8.1 学习目标

- 理解面板数据结构（个体 × 时间）
- 固定效应与随机效应模型
- 聚类稳健标准误

## 8.2 内容大纲

1. 面板数据 vs 截面 vs 时序
2. 混合 OLS 的问题
3. 固定效应（个体/时间）
4. 随机效应与 Hausman 检验
5. 标准误：聚类、稳健

## 8.3 练习

1. 构造一个公司-年度面板，估计杠杆率对盈利的影响（固定效应）。
2. 用 Hausman 检验在 FE 与 RE 之间选择。
