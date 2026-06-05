# 第11章 树模型与集成学习

!!! info "配套代码"
    `notebooks/ch11_ensemble_trees.ipynb`（使用 scikit-learn / xgboost，需 `--extra advanced`）

## 11.1 学习目标

- 决策树、随机森林、梯度提升
- XGBoost / LightGBM 在金融预测中的应用
- 特征重要性与 SHAP 解释

## 11.2 内容大纲

1. 决策树与过拟合
2. 随机森林（Bagging）
3. 梯度提升（Boosting）与 XGBoost
4. 超参调优与时序交叉验证
5. 模型解释：特征重要性、SHAP

## 11.3 练习

1. 用 XGBoost 预测涨跌方向，与逻辑回归比较 AUC。
2. 用 SHAP 解释哪些特征最重要，是否符合金融直觉。
