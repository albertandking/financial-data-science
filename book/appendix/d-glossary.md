# 附录D 术语与符号表

金融数据科学满是英文术语与数学符号。本附录供随时查阅：D.1 数学符号约定、D.2 缩写表、D.3 中英术语对照。

## D.1 数学符号约定

全书尽量遵循以下约定（个别章节按需扩展）：

| 符号 | 含义 |
|---|---|
| $P_t$ | 第 $t$ 期价格 |
| $r_t$ | 第 $t$ 期收益率（简单收益）；$\ell_t$ 表示对数收益 |
| $\mu,\ \sigma$ | 期望收益、波动率（标准差） |
| $\sigma^2$ | 方差 |
| $\rho_{XY}$ | $X$ 与 $Y$ 的相关系数 |
| $\Sigma$ | 协方差矩阵（对称半正定） |
| $\mathbf{w}$ | 组合权重向量；$\mathbf{1}$ 为全 1 向量 |
| $\beta$ | 贝塔（系统性风险暴露） |
| $\alpha$ | 阿尔法（超额收益）；在检验中也表示显著性水平 |
| $\mathbb{E}[\cdot],\ \mathrm{Var}(\cdot),\ \mathrm{Cov}(\cdot)$ | 期望、方差、协方差 |
| $\hat{\theta}$ | 参数 $\theta$ 的估计量 |
| $\mathbf{x},\ y$ | 特征向量、标签（监督学习） |
| $X,\ \boldsymbol{\beta},\ \boldsymbol{\varepsilon}$ | 设计矩阵、回归系数、误差项 |
| $\nabla f,\ H$ | 梯度、海森矩阵 |
| $\eta$ | 学习率 |
| $\lambda$ | 正则化强度；或拉格朗日乘子 |
| $L$ | 损失函数；或似然函数 |
| $\xrightarrow{p},\ \xrightarrow{d}$ | 依概率收敛、依分布收敛 |
| $A^\top,\ A^{-1}$ | 矩阵转置、逆 |

## D.2 缩写表

| 缩写 | 全称 | 中文 |
|---|---|---|
| NAV | Net Asset Value | 净值 |
| MDD | Maximum Drawdown | 最大回撤 |
| VaR | Value at Risk | 风险价值 |
| ES / CVaR | Expected Shortfall / Conditional VaR | 期望损失 |
| CAPM | Capital Asset Pricing Model | 资本资产定价模型 |
| SML / CML | Security / Capital Market Line | 证券市场线 / 资本市场线 |
| GMV | Global Minimum Variance | 全局最小方差（组合） |
| MPT | Modern Portfolio Theory | 现代组合理论 |
| FF3 / FF5 | Fama-French 3 / 5 factor | 三 / 五因子模型 |
| SMB / HML | Small Minus Big / High Minus Low | 规模 / 价值因子 |
| RMW / CMA / UMD | Robust-Weak / Conservative-Aggressive / Up-Down | 盈利 / 投资 / 动量因子 |
| AR / MA / ARIMA | Auto-Regressive / Moving Average / Integrated ARMA | 自回归 / 移动平均 / 差分自回归移动平均 |
| ARCH / GARCH | (Generalized) Auto-Regressive Conditional Heteroskedasticity | （广义）自回归条件异方差 |
| ADF / KPSS | Augmented Dickey-Fuller / Kwiatkowski–Phillips–Schmidt–Shin | 平稳性检验 |
| ACF / PACF | (Partial) Auto-Correlation Function | （偏）自相关函数 |
| OLS / GLS | Ordinary / Generalized Least Squares | 普通 / 广义最小二乘 |
| FE / RE | Fixed / Random Effects | 固定 / 随机效应 |
| IV | Instrumental Variable；Information Value | 工具变量；信息值（评分卡，注意同形） |
| WOE | Weight of Evidence | 证据权重 |
| PD / LGD / EAD / EL | Probability of Default / Loss Given Default / Exposure at Default / Expected Loss | 违约概率 / 违约损失率 / 违约暴露 / 预期损失 |
| KS | Kolmogorov-Smirnov | KS 统计量 |
| ROC / AUC | Receiver Operating Characteristic / Area Under Curve | ROC 曲线 / 曲线下面积 |
| IC / ICIR | Information Coefficient / IC Information Ratio | 信息系数 / IC 信息比 |
| PCA | Principal Component Analysis | 主成分分析 |
| MLE | Maximum Likelihood Estimation | 极大似然估计 |
| CLT / LLN | Central Limit Theorem / Law of Large Numbers | 中心极限定理 / 大数定律 |
| GBDT | Gradient Boosting Decision Tree | 梯度提升决策树 |
| SHAP | SHapley Additive exPlanations | 沙普利可加性解释 |
| MLP / RNN / LSTM / GRU | Multi-Layer Perceptron / Recurrent NN / Long Short-Term Memory / Gated Recurrent Unit | 多层感知机 / 循环网络 / 长短期记忆 / 门控循环单元 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| NLP | Natural Language Processing | 自然语言处理 |
| TF-IDF | Term Frequency–Inverse Document Frequency | 词频–逆文档频率 |
| LLM | Large Language Model | 大语言模型 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| MCP | Model Context Protocol | 模型上下文协议 |
| SFT / RLHF | Supervised Fine-Tuning / RL from Human Feedback | 有监督微调 / 人类反馈强化学习 |
| ReAct | Reasoning + Acting | 推理+行动（智能体范式） |
| HITL | Human-In-The-Loop | 人在回路 |
| T+1 | — | 当日买入次日才可卖出（A股交收制度） |
| ETF | Exchange-Traded Fund | 交易型开放式指数基金 |

## D.3 中英术语对照

### 方法论与陷阱

| 中文 | 英文 | 简释（详见章节） |
|---|---|---|
| 前视偏差 | look-ahead bias | 用到了预测时刻尚不可知的未来信息（第1、9、17章） |
| 幸存者偏差 | survivorship bias | 只看「活到现在」的样本，遗漏退市/失败者（第1章） |
| 数据窥探 | data snooping | 反复试参数直到「跑出」好结果，多重检验失真（第1、17章） |
| 信噪比 | signal-to-noise ratio | 金融数据信号弱、噪声大（第1章） |
| 平稳性 | stationarity | 统计特性不随时间变（第6章） |
| 厚尾 | fat / heavy tails | 极端事件比正态更频繁（第4、5章） |
| 波动率聚集 | volatility clustering | 大波动跟着大波动（第4、6章） |

### 收益与风险

| 中文 | 英文 | 简释 |
|---|---|---|
| 对数收益 | log return | $\ln(P_t/P_{t-1})$，时间可加（第5章） |
| 年化 | annualization | 日度统计量换算到年（第5章） |
| 夏普 / 索提诺 / 卡尔玛比率 | Sharpe / Sortino / Calmar ratio | 风险调整收益（第5章） |
| 最大回撤 | maximum drawdown | 净值从高点回落的最大幅度（第5章） |
| 风险价值 / 期望损失 | VaR / Expected Shortfall | 尾部风险度量（第5章） |

### 计量与因子

| 中文 | 英文 | 简释 |
|---|---|---|
| 贝塔 / 阿尔法 | beta / alpha | 系统性风险暴露 / 超额收益（第7章） |
| 有效前沿 | efficient frontier | 给定风险下收益最高的组合集合（第16章） |
| 切点组合 | tangency portfolio | 最大夏普组合（第16章） |
| 协方差收缩 | covariance shrinkage | Ledoit-Wolf 等稳健估计（第16章） |
| 风险平价 | risk parity | 按风险贡献而非资金等分（第16章） |
| 固定 / 随机效应 | fixed / random effects | 面板回归控制个体异质性（第8章） |
| 聚类稳健标准误 | clustered standard error | 允许同个体误差相关（第8章） |

### 机器学习

| 中文 | 英文 | 简释 |
|---|---|---|
| 时序交叉验证 | time-series cross-validation | 训练集始终在验证集之前（第9章） |
| 正则化 | regularization | 岭/Lasso/弹性网，抑制过拟合（第9章） |
| 偏差-方差权衡 | bias-variance tradeoff | 欠拟合与过拟合的平衡（第9章） |
| 特征工程 | feature engineering | 构造对模型有用的输入（第10章） |
| 集成学习 / 梯度提升 | ensemble / gradient boosting | 随机森林、XGBoost（第11章） |
| 信息系数 | information coefficient (IC) | 预测值与未来收益的相关性（第9、10章） |
| 评分卡 | scorecard | 信用风险的可解释模型（第18章） |
| 类别不平衡 | class imbalance | 正负样本悬殊（第18章） |

### 大模型与智能体

| 中文 | 英文 | 简释 |
|---|---|---|
| 提示工程 | prompt engineering | 设计输入以引导模型（第19章） |
| 检索增强生成 | retrieval-augmented generation (RAG) | 先检索再生成，缓解幻觉（第19章） |
| 微调 | fine-tuning | SFT/LoRA 适配下游任务（第19章） |
| 推理模型 | reasoning model | 回答前先「思考」（第19章） |
| 工具调用 | function / tool calling | 模型结构化地调用外部函数（第19、20章） |
| 智能体 | AI agent | 能自主规划-行动-观察完成任务（第20章） |
| 人在回路 | human-in-the-loop (HITL) | 高危操作须人工批准（第20章） |
| 幻觉 | hallucination | 模型编造看似合理实则错误的内容（第19章） |

### 中国市场制度

| 中文 | 英文 | 简释 |
|---|---|---|
| 复权 | adjusted price | 消除分红送股造成的价格跳变（第3章） |
| 停牌 | trading suspension | 暂停交易（第3章） |
| 涨跌停 | price limit | A股日涨跌幅限制（±10%/±20%等，第1章） |
| 换手率 | turnover ratio | 成交量 / 流通股本（第10章） |
