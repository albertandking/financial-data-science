# T

!!! example “例 1.1：信息比的量级推算与年化换算”
    **背景**：假设某量化策略在A股日频数据上，预测的超额收益（相对沪深300）均值为
    $\mu_d = 0.025\%$（即年化约 $0.025\% \times 252 \approx 6.3\%$），
    日度跟踪误差（策略收益率标准差）为 $\sigma_d = 1.2\%$。
    
    **第一步：计算日度信息比**
    
    $\text{IR}_{daily} = \frac{\mu_d}{\sigma_d} = \frac{0.025\%}{1.2\%} \approx 0.021$
    
    **第二步：年化信息比**
    
    由于一年约有252个交易日，假设每日收益率独立，年化标准差为 $\sigma_{annual} = \sigma_d \times \sqrt{252}$，年化均值为 $\mu_{annual} = \mu_d \times 252$，故：
    
    $\text{IR}_{annual} = \frac{\mu_{annual}}{\sigma_{annual}} = \frac{\mu_d \times 252}{\sigma_d \times \sqrt{252}} = \text{IR}_{daily} \times \sqrt{252} \approx 0.021 \times 15.87 \approx 0.33$
    
    **第三步：与主动管理基准对比**
    
    | 年化IR水平 | 业界评价 |
    |-----------|---------|
    | < 0.0 | 策略亏损 |
    | 0.0 – 0.5 | 较弱，难以覆盖管理成本 |
    | 0.5 – 1.0 | 一般（大多数公募基金在此区间） |
    | 1.0 – 2.0 | 优秀（头部量化机构） |
    | > 2.0 | 卓越（往往容量有限） |
    
    本例中 $\text{IR} \approx 0.33$，已属于有盈利可能的区间，但扣除管理费（1%~2%/年）后优势大为削减，这正是金融数据科学在实战中最大的挑战。
    
    **延伸：主动管理基本定律（Fundamental Law of Active Management）**
    
    Grinold (1989) 证明，信息比可以进一步分解：
    
    $\text{IR} \approx \text{IC} \times \sqrt{BR}$
    
    其中 $\text{IC}$（Information Coefficient）为预测值与实际收益率的截面相关系数（衡量「预测准不准」），$BR$（Breadth）为每年独立预测的次数（衡量「预测多少次」）。
    例如，若策略每月对300只股票截面排名选股，则 $BR \approx 300 \times 12 = 3600$；若 $\text{IC} = 0.04$，则 $\text{IR} \approx 0.04 \times \sqrt{3600} = 2.4$。
    
    这一公式的直觉是：**预测的准确度和覆盖广度同样重要**，仅靠少数几次「神准」的预测，远不如持续、广泛的中等准确预测。

