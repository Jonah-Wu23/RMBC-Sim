# Paper Outline (ICTLE-style, 6–8 pages)

**Title（建议）**  
Multi-level Robust Calibration for Bus Simulation via Bayesian Surrogates and Data Assimilation

**Keywords**  
Bus simulation; parameter inversion; multi-level calibration; Bayesian optimization; surrogate model; data assimilation; robustness

---

## 0. Abstract（150–250 words）

- 背景：公交仿真虚实鸿沟、参数不确定性、传统校准不稳/成本高
- 方法：L1 微观行为校准 + L2 宏观同化/约束；Kriging+RBF 双代理；BO（EI）选点；鲁棒性检验（K-S/分布指标）
- 数据：GPS/ETA（或站点到离站）、路段速度/检测器等（写你真实可用的）
- 结果：误差下降（平均+分位），跨时段/跨日稳定；仿真次数/成本
- 贡献点 3 条（见第 1 节）

---

## 1. Introduction

### 1.1 Problem Statement

- 公交仿真参数：停站行为、发车/速度、拥堵传播、控制策略等
- 难点：非凸、噪声、参数耦合、数据稀疏与时变

### 1.2 Research Gap

- 单层校准：只对齐一种观测（站点/路段），泛化差
- 纯黑箱优化：成本高、易过拟合
- 缺少“鲁棒性”评估：只报均值 RMSE 不够

### 1.3 Contributions（建议写成 3 点）

1. 提出 **多层次校准框架**：L1（线路/站点）+ L2（走廊/路段）闭环约束
2. 提出 **双代理 + BO** 的高效反演流程（小样本、低仿真预算）
3. 引入 **分布鲁棒评估**：跨时段/跨日 + K-S/分位误差指标，验证稳定性

---

## 2. Related Work

2.1 Bus simulation calibration (SUMO/微观仿真参数)  
2.2 Surrogate-based optimization (Kriging/RBF/BO)  
2.3 Data assimilation for traffic (EnKF/状态估计，简述即可)  
2.4 Robust calibration / distributional evaluation (K-S, quantile loss, etc.)

> 写法：每小节 6–10 句，最后一句落到“本文怎么补齐缺口”。

---

## 3. Problem Formulation

### 3.1 Simulation and Parameters

- 仿真平台：SUMO（写版本/接口）
- 参数向量 **θ**：分成两层
  - L1：dwell time（上下客时间模型参数）、accel/decel、car-following、lane-change、schedule adherence 等
  - L2：路段容量/拥堵参数、速度-流关系或同化的状态变量等

### 3.2 Observations and Metrics

- L1 观测：站点到离站时刻、站间行程时间、停站时长分布
- L2 观测：路段速度/流量（检测器/第三方速度）
- 损失函数：
  - **J1(θ)**：L1 误差（RMSE + 分位/分布项）
  - **J2(θ)**：L2 误差（RMSE + 分布项）
  - 总目标：**min J = w1·J1 + w2·J2**（权重可用归一化/经验/敏感性）

### 3.3 Robustness Definition（重点）

- 训练期 vs 测试期（跨小时段/跨日）
- 评价：K-S 统计量、P90/P95 误差、Worst-case over periods

---

## 4. Methodology

### 4.1 Multi-level Calibration Framework（框架图！）

- Step A：初始化采样（LHS/随机）跑仿真，得到 (θ, J1, J2)
- Step B：训练双代理（Kriging + RBF）拟合 J(θ) 或 (J1,J2)
- Step C：BO（EI）提出新 θ，追加仿真迭代
- Step D：L2 同化/约束（EnKF 或你使用的状态更新机制）
- Step E：输出 θ* 并做鲁棒验证

### 4.2 Dual Surrogate Model

- Kriging：全局趋势 + 不确定性
- RBF：局部逼近
- 融合方式：加权集成/stacking（写清楚你实现的那种）

### 4.3 Bayesian Optimization Details

- acquisition：Expected Improvement（EI）
- 预算：N0 初始 + N_iter 迭代
- 终止：最优改进 < ε 或预算用完

### 4.4 L2 Data Assimilation / Constraint（有则写）

- EnKF：状态 x、观测 y、更新公式（简写一行）
- 解释：让宏观速度/拥堵形态与现实同步，减少“只靠调参数硬拟合”

### 4.5 Complexity & Implementation

- 仿真一次耗时、总耗时、并行策略
- 可复现：参数范围表、随机种子、硬件

---

## 5. Experimental Setup

### 5.1 Study Area & Route Selection

- 线路信息：长度、站点数、主要走廊、早晚高峰特征
- 数据覆盖：几天、每 5s/30s/1min，缺失率

### 5.2 Data Pre-processing

- 地图匹配/异常点处理
- 对齐站点事件（arrive/depart）
- 路段速度聚合（时间窗）

### 5.3 Baselines（必须有，且至少 3 个）

- B1：手工经验参数 / 默认参数
- B2：单层校准（只 J1 或只 J2）
- B3：PSO/GA（黑箱优化）
- B4（可选）：单代理 BO（只有 Kriging 或只有 RBF）

### 5.4 Evaluation Protocol

- 划分：训练（2 天）+ 测试（1 天）或按时段
- 统计：mean、median、P90、worst-case、K-S

---

## 6. Results and Discussion

### 6.1 Calibration Convergence

- 最优值随迭代下降曲线（J、J1、J2）
- 仿真次数 vs 精度提升

### 6.2 Accuracy on L1 Metrics

- 站间时间误差、停站时长分布对齐（CDF 图）

### 6.3 Accuracy on L2 Metrics

- 速度时序对比、空间热力图（time–space diagram）

### 6.4 Robustness Tests（你论文的“卖点”）

- 跨小时段/跨日：P90、worst-case、K-S
- 讨论：为什么多层校准更稳（参数可辨识性、约束更强）

### 6.5 Ablation Study（强烈建议）

- 去掉 L2 同化 / 去掉双代理 / 去掉分布项
- 看鲁棒性掉多少

### 6.6 Limitations

- 数据偏差、假设（例如乘客上下客模型简化）
- 可扩展性（多线路、多走廊）

---

## 7. Conclusion

- 结论 3 句：效果、效率、鲁棒
- 展望：多线路联合反演、在线校准、调度控制闭环（为你后续 AI-Agent 铺垫）

---

## Acknowledgements（如有）

资助/实验室/数据来源

---

## References

- 交通仿真校准（SUMO & bus dwell/carfollowing calibration）
- BO & surrogate（Kriging/RBF/ensemble）
- EnKF 交通同化（traffic state estimation）
- Robust evaluation（distributional metrics）

---

## Figure & Table Checklist（写作时直接照做）

- **Fig.1** Multi-level calibration workflow（总框架图）
- **Fig.2** Study corridor + stops + detectors（区域示意）
- **Fig.3** Convergence curve（J/J1/J2 vs iteration）
- **Fig.4** L1 CDF 对比（dwell/segment time）
- **Fig.5** L2 time–space speed diagram 对比
- **Fig.6** Robustness boxplot（across days/periods）
- **Table 1** Parameter ranges & meaning（θ 每个参数的上下限）
- **Table 2** Baseline settings & budgets（各方法仿真次数/时间）
- **Table 3** Main results（mean/median/P90/worst/K-S）

