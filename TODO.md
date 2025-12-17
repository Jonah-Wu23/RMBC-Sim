# 项目 TODO LIST（6 周）
**项目**：基于多层次校准与贝叶斯代理模型的公交仿真参数鲁棒反演研究  
**English**：Robust Inversion of Bus Simulation Parameters Based on Multi-level Calibration and Bayesian Surrogate Models  
**仿真平台**：SUMO（建议安装路径：`D:\SUMO`）  

> 使用方式：每完成一项勾选 `[x]`；在每周末补充“产出物/结论/问题清单”，保证可追溯。

---

## 全周期里程碑（Week 1–6）
- [ ] Week 1：基础环境搭建（SUMO 场景 + TraCI + 高德真值提取/清洗）
- [ ] Week 2：L1 微观层校准最小闭环（目标/参数/评估指标 + BO 框架跑通）
- [ ] Week 3：L1 微观层校准完善（双代理 Kriging+RBF、采集函数、鲁棒性指标）
- [ ] Week 4：L2 宏观层融合（EnKF/同化模块 + 路段速度分布匹配）
- [ ] Week 5：对比实验与鲁棒性评估（K-S 检验、不同拥堵情景、敏感性分析）
- [ ] Week 6：论文/报告交付（方法复现细节、实验结果、可复用代码与说明）

---

## 论文写作“稳过型”5 个硬指标（贯穿 Week 2–6）
> 目标：把论文写成“工程可复现 + 贡献分层清晰 + 对比扎实 + 鲁棒可量化 + 成本说得透”的 ICTLE 口味。

### 1）分层贡献（必须做消融）
- [ ] 明确三组：`L1-only`、`L2-only`、`L1+L2`（同一数据划分、同一预算口径）
- [ ] 统一输出：`J1/J2/J` 收敛曲线 + 测试期指标表（mean/median/P90/worst/K-S）

### 2）强基线（至少 3 个，口径一致）
- [ ] B1：人工经验/默认参数（作为工程基线）
- [ ] B2：单层校准（只对齐 `J1` 或只对齐 `J2`）
- [ ] B3：黑箱优化 GA/PSO（同预算对比）
- [ ] 可选 B4：单代理 BO（只有 Kriging 或只有 RBF）

### 3）两类观测同时对齐（L1 + L2 同时成立）
- [ ] L1：站点到/离站时刻、站间行程时间、停站时长分布（CDF/分位误差）
- [ ] L2：路段速度分布（CDF/分位误差）+ 时空速度图（time–space diagram）

### 4）鲁棒性可证明（不是口号）
- [ ] 跨时段/跨日验证：训练期 vs 测试期（至少 2 天训 + 1 天测，或按时段划分）
- [ ] 分布一致性：K-S 检验统计量（或 p-value 口径固定其一）+ 分位数误差（P50/P90/P95）
- [ ] Worst-case 报告：按时段/日期取最差表现，给出 `worst-case` 指标

### 5）计算成本说透（工程效率叙事）
- [ ] 报告总仿真次数、墙钟耗时、单次仿真耗时、并行策略与硬件信息
- [ ] 收敛曲线：迭代次数 vs 最优值下降（`J/J1/J2`），并给“提升 1% 精度的仿真预算”统计
- [ ] 基线预算对齐：所有方法使用相同仿真预算/终止条件（避免不公平）

---

## Week 1（基础环境搭建）— 详细步骤

### 1）工作区与版本化（可追溯）
- [x] 建议建立目录骨架（后续脚本都按此约定放置）
  - [x] `docs/`：调研摘记、接口说明、实验记录
  - [x] `data/raw/`：高德原始响应（按日期/线路分文件）
  - [x] `data/processed/`：清洗后的真值（CSV/Parquet）
  - [x] `sumo/net/`：路网（`*.net.xml`）
  - [x] `sumo/additional/`：站点、检测器等附加文件
  - [x] `sumo/routes/`：线路与车辆（`*.rou.xml`）
  - [x] `sumo/config/`：`*.sumocfg`
  - [x] `src/`：核心代码（采集/清洗/校准/评估）
  - [x] `scripts/`：一次性工具脚本（下载、转换、批处理）
- [x] 建立实验记录模板（建议在 `docs/experiments.md` 里按日期记录）
  - [x] 本周选择的线路/方向/时间窗
  - [x] 数据来源与采样频率、缺失情况
  - [x] SUMO 场景版本、关键参数默认值
  - [x] 运行截图/日志路径与复现实验命令

### 2）SUMO 与 Python 环境（能跑、可复现）
- [x] 安装并验证 SUMO（建议：`D:\SUMO`）
  - [x] `sumo`, `sumo-gui`, `netconvert` 可在命令行运行
  - [x] 设置环境变量 `SUMO_HOME=D:\SUMO`，并将 `%SUMO_HOME%\\bin` 加入 `PATH`
- [x] Python 虚拟环境（建议 3.10+）
  - [x] 安装依赖：数据处理（`pandas/numpy`）、可视化（`matplotlib`）、统计（`scipy`）
  - [x] 校准/代理：`scikit-learn`（RBF 等）、Kriging/GP（确定库后固定版本）
  - [x] SUMO 接口：确保 `traci` 可导入（通常来自 `SUMO_HOME/tools`）
- [x] 最小 TraCI 冒烟测试
  - [x] 运行一个最小示例场景（1 条边、1 辆车），验证能取到 `speed/pos/edge` 等基本量
  - [x] 记录本机 SUMO 版本号与 Python 版本号到 `docs/experiments.md`

### 3）目标线路 SUMO 场景建模（先能对齐“站点”）
- [x] 明确研究对象
  - [x] 选定线路/时段：68X（基线稳态）+ 960（过海复杂工况），双向，时窗 2025-12-17 17:35–18:35，首轮测试数据已收集。
  - [ ] 列出站点序列、站间距离、首末站信息（从 KMB `/route-stop` + GeoJSON 形状累积里程获取）
- [ ] 路网准备（任选一条可执行路径）
  - [ ] Path A：OSM/公开路网 → `netconvert` 生成 `*.net.xml`
  - [ ] Path B：已有路网文件直接导入（若你已有现成 `*.net.xml`）
- [ ] 公交站点与线路文件
  - [ ] 在 `sumo/additional/` 生成公交站点（`busStop`）并绑定到 lane
  - [ ] 在 `sumo/routes/` 建立线路 route 与车辆 flows（先用少量车辆验证）
- [ ] 运行基线仿真并产出“站点层”日志
  - [ ] 至少能导出：到站/离站时刻、停站时间、站间运行时间
  - [ ] 保存：`sumo/config/*.sumocfg`、仿真输出（tripinfo/stopinfo 等）

### 4）真值数据提取（站点时刻 + 路况速度）— 改为香港 KMB/LWB + data.gov.hk
- [x] 明确 API 需求与字段映射（在 `docs/gaode_schema.md` 记录）
  - [x] 线路/站点：KMB/LWB `/route`、`/stop`、`/route-stop`；政府 GeoJSON（Routes and fares）获取 polyline/沿线里程
  - [x] 到离站/ETA：KMB ETA `/eta` 或 `/route-eta` 实时预测，反推到/离站时刻
  - [x] 路况/速度：由形状里程 + ETA 差分算平均速度；叠加 data.gov.hk 路况源（IRN 分段速度 + 原始探测器 rawSpeedVol-all.xml（可选）、JTI 走廊旅时；TDAS 暂不采集）
- [x] 编写“下载器”脚本（只做抓取与落盘，不做复杂逻辑）
  - [x] 拉取静态 route/stop/route-stop、GeoJSON 形状；ETA 定时采集（60s）
  - [x] 原始响应落 `data/raw/`（按 `线路-方向-日期-时段.json` 命名），记录请求参数/返回码/限流
- [x] 编写“清洗器”脚本（输出结构化真值）
  - [x] 站点序列/里程：`scripts/clean_kmb_shapes.py` 生成 `data/processed/kmb_route_stop_dist.csv`
  - [ ] 站点层：`station_eta.csv`（统一时区/格式、去重、补缺）
  - [ ] 路段层：`link_times.csv`、`link_speeds.csv`（站间运行时间/平均速度，含异常剔除/重采样/平滑可选）
  - [ ] 可选：合并路况/事件/天气特征，输出到 `data/processed/`（CSV/Parquet）

### 5）定义 Week 1 的“对齐指标”（为 Week 2 校准做准备）
- [ ] L1（站点层）目标量定义
  - [ ] `T_arrive(i)`, `T_depart(i)`, `T_dwell(i)`, `T_link(i→i+1)`
  - [ ] 初版误差：站点过站时间 RMSE（明确对齐到哪个站点集合/时间窗）
- [ ] L2（路段层）目标量定义
  - [ ] 路段速度分布/分位数（如 P10/P50/P90）与时变均值
  - [ ] 初版距离度量：KS 统计量/EMD（二选一，先能算出来）

### 6）本周产出物（Definition of Done）
- [ ] 一个能复现的 SUMO 场景：路网 + 站点 + 线路 + 配置，可一键运行
- [ ] 一套可复现的数据管线：高德原始数据落盘 + 清洗后真值文件
- [ ] 一个基线对比脚本：同一时段下输出“站点层误差 + 路段速度分布对比”
- [ ] 一页问题清单：数据缺失/接口限制/地图匹配难点/后续风险与应对

---

## Week 2（L1 最小闭环）
- [ ] 确定 L1 参数范围：`tau`, `sigma`, `minGap`, `t_board`
- [ ] 定义采样策略与初始样本（LHS/随机）
- [ ] 跑通 BO 框架：EI 采集 + 黑箱评估（SUMO 批跑 + 指标计算）
- [ ] 形成最小可用结果：站点层 RMSE 明显下降（记录对比表）

## Week 3（L1 完善：双代理 + 鲁棒性）
- [ ] 双代理：Kriging（GP）+ RBF 的融合策略（加权/stacking，先固定一种）
- [ ] 加入不确定性与鲁棒目标（均值 + 方差/分位数损失）
- [ ] 完成 L1 结果复现实验与消融（单代理 vs 双代理）

## Week 4（L2 同化：宏观拥堵融合）
- [ ] 固定 L1 最优/后验，定义 L2 参数：`a`, `b`, `capacityFactor`
- [ ] EnKF 状态/观测定义：路段速度观测、观测算子（map matching）
- [ ] 形成 L1+L2 协同校准闭环（交替/嵌套策略明确）

## Week 5（评估：鲁棒性与泛化）
- [ ] K-S 检验（95%）：速度分布同源性结论
- [ ] 不同拥堵情景/不同日期迁移测试
- [ ] 敏感性分析：关键参数扰动对指标影响

## Week 6（写作与交付）
- [ ] 方法章节：问题定义、两层参数、代理模型、BO、EnKF
- [ ] 实验章节：数据描述、场景、对比基线、结果图表、讨论
- [ ] 代码与复现说明：运行步骤、配置文件、随机种子、数据目录说明
- [ ] 论文大纲初稿（参见 `docs/paper_outline.md`，ICTLE 6–8 页结构）
