# 🚌 Week 2 Engineering Report: High-Precision Network Reconstruction & Signal Analysis

## 1. Executive Summary (项目摘要)

本周完成了从 OpenStreetMap (OSM) 到 **IRN 高精度路网** 的底层架构迁移。通过自定义 Python 管道解析 ArcGIS (GDB) 数据，成功解决了 OSM 存在的拓扑断裂和车道属性错误问题。初步实验显示，引入 IRN 路网和信号灯机制后，仿真行程时间误差从 **-74.8% 收敛至 -62.1%**。虽然误差有所改善，但结果表明仅靠静态路网重建不足以复现真实拥堵，必须在下一阶段引入微观参数校准和背景交通流。

---

## 2. Milestone: Network Reconstruction (里程碑：路网重建)

**目标**：构建可用于 L1 微观校准的高保真仿真底座，替换低质量的 OSM 路网。

### 核心技术实现

通过自研工具链 (`inspect_irn_gdb.py` -> `convert_irn_to_sumo_xml.py` -> `netconvert`) 实现了以下关键修复：

* **拓扑逻辑修复 (Topology Fixes)**：
* 解析 `TURN` 表，自动生成 **25,400+** 条显式连接（Explicit Connections），彻底解决了立交桥和隧道口的断路问题。
* 基于 `TRAVEL_DIRECTION` 属性自动识别并生成双向路（Bi-directional）和反向边。


* **坐标系精准对齐 (Coordinate Alignment)**：
* 配置 `proj.plain-geo` 并禁用归一化，完美保留了 HK 1980 Grid 原始坐标系。


* **站点映射 (Stop Mapping)**：
* 利用 `pyproj` 实现了 WGS84 到 HK1980 的高精度转换。
* **结果**：106 个公交站点吸附成功率达到 **100%**。



### 质量验证

| 验证指标 | 结果 | 说明 |
| --- | --- | --- |
| **连通性 (WCC)** | **99.2%** | 物理路网几乎全联通 |
| **Route 完整性** | **~9 断点** | 仅剩极少数私家路汇入点断裂，仿真引擎可容错 |
| **死锁情况** | **无** | 车辆按真实站点停靠，无严重死锁 |

> **结论**：`hk_irn.net.xml` 质量远超 Week 1 版本，已具备微观校准条件。

---

## 3. Experiment 1: Impact of Signals (实验分析：信号灯影响)

**假设**：在高精度路网上启用信号灯（TLS）会显著降低行程时间误差。

### 实验配置

* **Baseline**: 真实世界行程时间 (~149 min / 8962s).
* **Control Group**: Week 1 OSM Network.
* **Test Group A**: IRN Network (无信号灯).
* **Test Group B**: IRN Network (启用 `<tls.guess>` & `<tls.join>`).

### 结果数据对比 (End-to-End Travel Time)

| 版本 | 仿真耗时 (s) | 绝对误差 (vs Real) | 相对误差 (%) | 改善幅度 |
| --- | --- | --- | --- | --- |
| **Week 1 (OSM)** | 2256s | -6706s | **-74.8%** | N/A |
| **Week 2 (IRN No TLS)** | 2857s | -6105s | **-68.1%** | +6.7% |
| **Week 2 (IRN w/ TLS)** | 3394s | -5568s | **-62.1%** | **+12.7% (Total)** |

### 关键发现 (Key Insights)

1. **信号灯的边际效益**：引入信号灯增加了约 **540秒 (9分钟)** 的延误，使误差收敛了约 6 个百分点。
2. **"空车" 现象依然显著**：即便有了高精度路网和信号灯，**-62% 的误差** 说明仿真中的车辆仍在以接近“自由流”的速度行驶。
* *根本原因*：缺乏背景交通流 (Background Traffic) 导致的排队拥堵，以及缺乏停靠服务时间 (Dwell Time)。


3. **技术债务**：启用自动信号灯猜测导致 `Teleport` (车辆跳跃) 警告增加，表明在复杂路口（如禁止右转处）自动生成的信号相位存在冲突，后续可能需要人工修正关键路口。

---

## 4. Next Steps (下一步计划)

鉴于单纯的静态路网优化已进入边际效应递减区间，下周将正式启动 **L1 Micro-Calibration (微观参数校准)**：

1. **注入背景流 (Background Traffic injection)**：
* 这是解决 -62% 误差的最关键手段。需根据流量调查数据生成背景车流，制造真实的排队效应。


2. **公交行为建模 (Bus Behavior Modeling)**：
* **Dwell Time**: 增加乘客上下车时间分布。
* **Speed Factor**: 调整公交车的速度分布因子（公交车通常低于限速行驶）。


3. **信号灯优化**:
* 针对 Teleport 报错集中的路口，手动检查并修正 `.con.xml` 连接或 TLS 相位。