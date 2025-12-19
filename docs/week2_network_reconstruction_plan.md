# 第二周实施计划：高精度路网重建 (Week 2 Network Reconstruction Plan)

## 背景 (Context)
第一周（Week 1）的验证表明，基于 OSM 的基线仿真与真实数据存在巨大的系统性偏差（End-to-End 误差 -74.8%，仿真显著偏快）。主要原因在于：
1. **路网简陋**：OSM 在新界/九龙区域缺失大量连通细节，导致必须人工“猜”路由，且无法准确映射所有站点（丢失 ~15% 站点）。
2. **缺乏阻力**：缺失红绿灯、交叉口延误配置，导致车辆几乎全程以自由流速度通过。
3. **拓扑断裂**：多处路段不连通，导致车辆瞬移或路由失败。

本计划旨在利用 **Review High Precision Road Network (RdNet_IRNP.gdb)** 重建 SUMO 路网，以解决上述问题。

## 目标 (Objectives)
1. **全覆盖映射**：确保 100% 的 KMB 68X/960 站点能精确映射到路网上（误差 < 10m）。
2. **真实拓扑**：利用 IRN 的 Turn Tables (禁止/允许转向) 和 Route Geometry 建立无需“猜测”的物理连接。
3. **信号与车道**：恢复主要交叉口的信号灯逻辑（即使是固定相位），引入真实的行车阻力。

---

## 任务分解 (Tasks)

### 1. 数据解析与转换 (Data Parsing)
- **输入**: `data/RdNet_IRNP.gdb` (Esri File Geodatabase)
- **工具**: `gdal`/`fiona` 或 ArcGIS Pro (如果需要手动导出为 Shapefile)。
- **子任务**:
    - [ ] **提取 Layer**: 提取 `ROAD_NETWORK` (路中心线) 和 `TRAFFIC_AID` (交通灯/标志)。
    - [ ] **坐标转换**: IRN 使用 HK1980 Grid (EPSG:2326)，需无损转换为 WGS84 (EPSG:4326) 或直接转换为 SUMO 内部坐标。
    - [ ] **属性清洗**: 提取 `ROUTE_ID`, `FLOW_DIRECTION`, `SPEED_LIMIT` (如果有)。

### 2. SUMO 网络生成 (Network Generation)
- **工具**: `netconvert` (SUMO 核心工具)
- **子任务**:
    - [ ] **Import Config**: 编写 `irn.netccfg`，配置 `--shapefile-prefix` 或直接导入转换后的 XML/Edge files。
    - [ ] **Lane Number Inference**: 根据 IRN 属性（如 roadway width 或 road type）推断车道数。
    - [ ] **Junction Logic**: 启用 `--junctions.join` 优化复杂路口，启用 `--tls.guess` 猜测信号灯位置。

### 3. 站点与路由重映射 (Remapping)
- **脚本**: `scripts/generate_sumo_stops_irn.py` (新版)
- **逻辑**:
    - 不再使用最近邻搜索模糊匹配。
    - 使用 **Stop Name / Geo-Location** 强匹配 IRN 路段。
    - 针对 IRN 的 `Segment ID` 建立 `Stop -> Segment` 查找表。
- **验证**:
    - 运行静态检查：有多少站点成功 SNAP 到路网上？（目标 100%）

### 4. 仿真运行与再次校准 (Re-run & Calibration)
- **更新场景**: 创建 `sumo/net/hk_irn.net.xml` 及配套的 `routes/baseline_irn.rou.xml`。
- **再次对比**: 运行 `compare_week1.py` (兼容旧逻辑)，查看 Error 是否从 -74.8% 收敛到 -20% 以内。

---

## 执行时间表 (Timeline)

| 阶段 | 任务 | 预计耗时 | 输出 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | GDB 解析与初版 `.net.xml` 生成 | 1 天 | `hk_irn_v1.net.xml` |
| **Phase 2** | 站点重新映射脚本开发 | 0.5 天 | `bus_stops_irn.add.xml` |
| **Phase 3** | 路由生成与连通性修复 | 0.5 天 | `baseline_irn.rou.xml` |
| **Phase 4** | 完整运行与 L1/L2 验证 | 0.5 天 | 新的 `week2_comparison.csv` |

## 风险管理 (Risk Management)
- **GDB 解析失败**: 如果 `gdal` 无法完美读取 GDB 的拓扑关系，可以回退到使用 OSM 但手动利用 IRN 参考线修补（Hybrid Approach）。
- **路网过大**: IRN 包含全港，直接仿真实时性可能差。需使用 `netconvert --keep-edges.by-bbox` 裁剪出 九龙-新界西 走廊。

## 下一步 (Next Steps)
1. 确认 `data/RdNet_IRNP.gdb` 文件是否存在及其大小。
2. 编写 GDB 解析脚本原型。
