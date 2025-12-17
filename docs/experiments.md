# 实验记录模板
> 按日期记录，每次实验/验证补充一条，保持可复现。示例可复制后填写。

## 2025-12-16（示例）
- **线路/方向/时间窗**：`线路名称/方向`，时间窗 `HH:MM–HH:MM`
- **数据来源与采样**：高德接口（接口名/字段）；采样频率 `XX s`; 缺失/异常说明
- **SUMO 场景版本**：`sumo/config/<cfg>`；路网 `sumo/net/<net.xml>`；附加文件 `additional/`；路线 `routes/`
- **关键参数默认值**：`tau=?`, `sigma=?`, `minGap=?`, `t_board=?`, `a=?`, `b=?`, `capacityFactor=?`（未设可写 N/A）
- **运行命令**：`<python/sumo 命令>`（含工作目录、环境变量，尽量可复制粘贴）
- **输出/日志路径**：`<output 文件路径>`，日志 `<log 路径>`，截图 `<img 路径>`（如有）
- **观察/结论**：指标、现象、问题清单、下一步动作

---

## 待填写记录
- 日期：
- 线路/方向/时间窗：
- 数据来源与采样：
- SUMO 场景版本：
- 关键参数默认值：SUMO 版本 1.20.0；Python 3.11.4（venv，system-site-packages）
- 运行命令：
- 输出/日志路径：
- 观察/结论：

## 2025-12-16（记录）
- 线路/方向/时间窗：预定 KMB 68X（基线稳态）+ 960（过海复杂工况），双向，待采集窗口 1h
- 数据来源与采样：KMB ETA + data.gov.hk（IRN 分段速度、检测器/分段 CSV、JTI、STN、HKO）
- SUMO 场景版本：冒烟测试，内置网 `D:\SUMO\doc\examples\duarouter\flows2routes\input_net.net.xml`
- 关键参数默认值：SUMO 版本 1.20.0；Python 3.11.4（venv，system-site-packages）
- 运行命令：`python scripts/traci_smoke.py`
- 输出/日志路径：标准输出（无额外日志）
- 观察/结论：traci 冒烟通过，SUMO_HOME/PYTHONPATH 生效；下一步采集 68X/960 的 ETA+路况，并准备场景文件

## 2025-12-16（高德数据采集与验证）
- **线路/方向/时间窗**：西安 229 路双向，晚高峰约 18:20 起 1h 轮询（部分轮询被中断）。
- **数据来源与采样**：高德 `v3/bus/linename`+`v3/bus/lineid`（静态）；`v4/etd/driving` 尝试失败（10012 权限不足）；`v3/traffic/status/road` 成功（adcode=610100，长安南路，extensions=all），可 60s 轮询。
- **运行命令**：`python scripts/gaode_downloader.py --key <key> --city 610100 --keyword 229 --line 229 --etd --interval 60 --duration 3600 --count 48 --strategy 1 --firsttime-offset 300 --traffic-road --road-name 长安南路 --road-level 5`
- **输出/日志路径**：原始 `data/raw/229-*{linename|lineid|traffic}.json`；日志 `data/raw/229-fetch.log`；清洗 `data/processed/229_stops.csv`, `229_line_meta.csv`, `traffic_road.csv`。
- **观察/结论**：静态线路/站点获取正常；etd 无权限（10012）无法拿时间分布/路况；traffic/status/road 可用，返回 speed/status/polyline；需用路况 map-match 替代站间速度，ETA 需换数据源或升级权限。

## 2025-12-17 — HK ETA 全量站轮询器调整与运行

- 线路/方向/时间窗：KMB 68X + 960 双向，目标轮询 1h，起始约 17:35（每轮完整跑完全站 stop/stop-eta）。
- 数据来源与采样：KMB `/route` `/route-stop` `/route-eta` `/stop` `/stop-eta`；data.gov.hk `/traffic-speed-map/notification.csv`、IRN 下载版 `irnAvgSpeed-all.xml`、检测器信息 `traffic_speed_volume_occ_info.csv`、STN/JTI/HKO；过滤掉已知失败接口（irn_avg_speed direct、road_network_segments、tsm_raw/raw_dl、TDAS、JSON_BUS）。
- 运行命令：`python scripts/hk_collect.py --routes 68X,960 --service-type 1 --duration 3600 --interval 60`
- 输出/日志路径：`data/raw/kmb-route-*.json`、`kmb-route-stop-*-*.json`、`kmb-route-eta-*.json`、全量站点 `kmb-stop-*.json` / `kmb-stop-eta-*.json`；`tsm-notification-*.csv`、`irnAvgSpeed-all-*.xml`、`traffic_speed_volume_occ_info-*.csv`、`trafficnews-*.xml`、`Journeytimev2-*.xml`、`hko-rhrread-*.json`；终端打印每轮耗时与对齐信息。
- 观察/结论：轮询间隔按照设定间隔（默认 60s）对齐，每轮耗时单独记录；为保证单轮数据完整，启动后不在轮内截断，截止检查发生在轮与轮之间，总时长约 1h（最后一轮可能略跨过截止）；若个别接口出现 SSL/HTTP 报错仅打印 warn 不中断。

## 2025-12-17 — HK 数据组织与里程计算

- 任务：整理 HK 原始数据、生成站点序列与里程表。
- 操作：
  - 运行 `python scripts/organize_raw_files.py` 完成 5950 个 raw 文件归类（KMB ETA/route/stop，IRN，JTI，STN，HKO 等）。
  - 新增 `scripts/clean_kmb_shapes.py`，使用 IRN `CENTERLINE`（EPSG:2326）+ KMB route-stop/stop 计算站点累积里程，输出 `data/processed/kmb_route_stop_dist.csv`（68X/960 双向，110 行）。
- 结果：
  - 归档：`data/raw` 按类别整理（kmb/、irn_download/、detector_locations/ 等）。
  - 里程：基于路网最短路径，`cum_dist_m`、`link_dist_m` 均按中心线长度估算。
  - 运行日志：加载 CENTERLINE 约 36k 行，构图 ~7.86M 节点/10.69M 边，生成 CSV 完成；Shapely 距离计算出现若干 RuntimeWarning（空/无效几何），未影响输出。
