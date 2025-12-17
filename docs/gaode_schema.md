# 高德公交/路况接口字段映射（草稿）

> 目标线路：西安公交 229 路（双向）。数据仅作真值对齐，后续如遇字段缺失需补充替代方案。

## 接口概览
- `GET /v3/bus/linename`：公交线路关键字查询，获取线路 id、名称、首末站、途径站（extensions=all 时包含站点列表、polyline）。
- `GET /v3/bus/lineid`：公交线路 id 查询，按 id 拉取同样的线路详情，用于稳定字段、避免 keyword 歧义。
- `GET /v3/bus/stopname` / `GET /v3/bus/stopid`：站点关键字/id 查询，必要时用来补站点经纬度或校验。
- `GET /v4/etd/driving`：未来路径规划（含 time_infos，提供不同时间点的行程时长/收费/限行信息），可作为路段速度替代来源。

## 需要的真值字段 → 接口字段
- 线路标识：`buslines[].id`（linename/lineid）
- 线路名称：`buslines[].name`
- 首末站：`buslines[].start_stop` / `buslines[].end_stop`
- 站点序列：`buslines[].busstops[]`（按 `sequence` 或列表顺序）
  - 站点 id/名称：`id` / `name`
  - 站点坐标：`location`（lon,lat）
- 路网 polyline：`buslines[].polyline`（坐标串，用于 map-matching/截取路段）
- 反向线路 id：`buslines[].direc`
- 运营时间：`start_time` / `end_time`
- 票价：`basic_price` / `total_price`
- 路段/速度真值：
  - 官方接口无直接实时车速；可用 `v4/etd/driving` 返回的 `time_infos[].elements[].duration` + 路段距离近似速度
  - `time_infos[].starttime` 提供对应时间戳；`tmcs` 可提供路况状态/ polyline 切片（如返回）

## 缺失与替代方案
- 到离站时刻/ETA：公交信息接口未提供车辆实时 ETA；当前只能利用站点序列 + 路段速度近似推演。若后续拿不到 ETA，需定义“真值替代”：（1）线路 polyline 切分站间距离；（2）用 etd/driving 的时长分布推算站间时间；（3）或外部实测数据补充。
- 路段速度：etd/driving 返回的 `tmcs` 可能不覆盖全部站间路段，需要自行按 polyline 切分并插值；若缺失，先以整段平均速度作为占位。
- 路况补充：`/v3/traffic/status/road` 返回的 road 列表含 `speed`、`status`、`polyline`，可作为路段速度/拥堵等级来源（需 map-match 到线路 polyline）。

## 命名与落盘约定
- 原始响应：`data/raw/<线路>-<direction>-<YYYYMMDD>-<HHMMSS>-<type>.json`
  - direction 使用线路名称中的方向字样（如“(火车站—公交六公司)”）做安全化文件名。
  - type 示例：`linename`, `lineid`, `etd`.
- 日志：`data/raw/<线路>-fetch.log`（记录请求参数、响应状态、重试/限流信息）。

## 采集策略建议（229 路）
- 初次运行：先用 `linename`+`lineid` 拉取双向线路的静态结构；保存 polyline/站点序列。
- 路况抓取：选首末站经纬度调用 `v4/etd/driving`，`firsttime` 设为当前时间 + 5 分钟，`interval`=60s，`count`=48（最多 48 个时间点，覆盖 ~48 分钟；如需 1h，延长 count 或分两次调用）。
- 频率与限流：默认 60s 间隔；失败指数退避 1.5x，最多 3 重试；HTTP 429/500 记入日志。
- 权限提醒：当前 Key 调用 `v4/etd/driving` 返回 `errcode=10012 (INSUFFICIENT_PRIVILEGES)`，暂无法获得路况/时间分布；需要升级或改用可公开获取的速度源。`/v3/traffic/status/road` 已验证可用（2025-12-16，adcode=610100，长安南路 level=5 extensions=all 返回 10000，含 speed/status/polyline）。

## 清洗输出（已实现）
- `src/gaode_clean.py`：从 `*-lineid.json` 提取站点序列与线路元数据，输出 `data/processed/229_stops.csv`、`229_line_meta.csv`。
- `src/gaode_traffic_clean.py`：从 `*-traffic.json` 提取道路路况（speed/status/polyline 等），输出 `data/processed/traffic_road.csv`，供后续与线路 polyline 进行空间匹配。

## 明日执行计划（调整：转向香港 KMB/LWB ETA + HK 公开路况/事件）
- 目标源（实时 ETA）：`https://data.etabus.gov.hk/v1/transport/kmb`（或 lwb），获取站点级 ETA。
- 静态形状/距离：使用已下载的公交 GeoJSON（JSON_BUS）解析 route polyline、stop 坐标、沿线里程，生成站间距离表；路网用 IRN（`RdNet_IRNP.gdb`）支撑匹配。
- 基础静态：`/route`、`/stop`、`/route-stop/{route}/{bound}/{service_type}` → `kmb_routes.csv`、`kmb_stops.csv`、`kmb_route_stop.csv`。
- 实时采集：每 60s 轮询 `/route-eta/{route}/{service_type}` 或 `/eta/{stop}/{route}/{service_type}`，落 raw，清洗成 `station_eta.csv`（含 route/bound/service_type/seq/eta/data_timestamp/capture_ts）。优先线路：68X（基线稳态）+ 960（过海复杂工况），双向。
- 站间运行时间/速度：用 ETA 差分 + 形状里程计算 `link_times.csv`、`link_speeds.csv`。
- 路况特征：使用 IRN 分段平均速度 `irnAvgSpeed-all.xml`（processed 2min）和原始探测器 `rawSpeedVol-all.xml`（可选 raw 特征）；结合检测器/分段 CSV（`traffic_speed_volume_occ_info.csv`、`speed_segments_info.csv`）进行 map-match；JTI（Journeytimev2.xml，2min）作为走廊级旅时参考。TDAS 暂不采集。
- 事件与天气：拉取 Special Traffic News（trafficnews.xml）和 HKO Open Data（天气/警告），作为特征或筛除异常样本。
- 路网：参考 IRN 数据（方向/转向限制）做 map-matching，提升路段匹配质量。

---

## 香港 KMB/LWB + data.gov.hk 字段映射（执行中）

### KMB/LWB 公交（`https://data.etabus.gov.hk/v1/transport/kmb`，LWB 同结构）
- 所有响应含封面字段：`type`（Route/Stop/RouteStop/ETA）、`version`、`generated_timestamp`（ISO8601, UTC+8）。
- `/route` → 路线列表
  - 数据字段：`co`（KMB/LWB）、`route`、`bound`（I/O）、`service_type`（1=主线，2/3=特班）、`orig_en`/`orig_tc`/`orig_sc`、`dest_en`/`dest_tc`/`dest_sc`、`data_timestamp`。
- `/route-stop/{route}/{bound}/{service_type}` → 线路站点序列
  - 数据字段：`route`、`bound`、`service_type`、`seq`（1 开始）、`stop`、`data_timestamp`。
- `/stop/{stop_id}` → 站点详情
  - 数据字段：`stop`、`name_en`/`name_tc`/`name_sc`、`lat`、`long`、`address_en`/`address_tc`/`address_sc`。
- `/route-eta/{route}/{service_type}`（或 `/eta/{stop_id}/{route}/{service_type}`）→ ETA
  - 数据字段：`co`、`route`、`dir`（I/O）、`service_type`、`seq`（站序，Route ETA 有）、`stop`、`dest_en`/`dest_tc`/`dest_sc`、`eta_seq`（同站多车序号，最多 3 条）、`eta`、`rmk_en`/`rmk_tc`/`rmk_sc`、`data_timestamp`（同帧生成时间）、`gps`（Y/N）。
- 清洗命名/落盘约定
  - 原始：`data/raw/kmb-route-<ts>.json`、`kmb-route-stop-<route>-<bound>-<ts>.json`、`kmb-stop-<stop>-<ts>.json`、`kmb-route-eta-<route>-<ts>.json`（ts=`YYYYMMDD-HHMMSS`）。
  - 清洗：`data/processed/kmb_routes.csv`（route,bound,service_type,orig_en,dest_en,orig_tc,dest_tc）、`kmb_stops.csv`（stop_id,name_en,name_tc,lat,long,address_en,address_tc）、`kmb_route_stop.csv`（route,bound,service_type,seq,stop_id）、`station_eta.csv`（capture_ts,route,bound,service_type,stop_seq,stop_id,eta,data_timestamp,dest_en,dest_tc,eta_seq,gps,rmk_en,rmk_tc）。

### 公交形状与票价（GeoJSON）
- 源：`https://static.data.gov.hk/td/routes-fares-geojson/JSON_BUS.json`（已下载到 `data/JSON_BUS.json`）。
- 关键字段：
  - 路线：`routeId`、`companyCode`、`routeNameC/S/E`、`routeType`、`serviceMode`、`specialType`、`journeyTime`、`locStartName*`/`locEndName*`、`fullFare`、`lastUpdateDate`。
  - 站点：`coordinates`（lon,lat）、`routeSeq`、`stopSeq`、`stopId`、`stopPickDrop`（1/2/3）、`stopNameC/S/E`。
  - `fares`：段票价字段随公司/线路定义；保留全量。
  - `geometry`：LineString，多段 polyline（WGS84），可用于计算沿线里程/站间距离。
- 清洗目标：
  - 输出 `data/processed/kmb_shapes.parquet`（route_id,route,co,direction,service_type?,seq,lon,lat,cum_dist_m），累积里程用 polyline 逐段累计。
  - 站点表补充坐标与里程：join `kmb_route_stop` + GeoJSON polyline snapped/cum_dist。

### 路况/速度（data.gov.hk）
- IRN 分段平均速度（处理版）：`irnAvgSpeed-all.xml`
  - 字段：`road_section_id`、`avg_speed`（km/h）、`capture_date`、`region`、`direction` 等。落原始 `data/raw/irn_download/`，清洗后 `data/processed/irn_speed.csv`。
  - 需要用 `speed_segments_info.csv`（段定义）做 map-match 到公交 polyline。
- 原始探测器（可选）：`rawSpeedVol-all.xml`
  - 结构：`date`、`periods[].period_from/period_to`（30s 窗）、`periods[].detectors[].detector_id`、`direction`（1=North/2=East/3=South/4=West/5=NE/6=SE/7=NW/8=SW）、`lanes[].lane_id`（按车道数标号）、`speed`（km/h）、`occupancy`（%）、`volume`、`sd`、`valid`（Y/N）。
  - 定位辅助：`traffic_speed_volume_occ_info.csv`（探测器位置/道路，含 Device_ID, District 等）。
- JTI（走廊旅时）：`Journeytimev2.xml`
  - 字段：`LOCATION_ID`（传感器位置码，H1/K01/N01 等）、`DESTINATION_ID`（如 CH/EH/WH/ATL…）、`CAPTURE_DATE`（YYYY-MM-DDTHH:MM:SS）、`JOURNEY_TYPE`（1=journey time, 2/3/4=bitmap）、`JOURNEY_DATA`（分钟或 bitmap 码）。
- STN（Special Traffic News）：`trafficnews.xml`
  - 字段（v4.0）：`INCIDENT_NUMBER`、`INCIDENT_HEADING_EN/CN`、`INCIDENT_DETAIL_EN/CN`、`LOCATION_EN/CN`、`DISTRICT_EN/CN`（18 区枚举）、`DIRECTION_EN/CN`、`ANNOUNCEMENT_DATE`（YYYY-MM-DDTHH:MM:SS）、`INCIDENT_STATUS_EN/CN`（NEW/UPDATED/CLOSED）、`NEAR_LANDMARK_EN/CN`、`BETWEEN_LANDMARK_EN/CN` 等。用于事件标注/过滤。
- HKO 天气：`weatherAPI` `dataType=rhrread`
  - 字段：`updateTime`、`temperature.data[].value/unit/place`、`humidity.data[]`、`rainfall.data[]`、`lightning`, `warningMessage` 等，含 UTC+8 时间。用于天气特征或异常筛除。
- 命名/落盘约定
  - 原始：`data/raw/irnAvgSpeed-all-<ts>.xml`、`rawSpeedVol-all-<ts>.xml`、`traffic_speed_volume_occ_info-<ts>.csv`、`speed_segments_info-<ts>.csv`、`Journeytimev2-<ts>.xml`、`trafficnews-<ts>.xml`、`hko-rhrread-<ts>.json`。
  - 清洗：`data/processed/irn_speed.csv`、`irn_segments.csv`、`detectors.csv`、`detector_raw.csv`、`jti.csv`、`stn_events.csv`、`weather_rhrread.csv`。

### 计算逻辑对齐
- 站点 ETA → 到离站时刻：同站多车按 `eta_seq` 排序，取最近 ETA；结合 `capture_ts` 与下一帧差分推算到/离站时间。
- 站间运行时间/速度：利用 `cum_dist_m`（polyline 里程）与相邻站 ETA 差分得到行程时间/速度；异常值用 IQR 或分位截断。
- 路况匹配：IRN/JTI/探测器按空间最近路段/走廊匹配到公交 polyline 或站间段，时间对齐按 2min 窗。
