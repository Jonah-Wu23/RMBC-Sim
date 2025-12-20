# Week 2 状态报告：路网裁剪与仿真验证

**日期:** 2025/12/20
**状态:** 进行中

## 1. 目标
解决全港路网仿真速度过慢的问题，通过裁剪路网至市区核心区域（长沙湾 - 西营盘），在保证关键路线（68X, 960）实验有效性的前提下，提升仿真效率。

## 2. 已完成工作
- [x] **路网裁剪**: 使用 `netconvert` 的 `--keep-edges.in-boundary` 参数成功将 `hk_irn_v3.net.xml` (0.5GB) 裁剪为 `hk_cropped.net.xml` (38MB)。
- [x] **路由清洗**: 开发了 `crop_network_v2.py` 脚本，自动处理以下任务：
    - 识别裁剪后路网中的有效 Edge 和 Lane。
    - 过滤 `bus_stops.add.xml`，移除了 60+ 个位于边界外的无效站点，保留核心区 42 个站点。
    - 清洗 `fixed_routes.rou.xml`，自动分割和保留位于路网内的路线段。
    - 过滤 `background_clipped.rou.xml`，去除了引用无效路径的背景流。
- [x] **工具开发**: 创建了 `compare_week2_cropped.py`，支持在部分路线覆盖的情况下进行轨迹对齐和 L1/L2 指标对比。

## 3. 当前验证状态
- **仿真执行**: Experiment 2.3 (Cropped) 正在运行中。
- **参数**: 时长设定为 1200秒（20分钟），以快速验证数据流。
- **预期结果**:
    - 仿真速度显著提升（Real-time factor >> 1）。
    - 生成有效的 `stopinfo_exp2_cropped.xml` 和 `tripinfo_exp2_cropped.xml`。
    - 能够通过 `compare_week2_cropped.py` 生成轨迹对比图。

## 4. 遇到的问题与解决方案
- **问题**: SUMO 自带的 `cutRoutes.py` 和 `duarouter` 兼容性差，处理路网裁剪后的路由时频繁报错。
- **解决**: 放弃使用这些工具，完全使用自定义 Python 脚本 (`crop_network_v2.py`) 进行基于集合逻辑的过滤和清洗，效果良好。
- **问题**: 仿真初期只有 header 输出，无数据。
- **原因**: 车辆尚未完成行程或停靠。属于正常现象，需等待仿真进行。

## 5. 下一步
- 确认仿真输出数据。
- 运行验证脚本 `python scripts/compare_week2_cropped.py`。
- 如果数据有效，正式运行 1小时完整仿真。
