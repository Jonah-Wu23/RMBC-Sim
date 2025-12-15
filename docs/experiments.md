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
- 线路/方向/时间窗：待定（Week 1 选线阶段）
- 数据来源与采样：待定（高德接口确认中）
- SUMO 场景版本：冒烟测试，内置网 `D:\SUMO\doc\examples\duarouter\flows2routes\input_net.net.xml`
- 关键参数默认值：SUMO 版本 1.20.0；Python 3.11.4（venv，system-site-packages）
- 运行命令：`python scripts/traci_smoke.py`
- 输出/日志路径：标准输出（无额外日志）
- 观察/结论：traci 冒烟通过，SUMO_HOME/PYTHONPATH 生效；下一步需选定目标线路并准备场景文件
