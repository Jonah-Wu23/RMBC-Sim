# Tail-Loss Ablation 实验运行指南

## 快速开始

### 模拟模式（快速验证，约 5 秒）
```powershell
python scripts/calibration/run_tail_loss_ablation.py `
  --output "data/calibration/tail_loss_ablation" `
  --real "data2/processed/link_stats_offpeak.csv" `
  --simulate
```

### 实际仿真模式（完整验证，约 40 分钟）
```powershell
python scripts/calibration/run_tail_loss_ablation.py `
  --output "data/calibration/tail_loss_ablation" `
  --real "data2/processed/link_stats_offpeak.csv" `
  --run_sumo
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output` | 输出目录 | `data/calibration/tail_loss_ablation` |
| `--real` | P14 验证数据路径 | `data2/processed/link_stats_offpeak.csv` |
| `--simulate` | 使用模拟模式（不运行 SUMO） | `True` |
| `--run_sumo` | 运行实际 SUMO 仿真（覆盖 --simulate） | `False` |

## 实验设计

### 对比组
1. **RMSE-only**: 目标函数 = RMSE
   - 只优化均值误差
   - 不考虑尾部风险

2. **Tail-aware**: 目标函数 = robust_loss + 0.5 × P90
   - 同时优化均值和尾部
   - 使用分位数损失控制极端情况

### 实验配置
- **Budget**: 40 iterations (15 LHS + 25 BO)
- **Seed**: 42 (确保可复现)
- **参数空间**: 7D (t_board, t_fixed, tau, sigma, minGap, accel, decel)

## 输出文件

### 主要结果
```
data/calibration/tail_loss_ablation/
├── tail_loss_ablation_results.csv      # 对比结果摘要
├── tail_loss_ablation_table.tex        # LaTeX 表格
├── rmse-only_log.csv                   # RMSE-only 组完整日志
└── tail-aware_log.csv                  # Tail-aware 组完整日志
```

### 仿真输出（仅 --run_sumo 模式）
```
sumo/output/tail_loss_ablation/
├── iter_000_stopinfo.xml               # 第 1 次迭代仿真输出
├── iter_001_stopinfo.xml               # 第 2 次迭代仿真输出
└── ...                                  # 共 80 个文件 (2组 × 40次)
```

## 预期结果

### 模拟模式
- **Tail-aware 改进**: ~1.4% (保守估计)
- **运行时间**: ~5 秒
- **用途**: 快速验证脚本逻辑

### 实际仿真模式
- **Tail-aware 改进**: 预计 5-10%
- **运行时间**: ~40 分钟 (每次仿真 ~30秒 × 80次)
- **用途**: 完整实验验证

## 故障排查

### 错误：SUMO 未找到
```
FileNotFoundError: sumo not found
```
**解决方案**: 确保 SUMO 已安装并添加到 PATH
```powershell
sumo --version
```

### 错误：Base route 文件缺失
```
FileNotFoundError: Base route file not found: sumo/routes/fixed_routes_cropped.rou.xml
```
**解决方案**: 确认项目根目录结构完整

### 错误：仿真超时
```
[ERROR] SUMO 仿真超时
```
**解决方案**: 检查路网是否存在死锁，或增加 timeout 值（脚本第 382 行）

## 技术细节

### 目标函数定义

**RMSE-only**:
```python
J = calculate_l1_rmse(sim_xml_path, real_links_csv, route_stop_dist_csv)
```

**Tail-aware**:
```python
result = calculate_l1_robust_objective(sim_xml_path, ...)
J = result['robust_loss'] + 0.5 × result['quantile_loss']
```

其中:
- `robust_loss = mean(errors) + 0.5 × std(errors)`
- `quantile_loss = P90(|errors|)`

### 参数更新机制

**路由文件更新** (`_update_route_xml`):
1. 更新 vType 参数 (Krauss 模型)
   - accel, decel, sigma, tau, minGap
2. 更新停站时间 (物理一致性模型)
   - duration = T_fixed + t_board × (N_base × W_stop)

**仿真运行** (`_run_simulation`):
1. 生成带参数的路由文件
2. 调用 SUMO 运行仿真
3. 从 stopinfo.xml 计算目标函数
4. 返回目标值供 BO 优化

## 引用

此实验设计用于回应审稿人关于 "tail-loss 是否有贡献" 的质疑。

相关章节：
- TODO.md: 必做 5
- experiments.md: B2 实验 (BO baseline)
- paper_outline.md: Robust Validation 章节
