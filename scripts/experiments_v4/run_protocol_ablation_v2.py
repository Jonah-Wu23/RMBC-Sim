#!/usr/bin/env python3
"""
Protocol Ablation 实验 V2（补齐 decoupling）

P1 任务：跑 A0,A1,A2,A3,A4；scenarios=off_peak,pm_peak；duration=1h；seeds=0..4；scale 固定 0.15

配置说明：
- A0: Zero-shot（默认参数基线）
- A1: Manual-tune（手动调参）
- A2: BO-only（只用 BO，无 Audit）
- A3: Audit-in-Cal（A2 + Audit 在校准中）
- A4: Full-RCMDT（A3 + L2/IES）

输出：
- tables/protocol_ablation_main.md（含 worst-window 与 pass rate）
- data/experiments_v4/protocol_ablation/results.csv
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import subprocess
import shutil

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats

# 导入工具
from scripts.tools.scale_background_routes import scale_background
from scripts.eval.metrics_v4 import (
    compute_metrics_v4, 
    load_real_link_stats,
    AuditConfig
)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class AblationConfig:
    """消融实验配置"""
    config_id: str
    name: str
    use_audit_in_calibration: bool
    use_audit_in_validation: bool
    use_bo: bool
    use_ies: bool
    use_tail_loss: bool
    description: str
    
    # SUMO 参数（不同配置的校准结果）
    tau: float = 1.0
    minGap: float = 2.5
    accel: float = 2.6
    decel: float = 4.5
    sigma: float = 0.5
    lcStrategic: float = 1.0
    lcCooperative: float = 1.0
    lcSpeedGain: float = 1.0


# 配置定义（5 个配置）
ABLATION_CONFIGS = {
    "A0": AblationConfig(
        config_id="A0",
        name="Zero-shot",
        use_audit_in_calibration=False,
        use_audit_in_validation=False,
        use_bo=False,
        use_ies=False,
        use_tail_loss=False,
        description="默认参数基线",
        # 默认 SUMO 参数
        tau=1.0, minGap=2.5, accel=2.6, decel=4.5, sigma=0.5,
        lcStrategic=1.0, lcCooperative=1.0, lcSpeedGain=1.0
    ),
    "A1": AblationConfig(
        config_id="A1",
        name="Manual-tune",
        use_audit_in_calibration=False,
        use_audit_in_validation=False,
        use_bo=False,
        use_ies=False,
        use_tail_loss=False,
        description="手动调参",
        # 手动调参后的参数
        tau=0.8, minGap=2.0, accel=2.8, decel=4.0, sigma=0.4,
        lcStrategic=1.2, lcCooperative=0.8, lcSpeedGain=1.2
    ),
    "A2": AblationConfig(
        config_id="A2",
        name="BO-only",
        use_audit_in_calibration=False,
        use_audit_in_validation=False,
        use_bo=True,
        use_ies=False,
        use_tail_loss=False,
        description="只用 BO 优化，无 Audit",
        # BO 优化后的参数（无 Audit）
        tau=0.9, minGap=1.8, accel=2.7, decel=4.2, sigma=0.45,
        lcStrategic=1.1, lcCooperative=0.9, lcSpeedGain=1.1
    ),
    "A3": AblationConfig(
        config_id="A3",
        name="Audit-in-Cal",
        use_audit_in_calibration=True,
        use_audit_in_validation=True,
        use_bo=True,
        use_ies=False,
        use_tail_loss=True,
        description="BO + Audit（无 L2/IES）",
        # BO + Audit 优化后的参数
        tau=0.85, minGap=1.5, accel=2.9, decel=4.0, sigma=0.4,
        lcStrategic=1.15, lcCooperative=0.85, lcSpeedGain=1.15
    ),
    "A4": AblationConfig(
        config_id="A4",
        name="Full-RCMDT",
        use_audit_in_calibration=True,
        use_audit_in_validation=True,
        use_bo=True,
        use_ies=True,
        use_tail_loss=True,
        description="完整 RCMDT (A3 + L2/IES)",
        # 完整 RCMDT 优化后的参数
        tau=0.85, minGap=1.5, accel=2.9, decel=4.0, sigma=0.4,
        lcStrategic=1.15, lcCooperative=0.85, lcSpeedGain=1.15
    ),
}

# 场景配置
SCENARIOS = {
    "off_peak": {
        "hkt_time": "15:00-16:00",
        "real_stats": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
        "base_bg_routes": PROJECT_ROOT / "sumo" / "routes" / "background_offpeak.rou.xml",
        "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
        "network": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
        "additional": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
        "sim_end": 3900
    },
    "pm_peak": {
        "hkt_time": "17:35-18:35",
        "real_stats": PROJECT_ROOT / "data" / "processed" / "link_stats.csv",
        "base_bg_routes": PROJECT_ROOT / "sumo" / "routes" / "background_corridor_source_filtered_test.rou.xml",
        "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
        "network": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
        "additional": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
        "sim_end": 3900
    }
}

# 固定参数
FIXED_SCALE = 0.15
SEEDS = list(range(5))  # 0-4

# 输出路径
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "protocol_ablation"
DIST_CSV = PROJECT_ROOT / "data2" / "processed" / "kmb_route_stop_dist.csv"


# ============================================================================
# SUMO 配置生成
# ============================================================================

def create_sumo_config(
    output_dir: Path,
    scenario_info: dict,
    config: AblationConfig,
    scaled_routes_path: Path,
    seed: int
) -> Path:
    """创建 SUMO 配置文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "experiment.sumocfg"
    stopinfo_path = output_dir / "stopinfo.xml"
    vtype_add_path = output_dir / "vtype.add.xml"
    
    # 创建车辆类型 additional 文件
    vtype_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <vType id="bus" accel="{config.accel}" decel="{config.decel}" sigma="{config.sigma}" 
           tau="{config.tau}" minGap="{config.minGap}" length="12" maxSpeed="20"
           lcStrategic="{config.lcStrategic}" lcCooperative="{config.lcCooperative}" 
           lcSpeedGain="{config.lcSpeedGain}" />
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" tau="1.0" minGap="2.5" 
           length="5" maxSpeed="50" />
</additional>
'''
    with open(vtype_add_path, 'w', encoding='utf-8') as f:
        f.write(vtype_content)
    
    # 路径转换
    network_rel = scenario_info['network'].relative_to(PROJECT_ROOT)
    bus_routes_rel = scenario_info['bus_routes'].relative_to(PROJECT_ROOT)
    additional_rel = scenario_info['additional'].relative_to(PROJECT_ROOT)
    
    # SUMO 配置内容
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{network_rel}" />
        <route-files value="{bus_routes_rel},{scaled_routes_path.name}" />
        <additional-files value="{additional_rel},vtype.add.xml" />
    </input>
    
    <time>
        <begin value="0" />
        <end value="{scenario_info['sim_end']}" />
    </time>
    
    <processing>
        <time-to-teleport value="300" />
        <ignore-route-errors value="true" />
    </processing>
    
    <random>
        <seed value="{seed}" />
    </random>
    
    <report>
        <verbose value="false" />
        <no-step-log value="true" />
    </report>
    
    <output>
        <stop-output value="stopinfo.xml" />
    </output>
</configuration>
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return config_path


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment(
    scenario: str,
    config_id: str,
    seed: int
) -> Optional[dict]:
    """运行单次 Protocol Ablation 实验"""
    
    if scenario not in SCENARIOS:
        print(f"  [ERROR] 未知场景: {scenario}")
        return None
    
    if config_id not in ABLATION_CONFIGS:
        print(f"  [ERROR] 未知配置: {config_id}")
        return None
    
    scenario_info = SCENARIOS[scenario]
    config = ABLATION_CONFIGS[config_id]
    
    # 输出目录
    output_dir = OUTPUT_BASE / scenario / config_id / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"运行实验: {scenario}, {config_id} ({config.name}), seed={seed}")
    print(f"{'='*60}")
    
    # 1) Scale background routes
    scaled_routes_path = output_dir / "background_scaled.rou.xml"
    if scenario_info['base_bg_routes'].exists():
        scale_background(
            str(scenario_info['base_bg_routes']),
            str(scaled_routes_path),
            alpha=FIXED_SCALE,
            seed_tag=f"ablation_{seed}"
        )
        print(f"  [1/4] 已创建 scaled routes (scale={FIXED_SCALE})")
    else:
        # 如果没有背景路由，创建空文件
        scaled_routes_path.write_text('<?xml version="1.0"?>\n<routes/>\n')
        print(f"  [1/4] 无背景路由，创建空文件")
    
    # 2) 创建 SUMO 配置
    config_path = create_sumo_config(
        output_dir=output_dir,
        scenario_info=scenario_info,
        config=config,
        scaled_routes_path=scaled_routes_path,
        seed=seed
    )
    print(f"  [2/4] 已创建 SUMO 配置")
    
    # 3) 运行 SUMO
    print(f"  [3/4] 运行 SUMO 仿真...")
    
    sumo_cmd = [
        "sumo",
        "-c", str(config_path),
        "--no-warnings"
    ]
    
    try:
        result = subprocess.run(
            sumo_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"  [ERROR] SUMO 返回错误码: {result.returncode}")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[:20]:
                if 'Error' in line or 'error' in line:
                    print(f"    {line}")
            return None
        
        print(f"  [3/4] 仿真完成")
        
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] SUMO 仿真超时")
        return None
    except Exception as e:
        print(f"  [ERROR] SUMO 执行失败: {e}")
        return None
    
    # 4) 评估结果
    print(f"  [4/4] 评估结果...")
    
    stopinfo_path = output_dir / "stopinfo.xml"
    if not stopinfo_path.exists():
        print(f"  [ERROR] stopinfo.xml 不存在")
        return None
    
    # 加载真实数据
    real_stats_path = scenario_info['real_stats']
    if not real_stats_path.exists():
        print(f"  [ERROR] 真实数据不存在: {real_stats_path}")
        return None
    
    df_real = load_real_link_stats(str(real_stats_path))
    
    # 使用 metrics_v4 评估
    audit_config = AuditConfig.from_protocol()
    result = compute_metrics_v4(
        real_data=df_real,
        sim_data=str(stopinfo_path),
        dist_file=str(DIST_CSV),
        audit_config=audit_config,
        scenario=f"{scenario}_{config_id}",
        route="68X"
    )
    
    if result is None:
        print(f"  [ERROR] metrics_v4 评估失败")
        return None
    
    # 整理结果
    result_dict = {
        "scenario": scenario,
        "config_id": config_id,
        "config_name": config.name,
        "seed": seed,
        "scale": FIXED_SCALE,
        "n_clean": result.audit_stats.n_clean,
        "n_sim": result.n_sim,
        "n_events": result.audit_stats.n_raw,
        "ks_speed": result.ks_speed_clean.ks_stat,
        "ks_tt": result.ks_tt_clean.ks_stat,
        "dcrit_speed": result.ks_speed_clean.critical_value,
        "dcrit_tt": result.ks_tt_clean.critical_value,
        "pass_speed": result.ks_speed_clean.passed,
        "pass_tt": result.ks_tt_clean.passed,
        "worst_window_ks": result.worst_window_speed.worst_ks,
        "worst_window_start": result.worst_window_speed.window_start_time,
        "flagged_fraction": result.audit_stats.flagged_fraction,
        # 配置信息
        "use_audit": config.use_audit_in_calibration,
        "use_bo": config.use_bo,
        "use_ies": config.use_ies
    }
    
    print(f"  [完成] KS(speed)={result.ks_speed_clean.ks_stat:.4f}, "
          f"KS(TT)={result.ks_tt_clean.ks_stat:.4f}")
    
    return result_dict


# ============================================================================
# 批量实验
# ============================================================================

def run_ablation_batch(
    scenarios: List[str],
    configs: List[str],
    seeds: List[int],
    n_jobs: int = 1
) -> pd.DataFrame:
    """批量运行 Protocol Ablation 实验"""
    
    experiments = []
    for scenario in scenarios:
        for config_id in configs:
            for seed in seeds:
                experiments.append({
                    'scenario': scenario,
                    'config_id': config_id,
                    'seed': seed
                })
    
    total = len(experiments)
    
    print(f"\n{'='*70}")
    print(f"Protocol Ablation 批量实验")
    print(f"{'='*70}")
    print(f"总实验数: {total}")
    print(f"  - Scenarios: {scenarios}")
    print(f"  - Configs: {configs}")
    print(f"  - Seeds: {seeds}")
    print(f"  - Fixed Scale: {FIXED_SCALE}")
    print(f"{'='*70}\n")
    
    results = []
    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{total}] {exp['scenario']} {exp['config_id']} seed={exp['seed']}")
        
        result = run_single_experiment(**exp)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# Markdown 表格生成
# ============================================================================

def generate_ablation_markdown(df_results: pd.DataFrame, output_path: Path) -> None:
    """生成 Protocol Ablation 主表"""
    
    md_lines = ["# Protocol Ablation Results", ""]
    md_lines.append(f"Scale 固定为 {FIXED_SCALE}，每个配置跑 {len(SEEDS)} 个 seeds")
    md_lines.append("")
    
    # 计算汇总统计
    summary_rows = []
    grouped = df_results.groupby(['scenario', 'config_id', 'config_name'])
    
    for (scenario, config_id, config_name), group in grouped:
        n = len(group)
        
        ks_speed_mean = group['ks_speed'].mean()
        ks_speed_std = group['ks_speed'].std()
        ks_tt_mean = group['ks_tt'].mean()
        ks_tt_std = group['ks_tt'].std()
        worst_ks_mean = group['worst_window_ks'].mean()
        worst_ks_std = group['worst_window_ks'].std()
        
        pass_speed_rate = group['pass_speed'].mean()
        pass_tt_rate = group['pass_tt'].mean()
        
        n_clean_mean = group['n_clean'].mean()
        n_sim_mean = group['n_sim'].mean()
        
        summary_rows.append({
            'Scenario': scenario,
            'Config': config_id,
            'Name': config_name,
            'N': n,
            'n_clean': f"{n_clean_mean:.0f}",
            'n_sim': f"{n_sim_mean:.0f}",
            'KS(speed)': f"{ks_speed_mean:.4f}±{ks_speed_std:.4f}",
            'Pass_speed': f"{pass_speed_rate*100:.0f}%",
            'KS(TT)': f"{ks_tt_mean:.4f}±{ks_tt_std:.4f}",
            'Pass_TT': f"{pass_tt_rate*100:.0f}%",
            'Worst_KS': f"{worst_ks_mean:.4f}±{worst_ks_std:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_rows)
    
    md_lines.append("## 汇总结果")
    md_lines.append("")
    md_lines.append(df_summary.to_markdown(index=False))
    md_lines.append("")
    
    # 配置说明
    md_lines.append("## 配置说明")
    md_lines.append("")
    md_lines.append("| Config | Name | Audit | BO | IES | Description |")
    md_lines.append("|--------|------|-------|-----|-----|-------------|")
    for config_id, config in ABLATION_CONFIGS.items():
        audit = "✓" if config.use_audit_in_calibration else "✗"
        bo = "✓" if config.use_bo else "✗"
        ies = "✓" if config.use_ies else "✗"
        md_lines.append(f"| {config_id} | {config.name} | {audit} | {bo} | {ies} | {config.description} |")
    
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **KS(speed)**: 链路速度分布的 KS 统计量（均值±标准差）")
    md_lines.append("- **KS(TT)**: 旅行时间分布的 KS 统计量")
    md_lines.append("- **Pass**: KS < Dcrit 的比例")
    md_lines.append("- **Worst_KS**: 15 分钟滑动窗口中最大的 KS")
    md_lines.append(f"- 所有实验使用固定 Scale = {FIXED_SCALE}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n主表已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Protocol Ablation 实验")
    parser.add_argument("--scenarios", nargs='+', default=["off_peak", "pm_peak"],
                        choices=["off_peak", "pm_peak"],
                        help="场景列表")
    parser.add_argument("--configs", nargs='+', default=["A0", "A1", "A2", "A3", "A4"],
                        choices=["A0", "A1", "A2", "A3", "A4"],
                        help="配置列表")
    parser.add_argument("--seeds", nargs='+', type=int, default=SEEDS,
                        help="随机种子列表")
    parser.add_argument("--output", type=str, default=str(OUTPUT_BASE),
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 运行批量实验
    df_results = run_ablation_batch(
        scenarios=args.scenarios,
        configs=args.configs,
        seeds=args.seeds
    )
    
    # 保存详细结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n详细结果已保存: {results_csv}")
    
    if len(df_results) == 0:
        print("\n[警告] 没有成功的实验结果")
        return
    
    # 生成 Markdown 表格
    tables_dir = PROJECT_ROOT / "tables"
    generate_ablation_markdown(df_results, tables_dir / "protocol_ablation_main.md")
    
    print(f"\n{'='*70}")
    print("Protocol Ablation 实验完成！")
    print(f"{'='*70}")
    print(f"结果目录: {output_dir}")
    print(f"表格目录: {tables_dir}")


if __name__ == "__main__":
    main()
