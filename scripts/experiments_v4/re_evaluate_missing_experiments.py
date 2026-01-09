#!/usr/bin/env python3
"""
重新评估缺失的实验结果

从 stopinfo.xml 重新调用 metrics_v4 评估，补齐缺失的 off_peak 实验结果
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from scripts.eval.metrics_v4 import (
    compute_metrics_v4, 
    load_real_link_stats,
    AuditConfig
)


# 场景真实数据路径
REAL_DATA_PATHS = {
    "off_peak": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
    "pm_peak": PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
}

DIST_CSV = PROJECT_ROOT / "data2" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"


def evaluate_single_experiment(
    scenario: str,
    stopinfo_path: Path,
    real_stats_path: Path
) -> dict:
    """评估单个实验"""
    
    # 加载真实数据
    df_real = load_real_link_stats(str(real_stats_path))
    
    # 评估
    audit_config = AuditConfig.from_protocol()
    result = compute_metrics_v4(
        real_data=df_real,
        sim_data=str(stopinfo_path),
        dist_file=str(DIST_CSV),
        audit_config=audit_config,
        scenario=scenario,
        route="68X"
    )
    
    if result is None:
        return None
    
    # 提取结果
    return {
        'n_clean': result.audit_stats.n_clean,
        'n_sim': result.n_sim,
        'n_events': result.audit_stats.n_raw,
        'ks_speed': result.ks_speed_clean.ks_stat,
        'ks_tt': result.ks_tt_clean.ks_stat,
        'dcrit_speed': result.ks_speed_clean.critical_value,
        'dcrit_tt': result.ks_tt_clean.critical_value,
        'passed': result.ks_speed_clean.passed and result.ks_tt_clean.passed,
        'worst_window_ks': result.worst_window_speed.worst_ks,
        'worst_window_start': result.worst_window_speed.window_start_time,
        'flagged_fraction': result.audit_stats.flagged_fraction
    }


def main():
    print("="*70)
    print("重新评估缺失的实验结果")
    print("="*70)
    
    # 1. 加载现有结果
    results_csv = OUTPUT_BASE / "results.csv"
    df_existing = pd.read_csv(results_csv)
    print(f"\n现有结果: {len(df_existing)} 条")
    print(f"  - off_peak: {len(df_existing[df_existing['scenario']=='off_peak'])}")
    print(f"  - pm_peak: {len(df_existing[df_existing['scenario']=='pm_peak'])}")
    
    # 2. 扫描所有实验
    all_experiments = []
    
    for scenario_dir in OUTPUT_BASE.iterdir():
        if not scenario_dir.is_dir() or scenario_dir.name not in ['off_peak', 'pm_peak']:
            continue
        
        scenario = scenario_dir.name
        real_stats_path = REAL_DATA_PATHS[scenario]
        
        if not real_stats_path.exists():
            print(f"[警告] 真实数据不存在: {real_stats_path}")
            continue
        
        # 遍历目录结构
        for duration_dir in scenario_dir.iterdir():
            if not duration_dir.is_dir():
                continue
            
            duration_hours = int(duration_dir.name.replace('h', ''))
            
            for scale_dir in duration_dir.iterdir():
                if not scale_dir.is_dir():
                    continue
                
                scale = float(scale_dir.name.replace('scale', ''))
                
                for config_dir in scale_dir.iterdir():
                    if not config_dir.is_dir():
                        continue
                    
                    config_id = config_dir.name
                    
                    for seed_dir in config_dir.iterdir():
                        if not seed_dir.is_dir():
                            continue
                        
                        seed = int(seed_dir.name.replace('seed', ''))
                        stopinfo_path = seed_dir / "stopinfo.xml"
                        
                        if stopinfo_path.exists():
                            all_experiments.append({
                                'scenario': scenario,
                                'duration_hours': duration_hours,
                                'scale': scale,
                                'config_id': config_id,
                                'seed': seed,
                                'stopinfo_path': stopinfo_path,
                                'real_stats_path': real_stats_path
                            })
    
    print(f"\n扫描到 {len(all_experiments)} 个实验")
    
    # 3. 找出缺失的实验
    merge_keys = ['scenario', 'duration_hours', 'scale', 'config_id', 'seed']
    df_all = pd.DataFrame(all_experiments)
    
    df_merged = df_all.merge(
        df_existing[merge_keys],
        on=merge_keys,
        how='left',
        indicator=True
    )
    
    df_missing = df_merged[df_merged['_merge'] == 'left_only'].copy()
    df_missing = df_missing.drop('_merge', axis=1)
    
    print(f"缺失评估结果: {len(df_missing)} 个")
    print(f"  - off_peak: {len(df_missing[df_missing['scenario']=='off_peak'])}")
    print(f"  - pm_peak: {len(df_missing[df_missing['scenario']=='pm_peak'])}")
    
    if len(df_missing) == 0:
        print("\n所有实验都已有评估结果，无需重新评估")
        return
    
    # 4. 重新评估缺失的实验
    print(f"\n开始重新评估 {len(df_missing)} 个实验...")
    
    new_results = []
    failed = []
    
    for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing), desc="评估进度"):
        try:
            metrics = evaluate_single_experiment(
                scenario=row['scenario'],
                stopinfo_path=row['stopinfo_path'],
                real_stats_path=row['real_stats_path']
            )
            
            if metrics is not None:
                result = {
                    'scenario': row['scenario'],
                    'duration_hours': row['duration_hours'],
                    'scale': row['scale'],
                    'config_id': row['config_id'],
                    'seed': row['seed'],
                    **metrics
                }
                new_results.append(result)
            else:
                failed.append(row)
        
        except Exception as e:
            print(f"\n[错误] {row['scenario']} scale={row['scale']} {row['config_id']} seed={row['seed']}: {e}")
            failed.append(row)
    
    print(f"\n评估完成:")
    print(f"  - 成功: {len(new_results)}")
    print(f"  - 失败: {len(failed)}")
    
    if len(new_results) > 0:
        # 5. 合并并保存
        df_new = pd.DataFrame(new_results)
        print(f"\n新评估结果: {len(df_new)} 条")
        print(f"  - off_peak: {len(df_new[df_new['scenario']=='off_peak'])}")
        print(f"  - pm_peak: {len(df_new[df_new['scenario']=='pm_peak'])}")
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # 排序
        df_combined = df_combined.sort_values(merge_keys).reset_index(drop=True)
        
        print(f"\n合并后: {len(df_combined)} 条结果")
        print(f"  - off_peak: {len(df_combined[df_combined['scenario']=='off_peak'])}")
        print(f"  - pm_peak: {len(df_combined[df_combined['scenario']=='pm_peak'])}")
        
        # 保存
        df_combined.to_csv(results_csv, index=False)
        print(f"\n已保存到: {results_csv}")
        
        # 验证保存
        df_verify = pd.read_csv(results_csv)
        print(f"\n验证读取: {len(df_verify)} 条结果")
        print(f"  - off_peak: {len(df_verify[df_verify['scenario']=='off_peak'])}")
        print(f"  - pm_peak: {len(df_verify[df_verify['scenario']=='pm_peak'])}")
        
        print("\n现在可以运行 collect_scale_sweep_results.py 重新生成汇总统计和表格")
    else:
        print("\n[警告] 没有成功评估任何实验")
    
    if len(failed) > 0:
        print(f"\n失败的实验: {len(failed)} 个")
        for row in failed[:5]:
            print(f"  - {row['scenario']} scale={row['scale']} {row['config_id']} seed={row['seed']}")


if __name__ == "__main__":
    main()
