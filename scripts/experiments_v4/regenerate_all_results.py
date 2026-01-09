#!/usr/bin/env python3
"""
完整重新生成所有 scale_sweep 实验结果

从 stopinfo.xml 文件重新评估所有实验，确保数据完整
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from scripts.eval.metrics_v4 import (
    compute_metrics_v4, 
    load_real_link_stats,
    AuditConfig
)


# 配置
REAL_DATA_PATHS = {
    "off_peak": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
    "pm_peak": PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
}
DIST_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"

# 实验参数
SCENARIOS = ["off_peak", "pm_peak"]
SCALES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
CONFIGS = ["A3", "A4"]
SEEDS = list(range(10))


def main():
    print("="*70)
    print("完整重新生成 Scale Sweep 实验结果")
    print("="*70)
    
    results = []
    total = len(SCENARIOS) * len(SCALES) * len(CONFIGS) * len(SEEDS)
    
    print(f"\n需要评估 {total} 个实验")
    print(f"  - 场景: {SCENARIOS}")
    print(f"  - Scales: {SCALES}")
    print(f"  - Configs: {CONFIGS}")
    print(f"  - Seeds: {SEEDS}")
    print()
    
    count = 0
    failed = []
    
    for scenario in SCENARIOS:
        # 加载真实数据
        real_stats_path = REAL_DATA_PATHS[scenario]
        if not real_stats_path.exists():
            print(f"[错误] 真实数据不存在: {real_stats_path}")
            continue
        
        df_real = load_real_link_stats(str(real_stats_path))
        
        for scale in SCALES:
            for config_id in CONFIGS:
                for seed in SEEDS:
                    count += 1
                    
                    # 构建 stopinfo.xml 路径
                    stopinfo_path = (OUTPUT_BASE / scenario / "1h" / 
                                   f"scale{scale:.2f}" / config_id / 
                                   f"seed{seed}" / "stopinfo.xml")
                    
                    if not stopinfo_path.exists():
                        print(f"[{count}/{total}] 缺失: {scenario} scale={scale} {config_id} seed={seed}")
                        failed.append((scenario, scale, config_id, seed, "stopinfo缺失"))
                        continue
                    
                    try:
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
                            print(f"[{count}/{total}] 评估失败: {scenario} scale={scale} {config_id} seed={seed}")
                            failed.append((scenario, scale, config_id, seed, "评估返回None"))
                            continue
                        
                        # 记录结果
                        results.append({
                            'scenario': scenario,
                            'duration_hours': 1,
                            'scale': scale,
                            'config_id': config_id,
                            'seed': seed,
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
                        })
                        
                        if count % 20 == 0:
                            print(f"[{count}/{total}] 已完成 {len(results)} 个")
                    
                    except Exception as e:
                        print(f"[{count}/{total}] 异常: {scenario} scale={scale} {config_id} seed={seed}: {e}")
                        failed.append((scenario, scale, config_id, seed, str(e)))
    
    # 保存结果
    print(f"\n评估完成:")
    print(f"  - 成功: {len(results)}")
    print(f"  - 失败: {len(failed)}")
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # 排序
        df = df.sort_values(['scenario', 'scale', 'config_id', 'seed']).reset_index(drop=True)
        
        # 保存
        results_csv = OUTPUT_BASE / "results.csv"
        df.to_csv(results_csv, index=False)
        
        print(f"\n保存完成: {results_csv}")
        print(f"总计: {len(df)} 条结果")
        print(f"  - off_peak: {len(df[df['scenario']=='off_peak'])}")
        print(f"  - pm_peak: {len(df[df['scenario']=='pm_peak'])}")
        
        # 验证保存
        df_verify = pd.read_csv(results_csv)
        print(f"\n验证读取: {len(df_verify)} 条")
        print(f"  - off_peak: {len(df_verify[df_verify['scenario']=='off_peak'])}")
        print(f"  - pm_peak: {len(df_verify[df_verify['scenario']=='pm_peak'])}")
        
        if len(df_verify) == len(df):
            print("\n✓ 文件保存和读取验证成功")
        else:
            print(f"\n✗ 警告: 保存前 {len(df)} 条，读取后 {len(df_verify)} 条")
    
    else:
        print("\n[错误] 没有成功评估任何实验")
    
    if len(failed) > 0:
        print(f"\n失败的实验: {len(failed)} 个")
        for scenario, scale, config_id, seed, reason in failed[:10]:
            print(f"  - {scenario} scale={scale} {config_id} seed={seed}: {reason}")


if __name__ == "__main__":
    main()
