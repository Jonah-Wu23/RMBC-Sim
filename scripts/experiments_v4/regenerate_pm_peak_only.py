#!/usr/bin/env python3
"""只重新生成 pm_peak 的评估结果（off_peak 有问题暂时跳过）"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.eval.metrics_v4 import compute_metrics_v4, load_real_link_stats, AuditConfig

# 配置
REAL_DATA_PM = PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
DIST_CSV = PROJECT_ROOT / "data2" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"

SCALES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
CONFIGS = ["A3", "A4"]
SEEDS = list(range(10))

def main():
    print("="*70)
    print("重新生成 PM Peak 评估结果")
    print("="*70)
    
    # 加载真实数据
    df_real = load_real_link_stats(str(REAL_DATA_PM))
    print(f"\n真实数据: {len(df_real)} 条")
    
    results = []
    total = len(SCALES) * len(CONFIGS) * len(SEEDS)
    
    print(f"需要评估 {total} 个 pm_peak 实验\n")
    
    count = 0
    for scale in SCALES:
        for config_id in CONFIGS:
            for seed in SEEDS:
                count += 1
                
                stopinfo_path = (OUTPUT_BASE / "pm_peak" / "1h" / 
                               f"scale{scale:.2f}" / config_id / 
                               f"seed{seed}" / "stopinfo.xml")
                
                if not stopinfo_path.exists():
                    print(f"[{count}/{total}] 缺失: scale={scale} {config_id} seed={seed}")
                    continue
                
                try:
                    audit_config = AuditConfig.from_protocol()
                    result = compute_metrics_v4(
                        real_data=df_real,
                        sim_data=str(stopinfo_path),
                        dist_file=str(DIST_CSV),
                        audit_config=audit_config,
                        scenario="pm_peak",
                        route="68X"
                    )
                    
                    if result:
                        results.append({
                            'scenario': 'pm_peak',
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
                    print(f"[{count}/{total}] 错误: scale={scale} {config_id} seed={seed}: {e}")
    
    print(f"\n评估完成: {len(results)} 个成功")
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        df = df.sort_values(['scale', 'config_id', 'seed']).reset_index(drop=True)
        
        results_csv = OUTPUT_BASE / "results.csv"
        df.to_csv(results_csv, index=False)
        
        print(f"\n保存: {results_csv}")
        print(f"总计: {len(df)} 条")
        
        # 验证
        df_verify = pd.read_csv(results_csv)
        print(f"验证: {len(df_verify)} 条")
        
        if len(df_verify) == len(df):
            print("✓ 文件保存验证成功")
            
            # 立即生成汇总
            print("\n现在运行 collect_scale_sweep_results.py")
            import subprocess
            subprocess.run([
                "python", "scripts/experiments_v4/collect_scale_sweep_results.py"
            ], cwd=str(PROJECT_ROOT))
        else:
            print(f"✗ 警告: 保存 {len(df)} 条，读取 {len(df_verify)} 条")
    else:
        print("\n[错误] 没有成功评估任何实验")

if __name__ == "__main__":
    main()
