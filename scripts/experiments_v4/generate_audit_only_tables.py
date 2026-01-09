#!/usr/bin/env python3
"""
T0: 将现有 Scale Sweep 结果重新定义为 Audit-only Sensitivity
生成 flagged_fraction vs scale + KS(clean) vs scale 表格

注意：这些结果来自 A3/A4_fixed_params 配置（无真正 IES）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% 置信区间"""
    np.random.seed(seed)
    if len(data) == 0:
        return np.nan, np.nan
    samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = np.mean(samples, axis=1)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    results_csv = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep" / "results.csv"
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("T0: 生成 Audit-only Sensitivity 表格")
    print("="*70)
    
    df = pd.read_csv(results_csv)
    print(f"\n加载 {len(df)} 条结果")
    print(f"  - off_peak: {len(df[df['scenario']=='off_peak'])}")
    print(f"  - pm_peak: {len(df[df['scenario']=='pm_peak'])}")
    
    # 由于 A3 和 A4 结果相同（均为固定参数），合并处理
    # 按 scenario, scale 分组统计
    
    summary_rows = []
    
    for scenario in ["off_peak", "pm_peak"]:
        df_s = df[df['scenario'] == scenario]
        
        for scale in sorted(df_s['scale'].unique()):
            df_scale = df_s[df_s['scale'] == scale]
            
            # 取所有 seeds 的数据（A3 和 A4 相同，只取一个即可）
            df_a3 = df_scale[df_scale['config_id'] == 'A3']
            
            n_seeds = len(df_a3)
            
            # 采样量（应该每个 seed 都相同）
            n_events = df_a3['n_events'].mean()
            n_clean = df_a3['n_clean'].mean()
            n_sim = df_a3['n_sim'].mean()
            
            # flagged_fraction
            flagged_frac_mean = df_a3['flagged_fraction'].mean()
            flagged_frac_std = df_a3['flagged_fraction'].std()
            
            # KS(speed) clean
            ks_speed_mean = df_a3['ks_speed'].mean()
            ks_speed_std = df_a3['ks_speed'].std()
            ks_speed_ci = bootstrap_ci(df_a3['ks_speed'].dropna().values)
            
            # KS(TT) clean
            ks_tt_mean = df_a3['ks_tt'].mean()
            ks_tt_std = df_a3['ks_tt'].std()
            ks_tt_ci = bootstrap_ci(df_a3['ks_tt'].dropna().values)
            
            # Dcrit（取平均）
            dcrit_speed = df_a3['dcrit_speed'].mean()
            dcrit_tt = df_a3['dcrit_tt'].mean()
            
            # Pass rate
            pass_rate = df_a3['passed'].mean() if 'passed' in df_a3.columns else 0.0
            
            # worst-window（如果有）
            worst_ks = df_a3['worst_window_ks'].mean() if 'worst_window_ks' in df_a3.columns else np.nan
            
            summary_rows.append({
                'scenario': scenario,
                'scale': scale,
                'n_seeds': n_seeds,
                'n_events': n_events,
                'n_clean': n_clean,
                'n_sim': n_sim,
                'flagged_fraction': flagged_frac_mean,
                'flagged_fraction_std': flagged_frac_std,
                'ks_speed_mean': ks_speed_mean,
                'ks_speed_std': ks_speed_std,
                'ks_speed_ci_lower': ks_speed_ci[0],
                'ks_speed_ci_upper': ks_speed_ci[1],
                'ks_tt_mean': ks_tt_mean,
                'ks_tt_std': ks_tt_std,
                'ks_tt_ci_lower': ks_tt_ci[0],
                'ks_tt_ci_upper': ks_tt_ci[1],
                'dcrit_speed': dcrit_speed,
                'dcrit_tt': dcrit_tt,
                'pass_rate': pass_rate,
                'worst_window_ks': worst_ks
            })
    
    df_summary = pd.DataFrame(summary_rows)
    
    # 保存 CSV
    summary_csv = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep" / "audit_only_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n保存: {summary_csv}")
    
    # 生成 Markdown 表格
    md_lines = []
    md_lines.append("# Scale Sweep: Audit-only Sensitivity")
    md_lines.append("")
    md_lines.append("> **注意**: 本表格来自固定参数对比（A3/A4_fixed_params），**不包含 L2/IES 校准**。")
    md_lines.append("> A3 和 A4 使用了略微不同的 SUMO 参数，但未调用 IESLoop，因此结果近乎相同。")
    md_lines.append("> 本数据仅用于展示 **Audit Rule C 的 flagged_fraction 稳定性** 与 **KS(clean) vs scale 趋势**。")
    md_lines.append("")
    
    # 表格 1: flagged_fraction vs scale
    md_lines.append("## 1. Flagged Fraction vs Scale")
    md_lines.append("")
    md_lines.append("| Scenario | Scale | n_events | n_clean | flagged_frac | ±σ |")
    md_lines.append("|:---------|------:|--------:|--------:|-------------:|---:|")
    
    for _, row in df_summary.iterrows():
        md_lines.append(
            f"| {row['scenario']} | {row['scale']:.2f} | "
            f"{row['n_events']:.0f} | {row['n_clean']:.0f} | "
            f"{row['flagged_fraction']:.4f} | {row['flagged_fraction_std']:.4f} |"
        )
    
    md_lines.append("")
    
    # 表格 2: KS(clean) vs scale
    md_lines.append("## 2. KS Statistics vs Scale")
    md_lines.append("")
    md_lines.append("| Scenario | Scale | n_sim | KS(speed)_μ | ±σ | CI95 | D_crit | Pass? |")
    md_lines.append("|:---------|------:|------:|------------:|---:|:-----|-------:|:------|")
    
    for _, row in df_summary.iterrows():
        ci_str = f"[{row['ks_speed_ci_lower']:.3f}, {row['ks_speed_ci_upper']:.3f}]"
        pass_str = "✓" if row['pass_rate'] > 0.5 else "✗"
        md_lines.append(
            f"| {row['scenario']} | {row['scale']:.2f} | "
            f"{row['n_sim']:.0f} | {row['ks_speed_mean']:.4f} | {row['ks_speed_std']:.4f} | "
            f"{ci_str} | {row['dcrit_speed']:.4f} | {pass_str} |"
        )
    
    md_lines.append("")
    
    # 表格 3: KS(TT) vs scale
    md_lines.append("## 3. KS(TT) Statistics vs Scale")
    md_lines.append("")
    md_lines.append("| Scenario | Scale | KS(TT)_μ | ±σ | CI95 | D_crit | Pass? |")
    md_lines.append("|:---------|------:|---------:|---:|:-----|-------:|:------|")
    
    for _, row in df_summary.iterrows():
        ci_str = f"[{row['ks_tt_ci_lower']:.3f}, {row['ks_tt_ci_upper']:.3f}]"
        pass_str = "✓" if row['pass_rate'] > 0.5 else "✗"
        md_lines.append(
            f"| {row['scenario']} | {row['scale']:.2f} | "
            f"{row['ks_tt_mean']:.4f} | {row['ks_tt_std']:.4f} | "
            f"{ci_str} | {row['dcrit_tt']:.4f} | {pass_str} |"
        )
    
    md_lines.append("")
    
    # 表格 4: 趋势总结
    md_lines.append("## 4. 趋势观察")
    md_lines.append("")
    md_lines.append("### Off-Peak")
    off_peak = df_summary[df_summary['scenario'] == 'off_peak']
    if len(off_peak) > 0:
        ks_range = f"{off_peak['ks_speed_mean'].min():.4f} - {off_peak['ks_speed_mean'].max():.4f}"
        flag_range = f"{off_peak['flagged_fraction'].min():.2%} - {off_peak['flagged_fraction'].max():.2%}"
        md_lines.append(f"- **KS(speed) 范围**: {ks_range}")
        md_lines.append(f"- **flagged_fraction 范围**: {flag_range}")
        md_lines.append(f"- **趋势**: KS 随 scale 增加略有{'下降' if off_peak['ks_speed_mean'].iloc[-1] < off_peak['ks_speed_mean'].iloc[0] else '上升'}")
    
    md_lines.append("")
    md_lines.append("### PM Peak")
    pm_peak = df_summary[df_summary['scenario'] == 'pm_peak']
    if len(pm_peak) > 0:
        ks_range = f"{pm_peak['ks_speed_mean'].min():.4f} - {pm_peak['ks_speed_mean'].max():.4f}"
        flag_range = f"{pm_peak['flagged_fraction'].min():.2%} - {pm_peak['flagged_fraction'].max():.2%}"
        md_lines.append(f"- **KS(speed) 范围**: {ks_range}")
        md_lines.append(f"- **flagged_fraction 范围**: {flag_range}")
        md_lines.append(f"- **趋势**: KS 随 scale 增加略有{'下降' if pm_peak['ks_speed_mean'].iloc[-1] < pm_peak['ks_speed_mean'].iloc[0] else '上升'}")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*生成时间: 自动生成*")
    
    # 保存 Markdown
    md_path = tables_dir / "scale_sweep_audit_only.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"保存: {md_path}")
    print("\nT0 完成！")


if __name__ == "__main__":
    main()
