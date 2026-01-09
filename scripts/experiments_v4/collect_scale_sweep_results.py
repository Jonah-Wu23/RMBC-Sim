#!/usr/bin/env python3
"""
从 scale_sweep 子目录收集所有实验结果并重新生成汇总

扫描 off_peak/ 和 pm_peak/ 目录下的所有实验，
从 stopinfo.xml 重新提取评估结果，生成完整的：
- results.csv
- summary.csv
- delta.csv
- Markdown 表格
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET
from typing import Dict, Optional


# 输出路径
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"
TABLES_DIR = PROJECT_ROOT / "tables"
DIST_CSV = PROJECT_ROOT / "data2" / "processed" / "kmb_route_stop_dist.csv"

# 场景真实数据路径
REAL_DATA_PATHS = {
    "off_peak": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
    "pm_peak": PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
}


def extract_metrics_from_stopinfo(stopinfo_path: Path) -> Optional[Dict]:
    """从 stopinfo.xml 提取简化的评估指标（避免重新运行完整评估）"""
    if not stopinfo_path.exists():
        return None
    
    try:
        # 简单统计：计算停靠次数作为 n_sim
        tree = ET.parse(stopinfo_path)
        root = tree.getroot()
        stops = root.findall('.//stopinfo')
        n_sim = len(stops)
        
        # 注意：这里无法获取完整的 KS 等指标，需要从已有的 results.csv 匹配
        # 或者重新调用 metrics_v4
        return {'n_sim': n_sim}
    except Exception as e:
        print(f"  [警告] 解析 stopinfo.xml 失败: {e}")
        return None


def scan_experiment_directories(base_dir: Path) -> pd.DataFrame:
    """扫描实验目录结构，收集所有实验的元数据"""
    experiments = []
    
    for scenario_dir in base_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        
        scenario = scenario_dir.name
        
        # 遍历 duration 目录（目前只有 1h）
        for duration_dir in scenario_dir.iterdir():
            if not duration_dir.is_dir():
                continue
            
            duration_str = duration_dir.name  # "1h"
            duration_hours = int(duration_str.replace('h', ''))
            
            # 遍历 scale 目录
            for scale_dir in duration_dir.iterdir():
                if not scale_dir.is_dir():
                    continue
                
                scale_str = scale_dir.name  # "scale0.15"
                scale = float(scale_str.replace('scale', ''))
                
                # 遍历 config 目录
                for config_dir in scale_dir.iterdir():
                    if not config_dir.is_dir():
                        continue
                    
                    config_id = config_dir.name  # "A3" or "A4"
                    
                    # 遍历 seed 目录
                    for seed_dir in config_dir.iterdir():
                        if not seed_dir.is_dir():
                            continue
                        
                        seed_str = seed_dir.name  # "seed0"
                        seed = int(seed_str.replace('seed', ''))
                        
                        stopinfo_path = seed_dir / "stopinfo.xml"
                        
                        experiments.append({
                            'scenario': scenario,
                            'duration_hours': duration_hours,
                            'scale': scale,
                            'config_id': config_id,
                            'seed': seed,
                            'stopinfo_path': str(stopinfo_path),
                            'exists': stopinfo_path.exists()
                        })
    
    return pd.DataFrame(experiments)


def load_existing_results(results_path: Path) -> pd.DataFrame:
    """加载现有的 results.csv"""
    if results_path.exists():
        return pd.read_csv(results_path)
    return pd.DataFrame()


def merge_results(df_existing: pd.DataFrame, df_scanned: pd.DataFrame) -> pd.DataFrame:
    """
    合并扫描的实验元数据和现有的评估结果
    
    对于已有评估结果的实验，保留结果
    对于缺失的实验，标记为需要重新评估
    """
    if len(df_existing) == 0:
        print("[警告] 现有 results.csv 为空，需要重新评估所有实验")
        return df_scanned
    
    # 合并键
    merge_keys = ['scenario', 'duration_hours', 'scale', 'config_id', 'seed']
    
    # 确保类型一致
    for key in merge_keys:
        if key in df_existing.columns and key in df_scanned.columns:
            df_existing[key] = df_existing[key].astype(str)
            df_scanned[key] = df_scanned[key].astype(str)
    
    # 左连接：保留所有扫描到的实验
    df_merged = df_scanned.merge(
        df_existing,
        on=merge_keys,
        how='left',
        suffixes=('', '_existing')
    )
    
    # 恢复数值类型
    df_merged['duration_hours'] = df_merged['duration_hours'].astype(int)
    df_merged['scale'] = df_merged['scale'].astype(float)
    df_merged['seed'] = df_merged['seed'].astype(int)
    
    return df_merged


def compute_summary_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """计算汇总统计"""
    summary_rows = []
    
    grouped = df_results.groupby(['scenario', 'duration_hours', 'scale', 'config_id'])
    
    for (scenario, duration, scale, config_id), group in grouped:
        n = len(group)
        
        # 采样量统计
        n_events_mean = group['n_events'].mean()
        n_events_std = group['n_events'].std()
        n_sim_mean = group['n_sim'].mean()
        n_sim_std = group['n_sim'].std()
        n_clean_mean = group['n_clean'].mean()
        n_clean_std = group['n_clean'].std()
        
        # KS(speed) 统计
        ks_speed_vals = group['ks_speed'].dropna().values
        if len(ks_speed_vals) > 1:
            ks_speed_mean = np.mean(ks_speed_vals)
            ks_speed_std = np.std(ks_speed_vals, ddof=1)
            ks_speed_ci = bootstrap_ci(ks_speed_vals)
        else:
            ks_speed_mean = ks_speed_vals[0] if len(ks_speed_vals) == 1 else np.nan
            ks_speed_std = np.nan
            ks_speed_ci = (np.nan, np.nan)
        
        # KS(TT) 统计
        ks_tt_vals = group['ks_tt'].dropna().values
        if len(ks_tt_vals) > 1:
            ks_tt_mean = np.mean(ks_tt_vals)
            ks_tt_std = np.std(ks_tt_vals, ddof=1)
            ks_tt_ci = bootstrap_ci(ks_tt_vals)
        else:
            ks_tt_mean = ks_tt_vals[0] if len(ks_tt_vals) == 1 else np.nan
            ks_tt_std = np.nan
            ks_tt_ci = (np.nan, np.nan)
        
        # Pass rate
        pass_rate = group['passed'].mean() if 'passed' in group.columns else np.nan
        
        summary_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'config_id': config_id,
            'n_runs': n,
            'n_events_mean': n_events_mean,
            'n_events_std': n_events_std,
            'n_sim_mean': n_sim_mean,
            'n_sim_std': n_sim_std,
            'n_clean_mean': n_clean_mean,
            'n_clean_std': n_clean_std,
            'ks_speed_mean': ks_speed_mean,
            'ks_speed_std': ks_speed_std,
            'ks_speed_ci_low': ks_speed_ci[0],
            'ks_speed_ci_high': ks_speed_ci[1],
            'ks_tt_mean': ks_tt_mean,
            'ks_tt_std': ks_tt_std,
            'ks_tt_ci_low': ks_tt_ci[0],
            'ks_tt_ci_high': ks_tt_ci[1],
            'pass_rate': pass_rate
        })
    
    return pd.DataFrame(summary_rows)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
    """计算 bootstrap 95% CI"""
    if len(data) < 2:
        return (np.nan, np.nan)
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lower, upper)


def compute_delta_statistics(df_summary: pd.DataFrame) -> pd.DataFrame:
    """计算 ΔKS = KS(A3) - KS(A4)"""
    delta_rows = []
    
    grouped = df_summary.groupby(['scenario', 'duration_hours', 'scale'])
    
    for (scenario, duration, scale), group in grouped:
        a3_row = group[group['config_id'] == 'A3']
        a4_row = group[group['config_id'] == 'A4']
        
        if len(a3_row) == 0 or len(a4_row) == 0:
            continue
        
        a3 = a3_row.iloc[0]
        a4 = a4_row.iloc[0]
        
        # ΔKS(speed)
        delta_ks_speed = a3['ks_speed_mean'] - a4['ks_speed_mean']
        delta_ks_speed_ci_low = a3['ks_speed_ci_low'] - a4['ks_speed_ci_high']
        delta_ks_speed_ci_high = a3['ks_speed_ci_high'] - a4['ks_speed_ci_low']
        
        # ΔKS(TT)
        delta_ks_tt = a3['ks_tt_mean'] - a4['ks_tt_mean']
        delta_ks_tt_ci_low = a3['ks_tt_ci_low'] - a4['ks_tt_ci_high']
        delta_ks_tt_ci_high = a3['ks_tt_ci_high'] - a4['ks_tt_ci_low']
        
        delta_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'delta_ks_speed': delta_ks_speed,
            'delta_ks_speed_ci_low': delta_ks_speed_ci_low,
            'delta_ks_speed_ci_high': delta_ks_speed_ci_high,
            'delta_ks_tt': delta_ks_tt,
            'delta_ks_tt_ci_low': delta_ks_tt_ci_low,
            'delta_ks_tt_ci_high': delta_ks_tt_ci_high
        })
    
    return pd.DataFrame(delta_rows)


def generate_summary_markdown(df_summary: pd.DataFrame, output_path: Path) -> None:
    """生成汇总 Markdown 表格"""
    md_lines = ["# Scale Sweep Summary", ""]
    md_lines.append("每个 (scenario, duration, scale, config) 的统计汇总")
    md_lines.append("")
    
    df_display = df_summary.copy()
    
    # 格式化 CI
    df_display['ks_speed_ci'] = df_display.apply(
        lambda row: f"[{row['ks_speed_ci_low']:.4f}, {row['ks_speed_ci_high']:.4f}]" 
        if pd.notna(row['ks_speed_ci_low']) else "N/A", axis=1
    )
    df_display['ks_tt_ci'] = df_display.apply(
        lambda row: f"[{row['ks_tt_ci_low']:.4f}, {row['ks_tt_ci_high']:.4f}]"
        if pd.notna(row['ks_tt_ci_low']) else "N/A", axis=1
    )
    
    cols = ['scenario', 'scale', 'config_id', 'n_runs',
            'n_events_mean', 'n_sim_mean', 'n_clean_mean',
            'ks_speed_mean', 'ks_speed_std', 'ks_speed_ci',
            'ks_tt_mean', 'ks_tt_std', 'ks_tt_ci', 'pass_rate']
    
    df_table = df_display[cols].copy()
    df_table.columns = [
        'Scenario', 'Scale', 'Config', 'N',
        'Events', 'Sim', 'Clean',
        'KS(speed)_μ', 'KS(speed)_σ', 'KS(speed)_CI95',
        'KS(TT)_μ', 'KS(TT)_σ', 'KS(TT)_CI95', 'Pass Rate'
    ]
    
    md_lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **A3**: Audit-in-Calibration（无 L2/IES）")
    md_lines.append("- **A4**: Full-RCMDT（A3 + L2/IES）")
    md_lines.append("- **CI95**: Bootstrap 95% 置信区间")
    md_lines.append("- **Pass Rate**: KS 检验通过的比例")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"汇总表已保存: {output_path}")


def generate_delta_markdown(df_delta: pd.DataFrame, output_path: Path) -> None:
    """生成 ΔKS Markdown 表格"""
    md_lines = ["# Scale Sweep Delta (ΔKS = A3 - A4)", ""]
    md_lines.append("展示 RCMDT/IES 在不同拥堵强度下的改进效果")
    md_lines.append("")
    
    if len(df_delta) == 0:
        md_lines.append("（暂无数据）")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        return
    
    df_display = df_delta.copy()
    
    # 格式化 CI
    df_display['delta_speed_ci'] = df_display.apply(
        lambda row: f"[{row['delta_ks_speed_ci_low']:.4f}, {row['delta_ks_speed_ci_high']:.4f}]"
        if pd.notna(row['delta_ks_speed_ci_low']) else "N/A", axis=1
    )
    df_display['delta_tt_ci'] = df_display.apply(
        lambda row: f"[{row['delta_ks_tt_ci_low']:.4f}, {row['delta_ks_tt_ci_high']:.4f}]"
        if pd.notna(row['delta_ks_tt_ci_low']) else "N/A", axis=1
    )
    
    # 显著性标记
    df_display['sig_speed'] = df_display.apply(
        lambda row: "✓" if row['delta_ks_speed_ci_low'] > 0 else ("✗" if row['delta_ks_speed_ci_high'] < 0 else "~"),
        axis=1
    )
    df_display['sig_tt'] = df_display.apply(
        lambda row: "✓" if row['delta_ks_tt_ci_low'] > 0 else ("✗" if row['delta_ks_tt_ci_high'] < 0 else "~"),
        axis=1
    )
    
    cols = ['scenario', 'scale', 
            'delta_ks_speed', 'delta_speed_ci', 'sig_speed',
            'delta_ks_tt', 'delta_tt_ci', 'sig_tt']
    
    df_table = df_display[cols].copy()
    df_table.columns = [
        'Scenario', 'Scale',
        'ΔKS(speed)', 'CI95_speed', 'Sig',
        'ΔKS(TT)', 'CI95_TT', 'Sig'
    ]
    
    md_lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **ΔKS = KS(A3) - KS(A4)**")
    md_lines.append("- **正值** 表示 A4（Full-RCMDT）优于 A3（仅 Audit）")
    md_lines.append("- **Sig**: ✓=显著正, ✗=显著负, ~=不显著")
    md_lines.append("- Scale 越高 → 系统随机性越强 → RCMDT/IES 增益越显著（预期）")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Delta 表已保存: {output_path}")


def main():
    print("="*70)
    print("收集 Scale Sweep 实验结果")
    print("="*70)
    
    # 1. 加载现有结果
    print("\n加载 results.csv...")
    results_csv = OUTPUT_BASE / "results.csv"
    
    if not results_csv.exists():
        print(f"[错误] results.csv 不存在: {results_csv}")
        print("请先运行实验或重新评估")
        return
    
    df_complete = pd.read_csv(results_csv)
    print(f"加载: {len(df_complete)} 条结果")
    print(f"  - off_peak: {len(df_complete[df_complete['scenario']=='off_peak'])}")
    print(f"  - pm_peak: {len(df_complete[df_complete['scenario']=='pm_peak'])}")
    
    # 检查必需的列
    result_cols = ['n_clean', 'n_sim', 'n_events', 'ks_speed', 'ks_tt', 
                   'dcrit_speed', 'dcrit_tt', 'passed']
    missing_cols = [col for col in result_cols if col not in df_complete.columns]
    if missing_cols:
        print(f"\n[警告] 缺少列: {missing_cols}")
    
    # 检查数据完整性
    has_results = df_complete[result_cols].notna().all(axis=1)
    n_complete = has_results.sum()
    n_incomplete = len(df_complete) - n_complete
    
    if n_incomplete > 0:
        print(f"\n[警告] {n_incomplete} 条记录缺少评估结果")
        df_complete = df_complete[has_results].copy()
    
    print(f"\n有效结果: {len(df_complete)} 条")
    print(f"  - off_peak: {len(df_complete[df_complete['scenario']=='off_peak'])}")
    print(f"  - pm_peak: {len(df_complete[df_complete['scenario']=='pm_peak'])}")
    
    if len(df_complete) == 0:
        print("\n[错误] 没有找到任何有效的评估结果")
        return
    
    # 2. 计算汇总统计
    print("\n计算汇总统计...")
    df_summary = compute_summary_statistics(df_complete)
    summary_csv = OUTPUT_BASE / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"summary.csv 已保存: {summary_csv}")
    print(f"  - 汇总组数: {len(df_summary)}")
    
    # 3. 计算 Delta
    print("\n计算 Delta...")
    df_delta = compute_delta_statistics(df_summary)
    delta_csv = OUTPUT_BASE / "delta.csv"
    df_delta.to_csv(delta_csv, index=False)
    print(f"delta.csv 已保存: {delta_csv}")
    print(f"  - Delta 组数: {len(df_delta)}")
    
    # 4. 生成 Markdown 表格
    print("\n生成 Markdown 表格...")
    generate_summary_markdown(df_summary, TABLES_DIR / "scale_sweep_summary.md")
    generate_delta_markdown(df_delta, TABLES_DIR / "scale_sweep_delta.md")
    
    print("\n" + "="*70)
    print("收集完成！")
    print("="*70)
    print(f"详细结果: {results_csv} ({len(df_complete)} 条)")
    print(f"汇总统计: {summary_csv} ({len(df_summary)} 组)")
    print(f"Delta 表: {delta_csv} ({len(df_delta)} 组)")
    print(f"Markdown 表格: {TABLES_DIR}")


if __name__ == "__main__":
    main()
