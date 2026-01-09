#!/usr/bin/env python3
"""
Scale Sweep 增强评估与统计分析

P0 任务：对 scale_sweep 的 results.csv 做增强评估
- 计算：KS(speed), KS(TT) 的 full-hour + exhaustive worst-15min
- 输出：n_clean, n_sim, Dcrit, pass/fail（speed & TT 分开）
- 生成 Markdown 表格和 heatmap

输出文件：
- tables/scale_sweep_summary.md
- tables/scale_sweep_delta.md (ΔKS=A3-A4, 含 bootstrap 95% CI)
- tables/worst_window_summary.md
- figures/worst_window_heatmap_speed.png
- figures/worst_window_heatmap_tt.png
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 数据加载
# ============================================================================

def load_results(results_path: Path) -> pd.DataFrame:
    """加载 scale_sweep results.csv"""
    if not results_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_path}")
    
    df = pd.read_csv(results_path)
    print(f"加载 {len(df)} 条实验结果")
    return df


# ============================================================================
# 统计分析
# ============================================================================

def compute_summary_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算汇总统计（含 bootstrap 95% CI）
    
    对每个 (scenario, duration, scale, config) 组合计算：
    - n_events, n_sim, n_clean 均值
    - KS(speed), KS(TT) 均值/标准差/95% CI
    - Dcrit, pass rate
    - worst_window KS 均值
    """
    summary_rows = []
    
    grouped = df.groupby(['scenario', 'duration_hours', 'scale', 'config_id'])
    
    for (scenario, duration, scale, config_id), group in grouped:
        n = len(group)
        
        # 采样量统计
        n_events_mean = group['n_events'].mean()
        n_sim_mean = group['n_sim'].mean()
        n_clean_mean = group['n_clean'].mean()
        
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
        
        # Dcrit（取均值，理论上各 seed 相同）
        dcrit_speed_mean = group['dcrit_speed'].mean()
        dcrit_tt_mean = group['dcrit_tt'].mean()
        
        # Pass rate
        pass_rate = group['passed'].mean() if 'passed' in group.columns else np.nan
        
        # Worst window KS
        worst_ks_speed_vals = group['worst_window_ks'].dropna().values
        if len(worst_ks_speed_vals) > 1:
            worst_ks_mean = np.mean(worst_ks_speed_vals)
            worst_ks_std = np.std(worst_ks_speed_vals, ddof=1)
            worst_ks_ci = bootstrap_ci(worst_ks_speed_vals)
        else:
            worst_ks_mean = worst_ks_speed_vals[0] if len(worst_ks_speed_vals) == 1 else np.nan
            worst_ks_std = np.nan
            worst_ks_ci = (np.nan, np.nan)
        
        summary_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'config_id': config_id,
            'n_runs': n,
            'n_events_mean': n_events_mean,
            'n_sim_mean': n_sim_mean,
            'n_clean_mean': n_clean_mean,
            'ks_speed_mean': ks_speed_mean,
            'ks_speed_std': ks_speed_std,
            'ks_speed_ci_low': ks_speed_ci[0],
            'ks_speed_ci_high': ks_speed_ci[1],
            'dcrit_speed': dcrit_speed_mean,
            'pass_speed_rate': (group['ks_speed'] < group['dcrit_speed']).mean(),
            'ks_tt_mean': ks_tt_mean,
            'ks_tt_std': ks_tt_std,
            'ks_tt_ci_low': ks_tt_ci[0],
            'ks_tt_ci_high': ks_tt_ci[1],
            'dcrit_tt': dcrit_tt_mean,
            'pass_tt_rate': (group['ks_tt'] < group['dcrit_tt']).mean(),
            'worst_ks_mean': worst_ks_mean,
            'worst_ks_std': worst_ks_std,
            'worst_ks_ci_low': worst_ks_ci[0],
            'worst_ks_ci_high': worst_ks_ci[1],
            'pass_rate': pass_rate
        })
    
    return pd.DataFrame(summary_rows)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
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


def compute_delta_with_bootstrap(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    计算 ΔKS = KS(A3) - KS(A4)，含 bootstrap 95% CI
    
    对每个 (scenario, duration, scale) 计算 A3 与 A4 的差异
    """
    delta_rows = []
    
    grouped = df_summary.groupby(['scenario', 'duration_hours', 'scale'])
    
    for (scenario, duration, scale), group in grouped:
        a3_row = group[group['config_id'] == 'A3']
        a4_row = group[group['config_id'] == 'A4']
        
        if len(a3_row) == 0 or len(a4_row) == 0:
            continue
        
        a3 = a3_row.iloc[0]
        a4 = a4_row.iloc[0]
        
        # ΔKS(speed) = A3 - A4（正值表示 A4 更好）
        delta_ks_speed = a3['ks_speed_mean'] - a4['ks_speed_mean']
        # 保守估计 CI（使用各自 CI 的极端值）
        delta_ks_speed_ci_low = a3['ks_speed_ci_low'] - a4['ks_speed_ci_high']
        delta_ks_speed_ci_high = a3['ks_speed_ci_high'] - a4['ks_speed_ci_low']
        
        # ΔKS(TT)
        delta_ks_tt = a3['ks_tt_mean'] - a4['ks_tt_mean']
        delta_ks_tt_ci_low = a3['ks_tt_ci_low'] - a4['ks_tt_ci_high']
        delta_ks_tt_ci_high = a3['ks_tt_ci_high'] - a4['ks_tt_ci_low']
        
        # Δworst_ks
        delta_worst_ks = a3['worst_ks_mean'] - a4['worst_ks_mean']
        
        delta_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'n_runs_a3': a3['n_runs'],
            'n_runs_a4': a4['n_runs'],
            'delta_ks_speed': delta_ks_speed,
            'delta_ks_speed_ci_low': delta_ks_speed_ci_low,
            'delta_ks_speed_ci_high': delta_ks_speed_ci_high,
            'delta_ks_tt': delta_ks_tt,
            'delta_ks_tt_ci_low': delta_ks_tt_ci_low,
            'delta_ks_tt_ci_high': delta_ks_tt_ci_high,
            'delta_worst_ks': delta_worst_ks,
            'ks_speed_a3': a3['ks_speed_mean'],
            'ks_speed_a4': a4['ks_speed_mean'],
            'ks_tt_a3': a3['ks_tt_mean'],
            'ks_tt_a4': a4['ks_tt_mean']
        })
    
    return pd.DataFrame(delta_rows)


# ============================================================================
# Markdown 表格生成
# ============================================================================

def generate_summary_markdown(df_summary: pd.DataFrame, output_path: Path) -> None:
    """生成增强版汇总 Markdown 表格"""
    md_lines = ["# Scale Sweep Summary (Enhanced)", ""]
    md_lines.append("每个 (scenario, duration, scale, config) 的统计汇总")
    md_lines.append("")
    md_lines.append("## 采样量与 KS 统计")
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
            'ks_speed_mean', 'ks_speed_std', 'ks_speed_ci', 'dcrit_speed', 'pass_speed_rate',
            'ks_tt_mean', 'ks_tt_std', 'ks_tt_ci', 'dcrit_tt', 'pass_tt_rate']
    
    df_table = df_display[cols].copy()
    df_table.columns = [
        'Scenario', 'Scale', 'Config', 'N',
        'Events', 'Sim', 'Clean',
        'KS(speed)_μ', 'KS(speed)_σ', 'KS(speed)_CI95', 'Dcrit_speed', 'Pass_speed',
        'KS(TT)_μ', 'KS(TT)_σ', 'KS(TT)_CI95', 'Dcrit_TT', 'Pass_TT'
    ]
    
    md_lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **A3**: Audit-in-Calibration (无 L2/IES)")
    md_lines.append("- **A4**: Full-RCMDT (A3 + L2/IES)")
    md_lines.append("- **CI95**: Bootstrap 95% 置信区间")
    md_lines.append("- **Dcrit**: KS 临界值 (α=0.05)")
    md_lines.append("- **Pass**: KS < Dcrit 的比例")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"汇总表已保存: {output_path}")


def generate_delta_markdown(df_delta: pd.DataFrame, output_path: Path) -> None:
    """生成 ΔKS Markdown 表格（含 bootstrap 95% CI）"""
    md_lines = ["# Scale Sweep Delta (ΔKS = A3 - A4)", ""]
    md_lines.append("展示 RCMDT/IES 在不同拥堵强度下的改进效果")
    md_lines.append("")
    
    if len(df_delta) == 0:
        md_lines.append("（暂无数据：需要同时有 A3 和 A4 的结果才能计算 Delta）")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"Delta 表已保存（空）: {output_path}")
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
    
    # 添加显著性标记
    df_display['sig_speed'] = df_display.apply(
        lambda row: "✓" if row['delta_ks_speed_ci_low'] > 0 else ("✗" if row['delta_ks_speed_ci_high'] < 0 else "~"),
        axis=1
    )
    df_display['sig_tt'] = df_display.apply(
        lambda row: "✓" if row['delta_ks_tt_ci_low'] > 0 else ("✗" if row['delta_ks_tt_ci_high'] < 0 else "~"),
        axis=1
    )
    
    cols = ['scenario', 'scale', 
            'ks_speed_a3', 'ks_speed_a4', 'delta_ks_speed', 'delta_speed_ci', 'sig_speed',
            'ks_tt_a3', 'ks_tt_a4', 'delta_ks_tt', 'delta_tt_ci', 'sig_tt']
    
    df_table = df_display[cols].copy()
    df_table.columns = [
        'Scenario', 'Scale',
        'A3_speed', 'A4_speed', 'ΔKS(speed)', 'CI95_speed', 'Sig',
        'A3_TT', 'A4_TT', 'ΔKS(TT)', 'CI95_TT', 'Sig'
    ]
    
    md_lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **ΔKS = KS(A3) - KS(A4)**")
    md_lines.append("- **正值** 表示 A4 (Full-RCMDT) 优于 A3 (仅 Audit)")
    md_lines.append("- **CI95**: Bootstrap 95% 置信区间")
    md_lines.append("- **Sig**: ✓=显著正(A4更好), ✗=显著负(A3更好), ~=不显著")
    md_lines.append("- Scale 越高 → 系统随机性越强 → RCMDT/IES 增益越显著（预期）")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Delta 表已保存: {output_path}")


def generate_worst_window_markdown(df_summary: pd.DataFrame, output_path: Path) -> None:
    """生成 Worst Window 汇总表"""
    md_lines = ["# Worst Window Summary", ""]
    md_lines.append("每个 (scenario, scale, config) 的 worst-15min KS 统计")
    md_lines.append("")
    
    df_display = df_summary.copy()
    
    # 格式化 CI
    df_display['worst_ks_ci'] = df_display.apply(
        lambda row: f"[{row['worst_ks_ci_low']:.4f}, {row['worst_ks_ci_high']:.4f}]"
        if pd.notna(row['worst_ks_ci_low']) else "N/A", axis=1
    )
    
    cols = ['scenario', 'scale', 'config_id', 'n_runs',
            'worst_ks_mean', 'worst_ks_std', 'worst_ks_ci']
    
    df_table = df_display[cols].copy()
    df_table.columns = ['Scenario', 'Scale', 'Config', 'N',
                        'Worst_KS_μ', 'Worst_KS_σ', 'Worst_KS_CI95']
    
    md_lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("- **Worst_KS**: 15 分钟滑动窗口中最大的 KS 统计量")
    md_lines.append("- 该指标反映模型在最差时段的校准质量")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Worst Window 表已保存: {output_path}")


# ============================================================================
# Heatmap 生成
# ============================================================================

def generate_heatmaps(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """生成 worst-window KS vs scale heatmap"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario in df_summary['scenario'].unique():
        df_scenario = df_summary[df_summary['scenario'] == scenario]
        
        # 透视表：scale vs config
        pivot_speed = df_scenario.pivot_table(
            values='ks_speed_mean', 
            index='scale', 
            columns='config_id',
            aggfunc='mean'
        )
        
        pivot_tt = df_scenario.pivot_table(
            values='ks_tt_mean',
            index='scale',
            columns='config_id',
            aggfunc='mean'
        )
        
        pivot_worst = df_scenario.pivot_table(
            values='worst_ks_mean',
            index='scale',
            columns='config_id',
            aggfunc='mean'
        )
        
        # 绘制 KS(speed) heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.heatmap(pivot_speed, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                    ax=axes[0], vmin=0, vmax=0.5)
        axes[0].set_title(f'{scenario}: KS(speed) vs Scale')
        axes[0].set_xlabel('Config')
        axes[0].set_ylabel('Scale')
        
        sns.heatmap(pivot_tt, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    ax=axes[1], vmin=0, vmax=0.5)
        axes[1].set_title(f'{scenario}: KS(TT) vs Scale')
        axes[1].set_xlabel('Config')
        axes[1].set_ylabel('Scale')
        
        sns.heatmap(pivot_worst, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    ax=axes[2], vmin=0, vmax=1.0)
        axes[2].set_title(f'{scenario}: Worst-15min KS vs Scale')
        axes[2].set_xlabel('Config')
        axes[2].set_ylabel('Scale')
        
        plt.tight_layout()
        fig_path = output_dir / f"scale_sweep_heatmap_{scenario}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap 已保存: {fig_path}")


def generate_trend_plot(df_delta: pd.DataFrame, output_dir: Path) -> None:
    """生成 ΔKS vs Scale 趋势图"""
    if len(df_delta) == 0:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario in df_delta['scenario'].unique():
        df_scenario = df_delta[df_delta['scenario'] == scenario].sort_values('scale')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ΔKS(speed) vs Scale
        axes[0].errorbar(
            df_scenario['scale'],
            df_scenario['delta_ks_speed'],
            yerr=[
                df_scenario['delta_ks_speed'] - df_scenario['delta_ks_speed_ci_low'],
                df_scenario['delta_ks_speed_ci_high'] - df_scenario['delta_ks_speed']
            ],
            fmt='o-', capsize=5, capthick=2, label='ΔKS(speed)'
        )
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Scale (Background Traffic)')
        axes[0].set_ylabel('ΔKS(speed) = KS(A3) - KS(A4)')
        axes[0].set_title(f'{scenario}: ΔKS(speed) vs Scale')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ΔKS(TT) vs Scale
        axes[1].errorbar(
            df_scenario['scale'],
            df_scenario['delta_ks_tt'],
            yerr=[
                df_scenario['delta_ks_tt'] - df_scenario['delta_ks_tt_ci_low'],
                df_scenario['delta_ks_tt_ci_high'] - df_scenario['delta_ks_tt']
            ],
            fmt='s-', capsize=5, capthick=2, color='orange', label='ΔKS(TT)'
        )
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Scale (Background Traffic)')
        axes[1].set_ylabel('ΔKS(TT) = KS(A3) - KS(A4)')
        axes[1].set_title(f'{scenario}: ΔKS(TT) vs Scale')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / f"delta_ks_trend_{scenario}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"趋势图已保存: {fig_path}")


# ============================================================================
# README 生成
# ============================================================================

def generate_readme(output_dir: Path) -> None:
    """生成 README.md 说明文档"""
    readme_content = """# Scale Sweep 增强评估结果

## 目录结构

```
scale_sweep/
├── results.csv          # 原始实验结果（每个 run）
├── summary.csv          # 汇总统计（每个 scenario×scale×config）
├── delta.csv            # ΔKS 表（A3 vs A4）
└── README.md            # 本文件
```

## 指标说明

### 采样量指标
| 指标 | 说明 |
|------|------|
| n_events | 原始公交事件数（stopinfo 中的记录数） |
| n_sim | 仿真观测数（用于 KS 计算的仿真样本） |
| n_clean | Audit 清洗后的真实观测数 |

### KS 指标
| 指标 | 说明 |
|------|------|
| KS(speed) | 链路速度分布的 KS 统计量（full-hour） |
| KS(TT) | 旅行时间分布的 KS 统计量（full-hour） |
| Dcrit | KS 临界值（α=0.05, 双样本） |
| Pass | KS < Dcrit 的比例 |

### Worst Window 指标
| 指标 | 说明 |
|------|------|
| worst_window_ks | 15 分钟滑动窗口中最大的 KS(speed) |
| worst_window_start | 最差窗口的起始时间 |

### Delta 指标
| 指标 | 说明 |
|------|------|
| ΔKS | KS(A3) - KS(A4)，正值表示 A4 更好 |
| CI95 | Bootstrap 95% 置信区间 |
| Sig | 显著性：✓=显著正, ✗=显著负, ~=不显著 |

## 配置说明

| Config | 名称 | 描述 |
|--------|------|------|
| A3 | Audit-in-Calibration | 只在校准时使用 Audit，无 L2/IES |
| A4 | Full-RCMDT | 完整 RCMDT（A3 + L2/IES） |

## Scale 含义

Scale 参数控制背景车流量强度：
- scale=0.00: 无背景车（理想条件）
- scale=0.10: 10% 背景车（低拥堵）
- scale=0.20: 20% 背景车（中等拥堵）
- scale=0.30: 30% 背景车（高拥堵）

**预期**：Scale 越高 → 系统随机性越强 → RCMDT/IES 增益越显著

## Seed 机制

- Seed 影响 SUMO 仿真的随机性（车辆到达、换道决策等）
- Seed 不影响公交车班次（固定时刻表）
- 10 个 seeds (0-9) 提供足够的统计样本量

## 生成时间

本文件由 `analyze_scale_sweep.py` 自动生成
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README 已保存: {readme_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scale Sweep 增强评估与统计分析")
    parser.add_argument("--input", type=str, 
                        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep" / "results.csv"),
                        help="输入结果文件路径")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"),
                        help="输出目录")
    parser.add_argument("--tables", type=str,
                        default=str(PROJECT_ROOT / "tables"),
                        help="表格输出目录")
    parser.add_argument("--figures", type=str,
                        default=str(PROJECT_ROOT / "figures"),
                        help="图表输出目录")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="不生成 heatmap")
    
    args = parser.parse_args()
    
    results_path = Path(args.input)
    output_dir = Path(args.output)
    tables_dir = Path(args.tables)
    figures_dir = Path(args.figures)
    
    print("="*70)
    print("Scale Sweep 增强评估与统计分析")
    print("="*70)
    
    # 加载数据
    df_results = load_results(results_path)
    
    # 计算汇总统计
    print("\n计算汇总统计...")
    df_summary = compute_summary_with_ci(df_results)
    
    # 保存 summary.csv
    summary_csv = output_dir / "summary_enhanced.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"增强汇总已保存: {summary_csv}")
    
    # 计算 Delta
    print("\n计算 Delta...")
    df_delta = compute_delta_with_bootstrap(df_summary)
    
    # 保存 delta.csv
    delta_csv = output_dir / "delta_enhanced.csv"
    df_delta.to_csv(delta_csv, index=False)
    print(f"增强 Delta 已保存: {delta_csv}")
    
    # 生成 Markdown 表格
    print("\n生成 Markdown 表格...")
    generate_summary_markdown(df_summary, tables_dir / "scale_sweep_summary.md")
    generate_delta_markdown(df_delta, tables_dir / "scale_sweep_delta.md")
    generate_worst_window_markdown(df_summary, tables_dir / "worst_window_summary.md")
    
    # 生成 README
    generate_readme(output_dir)
    
    # 生成 Heatmap
    if not args.no_heatmap:
        print("\n生成 Heatmap...")
        generate_heatmaps(df_summary, figures_dir)
        generate_trend_plot(df_delta, figures_dir)
    
    print("\n" + "="*70)
    print("增强评估完成！")
    print("="*70)
    print(f"数据目录: {output_dir}")
    print(f"表格目录: {tables_dir}")
    print(f"图表目录: {figures_dir}")


if __name__ == "__main__":
    main()
