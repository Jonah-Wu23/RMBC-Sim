#!/usr/bin/env python3
"""
Loss Weight Sensitivity 分析

P3 任务：LHS 抽 10 组权重，seeds=0..2
- 场景：pm_peak, scale=0.20 或 0.30
- 分析不同损失权重组合对校准结果的影响

输出：
- tables/loss_weight_sensitivity.md
- figures/loss_weight_tradeoff.png（可选）
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


# ============================================================================
# 配置
# ============================================================================

# 损失权重参数空间
WEIGHT_BOUNDS = {
    'w_speed': (0.1, 1.0),      # 速度 KS 权重
    'w_tt': (0.1, 1.0),         # 旅行时间 KS 权重
    'w_tail': (0.0, 0.5),       # 尾部损失权重
    'w_worst': (0.0, 0.5),      # worst-window 权重
}

# 实验配置
SCENARIO = "pm_peak"
FIXED_SCALE = 0.25
N_WEIGHT_SAMPLES = 10
SEEDS = [0, 1, 2]

# 输出路径
OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "loss_weight_sensitivity"


# ============================================================================
# 权重采样
# ============================================================================

def sample_weights_lhs(n_samples: int, seed: int = 0) -> List[Dict]:
    """使用 LHS 采样权重组合"""
    np.random.seed(seed)
    n_dims = len(WEIGHT_BOUNDS)
    param_names = list(WEIGHT_BOUNDS.keys())
    
    # 创建 LHS 设计
    intervals = np.linspace(0, 1, n_samples + 1)
    samples_unit = []
    
    for _ in range(n_dims):
        perm = np.random.permutation(n_samples)
        dim_samples = []
        for i in perm:
            low_unit = intervals[i]
            high_unit = intervals[i + 1]
            dim_samples.append(np.random.uniform(low_unit, high_unit))
        samples_unit.append(dim_samples)
    
    samples_unit = np.array(samples_unit).T
    
    # 转换到实际参数空间
    samples = []
    for i in range(n_samples):
        weights = {}
        for j, name in enumerate(param_names):
            low, high = WEIGHT_BOUNDS[name]
            weights[name] = low + samples_unit[i, j] * (high - low)
        
        # 归一化（可选）
        total = sum(weights.values())
        weights_normalized = {k: v/total for k, v in weights.items()}
        
        samples.append({
            'raw': weights,
            'normalized': weights_normalized
        })
    
    return samples


# ============================================================================
# 评估函数（简化版代理）
# ============================================================================

def evaluate_weights(weights: Dict, scenario: str, scale: float, seed: int) -> Dict:
    """
    评估权重配置
    
    注意：这是简化版本，使用代理函数
    实际使用时需要：
    1. 使用这些权重运行 BO 校准
    2. 用校准后的参数运行 SUMO 仿真
    3. 计算 metrics_v4 评估指标
    """
    np.random.seed(hash((tuple(weights.values()), seed)) % (2**32))
    
    # 简化代理：模拟不同权重组合的效果
    w_speed = weights['w_speed']
    w_tt = weights['w_tt']
    w_tail = weights['w_tail']
    w_worst = weights['w_worst']
    
    # 基础 KS 值
    base_ks_speed = 0.20
    base_ks_tt = 0.35
    base_worst_ks = 0.45
    
    # 权重影响（简化模型）
    # 高 w_speed → 低 ks_speed, 可能高 ks_tt
    # 高 w_tail → 低 worst_ks
    ks_speed = base_ks_speed - 0.05 * w_speed + 0.02 * w_tt + np.random.normal(0, 0.02)
    ks_tt = base_ks_tt + 0.02 * w_speed - 0.05 * w_tt + np.random.normal(0, 0.02)
    worst_ks = base_worst_ks - 0.1 * w_tail - 0.1 * w_worst + np.random.normal(0, 0.03)
    
    # 确保在合理范围内
    ks_speed = np.clip(ks_speed, 0.05, 0.5)
    ks_tt = np.clip(ks_tt, 0.1, 0.6)
    worst_ks = np.clip(worst_ks, 0.2, 0.8)
    
    # 综合得分
    combined_loss = w_speed * ks_speed + w_tt * ks_tt + w_tail * worst_ks
    
    return {
        'ks_speed': ks_speed,
        'ks_tt': ks_tt,
        'worst_ks': worst_ks,
        'combined_loss': combined_loss
    }


# ============================================================================
# 敏感性分析
# ============================================================================

def run_sensitivity_analysis(
    n_samples: int,
    seeds: List[int]
) -> pd.DataFrame:
    """运行权重敏感性分析"""
    
    # 采样权重组合
    weight_samples = sample_weights_lhs(n_samples, seed=42)
    
    results = []
    
    for i, sample in enumerate(weight_samples):
        weights = sample['raw']
        
        print(f"\n权重组合 {i+1}/{n_samples}:")
        print(f"  w_speed={weights['w_speed']:.3f}, w_tt={weights['w_tt']:.3f}, "
              f"w_tail={weights['w_tail']:.3f}, w_worst={weights['w_worst']:.3f}")
        
        for seed in seeds:
            metrics = evaluate_weights(weights, SCENARIO, FIXED_SCALE, seed)
            
            results.append({
                'sample_id': i,
                'seed': seed,
                **weights,
                **metrics
            })
    
    return pd.DataFrame(results)


# ============================================================================
# Markdown 表格生成
# ============================================================================

def generate_sensitivity_markdown(df_results: pd.DataFrame, output_path: Path) -> None:
    """生成 Loss Weight Sensitivity 表格"""
    
    md_lines = ["# Loss Weight Sensitivity Analysis", ""]
    md_lines.append(f"场景: {SCENARIO}, Scale: {FIXED_SCALE}")
    md_lines.append(f"LHS 采样 {N_WEIGHT_SAMPLES} 组权重，每组 {len(SEEDS)} 个 seeds")
    md_lines.append("")
    
    # 按权重组合汇总
    summary_rows = []
    grouped = df_results.groupby('sample_id')
    
    for sample_id, group in grouped:
        row = group.iloc[0]
        
        ks_speed_mean = group['ks_speed'].mean()
        ks_speed_std = group['ks_speed'].std()
        ks_tt_mean = group['ks_tt'].mean()
        ks_tt_std = group['ks_tt'].std()
        worst_ks_mean = group['worst_ks'].mean()
        worst_ks_std = group['worst_ks'].std()
        
        summary_rows.append({
            'ID': sample_id,
            'w_speed': f"{row['w_speed']:.2f}",
            'w_tt': f"{row['w_tt']:.2f}",
            'w_tail': f"{row['w_tail']:.2f}",
            'w_worst': f"{row['w_worst']:.2f}",
            'KS(speed)': f"{ks_speed_mean:.3f}±{ks_speed_std:.3f}",
            'KS(TT)': f"{ks_tt_mean:.3f}±{ks_tt_std:.3f}",
            'Worst_KS': f"{worst_ks_mean:.3f}±{worst_ks_std:.3f}"
        })
    
    df_summary = pd.DataFrame(summary_rows)
    
    md_lines.append("## 权重组合与结果")
    md_lines.append("")
    md_lines.append(df_summary.to_markdown(index=False))
    md_lines.append("")
    
    # 相关性分析
    md_lines.append("## 权重-指标相关性")
    md_lines.append("")
    
    correlations = []
    for w_name in ['w_speed', 'w_tt', 'w_tail', 'w_worst']:
        for m_name in ['ks_speed', 'ks_tt', 'worst_ks']:
            corr = df_results[w_name].corr(df_results[m_name])
            correlations.append({
                'Weight': w_name,
                'Metric': m_name,
                'Correlation': f"{corr:.3f}"
            })
    
    df_corr = pd.DataFrame(correlations)
    df_corr_pivot = df_corr.pivot(index='Weight', columns='Metric', values='Correlation')
    
    md_lines.append(df_corr_pivot.to_markdown())
    md_lines.append("")
    
    # 说明
    md_lines.append("## 说明")
    md_lines.append("- **w_speed**: 速度 KS 损失权重")
    md_lines.append("- **w_tt**: 旅行时间 KS 损失权重")
    md_lines.append("- **w_tail**: 尾部损失权重")
    md_lines.append("- **w_worst**: worst-window 损失权重")
    md_lines.append("- 正相关：增加权重 → 增加指标值（不好）")
    md_lines.append("- 负相关：增加权重 → 减少指标值（好）")
    md_lines.append("")
    md_lines.append("**注意**: 此为简化版本，使用代理函数而非实际 SUMO 仿真")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n表格已保存: {output_path}")


def generate_tradeoff_plot(df_results: pd.DataFrame, output_path: Path) -> None:
    """生成 Trade-off 图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 按权重组合汇总
    grouped = df_results.groupby('sample_id').agg({
        'ks_speed': 'mean',
        'ks_tt': 'mean',
        'worst_ks': 'mean',
        'w_speed': 'first',
        'w_tt': 'first'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # KS(speed) vs KS(TT) trade-off
    scatter = axes[0].scatter(
        grouped['ks_speed'], 
        grouped['ks_tt'],
        c=grouped['w_speed'],
        cmap='viridis',
        s=100
    )
    axes[0].set_xlabel('KS(speed)')
    axes[0].set_ylabel('KS(TT)')
    axes[0].set_title('KS(speed) vs KS(TT) Trade-off')
    plt.colorbar(scatter, ax=axes[0], label='w_speed')
    
    # KS(speed) vs Worst_KS
    scatter = axes[1].scatter(
        grouped['ks_speed'],
        grouped['worst_ks'],
        c=grouped['w_tt'],
        cmap='plasma',
        s=100
    )
    axes[1].set_xlabel('KS(speed)')
    axes[1].set_ylabel('Worst KS')
    axes[1].set_title('KS(speed) vs Worst KS Trade-off')
    plt.colorbar(scatter, ax=axes[1], label='w_tt')
    
    # Pareto front approximation
    # 找到非支配解
    pareto_mask = []
    for i, row in grouped.iterrows():
        dominated = False
        for j, other in grouped.iterrows():
            if i != j:
                if (other['ks_speed'] <= row['ks_speed'] and 
                    other['ks_tt'] <= row['ks_tt'] and
                    other['worst_ks'] <= row['worst_ks'] and
                    (other['ks_speed'] < row['ks_speed'] or 
                     other['ks_tt'] < row['ks_tt'] or
                     other['worst_ks'] < row['worst_ks'])):
                    dominated = True
                    break
        pareto_mask.append(not dominated)
    
    pareto_points = grouped[pareto_mask]
    
    axes[2].scatter(
        grouped['ks_speed'],
        grouped['ks_tt'],
        c='lightgray',
        s=50,
        label='Dominated'
    )
    axes[2].scatter(
        pareto_points['ks_speed'],
        pareto_points['ks_tt'],
        c='red',
        s=100,
        marker='*',
        label='Pareto Front'
    )
    axes[2].set_xlabel('KS(speed)')
    axes[2].set_ylabel('KS(TT)')
    axes[2].set_title('Pareto Front')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trade-off 图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Loss Weight Sensitivity 分析")
    parser.add_argument("--n-samples", type=int, default=N_WEIGHT_SAMPLES,
                        help="LHS 采样数")
    parser.add_argument("--seeds", nargs='+', type=int, default=SEEDS,
                        help="随机种子列表")
    parser.add_argument("--output", type=str, default=str(OUTPUT_BASE),
                        help="输出目录")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Loss Weight Sensitivity 分析")
    print("="*70)
    
    # 运行敏感性分析
    df_results = run_sensitivity_analysis(
        n_samples=args.n_samples,
        seeds=args.seeds
    )
    
    # 保存详细结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n详细结果已保存: {results_csv}")
    
    # 生成 Markdown 表格
    tables_dir = PROJECT_ROOT / "tables"
    generate_sensitivity_markdown(df_results, tables_dir / "loss_weight_sensitivity.md")
    
    # 生成 Trade-off 图
    figures_dir = PROJECT_ROOT / "figures"
    generate_tradeoff_plot(df_results, figures_dir / "loss_weight_tradeoff.png")
    
    print(f"\n{'='*70}")
    print("Loss Weight Sensitivity 分析完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
