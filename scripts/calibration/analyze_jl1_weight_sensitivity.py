"""
analyze_jl1_weight_sensitivity.py
==================================
JL1 权重敏感性分析（离线重算）

功能：
1. 从 recalculate_jl1_from_b2.py 的输出读取基线 JL1 分项
2. 构建 3 组权重配置：
   - Balanced (原始): α=1.0, λ=0.5, β=0.3
   - Mean-Heavy (RMSE优先): α=0.6, λ=0.5, β=0.18 (α×0.6, β×0.6, 保持相对比例)
   - Tail-Heavy (鲁棒性优先): α=1.0, λ=0.5, β=0.6 (β×2)
3. 离线重算每个候选的 JL1 值（利用已有分项）
4. 报告：
   - 原始最优候选在其他权重下的排名
   - Top-5 候选重叠情况
   - 一句话结论

作者：RCMDT Project
日期：2026-01-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 输入输出路径
INPUT_CSV = PROJECT_ROOT / "data" / "calibration" / "B2_jl1_recalculated.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration"
OUTPUT_TABLE = OUTPUT_DIR / "jl1_weight_sensitivity_table.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "jl1_weight_sensitivity_summary.txt"

# 3 组权重配置
WEIGHT_CONFIGS = {
    'balanced': {
        'name': 'Balanced (原始论文)',
        'alpha': 1.0,
        'lambda_std': 0.5,
        'beta': 0.3,
        'desc': '论文默认配置，平衡 RMSE、分散度、尾部风险'
    },
    'mean_heavy': {
        'name': 'Mean-Heavy (RMSE优先)',
        'alpha': 0.6,
        'lambda_std': 0.5,
        'beta': 0.18,
        'desc': 'RMSE 主导，降低 α 和 β 使复合项权重减小'
    },
    'tail_heavy': {
        'name': 'Tail-Heavy (鲁棒性优先)',
        'alpha': 1.0,
        'lambda_std': 0.5,
        'beta': 0.6,
        'desc': '尾部风险加倍，强调极端情况鲁棒性'
    }
}


def recalculate_jl1_with_weights(
    df: pd.DataFrame,
    alpha: float,
    lambda_std: float,
    beta: float
) -> pd.Series:
    """
    利用已有分项离线重算 JL1
    
    JL1 = RMSE + α(MAE + λ·std) + β·P90
        = rmse_term + α·dispersion_term + β·tail_term
    
    注意：dispersion_term 已经包含了 λ，所以这里直接用
    
    Args:
        df: 包含 rmse_term, dispersion_term, tail_term 的 DataFrame
        alpha: MAE+dispersion 项权重
        lambda_std: 未使用（因为 dispersion_term 已包含）
        beta: 尾部风险权重
        
    Returns:
        重算后的 JL1 Series
    """
    # 需要重新计算 dispersion_term，因为原始的 lambda_std 可能不同
    # dispersion_term = mae_term + lambda_std * std_term
    dispersion_new = df['mae_term'] + lambda_std * df['std_term']
    
    jl1_new = df['rmse_term'] + alpha * dispersion_new + beta * df['tail_term']
    
    return jl1_new


def analyze_sensitivity() -> Tuple[pd.DataFrame, Dict]:
    """
    执行敏感性分析
    
    Returns:
        (结果表, 统计摘要字典)
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到重算结果: {INPUT_CSV}\n"
            f"请先运行: python scripts/calibration/recalculate_jl1_from_b2.py"
        )
    
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] 读取 {len(df)} 个候选")
    
    # 过滤有效样本
    df_valid = df[df['jl1'].notna()].copy()
    print(f"[INFO] 有效样本: {len(df_valid)}/{len(df)}")
    
    if len(df_valid) == 0:
        raise ValueError("没有有效样本，无法进行敏感性分析")
    
    # 为每组权重重算 JL1
    results = df_valid[['iter', 'type']].copy()
    
    for config_key, config in WEIGHT_CONFIGS.items():
        jl1_new = recalculate_jl1_with_weights(
            df_valid,
            alpha=config['alpha'],
            lambda_std=config['lambda_std'],
            beta=config['beta']
        )
        results[f'jl1_{config_key}'] = jl1_new
        results[f'rank_{config_key}'] = jl1_new.rank(method='min').astype(int)
    
    # 添加原始分项（用于参考）
    results['rmse_term'] = df_valid['rmse_term'].values
    results['mae_term'] = df_valid['mae_term'].values
    results['std_term'] = df_valid['std_term'].values
    results['tail_term'] = df_valid['tail_term'].values
    
    # 分析统计
    stats = {}
    
    for config_key, config in WEIGHT_CONFIGS.items():
        jl1_col = f'jl1_{config_key}'
        rank_col = f'rank_{config_key}'
        
        best_idx = results[jl1_col].idxmin()
        best_iter = results.loc[best_idx, 'iter']
        best_jl1 = results.loc[best_idx, jl1_col]
        
        stats[config_key] = {
            'name': config['name'],
            'best_iter': int(best_iter),
            'best_jl1': float(best_jl1),
            'top5_iters': results.nsmallest(5, jl1_col)['iter'].tolist()
        }
    
    # 计算交叉分析
    baseline_best_iter = stats['balanced']['best_iter']
    
    # 原始最优候选在其他权重下的排名
    baseline_best_row = results[results['iter'] == baseline_best_iter].iloc[0]
    
    stats['cross_analysis'] = {
        'baseline_best_iter': baseline_best_iter,
        'baseline_best_jl1': float(baseline_best_row['jl1_balanced']),
        'rank_in_mean_heavy': int(baseline_best_row['rank_mean_heavy']),
        'rank_in_tail_heavy': int(baseline_best_row['rank_tail_heavy']),
    }
    
    # Top-5 重叠
    top5_balanced = set(stats['balanced']['top5_iters'])
    top5_mean = set(stats['mean_heavy']['top5_iters'])
    top5_tail = set(stats['tail_heavy']['top5_iters'])
    
    stats['overlap'] = {
        'balanced_vs_mean': len(top5_balanced & top5_mean),
        'balanced_vs_tail': len(top5_balanced & top5_tail),
        'mean_vs_tail': len(top5_mean & top5_tail),
        'all_three': len(top5_balanced & top5_mean & top5_tail)
    }
    
    return results, stats


def generate_summary(stats: Dict) -> str:
    """生成文本摘要"""
    lines = []
    lines.append("=" * 70)
    lines.append("JL1 权重敏感性分析摘要")
    lines.append("=" * 70)
    lines.append("")
    
    # 各权重配置的最优候选
    lines.append("## 各权重配置的最优候选")
    lines.append("")
    for config_key, config in WEIGHT_CONFIGS.items():
        s = stats[config_key]
        lines.append(f"### {s['name']}")
        lines.append(f"  最佳迭代: Iter {s['best_iter']}")
        lines.append(f"  最佳 JL1: {s['best_jl1']:.2f}")
        lines.append(f"  Top-5: {s['top5_iters']}")
        lines.append("")
    
    # 交叉分析
    cross = stats['cross_analysis']
    lines.append("## 原始最优候选在其他权重下的表现")
    lines.append("")
    lines.append(f"原始最优 (Balanced): Iter {cross['baseline_best_iter']}, JL1={cross['baseline_best_jl1']:.2f}")
    lines.append(f"  - 在 Mean-Heavy 权重下排名: #{cross['rank_in_mean_heavy']}")
    lines.append(f"  - 在 Tail-Heavy 权重下排名: #{cross['rank_in_tail_heavy']}")
    lines.append("")
    
    # Top-5 重叠
    overlap = stats['overlap']
    lines.append("## Top-5 候选重叠情况")
    lines.append("")
    lines.append(f"Balanced vs Mean-Heavy: {overlap['balanced_vs_mean']}/5 重叠")
    lines.append(f"Balanced vs Tail-Heavy: {overlap['balanced_vs_tail']}/5 重叠")
    lines.append(f"Mean-Heavy vs Tail-Heavy: {overlap['mean_vs_tail']}/5 重叠")
    lines.append(f"三组共同 Top-5: {overlap['all_three']}/5")
    lines.append("")
    
    # 一句话结论
    lines.append("## 一句话结论（供论文使用）")
    lines.append("")
    
    if cross['rank_in_mean_heavy'] <= 3 and cross['rank_in_tail_heavy'] <= 3:
        conclusion = (
            f"论文采用的 Balanced 权重（α=1.0, λ=0.5, β=0.3）所选最优候选 (Iter {cross['baseline_best_iter']}) "
            f"在 Mean-Heavy 和 Tail-Heavy 权重下仍保持前 3 名，表明该权重配置对不同优化目标具有良好的鲁棒性。"
        )
    elif overlap['all_three'] >= 3:
        conclusion = (
            f"三组权重配置的 Top-5 候选有 {overlap['all_three']} 个共同元素，"
            f"表明 JL1 复合损失对权重调整具有一定稳健性，不同权重倾向不会导致完全不同的候选排序。"
        )
    else:
        conclusion = (
            f"权重调整导致最优候选发生变化：原 Balanced 最优 (Iter {cross['baseline_best_iter']}) "
            f"在 Mean-Heavy 下排名 #{cross['rank_in_mean_heavy']}，在 Tail-Heavy 下排名 #{cross['rank_in_tail_heavy']}，"
            f"提示权重选择对校准结果有显著影响。"
        )
    
    lines.append(conclusion)
    lines.append("")
    
    return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JL1 权重敏感性分析（离线重算）"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help=f"输入 CSV 路径 (默认: {INPUT_CSV})"
    )
    parser.add_argument(
        "--output-table", type=str, default=None,
        help=f"输出表格 CSV 路径 (默认: {OUTPUT_TABLE})"
    )
    parser.add_argument(
        "--output-summary", type=str, default=None,
        help=f"输出摘要 TXT 路径 (默认: {OUTPUT_SUMMARY})"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input) if args.input else INPUT_CSV
    table_path = Path(args.output_table) if args.output_table else OUTPUT_TABLE
    summary_path = Path(args.output_summary) if args.output_summary else OUTPUT_SUMMARY
    
    print("=" * 70)
    print("JL1 权重敏感性分析")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"输出表格: {table_path}")
    print(f"输出摘要: {summary_path}")
    print()
    
    # 执行分析
    results, stats = analyze_sensitivity()
    
    # 保存结果表
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(table_path, index=False)
    print(f"[DONE] 结果表已保存: {table_path}")
    
    # 生成并保存摘要
    summary_text = generate_summary(stats)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"[DONE] 摘要已保存: {summary_path}")
    
    # 打印摘要
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
