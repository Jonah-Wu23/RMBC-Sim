"""
recalculate_jl1_from_b2.py
==========================
从 B2 历史运行的 SUMO 输出离线重新计算 JL1 复合损失

功能：
1. 读取 B2_log.csv 获取 40 个候选的参数
2. 对每个候选，从对应的 SUMO 输出计算完整 JL1 分项
3. 保存到 B2_jl1_recalculated.csv，包含所有分项
4. 支持敏感性分析：可调整 α, λ, β 权重

使用场景：
- B2 原始实验只记录了 rmse_68x，没有 JL1 分项
- 如果 SUMO 输出文件仍然存在，可离线重算
- 如果输出已删除，此脚本会跳过缺失文件并报告

作者：RCMDT Project
日期：2026-01-11
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from calibration.objective import calculate_jl1_loss

# 路径配置
B2_LOG = PROJECT_ROOT / "data" / "calibration" / "B2_log.csv"
REAL_LINKS = PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
ROUTE_DIST = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
SUMO_OUTPUT_BASE = PROJECT_ROOT / "sumo" / "output"
OUTPUT_CSV = PROJECT_ROOT / "data" / "calibration" / "B2_jl1_recalculated.csv"

# JL1 默认权重（来自论文）
DEFAULT_ALPHA = 1.0
DEFAULT_LAMBDA = 0.5
DEFAULT_BETA = 0.3


def find_sumo_output(iter_num: int, label: str = "B2") -> Path:
    """
    根据迭代号查找对应的 SUMO 输出文件
    
    可能的命名规则：
    - sumo/output/stopinfo_{label}_iter{iter}.xml
    - sumo/output/experiment2_cropped_stopinfo.xml (如果是单次运行)
    - sumo/output/B2_iter{iter}_stopinfo.xml
    
    返回第一个存在的文件，如果都不存在则返回 None
    """
    possible_paths = [
        SUMO_OUTPUT_BASE / f"stopinfo_{label}_iter{iter_num}.xml",
        SUMO_OUTPUT_BASE / f"{label}_iter{iter_num}_stopinfo.xml",
        SUMO_OUTPUT_BASE / f"experiment2_cropped_stopinfo.xml",
        SUMO_OUTPUT_BASE / "experiment2_cropped_stopinfo.xml"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def recalculate_jl1_for_b2(
    alpha: float = DEFAULT_ALPHA,
    lambda_std: float = DEFAULT_LAMBDA,
    beta: float = DEFAULT_BETA,
    route: str = '68X'
) -> pd.DataFrame:
    """
    为 B2 的所有候选重新计算 JL1
    
    Args:
        alpha: MAE+dispersion 项权重
        lambda_std: dispersion 内部权重
        beta: 尾部风险权重
        route: 目标路线
        
    Returns:
        DataFrame with columns: iter, jl1, rmse_term, mae_term, std_term, 
                                dispersion_term, tail_term, alpha, lambda, beta
    """
    if not B2_LOG.exists():
        raise FileNotFoundError(f"B2 日志文件不存在: {B2_LOG}")
    
    df_log = pd.read_csv(B2_LOG)
    print(f"[INFO] 从 B2_log.csv 读取 {len(df_log)} 个候选")
    print(f"[INFO] JL1 权重: α={alpha}, λ={lambda_std}, β={beta}")
    print()
    
    results = []
    missing_count = 0
    
    for idx, row in df_log.iterrows():
        iter_num = int(row['iter'])
        
        # 查找 SUMO 输出
        sim_xml = find_sumo_output(iter_num)
        
        if sim_xml is None or not sim_xml.exists():
            missing_count += 1
            print(f"[WARN] Iter {iter_num}: SUMO 输出文件缺失，跳过")
            # 填充缺失值
            results.append({
                'iter': iter_num,
                'type': row['type'],
                'jl1': np.nan,
                'rmse_term': np.nan,
                'mae_term': np.nan,
                'std_term': np.nan,
                'dispersion_term': np.nan,
                'tail_term': np.nan,
                'alpha': alpha,
                'lambda_std': lambda_std,
                'beta': beta,
                'n_errors': 0,
                'sumo_output': 'MISSING'
            })
            continue
        
        # 计算 JL1
        try:
            jl1_metrics = calculate_jl1_loss(
                sim_xml_path=str(sim_xml),
                real_links_csv=str(REAL_LINKS),
                route_stop_dist_csv=str(ROUTE_DIST),
                route=route,
                bound='I',
                alpha=alpha,
                lambda_std=lambda_std,
                beta=beta
            )
            
            result = {
                'iter': iter_num,
                'type': row['type'],
                'jl1': jl1_metrics['jl1'],
                'rmse_term': jl1_metrics['rmse_term'],
                'mae_term': jl1_metrics['mae_term'],
                'std_term': jl1_metrics['std_term'],
                'dispersion_term': jl1_metrics['dispersion_term'],
                'tail_term': jl1_metrics['tail_term'],
                'alpha': alpha,
                'lambda_std': lambda_std,
                'beta': beta,
                'n_errors': jl1_metrics['n_errors'],
                'sumo_output': str(sim_xml.name)
            }
            results.append(result)
            
            if iter_num % 5 == 0:
                print(f"[OK] Iter {iter_num}: JL1={jl1_metrics['jl1']:.2f} "
                      f"(RMSE={jl1_metrics['rmse_term']:.2f}, "
                      f"MAE={jl1_metrics['mae_term']:.2f}, "
                      f"P90={jl1_metrics['tail_term']:.2f})")
                
        except Exception as e:
            print(f"[ERROR] Iter {iter_num}: {e}")
            results.append({
                'iter': iter_num,
                'type': row['type'],
                'jl1': np.nan,
                'rmse_term': np.nan,
                'mae_term': np.nan,
                'std_term': np.nan,
                'dispersion_term': np.nan,
                'tail_term': np.nan,
                'alpha': alpha,
                'lambda_std': lambda_std,
                'beta': beta,
                'n_errors': 0,
                'sumo_output': f'ERROR: {str(e)[:50]}'
            })
    
    print()
    print(f"[SUMMARY] 成功: {len(results) - missing_count}/{len(df_log)}, 缺失: {missing_count}")
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从 B2 历史 SUMO 输出离线重算 JL1 复合损失"
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"MAE+dispersion 项权重 (默认: {DEFAULT_ALPHA})"
    )
    parser.add_argument(
        "--lambda-std", type=float, default=DEFAULT_LAMBDA,
        help=f"Dispersion 内部权重 (默认: {DEFAULT_LAMBDA})"
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_BETA,
        help=f"尾部风险权重 (默认: {DEFAULT_BETA})"
    )
    parser.add_argument(
        "--route", type=str, default='68X',
        help="目标路线 (默认: 68X)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"输出 CSV 路径 (默认: {OUTPUT_CSV})"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else OUTPUT_CSV
    
    print("=" * 70)
    print("B2 JL1 离线重算")
    print("=" * 70)
    print(f"输入日志: {B2_LOG}")
    print(f"真实数据: {REAL_LINKS}")
    print(f"输出路径: {output_path}")
    print()
    
    # 执行重算
    df_jl1 = recalculate_jl1_for_b2(
        alpha=args.alpha,
        lambda_std=args.lambda_std,
        beta=args.beta,
        route=args.route
    )
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_jl1.to_csv(output_path, index=False)
    print(f"\n[DONE] 结果已保存到: {output_path}")
    
    # 统计摘要
    valid = df_jl1[df_jl1['jl1'].notna()]
    if len(valid) > 0:
        print()
        print("=" * 70)
        print("JL1 统计摘要")
        print("=" * 70)
        print(f"有效样本: {len(valid)}/{len(df_jl1)}")
        print(f"JL1 范围: {valid['jl1'].min():.2f} - {valid['jl1'].max():.2f}")
        print(f"JL1 均值: {valid['jl1'].mean():.2f} ± {valid['jl1'].std():.2f}")
        print(f"最佳 JL1: {valid['jl1'].min():.2f} (iter={valid.loc[valid['jl1'].idxmin(), 'iter']:.0f})")
        print()
        print("分项均值:")
        print(f"  RMSE term:       {valid['rmse_term'].mean():.2f}")
        print(f"  MAE term:        {valid['mae_term'].mean():.2f}")
        print(f"  Std term:        {valid['std_term'].mean():.2f}")
        print(f"  Dispersion term: {valid['dispersion_term'].mean():.2f}")
        print(f"  Tail term (P90): {valid['tail_term'].mean():.2f}")


if __name__ == "__main__":
    main()
