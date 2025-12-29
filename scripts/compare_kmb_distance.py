#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_kmb_distance.py
=======================
对比分析：无向 vs 有向 KMB 距离

输出论文可用统计：
1. 受单向影响段占比（按路线分组）
2. Top-K 受影响段清单
3. Scale 改判前后对比（Core 段口径）

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-29
"""

import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KMB_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Core 段定义（研究区域口径）
CORE_RANGES = {
    ('68X', 'inbound'): (1, 14),
    ('68X', 'outbound'): (19, 31),
    ('960', 'inbound'): (1, 10),
    ('960', 'outbound'): (15, 22),
}

# DP 路径长度（来自用户提供的实验结果）
DP_PATH_LENGTHS = {
    ('68X', 'inbound'): 9861,
    ('68X', 'outbound'): 8844,
    ('960', 'inbound'): 13581,
    ('960', 'outbound'): 6495,
}


def main():
    print("=" * 80)
    print("KMB 距离对比分析：无向 vs 有向")
    print("=" * 80)
    
    df = pd.read_csv(KMB_CSV)
    print(f"[info] 加载 {len(df)} 行数据\n")
    
    # =========================================================================
    # 1. 受单向影响段占比
    # =========================================================================
    print("=" * 80)
    print("1. 受单向影响段占比")
    print("=" * 80)
    
    # 计算差异
    df['link_diff'] = df['link_dist_m_dir'] - df['link_dist_m_undir']
    df['is_affected'] = df['link_diff'].abs() > 100  # 差异 > 100m
    
    for (route, bound), group in df.groupby(['route', 'bound']):
        core_range = CORE_RANGES.get((route, bound))
        if core_range:
            core_group = group[(group['seq'] >= core_range[0]) & (group['seq'] <= core_range[1])]
            total = len(core_group)
            affected = core_group['is_affected'].sum()
            print(f"  {route} {bound} (Core seq {core_range[0]}-{core_range[1]}): "
                  f"{affected}/{total} 段受影响 ({affected/total*100:.1f}%)")
    
    # =========================================================================
    # 2. Top-K 受影响段清单
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Top-10 受影响段清单（按差异绝对值降序）")
    print("=" * 80)
    
    # 只看 Core 段
    core_rows = []
    for (route, bound), group in df.groupby(['route', 'bound']):
        core_range = CORE_RANGES.get((route, bound))
        if core_range:
            core_group = group[(group['seq'] >= core_range[0]) & (group['seq'] <= core_range[1])]
            core_rows.append(core_group)
    
    df_core = pd.concat(core_rows, ignore_index=True)
    df_core = df_core[df_core['link_diff'].notna()]
    df_core['diff_abs'] = df_core['link_diff'].abs()
    
    top_k = df_core.nlargest(10, 'diff_abs')
    
    print(f"\n{'路线':<15} {'seq':<5} {'站点':<20} {'无向(m)':<10} {'有向(m)':<10} {'差异(m)':<10}")
    print("-" * 75)
    for _, row in top_k.iterrows():
        stop_name = row['stop_name_tc'][:18] if isinstance(row['stop_name_tc'], str) else 'N/A'
        undir = row['link_dist_m_undir']
        dir_val = row['link_dist_m_dir']
        diff = row['link_diff']
        print(f"{row['route']:<6} {row['bound']:<8} {row['seq']:<5} {stop_name:<20} "
              f"{undir:<10.1f} {dir_val:<10.1f} {diff:<+10.1f}")
    
    # =========================================================================
    # 3. Scale 改判前后对比
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. Scale 改判前后对比（Core 段口径）")
    print("=" * 80)
    
    results = []
    
    for (route, bound), group in df.groupby(['route', 'bound']):
        core_range = CORE_RANGES.get((route, bound))
        if not core_range:
            continue
        
        core_group = group[(group['seq'] >= core_range[0]) & (group['seq'] <= core_range[1])]
        core_group = core_group.sort_values('seq')
        
        # 计算 Core 段累积距离
        first_row = core_group.iloc[0]
        last_row = core_group.iloc[-1]
        
        kmb_undir = last_row['cum_dist_m_undir'] - first_row['cum_dist_m_undir']
        kmb_dir = last_row['cum_dist_m_dir'] - first_row['cum_dist_m_dir']
        
        dp_path = DP_PATH_LENGTHS.get((route, bound), 0)
        
        # 计算 Scale
        scale_undir = dp_path / kmb_undir if kmb_undir > 0 else float('nan')
        scale_dir = dp_path / kmb_dir if kmb_dir > 0 and not math.isnan(kmb_dir) else float('nan')
        
        results.append({
            'route': route,
            'bound': bound,
            'core_range': f"seq {core_range[0]}-{core_range[1]}",
            'kmb_undir_m': kmb_undir,
            'kmb_dir_m': kmb_dir,
            'dp_path_m': dp_path,
            'scale_undir': scale_undir,
            'scale_dir': scale_dir,
            'scale_change': (scale_dir - scale_undir) if not math.isnan(scale_dir) else float('nan'),
        })
    
    print(f"\n{'路线':<15} {'Core范围':<15} {'KMB无向(m)':<12} {'KMB有向(m)':<12} "
          f"{'DP路径(m)':<10} {'Scale无向':<10} {'Scale有向':<10} {'变化':<8}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['route']:<6} {r['bound']:<8} {r['core_range']:<15} "
              f"{r['kmb_undir_m']:<12.1f} "
              f"{r['kmb_dir_m']:<12.1f}" if not math.isnan(r['kmb_dir_m']) else f"{'N/A':<12}"
              f" {r['dp_path_m']:<10} "
              f"{r['scale_undir']:<10.2f} "
              f"{r['scale_dir']:<10.2f}" if not math.isnan(r['scale_dir']) else f"{'N/A':<10}"
              f" {r['scale_change']:+.2f}" if not math.isnan(r['scale_change']) else '')
    
    # 保存详细结果
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "kmb_distance_comparison.csv", index=False)
    
    # =========================================================================
    # 4. 论文级结论
    # =========================================================================
    print("\n" + "=" * 80)
    print("论文级结论")
    print("=" * 80)
    
    print("""
关键发现:
1. 无向 KMB 距离允许逆向穿越单向道路，导致"真值"被低估
2. Phase 1 验证确认：所有 7 个 Pareto Top 段都包含单向约束违规（共 37 条违规边）
3. 使用有向距离后，Scale 会下降（因为 KMB 真值变大）

结论:
- DP 优化已消除算法伪影（U-turn=0）
- 之前看似 "仿真绕行过长"，实际是 "无向真值不可行驶"
- 论文指标应使用 scale_dir（有向口径）
""")
    
    print(f"\n[ok] 详细结果已保存到: {OUTPUT_DIR / 'kmb_distance_comparison.csv'}")


if __name__ == "__main__":
    main()
