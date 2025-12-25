#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_corridor_mask.py
=====================
自适应走廊截断：用数据可观测性定义研究走廊。

核心逻辑：
1. 用 coverage + sampledSeconds + n_matched 定义"1h 可达/可观测"
2. 对每条 (route, bound) 取满足阈值的最长连续前缀（prefix corridor）
3. 冻结 corridor_ids，保证 IES 维度固定

约束（防翻车）：
- 约束1：走廊必须"连续前缀"，不能是碎片
- 约束2：NO_MAPPING 显式处理，不能无声截断
- 约束3：sampledSeconds 用占比阈值更稳

Author: RMBC-Sim project
Date: 2025-12-25
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import List, Set, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_mapping_ids(mapping_csv: str) -> Set[int]:
    """加载映射表中的 observation_id 集合"""
    mp = pd.read_csv(mapping_csv)
    return set(mp["observation_id"].astype(int).tolist())


def make_threshold_mask(
    df: pd.DataFrame,
    min_cov: float,
    min_sampled: float,
    interval_s: float
) -> pd.DataFrame:
    """
    为每行计算是否满足阈值条件。
    
    sampledSeconds 阈值使用"固定值 vs 占比"取 max，对稀发车线路更鲁棒。
    """
    # sampled 的占比阈值：1% of interval（如 3600s → 36s）
    min_sampled_eff = max(min_sampled, 0.01 * interval_s)
    
    df = df.copy()
    
    # 确保 sampledSeconds_sum 列存在（诊断文件应该有）
    if "sampledSeconds_sum" not in df.columns:
        # fallback: 尝试用其他列名
        if "sampled_seconds_sum" in df.columns:
            df["sampledSeconds_sum"] = df["sampled_seconds_sum"]
        else:
            df["sampledSeconds_sum"] = 0.0
    
    # 确保 n_matched 列存在
    if "n_matched" not in df.columns:
        if "n_edges_matched" in df.columns:
            df["n_matched"] = df["n_edges_matched"]
        else:
            df["n_matched"] = 0
    
    # 确保 coverage_raw 列存在
    if "coverage_raw" not in df.columns:
        if "coverage" in df.columns:
            df["coverage_raw"] = df["coverage"]
        else:
            df["coverage_raw"] = 0.0
    
    # 阈值判定
    df["ok_by_threshold"] = (
        (df["coverage_raw"] >= min_cov) &
        (df["sampledSeconds_sum"] >= min_sampled_eff) &
        (df["n_matched"] >= 1)
    )
    
    return df


def prefix_corridor(df: pd.DataFrame, verbose: bool = True) -> Tuple[List[int], dict]:
    """
    对每条 (route, bound) 取满足阈值的最长连续前缀。
    
    逻辑：
    - 按 from_seq/to_seq 排序
    - 从最小 seq 开始，只要当前 link 满足阈值就纳入
    - 一旦遇到第一个不满足阈值的 link，就停止
    
    Returns:
        (corridor_ids, metadata): corridor_ids 列表 + 每条线路的截断信息
    """
    keep = []
    metadata = {}  # 记录每条线路的截断信息
    
    for (route, bound), g in df.groupby(["route", "bound"], dropna=False):
        g2 = g.sort_values(["from_seq", "to_seq"]).copy()
        
        prefix_ids = []
        cutoff_reason = None
        cutoff_link = None
        max_to_seq = None
        
        for _, r in g2.iterrows():
            obs_id = int(r["observation_id"])
            
            # 检查 mapping 状态（NO_MAPPING 是硬断点）
            has_mapping = r.get("has_mapping", True)
            if not has_mapping:
                cutoff_reason = "NO_MAPPING"
                cutoff_link = obs_id
                break
            
            # 检查阈值条件
            if bool(r["ok_by_threshold"]):
                prefix_ids.append(obs_id)
                max_to_seq = r["to_seq"]
            else:
                # 记录截断原因
                if r.get("reason", "") == "NO_SAMPLED_EDGE":
                    cutoff_reason = "NO_SAMPLED_EDGE"
                elif r.get("reason", "") == "LOW_COVERAGE":
                    cutoff_reason = "LOW_COVERAGE"
                else:
                    cutoff_reason = "THRESHOLD_NOT_MET"
                cutoff_link = obs_id
                break
        
        keep.extend(prefix_ids)
        
        # 记录元数据
        key = f"{route}_{bound}"
        metadata[key] = {
            "route": route,
            "bound": bound,
            "n_total": len(g2),
            "n_keep": len(prefix_ids),
            "max_to_seq": max_to_seq,
            "cutoff_reason": cutoff_reason,
            "cutoff_link": cutoff_link,
        }
        
        if verbose:
            status = f"keep {len(prefix_ids)}/{len(g2)}"
            if cutoff_reason:
                status += f", cutoff at link {cutoff_link} ({cutoff_reason})"
            else:
                status += ", full corridor"
            print(f"  [{route} {bound}] {status}")
    
    return keep, metadata


def main():
    parser = argparse.ArgumentParser(
        description="自适应走廊截断：生成 corridor mask 文件"
    )
    parser.add_argument(
        "--diagnosis", "-d",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / "l2_sim_vector_diagnosis.csv"),
        help="诊断文件路径（build_l2_simulation_vector.py 输出）"
    )
    parser.add_argument(
        "--observation", "-o",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector.csv"),
        help="观测向量 CSV 路径"
    )
    parser.add_argument(
        "--mapping", "-m",
        type=str,
        default=str(PROJECT_ROOT / "config" / "calibration" / "link_edge_mapping.csv"),
        help="link-edge 映射表路径"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration"),
        help="输出目录"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.7,
        help="最小覆盖率阈值 (默认: 0.7)"
    )
    parser.add_argument(
        "--min-sampled-seconds",
        type=float,
        default=60.0,
        help="最小采样秒数阈值 (默认: 60s)"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=3600.0,
        help="仿真时间窗口 (默认: 3600s)"
    )
    parser.add_argument(
        "--freeze-meta",
        type=str,
        default="B4_v2 iter01_run00 seed=FIXED",
        help="元信息标签，写入 corridor_ids.txt 便于复现"
    )
    parser.add_argument(
        "--fail-on-no-mapping",
        action="store_true",
        help="如果遇到 NO_MAPPING 就报错退出"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    print("[Corridor Mask] 开始生成...")
    print(f"  - 诊断文件: {args.diagnosis}")
    print(f"  - 观测向量: {args.observation}")
    print(f"  - 映射表: {args.mapping}")
    print(f"  - 输出目录: {args.outdir}")
    print(f"  - 阈值: coverage >= {args.min_coverage}, sampledSeconds >= {args.min_sampled_seconds}s")
    
    # 加载数据
    diag = pd.read_csv(args.diagnosis)
    obs = pd.read_csv(args.observation)
    mp = pd.read_csv(args.mapping)
    
    # 必需列检查
    required_cols = ["observation_id", "route", "bound", "from_seq", "to_seq"]
    for col in required_cols:
        if col not in diag.columns:
            raise ValueError(f"诊断文件缺少列: {col}")
    
    # 检查 NO_MAPPING 情况
    mapping_ids = load_mapping_ids(args.mapping)
    diag["has_mapping"] = diag["observation_id"].astype(int).isin(mapping_ids)
    n_no_mapping = int((~diag["has_mapping"]).sum())
    
    print(f"\n[预检查] NO_MAPPING 数量: {n_no_mapping}")
    if n_no_mapping > 0:
        no_mapping_links = diag.loc[~diag["has_mapping"], "observation_id"].tolist()
        print(f"  - 涉及 links: {no_mapping_links}")
        
        if args.fail_on_no_mapping:
            raise ValueError(f"存在 NO_MAPPING 的 links: {no_mapping_links}，请先修复映射表")
    
    # 阈值 mask
    diag2 = make_threshold_mask(
        diag,
        args.min_coverage,
        args.min_sampled_seconds,
        args.interval_seconds
    )
    
    # 把无 mapping 视作不 ok（会在 prefix 逻辑中作为硬断点）
    diag2.loc[~diag2["has_mapping"], "ok_by_threshold"] = False
    
    # 计算 prefix corridor（连续前缀）
    print("\n[Prefix Corridor] 计算各线路走廊范围:")
    corridor_ids, metadata = prefix_corridor(diag2, verbose=True)
    
    # 去重并排序
    corridor_ids = sorted(set(corridor_ids))
    
    # 输出 corridor_ids.txt（带元信息）
    corridor_txt_path = os.path.join(args.outdir, "corridor_ids.txt")
    with open(corridor_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Corridor Mask - 自适应走廊定义\n")
        f.write(f"# 生成基准: {args.freeze_meta}\n")
        f.write(f"# 阈值: min_cov={args.min_coverage}, min_sampled={args.min_sampled_seconds}s, interval={args.interval_seconds}s\n")
        f.write(f"# NO_MAPPING in diagnosis: {n_no_mapping}\n")
        f.write(f"# Corridor 维度: {len(corridor_ids)}\n")
        f.write(f"#\n")
        for key, meta in sorted(metadata.items()):
            cutoff_info = f", cutoff={meta['cutoff_reason']} at link {meta['cutoff_link']}" if meta['cutoff_reason'] else ""
            f.write(f"# {key}: keep {meta['n_keep']}/{meta['n_total']}, max_to_seq={meta['max_to_seq']}{cutoff_info}\n")
        f.write(f"#\n")
        for cid in corridor_ids:
            f.write(str(cid) + "\n")
    
    print(f"\n[输出] corridor_ids.txt: {corridor_txt_path}")
    
    # 裁剪 observation & mapping
    obs2 = obs[obs["observation_id"].astype(int).isin(corridor_ids)].copy()
    mp2 = mp[mp["observation_id"].astype(int).isin(corridor_ids)].copy()
    
    obs_corridor_path = os.path.join(args.outdir, "l2_observation_vector_corridor.csv")
    obs2.to_csv(obs_corridor_path, index=False)
    print(f"[输出] l2_observation_vector_corridor.csv: {obs_corridor_path} ({len(obs2)} rows)")
    
    # mapping 文件输出到 config 目录
    config_outdir = str(PROJECT_ROOT / "config" / "calibration")
    os.makedirs(config_outdir, exist_ok=True)
    mp_corridor_path = os.path.join(config_outdir, "link_edge_mapping_corridor.csv")
    mp2.to_csv(mp_corridor_path, index=False)
    print(f"[输出] link_edge_mapping_corridor.csv: {mp_corridor_path} ({len(mp2)} rows)")
    
    # 统计摘要
    print("\n" + "=" * 50)
    print("[Corridor Mask 完成]")
    print(f"  - 原始维度: {len(diag)}")
    print(f"  - Corridor 维度: {len(corridor_ids)}")
    print(f"  - 裁剪比例: {len(corridor_ids)/len(diag)*100:.1f}%")
    print("=" * 50)
    
    # 连续性检查（验证 prefix 逻辑）
    print("\n[连续性检查] 验证 corridor 是否为连续前缀:")
    for (route, bound), g in obs2.groupby(["route", "bound"], dropna=False):
        seqs = sorted(g["from_seq"].tolist())
        is_continuous = True
        for i in range(1, len(seqs)):
            if seqs[i] - seqs[i-1] > 10:  # 允许一定间隔（stop 编号可能不连续）
                is_continuous = False
                break
        status = "✓ 连续" if is_continuous else "✗ 有跳跃"
        print(f"  [{route} {bound}] {status}, seq range: {min(seqs)}-{max(seqs)}")


if __name__ == "__main__":
    main()
