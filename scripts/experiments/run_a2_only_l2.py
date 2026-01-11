#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_a2_only_l2.py
=================
运行 A2 (Only L2) Pipeline

配置说明：
- 使用 A0 默认参数（不做 L1 校准）
- 只运行 L2 状态同化（IES）
- 优化状态向量：[capacityFactor, minGap, impatience]

严格复用论文 L2 定义：
- 观测向量: M11 走廊 11 维 moving 速度
- IES: Ne=10, K=3, β=0.3
- Rule C: T*=325s, v*=5km/h

Author: RCMDT Project
Date: 2026-01-11
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "calibration"))

from run_ies_loop import IESLoop
from build_l2_sim_vector_traveltime import build_simulation_vector_tt


# =============================================================================
# A2 配置
# =============================================================================

def load_baseline_l1_params() -> Dict[str, float]:
    """从 baseline_parameters.json 读取 L1 默认参数（t_board/t_fixed 使用默认值）"""
    baseline_path = PROJECT_ROOT / "config" / "calibration" / "baseline_parameters.json"
    if not baseline_path.exists():
        return {
            "tau": 1.0,
            "sigma": 0.5,
            "minGap": 2.5,
            "accel": 2.5,
            "decel": 4.5,
            "t_board": 2.0,
            "t_fixed": 5.0
        }
    
    import json
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    
    bus = baseline.get("micro_parameters", {}).get("kmb_double_decker", {})
    return {
        "tau": float(bus.get("tau", 1.0)),
        "sigma": float(bus.get("sigma", 0.5)),
        "minGap": float(bus.get("minGap", 2.5)),
        "accel": float(bus.get("accel", 2.5)),
        "decel": float(bus.get("decel", 4.5)),
        "t_board": 2.0,
        "t_fixed": 5.0
    }


BASELINE_L1_PARAMS = load_baseline_l1_params()

A2_CONFIG = {
    "config_id": "A2",
    "name": "Only-L2",
    "description": "固定默认参数 + L2 状态同化",
    
    # L1 参数（固定为默认值，不优化）
    "bus_params": BASELINE_L1_PARAMS,
    
    # L2 协议
    "l2_protocol": {
        "observation_file": PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11_moving_irn.csv",
        "observation_dim": 11,
        "state_params": ["capacityFactor", "minGap", "impatience"],
        "ensemble_size": 10,
        "max_iterations": 3,
        "damping": 0.3,
        "tt_mode": "moving",
        "t_min": 0.0,
        "t_max": 3600.0
    },
    "dist_csv": PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv",
    
    # 运行配置
    "seeds": [0, 1, 2],
    "output_dir": PROJECT_ROOT / "data" / "experiments_v4" / "unified_l2" / "protocol_ablation" / "A2"
}


# =============================================================================
# 运行函数
# =============================================================================

def run_a2_single_seed(seed: int, output_dir: Path) -> None:
    """
    运行单个 seed 的 A2 实验
    
    Parameters:
        seed: 随机种子
        output_dir: 输出根目录
    """
    print("=" * 70)
    print(f"A2 (Only L2) - Seed {seed}")
    print("=" * 70)
    print(f"L1 参数: 固定为默认值（不优化）")
    print(f"L2 IES: Ne={A2_CONFIG['l2_protocol']['ensemble_size']}, " 
          f"K={A2_CONFIG['l2_protocol']['max_iterations']}, "
          f"β={A2_CONFIG['l2_protocol']['damping']}")
    print(f"观测向量: M11 走廊 {A2_CONFIG['l2_protocol']['observation_dim']} 维")
    print()
    
    # 创建 IES Loop
    ies = IESLoop(
        project_root=str(PROJECT_ROOT),
        label=f"a2_only_l2_seed{seed}",
        ensemble_size=A2_CONFIG['l2_protocol']['ensemble_size'],
        max_iters=A2_CONFIG['l2_protocol']['max_iterations'],
        seed=seed,
        use_baseline=True,  # 使用 baseline (默认) L1 参数
        l1_params_override=A2_CONFIG["bus_params"],
        apply_l1_params=True,
        update_damping=A2_CONFIG['l2_protocol']['damping'],
        use_group_weights=True,
        t_min=A2_CONFIG['l2_protocol']['t_min'],
        t_max=A2_CONFIG['l2_protocol']['t_max'],
        tt_mode=A2_CONFIG['l2_protocol']['tt_mode']
    )
    
    # 覆盖观测向量为 M11
    import pandas as pd
    obs_df = pd.read_csv(A2_CONFIG['l2_protocol']['observation_file']).reset_index(drop=True)
    ies.obs_df = obs_df
    ies.Y_obs = obs_df['mean_speed_kmh'].values
    ies.obs_csv_path = str(A2_CONFIG['l2_protocol']['observation_file'])
    
    print("[IES] 开始迭代优化...")
    print(f"[IES] 状态向量: {A2_CONFIG['l2_protocol']['state_params']}")
    print()
    
    # 运行 IES
    best_params = ies.run()
    
    print("\n[IES] 优化完成")
    print("最优参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")
    
    # 读取最终日志
    log_path = PROJECT_ROOT / "data" / "calibration" / f"a2_only_l2_seed{seed}_ies_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        final_ks = log_df['ks_distance'].iloc[-1] if len(log_df) > 0 else None
        print(f"\n最终 KS(speed): {final_ks:.4f}" if final_ks else "\n最终 KS: N/A")
    
    # 将最终的 stopinfo.xml 复制到输出目录（选择最后一轮 RMSE 最优成员）
    seed_output_dir = output_dir / f"seed{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    
    best_member_stopinfo = select_best_member_stopinfo(
        ies_output_dir=ies.output_dir,
        last_iter=A2_CONFIG['l2_protocol']['max_iterations'],
        ensemble_size=A2_CONFIG['l2_protocol']['ensemble_size'],
        obs_csv=A2_CONFIG['l2_protocol']['observation_file'],
        dist_csv=A2_CONFIG['dist_csv'],
        t_min=A2_CONFIG['l2_protocol']['t_min'],
        t_max=A2_CONFIG['l2_protocol']['t_max'],
        tt_mode=A2_CONFIG['l2_protocol']['tt_mode']
    )
    
    if best_member_stopinfo is None:
        print("\n[WARN] 未找到可用的 stopinfo.xml，跳过保存。")
    else:
        import shutil
        dst_stopinfo = seed_output_dir / "stopinfo.xml"
        shutil.copy(best_member_stopinfo, dst_stopinfo)
        print(f"\nStopinfo 已保存: {dst_stopinfo}")
    
    print("\n" + "=" * 70)


def run_all_a2_seeds(seeds: list, output_dir: Path) -> None:
    """
    运行所有 seeds 的 A2 实验
    
    Parameters:
        seeds: 随机种子列表
        output_dir: 输出根目录
    """
    print("\n")
    print("=" * 70)
    print("A2 (Only L2) Pipeline - 批量运行")
    print("=" * 70)
    print(f"配置: {A2_CONFIG['name']}")
    print(f"Seeds: {seeds}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    print()
    
    for seed in seeds:
        run_a2_single_seed(seed, output_dir)
        print("\n" + "-" * 70 + "\n")
    
    print("=" * 70)
    print("所有 A2 实验完成！")
    print("=" * 70)
    print()
    print("下一步：运行指标计算")
    print(f"  python scripts/experiments/compute_pipeline_metrics.py")
    print()


def compute_rmse(sim_values: np.ndarray, obs_values: np.ndarray) -> Optional[float]:
    """计算 RMSE（只使用有效样本）"""
    valid_mask = ~np.isnan(sim_values) & ~np.isnan(obs_values)
    if valid_mask.sum() < 3:
        return None
    return float(np.sqrt(np.mean((sim_values[valid_mask] - obs_values[valid_mask]) ** 2)))


def select_best_member_stopinfo(
    ies_output_dir: Path,
    last_iter: int,
    ensemble_size: int,
    obs_csv: Path,
    dist_csv: Path,
    t_min: float,
    t_max: float,
    tt_mode: str
) -> Optional[Path]:
    """从最后一轮 IES 成员中选择 RMSE 最优的 stopinfo.xml"""
    import pandas as pd
    
    obs_df = pd.read_csv(obs_csv)
    obs_speeds = obs_df["mean_speed_kmh"].to_numpy()
    
    best_rmse = None
    best_path = None
    
    for member_id in range(ensemble_size):
        run_dir = ies_output_dir / f"iter{last_iter:02d}_run{member_id:02d}"
        stopinfo_path = run_dir / "stopinfo.xml"
        if not stopinfo_path.exists():
            continue
        
        try:
            sim_df = build_simulation_vector_tt(
                stopinfo_path=str(stopinfo_path),
                observation_csv=str(obs_csv),
                route_stop_csv=str(dist_csv),
                output_csv=None,
                max_gap_seconds=1800.0,
                verbose=False,
                tmin=t_min,
                tmax=t_max,
                tt_mode=tt_mode
            )
        except Exception:
            continue
        
        if "sim_speed_kmh" not in sim_df.columns:
            continue
        
        sim_speeds = sim_df["sim_speed_kmh"].to_numpy()
        rmse = compute_rmse(sim_speeds, obs_speeds)
        if rmse is None:
            continue
        
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_path = stopinfo_path
    
    if best_path is not None:
        print(f"[A2] 最优成员 RMSE(speed)={best_rmse:.4f} -> {best_path.parent.name}")
    return best_path


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="运行 A2 (Only L2) Pipeline")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=A2_CONFIG["seeds"],
        help="随机种子列表"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(A2_CONFIG["output_dir"]),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 运行
    output_dir = Path(args.output_dir)
    run_all_a2_seeds(args.seeds, output_dir)


if __name__ == "__main__":
    main()
