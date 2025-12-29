#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_capfactor_test.py
=====================
capacityFactor 三点敏感性试验

固定 L1=B2 参数，分别跑 capFactor=1/2/4，输出 TT 对比表。
"""

import subprocess
import sys
import json
import pandas as pd
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/calibration'))

from build_l2_sim_vector_traveltime import build_simulation_vector_tt


def run_single_simulation(cap_factor: float, label: str) -> Path:
    """运行单次仿真，返回 stopinfo.xml 路径"""
    
    output_dir = PROJECT_ROOT / f'sumo/output/capfactor_test/{label}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 L1 参数（B2 最优）
    l1_params_file = PROJECT_ROOT / 'config/calibration/best_l1_parameters.json'
    with open(l1_params_file) as f:
        l1_params = json.load(f)
    
    # 创建临时背景路由文件（只修改 capacityFactor 相关的 scale）
    # 注意：capacityFactor 通过 SUMO 的 --scale 参数实现
    
    # 构建 SUMO 配置
    sumocfg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{PROJECT_ROOT}/sumo/net/hk_cropped.net.xml"/>
        <route-files value="{PROJECT_ROOT}/sumo/routes/fixed_routes_cropped.rou.xml,{PROJECT_ROOT}/sumo/routes/background_cropped.rou.xml"/>
        <additional-files value="{PROJECT_ROOT}/sumo/additional/bus_stops_cropped.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <scale value="{cap_factor}"/>
        <time-to-teleport value="300"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
    <output>
        <stop-output value="{output_dir}/stopinfo.xml"/>
    </output>
</configuration>'''
    
    sumocfg_path = output_dir / 'test.sumocfg'
    with open(sumocfg_path, 'w') as f:
        f.write(sumocfg_content)
    
    # 运行 SUMO
    print(f"\n[{label}] 运行仿真 (capFactor={cap_factor})...")
    cmd = ['sumo', '-c', str(sumocfg_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] SUMO 运行失败: {result.stderr}")
        return None
    
    print(f"[{label}] 仿真完成")
    return output_dir / 'stopinfo.xml'


def main():
    cap_factors = [1.0, 2.0, 4.0]
    results = []
    
    for cap in cap_factors:
        label = f'cap{int(cap)}'
        stopinfo_path = run_single_simulation(cap, label)
        
        if stopinfo_path and stopinfo_path.exists():
            # 提取 TT
            df = build_simulation_vector_tt(
                stopinfo_path=str(stopinfo_path),
                observation_csv=str(PROJECT_ROOT / 'data/calibration/l2_observation_vector_corridor_M11_TT.csv'),
                route_stop_csv=str(PROJECT_ROOT / 'data/processed/kmb_route_stop_dist.csv'),
                verbose=False
            )
            
            # 计算统计量
            median_tt_sim = df['travel_time_sim_s'].median()
            median_tt_obs = df['travel_time_obs_s'].median()
            median_ratio = df['ratio'].median()
            iqr_sim = df['travel_time_sim_s'].quantile(0.75) - df['travel_time_sim_s'].quantile(0.25)
            
            results.append({
                'capFactor': cap,
                'median_TT_sim_s': median_tt_sim,
                'median_TT_obs_s': median_tt_obs,
                'median_ratio': median_ratio,
                'IQR_sim': iqr_sim,
                'matched_count': df['travel_time_sim_s'].notna().sum()
            })
            
            print(f"\n[结果] capFactor={cap}")
            print(f"  median(TT_sim) = {median_tt_sim:.1f}s")
            print(f"  median(TT_obs) = {median_tt_obs:.1f}s")  
            print(f"  ratio = {median_ratio:.3f}")
    
    # 输出对比表
    print("\n" + "="*60)
    print("capacityFactor 三点试验结果")
    print("="*60)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    # 保存结果
    output_path = PROJECT_ROOT / 'data/calibration/capfactor_test_results.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n[输出] 已保存到 {output_path}")
    
    # 判读
    print("\n[判读]")
    if len(results) >= 2:
        ratio_change = results[-1]['median_ratio'] - results[0]['median_ratio']
        if ratio_change > 0.2:
            print("✓ capacityFactor 是主旋钮 (ratio 随 capFactor 上升)")
            print("  建议: 继续用扩展后的先验范围跑 IES")
        else:
            print("✗ capacityFactor 效果有限")
            print("  建议: 检查统计窗口对齐 / 背景流注入位置")


if __name__ == "__main__":
    main()
