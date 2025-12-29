#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_corridor_load_test.py
=========================
走廊压载三点试验：验证"走廊压载"是否是 TT 的主旋钮

固定背景流（原地流），只改变走廊压载量（0/300/900 vph）
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/calibration'))

from build_l2_sim_vector_traveltime import build_simulation_vector_tt


def create_corridor_load_file(vph: int, output_path: Path):
    """创建指定流量的走廊压载文件"""
    
    # 68X inbound route edges
    route_edges = "105735 105735_rev 273264_rev 105528 105501 105502 106883 106884_rev 106894_rev 105511_rev 105609_rev 284967_rev 284974 284930_rev 106938 261602 106935_rev 107180_rev 105753_rev 105729_rev 106963 106955 105653 105653_rev 105653 107154 106838 106831_rev 106831 106838_rev 107154_rev 107002 105952_rev 105929 115859 105926 105922 105923 105925 105910 105819 137853 137854 285166 105817 105832 105830 105829 105827 106986_rev 106985_rev 105836_rev 105886_rev 105866_rev 105880_rev 106995 106996 106993 106991 106728 106729 106073 105770_rev 106116_rev 106062_rev 106056_rev 105785_rev 105786_rev 106028 106053_rev 106077 106285_rev 106243_rev 105334_rev 105335 105335_rev 105336_rev 105351 105343 106429 107083 106344 106365 106366_rev 106367 106272_rev 106270_rev 106320 284524 106580 106608 106627 106628 107088 107087 106628 107088 106625 106623 106624 106537_rev 105377 106535 9343 9029 272309 9623_rev 8312_rev 9639_rev 8006_rev 9640_rev 8991_rev 8998 7993 8997 8303_rev 9023_rev 9023 8303 8302 9638 8337 9357 8327 116396 8024 8044 8663_rev 7654_rev 9043_rev 8975_rev 8696_rev 9677_rev"
    
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="bg_corridor" vClass="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="33.3" color="0,1,0"/>
    <route id="r_68x_in_corridor" edges="{route_edges}"/>
'''
    
    if vph > 0:
        content += f'''    <flow id="bg_corridor_{vph}vph" type="bg_corridor" route="r_68x_in_corridor"
          begin="0" end="3600" vehsPerHour="{vph}"
          departLane="best" departPos="random" departSpeed="max"/>
'''
    
    content += '</routes>'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_single_simulation(vph: int, label: str) -> Path:
    """运行单次仿真"""
    
    output_dir = PROJECT_ROOT / f'sumo/output/corridor_load_test/{label}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对应流量的走廊压载文件
    corridor_load_path = output_dir / 'corridor_load.rou.xml'
    create_corridor_load_file(vph, corridor_load_path)
    
    # 构建 SUMO 配置
    sumocfg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{PROJECT_ROOT}/sumo/net/hk_cropped.net.xml"/>
        <route-files value="{PROJECT_ROOT}/sumo/routes/fixed_routes_cropped.rou.xml,{PROJECT_ROOT}/sumo/routes/background_cropped.rou.xml,{corridor_load_path}"/>
        <additional-files value="{PROJECT_ROOT}/sumo/additional/bus_stops_cropped.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
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
    print(f"\n[{label}] 运行仿真 (corridor load = {vph} vph)...")
    cmd = ['sumo', '-c', str(sumocfg_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] SUMO 运行失败: {result.stderr[:500]}")
        return None
    
    print(f"[{label}] 仿真完成")
    return output_dir / 'stopinfo.xml'


def main():
    # 三点压载：0 / 300 / 900 vph
    vphs = [0, 300, 900]
    results = []
    
    for vph in vphs:
        label = f'vph{vph}'
        stopinfo_path = run_single_simulation(vph, label)
        
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
                'corridor_vph': vph,
                'median_TT_sim_s': median_tt_sim,
                'median_TT_obs_s': median_tt_obs,
                'median_ratio': median_ratio,
                'IQR_sim': iqr_sim,
                'matched_count': df['travel_time_sim_s'].notna().sum()
            })
            
            print(f"\n[结果] corridor_vph={vph}")
            print(f"  median(TT_sim) = {median_tt_sim:.1f}s")
            print(f"  median(TT_obs) = {median_tt_obs:.1f}s")  
            print(f"  ratio = {median_ratio:.3f}")
    
    # 输出对比表
    print("\n" + "="*60)
    print("走廊压载三点试验结果")
    print("="*60)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    # 保存结果
    output_path = PROJECT_ROOT / 'data/calibration/corridor_load_test_results.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n[输出] 已保存到 {output_path}")
    
    # 判读
    print("\n[判读]")
    if len(results) >= 2:
        ratio_0 = results[0]['median_ratio']
        ratio_last = results[-1]['median_ratio']
        ratio_change = ratio_last - ratio_0
        
        if ratio_change > 0.1:
            print(f"✓ 走廊压载是主旋钮 (ratio 从 {ratio_0:.3f} → {ratio_last:.3f}, +{ratio_change:.3f})")
            print("  建议: 修复背景流 OD，确保足够比例的车穿越走廊")
        elif ratio_change > 0.02:
            print(f"△ 走廊压载有效但不够强 (ratio 变化 {ratio_change:.3f})")
            print("  建议: 可能需要更高压载 / 检查 teleport 抹平")
        else:
            print(f"✗ 走廊压载效果仍有限 (ratio 变化 {ratio_change:.3f})")
            print("  建议: 检查车道权限 / 并道冲突 / teleport")


if __name__ == "__main__":
    main()
