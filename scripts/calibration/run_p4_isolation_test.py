#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_p4_isolation_test.py
========================
P4 隔离验证：禁用背景流，验证走廊压载是否有效

分三阶段：
- P4-0: 基线（公交-only）
- P4-1: 插车验证（900vph × 10min）
- P4-2: 正式三点（0/300/900 vph × 1h）
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/calibration'))

from build_l2_sim_vector_traveltime import build_simulation_vector_tt

# 68X inbound route edges
CORRIDOR_EDGES = "105735 105735_rev 273264_rev 105528 105501 105502 106883 106884_rev 106894_rev 105511_rev 105609_rev 284967_rev 284974 284930_rev 106938 261602 106935_rev 107180_rev 105753_rev 105729_rev 106963 106955 105653 105653_rev 105653 107154 106838 106831_rev 106831 106838_rev 107154_rev 107002 105952_rev 105929 115859 105926 105922 105923 105925 105910 105819 137853 137854 285166 105817 105832 105830 105829 105827 106986_rev 106985_rev 105836_rev 105886_rev 105866_rev 105880_rev 106995 106996 106993 106991 106728 106729 106073 105770_rev 106116_rev 106062_rev 106056_rev 105785_rev 105786_rev 106028 106053_rev 106077 106285_rev 106243_rev 105334_rev 105335 105335_rev 105336_rev 105351 105343 106429 107083 106344 106365 106366_rev 106367 106272_rev 106270_rev 106320 284524 106580 106608 106627 106628 107088 107087 106628 107088 106625 106623 106624 106537_rev 105377 106535 9343 9029 272309 9623_rev 8312_rev 9639_rev 8006_rev 9640_rev 8991_rev 8998 7993 8997 8303_rev 9023_rev 9023 8303 8302 9638 8337 9357 8327 116396 8024 8044 8663_rev 7654_rev 9043_rev 8975_rev 8696_rev 9677_rev"


def create_corridor_load_file(vph: int, output_path: Path):
    """创建走廊压载文件（带防插车失败参数）"""
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="bg_corridor" vClass="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="33.3" color="0,1,0"/>
    <route id="r_68x_in_corridor" edges="{CORRIDOR_EDGES}"/>
'''
    if vph > 0:
        content += f'''    <flow id="bg_corridor_{vph}vph" type="bg_corridor" route="r_68x_in_corridor"
          begin="0" end="3600" vehsPerHour="{vph}"
          departLane="free" departPos="random_free" departSpeed="max"/>
'''
    content += '</routes>'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_simulation(label: str, vph: int, duration: int = 3600) -> Path:
    """
    运行隔离仿真（直接用命令行参数，不依赖 sumocfg）
    
    Args:
        label: 输出目录标签
        vph: 走廊压载流量
        duration: 仿真时长（秒）
    """
    output_dir = PROJECT_ROOT / f'sumo/output/p4_isolation/{label}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建走廊压载文件
    corridor_load_path = output_dir / 'corridor_load.rou.xml'
    create_corridor_load_file(vph, corridor_load_path)
    
    # 构建路线文件列表
    route_files = [str(PROJECT_ROOT / 'sumo/routes/fixed_routes_cropped.rou.xml')]
    if vph > 0:
        route_files.append(str(corridor_load_path))
    
    # 直接用命令行参数（绕开 sumocfg，避免暗加载）
    cmd = [
        'sumo',
        '-n', str(PROJECT_ROOT / 'sumo/net/hk_cropped.net.xml'),
        '-r', ','.join(route_files),
        '-a', str(PROJECT_ROOT / 'sumo/additional/bus_stops_cropped.add.xml'),
        '-b', '0',
        '-e', str(duration),
        '--ignore-route-errors', 'true',
        '--time-to-teleport', '300',
        '--no-step-log', 'true',
        '--stop-output', str(output_dir / 'stopinfo.xml'),
        '--tripinfo-output', str(output_dir / 'tripinfo.xml'),
        '--summary-output', str(output_dir / 'summary.xml'),
        '--statistic-output', str(output_dir / 'statistics.xml'),
    ]
    
    print(f"\n[{label}] 运行仿真 (vph={vph}, duration={duration}s)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] SUMO 失败: {result.stderr[:500] if result.stderr else '(no stderr)'}")
        return None
    
    print(f"[{label}] 仿真完成")
    return output_dir


def check_insertion_metrics(output_dir: Path, expected_vph: int, duration: int) -> dict:
    """检查插车硬门槛指标（使用 statistics.xml）"""
    metrics = {
        'inserted': 0,
        'waiting': 0,
        'expected': int(expected_vph * duration / 3600),
        'insertion_rate': 0.0,
        'pass_threshold': False
    }
    
    # 解析 statistics.xml
    stats_path = output_dir / 'statistics.xml'
    if stats_path.exists():
        tree = ET.parse(stats_path)
        root = tree.getroot()
        
        veh_elem = root.find('.//vehicles')
        if veh_elem is not None:
            metrics['inserted'] = int(veh_elem.get('inserted', 0))
            metrics['waiting'] = int(veh_elem.get('waiting', 0))
    
    # 计算插入率（相对于预期）
    if metrics['expected'] > 0:
        metrics['insertion_rate'] = metrics['inserted'] / metrics['expected']
    
    # 判断是否通过硬门槛：需要足够多的压载车插入
    # 注意：公交车也包含在 inserted 中，所以 bg_corridor 实际插入 ≈ inserted - 5
    bg_inserted = max(0, metrics['inserted'] - 5)  # 减去公交车数量
    bg_expected = metrics['expected']
    bg_insertion_rate = bg_inserted / bg_expected if bg_expected > 0 else 0
    
    metrics['bg_inserted'] = bg_inserted
    metrics['bg_insertion_rate'] = bg_insertion_rate
    
    # 硬门槛：bg 插入率 >= 80% 且等待数合理（< 预期的 30%）
    metrics['pass_threshold'] = (
        bg_insertion_rate >= 0.80 and
        metrics['waiting'] < metrics['expected'] * 0.30
    )
    
    return metrics


def run_p4_0_baseline():
    """P4-0: 基线（公交-only）"""
    print("\n" + "="*60)
    print("P4-0: 基线（公交-only）")
    print("="*60)
    
    output_dir = run_simulation('p4_0_baseline', vph=0, duration=3600)
    if not output_dir:
        return None
    
    # 提取 TT
    df = build_simulation_vector_tt(
        stopinfo_path=str(output_dir / 'stopinfo.xml'),
        observation_csv=str(PROJECT_ROOT / 'data/calibration/l2_observation_vector_corridor_M11_TT.csv'),
        route_stop_csv=str(PROJECT_ROOT / 'data/processed/kmb_route_stop_dist.csv'),
        verbose=False
    )
    
    ratio = df['ratio'].median()
    print(f"  基线 ratio = {ratio:.3f}")
    return ratio


def run_p4_1_insertion_test():
    """P4-1: 插车验证（900vph × 10min）"""
    print("\n" + "="*60)
    print("P4-1: 插车验证（900vph × 10min）")
    print("="*60)
    
    output_dir = run_simulation('p4_1_insertion', vph=900, duration=600)
    if not output_dir:
        return None
    
    # 检查硬门槛
    metrics = check_insertion_metrics(output_dir, expected_vph=900, duration=600)
    
    print(f"  total inserted: {metrics['inserted']} (bg_corridor: {metrics['bg_inserted']} / {metrics['expected']}, rate: {metrics['bg_insertion_rate']*100:.1f}%)")
    print(f"  waiting: {metrics['waiting']}")
    
    if metrics['pass_threshold']:
        print("  ✓ 通过硬门槛，可进行 P4-2")
    else:
        print("  ✗ 未通过硬门槛，需进一步诊断")
    
    return metrics


def run_p4_2_three_point():
    """P4-2: 正式三点压载（0/300/900 vph × 1h）"""
    print("\n" + "="*60)
    print("P4-2: 正式三点压载（0/300/900 vph × 1h）")
    print("="*60)
    
    vphs = [0, 300, 900]
    results = []
    
    for vph in vphs:
        label = f'p4_2_vph{vph}'
        output_dir = run_simulation(label, vph=vph, duration=3600)
        if not output_dir:
            continue
        
        # 检查插车指标
        if vph > 0:
            metrics = check_insertion_metrics(output_dir, expected_vph=vph, duration=3600)
            print(f"  [{vph}vph] bg_inserted: {metrics['bg_inserted']}/{metrics['expected']} ({metrics['bg_insertion_rate']*100:.1f}%), waiting: {metrics['waiting']}")
        
        # 提取 TT
        df = build_simulation_vector_tt(
            stopinfo_path=str(output_dir / 'stopinfo.xml'),
            observation_csv=str(PROJECT_ROOT / 'data/calibration/l2_observation_vector_corridor_M11_TT.csv'),
            route_stop_csv=str(PROJECT_ROOT / 'data/processed/kmb_route_stop_dist.csv'),
            verbose=False
        )
        
        results.append({
            'vph': vph,
            'median_TT_sim_s': df['travel_time_sim_s'].median(),
            'median_TT_obs_s': df['travel_time_obs_s'].median(),
            'median_ratio': df['ratio'].median(),
            'matched_count': df['travel_time_sim_s'].notna().sum()
        })
        
        print(f"  [{vph}vph] ratio = {results[-1]['median_ratio']:.3f}")
    
    # 输出对比表
    print("\n" + "-"*60)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    # 保存结果
    output_path = PROJECT_ROOT / 'data/calibration/p4_isolation_results.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n[输出] 已保存到 {output_path}")
    
    return results


def main():
    print("="*60)
    print("P4 隔离验证：禁用背景流，验证走廊压载效果")
    print("="*60)
    
    # P4-0: 基线
    baseline_ratio = run_p4_0_baseline()
    
    # P4-1: 插车验证
    insertion_metrics = run_p4_1_insertion_test()
    
    if insertion_metrics and insertion_metrics['pass_threshold']:
        # P4-2: 正式三点
        results = run_p4_2_three_point()
        
        # 判读
        print("\n" + "="*60)
        print("P4 判读")
        print("="*60)
        if results and len(results) >= 2:
            ratio_0 = results[0]['median_ratio']
            ratio_900 = results[-1]['median_ratio']
            delta = ratio_900 - ratio_0
            
            if delta > 0.15:
                print(f"✓ 走廊压载是主旋钮 (ratio 从 {ratio_0:.3f} → {ratio_900:.3f}, Δ={delta:+.3f})")
                print("  建议: 进入 P5 修复背景流 OD")
            elif delta > 0.05:
                print(f"△ 走廊压载有效但不够强 (Δ={delta:+.3f})")
                print("  建议: 检查公交与压载车是否同车道交互")
            else:
                print(f"✗ 走廊压载效果有限 (Δ={delta:+.3f})")
                print("  建议: 检查车道权限 / 公交专用道 / 压载路线")
    else:
        print("\n[警告] P4-1 插车验证未通过，跳过 P4-2")
        print("  需进一步诊断插车失败原因")


if __name__ == "__main__":
    main()
