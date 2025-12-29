#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_teleport.py
====================
诊断走廊压载仿真中的 teleport 情况
"""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    # 68X inbound route edges
    route_edges = "105735 105735_rev 273264_rev 105528 105501 105502 106883 106884_rev 106894_rev 105511_rev 105609_rev 284967_rev 284974 284930_rev 106938 261602 106935_rev 107180_rev 105753_rev 105729_rev 106963 106955 105653 105653_rev 105653 107154 106838 106831_rev 106831 106838_rev 107154_rev 107002 105952_rev 105929 115859 105926 105922 105923 105925 105910 105819 137853 137854 285166 105817 105832 105830 105829 105827 106986_rev 106985_rev 105836_rev 105886_rev 105866_rev 105880_rev 106995 106996 106993 106991 106728 106729 106073 105770_rev 106116_rev 106062_rev 106056_rev 105785_rev 105786_rev 106028 106053_rev 106077 106285_rev 106243_rev 105334_rev 105335 105335_rev 105336_rev 105351 105343 106429 107083 106344 106365 106366_rev 106367 106272_rev 106270_rev 106320 284524 106580 106608 106627 106628 107088 107087 106628 107088 106625 106623 106624 106537_rev 105377 106535 9343 9029 272309 9623_rev 8312_rev 9639_rev 8006_rev 9640_rev 8991_rev 8998 7993 8997 8303_rev 9023_rev 9023 8303 8302 9638 8337 9357 8327 116396 8024 8044 8663_rev 7654_rev 9043_rev 8975_rev 8696_rev 9677_rev"

    output_dir = PROJECT_ROOT / 'sumo/output/corridor_load_test/diag900'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建走廊压载文件
    corridor_load_path = output_dir / 'corridor_load.rou.xml'
    with open(corridor_load_path, 'w') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="bg_corridor" vClass="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="33.3" color="0,1,0"/>
    <route id="r_68x_in_corridor" edges="{route_edges}"/>
    <flow id="bg_corridor_900vph" type="bg_corridor" route="r_68x_in_corridor"
          begin="0" end="3600" vehsPerHour="900"
          departLane="best" departPos="random" departSpeed="max"/>
</routes>''')

    # SUMO 配置
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
        <verbose value="true"/>
        <no-warnings value="false"/>
        <no-step-log value="true"/>
    </report>
    <output>
        <stop-output value="{output_dir}/stopinfo.xml"/>
        <tripinfo-output value="{output_dir}/tripinfo.xml"/>
        <statistic-output value="{output_dir}/statistics.xml"/>
    </output>
</configuration>'''

    sumocfg_path = output_dir / 'test.sumocfg'
    with open(sumocfg_path, 'w') as f:
        f.write(sumocfg_content)

    print(f'[开始] 运行诊断仿真 (900 vph)...')
    result = subprocess.run(['sumo', '-c', str(sumocfg_path)], capture_output=True, text=True)
    print(f'[完成] returncode={result.returncode}')
    
    if result.returncode != 0:
        print(f'[ERROR] stderr: {result.stderr[:1000] if result.stderr else "(empty)"}')
        return
    
    # 解析 tripinfo
    tripinfo_path = output_dir / 'tripinfo.xml'
    if tripinfo_path.exists():
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
        
        # 统计车辆类型
        vclass_counts = Counter()
        teleport_counts = Counter()
        duration_by_type = {}
        
        for trip in root.findall('.//tripinfo'):
            vtype = trip.get('vType', 'unknown')
            vclass_counts[vtype] += 1
            
            # 检查 teleport
            reroute_no = int(trip.get('rerouteNo', 0))
            if reroute_no > 0:
                teleport_counts[vtype] += 1
            
            # 记录行驶时间
            duration = float(trip.get('duration', 0))
            if vtype not in duration_by_type:
                duration_by_type[vtype] = []
            duration_by_type[vtype].append(duration)
        
        print(f'\n=== 车辆类型统计 ===')
        for vtype, count in vclass_counts.most_common():
            print(f'  {vtype}: {count} 辆')
        
        print(f'\n=== Teleport/Reroute 统计 ===')
        for vtype, count in teleport_counts.most_common():
            print(f'  {vtype}: {count} 辆发生 reroute')
        if not teleport_counts:
            print('  (没有 reroute 记录)')
        
        print(f'\n=== 行驶时间统计 (秒) ===')
        for vtype in ['bg_corridor', 'bus', 'Bus']:
            if vtype in duration_by_type:
                durations = duration_by_type[vtype]
                avg = sum(durations) / len(durations)
                print(f'  {vtype}: count={len(durations)}, avg={avg:.0f}s, min={min(durations):.0f}s, max={max(durations):.0f}s')
    else:
        print(f'[WARNING] tripinfo.xml 不存在')
    
    # 解析 statistics
    stats_path = output_dir / 'statistics.xml'
    if stats_path.exists():
        tree = ET.parse(stats_path)
        root = tree.getroot()
        
        print(f'\n=== 仿真统计 ===')
        for child in root:
            if child.tag == 'vehicleTripStatistics':
                for key in ['count', 'routeLength', 'duration', 'speed', 'timeLoss', 'departDelay', 'totalTravelTime', 'totalDepartDelay']:
                    if key in child.attrib:
                        print(f'  {key}: {child.get(key)}')
            elif child.tag == 'teleports':
                for key in ['total', 'jam', 'yield', 'wrongLane']:
                    if key in child.attrib:
                        print(f'  teleport_{key}: {child.get(key)}')
    else:
        print(f'[WARNING] statistics.xml 不存在')

if __name__ == '__main__':
    main()
