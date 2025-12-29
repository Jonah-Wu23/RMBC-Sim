#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_topology_p0.py
==================
P0 拓扑修复：为 106831_rev → 105817 段添加桥接边

方案：
1. 导出 plain 格式网络
2. 在 edg.xml 添加桥接边 bridge_68X_P0（带 shape，约 420m）
3. 在 con.xml 添加两端 connections
4. 重建 patched 网络
5. 验证连通性

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-27
"""

import os
import subprocess
import sys
import math
from pathlib import Path

try:
    import sumolib
except ImportError:
    print("⚠️ sumolib 未安装")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd, cwd=None):
    """运行命令并返回结果"""
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️ Error: {result.stderr}")
    else:
        print(f"  ✓ Success")
    return result.returncode == 0


def main():
    net_path = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3.net.xml"
    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    plain_prefix = tmp_dir / "hk_plain"
    patched_net = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched.net.xml"
    
    print("=" * 80)
    print("[P0 拓扑修复] 106831_rev → 105817 桥接边")
    print("=" * 80)
    
    # Step 1: 验证节点信息
    print("\n[Step 1] 验证节点信息...")
    net = sumolib.net.readNet(str(net_path))
    
    e1 = net.getEdge('106831_rev')
    e2 = net.getEdge('105817')
    
    from_node_id = e1.getToNode().getID()      # 7178
    to_node_id = e2.getFromNode().getID()      # 12808
    from_coord = e1.getToNode().getCoord()     # (22760.24, 7773.94)
    to_coord = e2.getFromNode().getCoord()     # (22890.47, 8068.15)
    
    print(f"  from_node: {from_node_id} @ ({from_coord[0]:.2f}, {from_coord[1]:.2f})")
    print(f"  to_node: {to_node_id} @ ({to_coord[0]:.2f}, {to_coord[1]:.2f})")
    
    # 计算直线距离
    direct_dist = math.sqrt((to_coord[0] - from_coord[0])**2 + (to_coord[1] - from_coord[1])**2)
    print(f"  直线距离: {direct_dist:.2f}m")
    
    # 目标长度：KMB 段长 420m
    target_length = 420.0
    print(f"  目标长度: {target_length:.2f}m (KMB 段长)")
    
    # 计算 shape：插入中间拐点使长度接近 420m
    # 先简单做：往侧边偏移，形成弧线
    mid_x = (from_coord[0] + to_coord[0]) / 2
    mid_y = (from_coord[1] + to_coord[1]) / 2
    
    # 偏移量：使总长度接近 420m
    # 两条弦长各约 210m，直线距离 321m/2 = 160m
    # sqrt(210^2 - 160^2) ≈ 136m 横向偏移
    offset = 100.0  # 保守值，约使总长 ~400m
    
    # 计算偏移方向（垂直于连线）
    dx = to_coord[0] - from_coord[0]
    dy = to_coord[1] - from_coord[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    # 单位法向量（左侧）
    nx = -dy / length
    ny = dx / length
    
    mid_offset_x = mid_x + nx * offset
    mid_offset_y = mid_y + ny * offset
    
    # shape 字符串
    shape_str = f"{from_coord[0]:.2f},{from_coord[1]:.2f} {mid_offset_x:.2f},{mid_offset_y:.2f} {to_coord[0]:.2f},{to_coord[1]:.2f}"
    
    # 计算实际长度
    seg1 = math.sqrt((mid_offset_x - from_coord[0])**2 + (mid_offset_y - from_coord[1])**2)
    seg2 = math.sqrt((to_coord[0] - mid_offset_x)**2 + (to_coord[1] - mid_offset_y)**2)
    actual_length = seg1 + seg2
    print(f"  桥接边实际长度: {actual_length:.2f}m")
    print(f"  shape: {shape_str}")
    
    # Step 2: 导出 plain 格式
    print("\n[Step 2] 导出 plain 格式...")
    cmd = f'netconvert -s "{net_path}" --plain-output-prefix "{plain_prefix}"'
    if not run_cmd(cmd, cwd=PROJECT_ROOT):
        print("  ❌ 导出失败")
        return
    
    # Step 3: 修改 edg.xml 添加桥接边
    print("\n[Step 3] 添加桥接边到 edg.xml...")
    edg_file = Path(str(plain_prefix) + ".edg.xml")
    
    if not edg_file.exists():
        print(f"  ❌ 文件不存在: {edg_file}")
        return
    
    # 读取并修改
    with open(edg_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在 </edges> 前插入桥接边
    bridge_edge = f'''
    <!-- P0 桥接边: 106831_rev → 105817 -->
    <edge id="bridge_68X_P0" from="{from_node_id}" to="{to_node_id}" 
          numLanes="1" speed="5.56" priority="1" allow="bus"
          shape="{shape_str}"/>
'''
    
    if 'bridge_68X_P0' not in content:
        content = content.replace('</edges>', bridge_edge + '</edges>')
        with open(edg_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ 桥接边已添加")
    else:
        print("  ⚠️ 桥接边已存在，跳过")
    
    # Step 4: 修改 con.xml 添加 connections
    print("\n[Step 4] 添加 connections 到 con.xml...")
    con_file = Path(str(plain_prefix) + ".con.xml")
    
    if not con_file.exists():
        print(f"  ❌ 文件不存在: {con_file}")
        return
    
    with open(con_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    connections = '''
    <!-- P0 桥接边 connections -->
    <connection from="106831_rev" to="bridge_68X_P0" fromLane="0" toLane="0"/>
    <connection from="bridge_68X_P0" to="105817" fromLane="0" toLane="0"/>
'''
    
    if 'bridge_68X_P0' not in content:
        content = content.replace('</connections>', connections + '</connections>')
        with open(con_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Connections 已添加")
    else:
        print("  ⚠️ Connections 已存在，跳过")
    
    # Step 5: 重建 patched 网络
    print("\n[Step 5] 重建 patched 网络...")
    
    # 检查是否有 typ.xml
    typ_file = Path(str(plain_prefix) + ".typ.xml")
    typ_arg = f'--type-files "{typ_file}"' if typ_file.exists() else ""
    
    # 检查是否有 tll.xml
    tll_file = Path(str(plain_prefix) + ".tll.xml")
    tll_arg = f'--tllogic-files "{tll_file}"' if tll_file.exists() else ""
    
    cmd = f'''netconvert \
      --node-files "{plain_prefix}.nod.xml" \
      --edge-files "{plain_prefix}.edg.xml" \
      --connection-files "{plain_prefix}.con.xml" \
      {typ_arg} {tll_arg} \
      -o "{patched_net}"'''
    
    cmd = ' '.join(cmd.split())  # 合并换行
    
    if not run_cmd(cmd, cwd=PROJECT_ROOT):
        print("  ❌ 网络重建失败")
        return
    
    # Step 6: 验证桥接边连通性
    print("\n[Step 6] 验证桥接边连通性...")
    net_patched = sumolib.net.readNet(str(patched_net))
    
    try:
        bridge = net_patched.getEdge('bridge_68X_P0')
        print(f"  ✓ 桥接边存在: {bridge.getID()}")
        print(f"    长度: {bridge.getLength():.2f}m")
        print(f"    速度限制: {bridge.getSpeed():.2f} m/s ({bridge.getSpeed()*3.6:.1f} km/h)")
        print(f"    允许车型: {bridge.getAllowedStr()}")
        
        # 检查 connection
        incoming = list(bridge.getIncoming())
        outgoing = list(bridge.getOutgoing())
        print(f"    入边: {[e.getID() for e in incoming]}")
        print(f"    出边: {[e.getID() for e in outgoing]}")
        
        if '106831_rev' in [e.getID() for e in incoming]:
            print("  ✓ 106831_rev → bridge_68X_P0 连接正常")
        else:
            print("  ⚠️ 106831_rev → bridge_68X_P0 连接缺失")
        
        if '105817' in [e.getID() for e in outgoing]:
            print("  ✓ bridge_68X_P0 → 105817 连接正常")
        else:
            print("  ⚠️ bridge_68X_P0 → 105817 连接缺失")
            
    except Exception as e:
        print(f"  ❌ 桥接边不存在: {e}")
        return
    
    # Step 7: 测试路由
    print("\n[Step 7] 测试 106831_rev → 105817 最短路...")
    try:
        from_edge = net_patched.getEdge('106831_rev')
        to_edge = net_patched.getEdge('105817')
        route, cost = net_patched.getShortestPath(from_edge, to_edge)
        
        if route:
            edge_ids = [e.getID() for e in route]
            total_len = sum(e.getLength() for e in route)
            print(f"  ✓ 路径可达!")
            print(f"    路径: {edge_ids}")
            print(f"    总长: {total_len:.2f}m")
            print(f"    ratio: {total_len/420:.2f} (KMB段长420m)")
            
            # 检查是否使用了桥接边
            if 'bridge_68X_P0' in edge_ids:
                print("  ✓ 路径使用了桥接边!")
            else:
                print("  ⚠️ 路径未使用桥接边（可能有其它更短路径）")
        else:
            print("  ❌ 仍然无路径")
    except Exception as e:
        print(f"  ❌ 路由测试失败: {e}")
    
    print("\n" + "=" * 80)
    print("[完成] patched 网络已生成: sumo/net/hk_irn_v3_patched.net.xml")
    print("=" * 80)


if __name__ == '__main__':
    main()
