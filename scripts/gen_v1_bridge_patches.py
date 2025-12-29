#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_v1_bridge_patches.py
========================
V1阶段：生成2条NO_ALT_PATH的bridge edge补丁

自动从网络获取正确的node ID和坐标，生成XML片段供插入plain文件
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sumolib
except ImportError:
    print("❌ sumolib 未安装")
    sys.exit(1)

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched.net.xml"

# 需要修复的2条 NO_ALT_PATH
BRIDGES = [
    # (new_edge_id, from_edge, to_edge, comment)
    ("bridge_68X_GAP_11", "105734", "105735_rev", "68X outbound 30→31, node_dist≈150m"),
    ("bridge_960_GAP_13", "94236_rev", "94664_rev", "960 outbound 2→3, node_dist≈101m"),
]

def main():
    print(f"📖 读取网络: {NET_FILE}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    edge_snippets = []
    conn_snippets = []
    
    for new_id, fe, te, comment in BRIDGES:
        try:
            e_from = net.getEdge(fe)
            e_to = net.getEdge(te)
        except Exception as ex:
            print(f"❌ 找不到边 {fe} 或 {te}: {ex}")
            continue
        
        n_from = e_from.getToNode()
        n_to = e_to.getFromNode()
        
        x1, y1 = n_from.getCoord()
        x2, y2 = n_to.getCoord()
        
        # 计算距离
        import math
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        shape = f"{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}"
        
        print(f"\n✅ {new_id}")
        print(f"   {fe} (to_node={n_from.getID()}) → {te} (from_node={n_to.getID()})")
        print(f"   距离: {dist:.1f}m")
        
        # 生成 edge XML
        edge_xml = (
            f'    <!-- {comment} -->\n'
            f'    <edge id="{new_id}" from="{n_from.getID()}" to="{n_to.getID()}" '
            f'numLanes="1" speed="5.56" priority="1" allow="bus" shape="{shape}"/>'
        )
        edge_snippets.append(edge_xml)
        
        # 生成 connection XML
        conn_xml = (
            f'    <connection from="{fe}" to="{new_id}" fromLane="0" toLane="0"/>\n'
            f'    <connection from="{new_id}" to="{te}" fromLane="0" toLane="0"/>'
        )
        conn_snippets.append(conn_xml)
    
    print("\n" + "="*60)
    print("📝 Edge XML (插入到 tmp/hk_plain.edg.xml 的 </edges> 之前):")
    print("="*60)
    for s in edge_snippets:
        print(s)
    
    print("\n" + "="*60)
    print("📝 Connection XML (插入到 tmp/hk_plain.con.xml 的 </connections> 之前):")
    print("="*60)
    for s in conn_snippets:
        print(s)
    
    # 保存到临时文件供后续使用
    output_dir = PROJECT_ROOT / "tmp"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "v1_bridge_edges.xml", "w", encoding="utf-8") as f:
        f.write("<!-- V1 Bridge Edge Patches -->\n")
        for s in edge_snippets:
            f.write(s + "\n")
    
    with open(output_dir / "v1_bridge_connections.xml", "w", encoding="utf-8") as f:
        f.write("<!-- V1 Bridge Connection Patches -->\n")
        for s in conn_snippets:
            f.write(s + "\n")
    
    print(f"\n✅ 已保存到:")
    print(f"   {output_dir / 'v1_bridge_edges.xml'}")
    print(f"   {output_dir / 'v1_bridge_connections.xml'}")

if __name__ == "__main__":
    main()
