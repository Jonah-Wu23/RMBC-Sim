#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_v3_patches.py
=================
V3: Generate patches for high-ratio LONG_DETOUR segments

Focus on ratio > 5 segments that need bridge edges
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sumolib
except ImportError:
    print("[ERROR] sumolib not installed")
    sys.exit(1)

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"

# High ratio segments that need bridge edges (ratio > 5)
HIGH_RATIO_SEGMENTS = [
    # (bridge_id, from_edge, to_edge, description)
    ("bridge_68X_GAP_12", "97407", "96273_rev", "68X outbound 6->7, ratio=121.8, direct_dist=13m"),
    ("bridge_68X_GAP_13", "105735_rev", "105653_rev", "68X inbound 1->2, ratio=6.0, direct_dist=397m"),
]


def main():
    print("="*80)
    print("[V3] Generate Bridge Edge Patches")
    print("="*80)
    
    print(f"\n[LOAD] Network: {NET_FILE.name}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    edge_snippets = []
    conn_snippets = []
    
    for bridge_id, fe, te, desc in HIGH_RATIO_SEGMENTS:
        print(f"\n[GEN] {bridge_id}")
        print(f"      {fe} -> {te}")
        print(f"      {desc}")
        
        try:
            e_from = net.getEdge(fe)
            e_to = net.getEdge(te)
        except Exception as ex:
            print(f"      [ERROR] Cannot find edge: {ex}")
            continue
        
        n_from = e_from.getToNode()
        n_to = e_to.getFromNode()
        
        x1, y1 = n_from.getCoord()
        x2, y2 = n_to.getCoord()
        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        
        print(f"      Nodes: {n_from.getID()} -> {n_to.getID()}")
        print(f"      Distance: {dist:.1f}m")
        
        shape = f"{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}"
        
        # Generate edge XML
        edge_xml = (
            f'    <!-- {desc} -->\n'
            f'    <edge id="{bridge_id}" from="{n_from.getID()}" to="{n_to.getID()}" '
            f'numLanes="1" speed="5.56" priority="1" allow="bus" shape="{shape}"/>'
        )
        edge_snippets.append(edge_xml)
        
        # Generate connection XML
        conn_xml = (
            f'    <connection from="{fe}" to="{bridge_id}" fromLane="0" toLane="0"/>\n'
            f'    <connection from="{bridge_id}" to="{te}" fromLane="0" toLane="0"/>'
        )
        conn_snippets.append(conn_xml)
    
    # Output
    output_dir = PROJECT_ROOT / "tmp"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("Edge patches:")
    print("="*80)
    for s in edge_snippets:
        print(s)
    
    print("\n" + "="*80)
    print("Connection patches:")
    print("="*80)
    for s in conn_snippets:
        print(s)
    
    # Save
    with open(output_dir / "v3_bridge_edges.xml", "w", encoding="utf-8") as f:
        f.write("<!-- V3 Bridge Edge Patches -->\n")
        for s in edge_snippets:
            f.write(s + "\n")
    
    with open(output_dir / "v3_bridge_connections.xml", "w", encoding="utf-8") as f:
        f.write("<!-- V3 Bridge Connection Patches -->\n")
        for s in conn_snippets:
            f.write(s + "\n")
    
    print(f"\n[SAVED]")
    print(f"  {output_dir / 'v3_bridge_edges.xml'}")
    print(f"  {output_dir / 'v3_bridge_connections.xml'}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
