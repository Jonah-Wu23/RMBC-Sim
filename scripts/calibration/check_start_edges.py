#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查走廊起点边的情况"""

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
net_path = PROJECT_ROOT / 'sumo/net/hk_cropped.net.xml'

tree = ET.parse(net_path)
root = tree.getroot()

# 检查走廊起点边
start_edges = ['105735', '105735_rev', '273264_rev']
print("=== 走廊起点边信息 ===")
for edge_id in start_edges:
    edge = root.find(f".//edge[@id='{edge_id}']")
    if edge is not None:
        lanes = edge.findall('lane')
        print(f"\nEdge {edge_id}: {len(lanes)} lanes")
        for lane in lanes:
            print(f"  lane {lane.get('id')}:")
            print(f"    length={lane.get('length')}, speed={lane.get('speed')}")
            print(f"    allow={lane.get('allow', 'all')}, disallow={lane.get('disallow', 'none')}")
    else:
        print(f"\nEdge {edge_id}: NOT FOUND")

# 检查前 5 条走廊边
route_edges = "105735 105735_rev 273264_rev 105528 105501".split()
print("\n\n=== 走廊前 5 条边 ===")
for edge_id in route_edges:
    edge = root.find(f".//edge[@id='{edge_id}']")
    if edge is not None:
        lanes = edge.findall('lane')
        lane_summary = [f"{l.get('id')}(allow={l.get('allow','all')})" for l in lanes]
        print(f"  {edge_id}: {len(lanes)} lanes - {lane_summary}")
    else:
        print(f"  {edge_id}: NOT FOUND")
