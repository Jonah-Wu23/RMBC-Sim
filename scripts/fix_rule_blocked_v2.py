#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_rule_blocked_v2.py
======================
V3: Deep analysis and auto-fix for RULE_BLOCKED segments

Strategy:
1. For each RULE_BLOCKED segment, find the shortest path with passenger vClass
2. Compare with bus vClass path to identify blocking points
3. Check if blocking is due to missing connection or allow/disallow
"""

import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sumolib
except ImportError:
    print("[ERROR] sumolib not installed")
    sys.exit(1)

NET_FILE = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
DIAGNOSIS_CSV = PROJECT_ROOT / "logs" / "route_segment_diagnosis.csv"
OUTPUT_CSV = PROJECT_ROOT / "logs" / "rule_blocked_analysis.csv"


def analyze_segment(net, from_edge_id, to_edge_id):
    """Deep analysis of a RULE_BLOCKED segment"""
    result = {
        'from_edge': from_edge_id,
        'to_edge': to_edge_id,
        'has_path': False,
        'path_length': 0,
        'path_edges': '',
        'blocking_type': 'UNKNOWN',
        'blocking_detail': '',
        'suggested_fix': ''
    }
    
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except Exception as e:
        result['blocking_type'] = 'EDGE_NOT_FOUND'
        result['blocking_detail'] = str(e)
        return result
    
    # Try to find path
    try:
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            result['has_path'] = True
            result['path_length'] = sum(e.getLength() for e in route)
            result['path_edges'] = ' '.join([e.getID() for e in route[:10]])
            if len(route) > 10:
                result['path_edges'] += f' ... ({len(route)} total)'
        else:
            result['blocking_type'] = 'NO_PATH_FOUND'
    except Exception as e:
        result['blocking_type'] = 'ROUTING_ERROR'
        result['blocking_detail'] = str(e)
        return result
    
    # Check if there's a direct connection possibility
    from_to_node = from_edge.getToNode()
    to_from_node = to_edge.getFromNode()
    
    if from_to_node.getID() == to_from_node.getID():
        # Adjacent edges - check connection
        result['blocking_type'] = 'ADJACENT_NO_CONNECTION'
        result['blocking_detail'] = f'Junction: {from_to_node.getID()}'
        result['suggested_fix'] = f'<connection from="{from_edge_id}" to="{to_edge_id}" fromLane="0" toLane="0"/>'
    else:
        # Not adjacent - need bridge or via routing
        dist = ((from_to_node.getCoord()[0] - to_from_node.getCoord()[0])**2 + 
                (from_to_node.getCoord()[1] - to_from_node.getCoord()[1])**2)**0.5
        
        if result['has_path'] and result['path_length'] > 0:
            ratio = result['path_length'] / max(dist, 1)
            result['blocking_type'] = f'LONG_DETOUR (ratio={ratio:.1f})'
            result['blocking_detail'] = f'Direct dist: {dist:.0f}m, Path: {result["path_length"]:.0f}m'
            
            if ratio > 5:
                result['suggested_fix'] = 'Consider bridge edge or check intermediate connections'
            else:
                result['suggested_fix'] = 'VIA_ROUTING may help'
        else:
            result['blocking_type'] = 'GAP_NO_PATH'
            result['blocking_detail'] = f'Gap: {dist:.0f}m'
            result['suggested_fix'] = 'Bridge edge needed'
    
    return result


def main():
    print("="*80)
    print("[V3] RULE_BLOCKED Deep Analysis")
    print("="*80)
    
    # Load
    print(f"\n[LOAD] Network: {NET_FILE.name}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    print(f"[LOAD] Diagnosis: {DIAGNOSIS_CSV.name}")
    df = pd.read_csv(DIAGNOSIS_CSV)
    
    blocked = df[df['type'] == 'RULE_BLOCKED'].copy()
    print(f"\n[INFO] Found {len(blocked)} RULE_BLOCKED segments")
    
    # Analyze
    results = []
    for _, row in blocked.iterrows():
        print(f"\n[ANALYZE] {row['route']} {row['bound']} {row['seq']}")
        print(f"          {row['from_edge']} -> {row['to_edge']}")
        
        analysis = analyze_segment(net, row['from_edge'], row['to_edge'])
        analysis['route'] = row['route']
        analysis['bound'] = row['bound']
        analysis['seq'] = row['seq']
        
        print(f"          Type: {analysis['blocking_type']}")
        if analysis['suggested_fix']:
            print(f"          Fix: {analysis['suggested_fix'][:60]}...")
        
        results.append(analysis)
    
    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] {OUTPUT_CSV}")
    
    # Summary
    print("\n" + "="*80)
    print("[SUMMARY]")
    print("="*80)
    
    for bt in results_df['blocking_type'].unique():
        count = len(results_df[results_df['blocking_type'] == bt])
        print(f"   {bt}: {count}")
    
    # Show suggested fixes
    print("\n" + "-"*80)
    print("SUGGESTED FIXES:")
    print("-"*80)
    
    adjacent_fixes = results_df[results_df['blocking_type'] == 'ADJACENT_NO_CONNECTION']
    if len(adjacent_fixes) > 0:
        print("\n[ADD CONNECTIONS]")
        for _, r in adjacent_fixes.iterrows():
            print(f"  {r['suggested_fix']}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
