#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_rule_blocked.py
========================
V3: Diagnose RULE_BLOCKED segments

For each RULE_BLOCKED segment, determine the root cause:
- CASE A: Missing connection between edges
- CASE B: Edge/lane disallows bus vClass

Output: Specific fix recommendations (add connection vs modify allow)
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
OUTPUT_CSV = PROJECT_ROOT / "logs" / "rule_blocked_fixes.csv"
OUTPUT_CON_XML = PROJECT_ROOT / "tmp" / "v3_fix_connections.xml"
OUTPUT_EDGE_XML = PROJECT_ROOT / "tmp" / "v3_fix_edges.xml"


def check_connection_exists(net, from_edge_id, to_edge_id):
    """Check if a connection exists between two edges in any lane combination"""
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except:
        return False, None, None
    
    # Check all lane combinations
    for from_lane in from_edge.getLanes():
        outgoing = from_lane.getOutgoing()
        for conn in outgoing:
            # conn.getTo() returns the target lane, getEdge() gets its parent edge
            target_lane = conn.getTo()
            if hasattr(target_lane, 'getEdge'):
                target_edge_id = target_lane.getEdge().getID()
            else:
                # In some sumolib versions, getTo() returns edge directly
                target_edge_id = target_lane.getID()
            if target_edge_id == to_edge_id:
                return True, from_lane.getIndex(), conn.getToLane().getIndex() if hasattr(conn, 'getToLane') else 0
    
    return False, None, None


def check_bus_allowed(net, edge_id):
    """Check if bus is allowed on the edge"""
    try:
        edge = net.getEdge(edge_id)
    except:
        return None, "EDGE_NOT_FOUND"
    
    for lane in edge.getLanes():
        # Use private attributes for sumolib compatibility
        allowed = getattr(lane, '_allowed', None) or getattr(lane, 'allowed', [])
        disallowed = getattr(lane, '_disallowed', None) or getattr(lane, 'disallowed', [])
        
        # If allow list exists and bus is not in it
        if allowed and 'bus' not in allowed:
            return False, f"lane {lane.getIndex()}: allow={allowed}"
        
        # If disallow list contains bus
        if disallowed and 'bus' in disallowed:
            return False, f"lane {lane.getIndex()}: disallow contains bus"
    
    return True, "bus allowed"


def find_path_with_passenger(net, from_edge_id, to_edge_id):
    """Try to find a path using passenger vClass (ignoring bus restrictions)"""
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except:
        return None, []
    
    # Standard shortest path
    route, cost = net.getShortestPath(from_edge, to_edge)
    if route:
        return cost, [e.getID() for e in route]
    return None, []


def diagnose_segment(net, from_edge_id, to_edge_id):
    """Diagnose why a segment is RULE_BLOCKED"""
    result = {
        'from_edge': from_edge_id,
        'to_edge': to_edge_id,
        'root_cause': 'UNKNOWN',
        'fix_type': '',
        'fix_detail': '',
        'blocking_edge': '',
        'blocking_reason': ''
    }
    
    # Check if direct connection exists
    conn_exists, from_lane, to_lane = check_connection_exists(net, from_edge_id, to_edge_id)
    
    if not conn_exists:
        # Check if edges are adjacent (could add direct connection)
        try:
            from_edge = net.getEdge(from_edge_id)
            to_edge = net.getEdge(to_edge_id)
            
            # Check if from_edge's to_node is same as to_edge's from_node
            if from_edge.getToNode().getID() == to_edge.getFromNode().getID():
                result['root_cause'] = 'MISSING_DIRECT_CONNECTION'
                result['fix_type'] = 'ADD_CONNECTION'
                result['fix_detail'] = f'<connection from="{from_edge_id}" to="{to_edge_id}" fromLane="0" toLane="0"/>'
                return result
        except:
            pass
    
    # Try to find a path and identify blocking point
    cost, path = find_path_with_passenger(net, from_edge_id, to_edge_id)
    
    if path:
        # Check each edge in path for bus restriction
        for edge_id in path:
            bus_ok, reason = check_bus_allowed(net, edge_id)
            if bus_ok is False:
                result['root_cause'] = 'BUS_DISALLOWED'
                result['fix_type'] = 'MODIFY_ALLOW'
                result['blocking_edge'] = edge_id
                result['blocking_reason'] = reason
                result['fix_detail'] = f'Add bus to allow on edge {edge_id}'
                return result
        
        # Check connections between edges in path
        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i+1]
            conn_ok, _, _ = check_connection_exists(net, e1, e2)
            if not conn_ok:
                result['root_cause'] = 'MISSING_PATH_CONNECTION'
                result['fix_type'] = 'ADD_CONNECTION'
                result['blocking_edge'] = f'{e1} -> {e2}'
                result['fix_detail'] = f'<connection from="{e1}" to="{e2}" fromLane="0" toLane="0"/>'
                return result
    
    result['root_cause'] = 'COMPLEX_ISSUE'
    result['fix_type'] = 'MANUAL_REVIEW'
    result['fix_detail'] = 'Requires manual investigation'
    
    return result


def main():
    print("="*80)
    print("[V3] RULE_BLOCKED Diagnosis")
    print("="*80)
    
    # Load network
    print(f"\n[LOAD] Network: {NET_FILE.name}")
    net = sumolib.net.readNet(str(NET_FILE))
    
    # Load diagnosis
    print(f"[LOAD] Diagnosis: {DIAGNOSIS_CSV.name}")
    df = pd.read_csv(DIAGNOSIS_CSV)
    
    # Filter RULE_BLOCKED
    blocked = df[df['type'] == 'RULE_BLOCKED'].copy()
    print(f"\n[INFO] Found {len(blocked)} RULE_BLOCKED segments")
    
    if len(blocked) == 0:
        print("[INFO] No RULE_BLOCKED segments to fix!")
        return
    
    # Diagnose each segment
    results = []
    connection_fixes = []
    edge_fixes = []
    
    for _, row in blocked.iterrows():
        from_edge = row['from_edge']
        to_edge = row['to_edge']
        
        print(f"\n[DIAG] {row['route']} {row['bound']} {row['seq']}")
        print(f"       {from_edge} -> {to_edge}")
        
        diag = diagnose_segment(net, from_edge, to_edge)
        diag['route'] = row['route']
        diag['bound'] = row['bound']
        diag['seq'] = row['seq']
        
        print(f"       Root cause: {diag['root_cause']}")
        print(f"       Fix: {diag['fix_type']}")
        if diag['blocking_edge']:
            print(f"       Blocking: {diag['blocking_edge']}")
        
        results.append(diag)
        
        # Collect fixes
        if diag['fix_type'] == 'ADD_CONNECTION' and diag['fix_detail']:
            connection_fixes.append(diag['fix_detail'])
        elif diag['fix_type'] == 'MODIFY_ALLOW' and diag['blocking_edge']:
            edge_fixes.append({
                'edge': diag['blocking_edge'],
                'reason': diag['blocking_reason']
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] Diagnosis results: {OUTPUT_CSV}")
    
    # Generate fix files
    if connection_fixes:
        print(f"\n[GEN] Connection fixes ({len(connection_fixes)} connections)")
        with open(OUTPUT_CON_XML, 'w', encoding='utf-8') as f:
            f.write('<!-- V3 Connection Fixes -->\n')
            f.write('<connections>\n')
            for fix in set(connection_fixes):
                f.write(f'    {fix}\n')
            f.write('</connections>\n')
        print(f"[SAVED] {OUTPUT_CON_XML}")
    
    if edge_fixes:
        print(f"\n[INFO] Edge allow fixes needed ({len(edge_fixes)} edges):")
        unique_edges = {}
        for fix in edge_fixes:
            unique_edges[fix['edge']] = fix['reason']
        for edge, reason in unique_edges.items():
            print(f"       {edge}: {reason}")
    
    # Summary
    print("\n" + "="*80)
    print("[SUMMARY]")
    print("="*80)
    cause_counts = results_df['root_cause'].value_counts()
    for cause, count in cause_counts.items():
        print(f"   {cause}: {count}")
    
    print("\n[DONE]")

if __name__ == "__main__":
    main()
