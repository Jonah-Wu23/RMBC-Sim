#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v31_quality_gate.py
===================
V3.1 Quality Gate: Re-classify RULE_BLOCKED segments

For each RULE_BLOCKED segment, determine:
1. bus_reachable: Can bus actually reach from A to B?
2. bus_path_len: If reachable, what's the path length?
3. passenger_path_len: Path length ignoring vClass
4. Is this a real unreachable issue or just a long detour?
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sumolib
except ImportError:
    print("[ERROR] sumolib not installed")
    sys.exit(1)

# Use V1 network (without V3 bridge patches) to see the real issues
NET_V1 = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v1.net.xml"
NET_V3 = PROJECT_ROOT / "sumo" / "net" / "hk_irn_v3_patched_v3.net.xml"
DIAGNOSIS_V1 = PROJECT_ROOT / "logs" / "route_segment_diagnosis_v1.csv"
OUTPUT_CSV = PROJECT_ROOT / "logs" / "v31_rule_blocked_reclassify.csv"

# The 12.5m bridge that needs investigation
SUSPECT_BRIDGE = {
    'from_edge': '97407',
    'to_edge': '96273_rev',
    'bridge_id': 'bridge_68X_GAP_12',
    'node_dist': 12.5,
}


def check_adjacent_connection(net, from_edge_id, to_edge_id):
    """Check if two edges share a junction and if connection exists"""
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
    except:
        return None
    
    from_to_node = from_edge.getToNode()
    to_from_node = to_edge.getFromNode()
    
    result = {
        'from_to_node': from_to_node.getID(),
        'to_from_node': to_from_node.getID(),
        'same_junction': from_to_node.getID() == to_from_node.getID(),
        'connection_exists': False,
        'connection_info': ''
    }
    
    if result['same_junction']:
        # Check if connection exists
        for from_lane in from_edge.getLanes():
            for conn in from_lane.getOutgoing():
                target = conn.getTo()
                if hasattr(target, 'getEdge'):
                    target_edge_id = target.getEdge().getID()
                else:
                    target_edge_id = target.getID()
                if target_edge_id == to_edge_id:
                    result['connection_exists'] = True
                    result['connection_info'] = f'{from_edge_id}_L{from_lane.getIndex()} -> {to_edge_id}'
                    break
    
    return result


def get_path_length(net, from_edge_id, to_edge_id):
    """Get shortest path length between two edges"""
    try:
        from_edge = net.getEdge(from_edge_id)
        to_edge = net.getEdge(to_edge_id)
        route, cost = net.getShortestPath(from_edge, to_edge)
        if route:
            return sum(e.getLength() for e in route), len(route)
        return None, 0
    except:
        return None, 0


def main():
    print("="*80)
    print("[V3.1] Quality Gate - RULE_BLOCKED Re-classification")
    print("="*80)
    
    # Load V1 network (to see issues without V3 bridges)
    print(f"\n[LOAD] V1 Network: {NET_V1.name}")
    net_v1 = sumolib.net.readNet(str(NET_V1))
    
    # First, investigate the 12.5m suspect bridge
    print("\n" + "="*80)
    print("[INVESTIGATE] Suspect Bridge: bridge_68X_GAP_12 (12.5m)")
    print("="*80)
    
    fe, te = SUSPECT_BRIDGE['from_edge'], SUSPECT_BRIDGE['to_edge']
    print(f"  Edge pair: {fe} -> {te}")
    
    adj = check_adjacent_connection(net_v1, fe, te)
    if adj:
        print(f"  From edge to_node: {adj['from_to_node']}")
        print(f"  To edge from_node: {adj['to_from_node']}")
        print(f"  Same junction: {adj['same_junction']}")
        print(f"  Connection exists: {adj['connection_exists']}")
        if adj['connection_info']:
            print(f"  Connection info: {adj['connection_info']}")
        
        if adj['same_junction'] and not adj['connection_exists']:
            print(f"\n  [VERDICT] This is a MISSING CONNECTION issue!")
            print(f"  [FIX] Add: <connection from=\"{fe}\" to=\"{te}\" fromLane=\"0\" toLane=\"0\"/>")
        elif not adj['same_junction']:
            print(f"\n  [VERDICT] Not adjacent (nodes differ)")
            path_len, edge_count = get_path_length(net_v1, fe, te)
            if path_len:
                print(f"  Path exists: {path_len:.0f}m, {edge_count} edges")
            else:
                print(f"  No path found - bridge may be needed")
    
    # Now scan for V1 RULE_BLOCKED segments
    print("\n" + "="*80)
    print("[SCAN] V1 Network RULE_BLOCKED segments")
    print("="*80)
    
    # Need to run scan on V1 first
    print("  Running scan on V1 network...")
    
    import subprocess
    cmd = [
        "python", str(PROJECT_ROOT / "scripts" / "scan_all_route_segments.py"),
        "--net", str(NET_V1)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    
    # Rename output to V1
    diagnosis_file = PROJECT_ROOT / "logs" / "route_segment_diagnosis.csv"
    if diagnosis_file.exists():
        diagnosis_file.rename(DIAGNOSIS_V1)
        print(f"  Saved: {DIAGNOSIS_V1.name}")
    
    # Load and reclassify
    df = pd.read_csv(DIAGNOSIS_V1)
    blocked = df[df['type'] == 'RULE_BLOCKED'].copy()
    print(f"  Found {len(blocked)} RULE_BLOCKED segments in V1")
    
    # Reclassify each
    results = []
    for _, row in blocked.iterrows():
        fe, te = row['from_edge'], row['to_edge']
        kmb_len = row['kmb_len']
        
        rec = {
            'route': row['route'],
            'bound': row['bound'],
            'seq': row['seq'],
            'from_edge': fe,
            'to_edge': te,
            'kmb_len': kmb_len,
        }
        
        # Check adjacency
        adj = check_adjacent_connection(net_v1, fe, te)
        if adj:
            rec['same_junction'] = adj['same_junction']
            rec['connection_exists'] = adj['connection_exists']
        else:
            rec['same_junction'] = None
            rec['connection_exists'] = None
        
        # Check path
        path_len, edge_count = get_path_length(net_v1, fe, te)
        rec['bus_reachable'] = path_len is not None
        rec['bus_path_len'] = path_len
        rec['edge_count'] = edge_count
        rec['ratio'] = path_len / kmb_len if path_len and kmb_len > 0 else None
        
        # Determine true category
        if rec['same_junction'] and not rec['connection_exists']:
            rec['true_category'] = 'MISSING_CONNECTION'
            rec['fix'] = f'<connection from="{fe}" to="{te}" fromLane="0" toLane="0"/>'
        elif not rec['bus_reachable']:
            rec['true_category'] = 'TRUE_UNREACHABLE'
            rec['fix'] = 'Bridge edge needed'
        elif rec['ratio'] and rec['ratio'] > 3:
            rec['true_category'] = 'LONG_DETOUR'
            rec['fix'] = 'VIA_ROUTING or check intermediate connections'
        else:
            rec['true_category'] = 'MINOR_DETOUR'
            rec['fix'] = 'Acceptable or VIA_ROUTING'
        
        results.append(rec)
        
        print(f"\n  {row['route']} {row['bound']} {row['seq']}: {fe} -> {te}")
        print(f"    Same junction: {rec['same_junction']}, Conn exists: {rec['connection_exists']}")
        print(f"    Bus reachable: {rec['bus_reachable']}, Path: {rec['bus_path_len'] or 'N/A'}m")
        print(f"    True category: {rec['true_category']}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] {OUTPUT_CSV}")
    
    # Summary
    print("\n" + "="*80)
    print("[SUMMARY] True Categories of RULE_BLOCKED")
    print("="*80)
    
    for cat in results_df['true_category'].unique():
        count = len(results_df[results_df['true_category'] == cat])
        print(f"  {cat}: {count}")
    
    # Show MISSING_CONNECTION fixes
    missing_conn = results_df[results_df['true_category'] == 'MISSING_CONNECTION']
    if len(missing_conn) > 0:
        print("\n" + "-"*80)
        print("[ACTION REQUIRED] Missing Connections to Add:")
        print("-"*80)
        for _, r in missing_conn.iterrows():
            print(f"  {r['fix']}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
