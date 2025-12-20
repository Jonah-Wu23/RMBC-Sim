import xml.etree.ElementTree as ET
import sys
import os

def load_network(net_file):
    print(f"Loading network {net_file}...")
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Map: lane_id -> length
    lane_lengths = {}
    # Map: edge_id -> list of lane_ids
    edge_lanes = {}
    
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        e_lanes = []
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            length = float(lane.get('length'))
            lane_lengths[lane_id] = length
            e_lanes.append(lane_id)
        edge_lanes[edge_id] = e_lanes
        
    # Map: edge_id -> Set of connected edge_ids
    connections = {} 
    for conn in root.findall('connection'):
        from_e = conn.get('from')
        to_e = conn.get('to')
        if from_e not in connections: connections[from_e] = set()
        if to_edge := to_e: connections[to_edge] = connections.get(to_edge, set()) | {from_e}
        connections[from_e].add(to_e)
        
    print(f"Loaded {len(lane_lengths)} lanes.")
    return lane_lengths, edge_lanes, connections

def fix_bus_stops(input_file, net_file):
    lane_lengths, edge_lanes, network_adj = load_network(net_file)
    
    print(f"Processing stops in {input_file}...")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    modified_count = 0
    removed_count = 0
    min_len_req = 20.0
    
    # Stops to REMOVE (Stubborn "not downstream" warnings)
    remove_ids = [
        "447C1D79A10C83E2", # On 261287_rev
        "47A2AA98F8E20AE2", # On 5291
        "2B3A6C000BB085CD"  # On 95356
    ]

    # Manual fixes: ID -> {attr: val}
    manual_fixes = {
        # Fix "Too Short" (Out of bounds) by clamping/shifting
        "B113E1371D5AC497": {"startPos": "88.00", "endPos": "108.00"}, 
        "0BFC5F7FC4A2D431": {"startPos": "247.00", "endPos": "267.00"}, 
        "13E0A0449C73F4DE": {"startPos": "95.00", "endPos": "115.00"}, 
        "6AE148CA74FC4D53": {"startPos": "88.00", "endPos": "108.00"},
        
        # Previous fixes (Internal lanes / relocations)
        "AB0F79ED382081D0": {"lane": ":14935_5_0"}, 
    }
    
    # We need to collect stops to remove then remove them from root
    stops_to_remove = []

    for bus_stop in root.findall('busStop'):
        stop_id = bus_stop.get('id')
        
        # Check removal
        if stop_id in remove_ids:
            print(f"Removing stubborn stop: {stop_id}")
            stops_to_remove.append(bus_stop)
            removed_count += 1
            continue

        lane_id = bus_stop.get('lane')
        
        # Apply Manual Fixes
        if stop_id in manual_fixes:
            print(f"Applying manual fix for {stop_id}...")
            for k, v in manual_fixes[stop_id].items():
                bus_stop.set(k, v)
            modified_count += 1
            lane_id = bus_stop.get('lane') 

        # Generic Logic: Check Lane Length and Relocate if needed
        current_len = lane_lengths.get(lane_id, 999.0)
        
        if current_len < min_len_req:
            print(f"Stop {stop_id} on {lane_id} is too short ({current_len:.2f}m). Searching for relocation...")
            curr_edge = lane_id.rpartition('_')[0]
            found_new_lane = None
            
            neighbors = network_adj.get(curr_edge, [])
            for neighbor_edge in neighbors:
                cand_lanes = edge_lanes.get(neighbor_edge, [])
                for cand_lane in cand_lanes:
                    if lane_lengths.get(cand_lane, 0) > min_len_req:
                        found_new_lane = cand_lane
                        break
                if found_new_lane: break
            
            if found_new_lane:
                print(f"  -> Relocating to {found_new_lane}")
                bus_stop.set('lane', found_new_lane)
                bus_stop.set('startPos', "0.00")
                bus_stop.set('endPos', "20.00")
                modified_count += 1
        
        # Final sanity check: Ensure stop fits in lane (Clamp)
        # Re-get checks in case manual fix changed things
        lane_id = bus_stop.get('lane')
        current_len = lane_lengths.get(lane_id, 9999.0)
        start_pos = float(bus_stop.get('startPos'))
        end_pos = float(bus_stop.get('endPos'))
        
        if end_pos > current_len:
            diff = end_pos - current_len
            new_end = current_len - 0.1 
            new_start = max(0.0, new_end - 20.0)
            if new_start < 0.0: new_start = 0.0
            
            bus_stop.set('startPos', f"{new_start:.2f}")
            bus_stop.set('endPos', f"{new_end:.2f}")
            print(f"  -> Clamped {stop_id} to fit lane {lane_id} (Len {current_len:.2f})")
            modified_count += 1
            
        elif (end_pos - start_pos) < 15.0:
             new_end = start_pos + 20.0
             if new_end <= current_len:
                 bus_stop.set('endPos', f"{new_end:.2f}")
                 modified_count += 1

    # Perform removal
    for stop in stops_to_remove:
        root.remove(stop)

    if modified_count > 0 or removed_count > 0:
        tree.write(input_file, encoding='utf-8', xml_declaration=True)
        print(f"Successfully fixed {modified_count} stops and removed {removed_count} stops.")
    else:
        print("No bus stops needed fixing.")

if __name__ == "__main__":
    net_path = "sumo/net/hk_irn_v3.net.xml"
    stops_path = "sumo/additional/bus_stops.add.xml"
    
    if not os.path.exists(net_path):
        net_path = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_irn_v3.net.xml"
    if not os.path.exists(stops_path):
        stops_path = r"d:\Documents\Bus Project\Sorce code\sumo\additional\bus_stops.add.xml"
        
    if os.path.exists(net_path) and os.path.exists(stops_path):
        fix_bus_stops(stops_path, net_path)
    else:
        print("Files not found.")
