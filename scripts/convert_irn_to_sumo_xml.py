"""
Script to convert Hong Kong IRN data directly to SUMO XML format (Nodes and Edges).
This avoids the need for GDAL support in SUMO netconvert.
"""

import geopandas as gpd
import pandas as pd
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Paths
INPUT_GEOJSON = os.path.abspath(os.path.join("data", "processed", "hk_irn_edges.geojson"))
OUTPUT_NODES = os.path.abspath(os.path.join("data", "processed", "hk_irn.nod.xml"))
OUTPUT_EDGES = os.path.abspath(os.path.join("data", "processed", "hk_irn.edg.xml"))

def convert_to_sumo_xml():
    print(f"Reading {INPUT_GEOJSON}...")
    gdf = gpd.read_file(INPUT_GEOJSON)
    
    # Ensure WGS84 or UTM. 
    # SUMO XMLs usually expect UTM (x,y in meters) unless proj parameters are passed.
    # IRN is in EPSG:2326 (HK Grid, meters), which is perfect for SUMO!
    # BUT we converted it to 4326 (WGS84) in previous step.
    # Let's convert back to EPSG:2326 or use UTM 50N (EPSG:32650).
    # EPSG:2326 is native HK 1980 Grid, unit: meter. 
    # Let's use EPSG:2326 for coordinates.
    target_crs = "EPSG:2326"
    print(f"Reprojecting to {target_crs} (HK 1980 Grid) for xy-coordinates...")
    gdf = gdf.to_crs(target_crs)

    nodes = {} # (x, y) -> node_id
    edges = []
    
    print("Building Node/Edge graph...")
    
    node_counter = 0
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'MultiLineString':
            # Take the longest part or iterate parts? Usually roads are single linestrings.
            # Explode or just take largest? Let's iterate parts to be safe.
            parts = list(geom.geoms)
        elif geom.geom_type == 'LineString':
            parts = [geom]
        else:
            continue
            
        edge_id_base = str(row['ROUTE_ID'])
        speed = row['speed_ms']
        lanes = int(row['lanes'])
        name = str(row.get('STREET_ENAME', ''))
        travel_dir = int(row.get('TRAVEL_DIRECTION', 1)) # Default 1 if missing? Or 3?
        
        for part_idx, line in enumerate(parts):
            coords = list(line.coords)
            if len(coords) < 2: continue
            
            # Identify Start/End Nodes
            start_coord = coords[0][:2] # (x, y)
            end_coord = coords[-1][:2]
            
            # Get or Create Node IDs
            if start_coord not in nodes:
                node_counter += 1
                nodes[start_coord] = str(node_counter)
            from_node = nodes[start_coord]
            
            if end_coord not in nodes:
                node_counter += 1
                nodes[end_coord] = str(node_counter)
            to_node = nodes[end_coord]
            
            # Shape string (all coords space-separated "x,y x,y")
            shape_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in coords])
            shape_rev_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in reversed(coords)])
            
            # Unique Edge ID suffix
            eid_suffix = ""
            if len(parts) > 1:
                eid_suffix = f"_{part_idx}"
            
            # Logic:
            # 1: One Way (Forward)
            # 2: One Way (Backward) - Though count is 0 based on check
            # 3: Two Way
            
            # Forward Edge (if dir 1 or 3)
            if travel_dir in [1, 3]:
                eid = edge_id_base + eid_suffix
                edges.append({
                    'id': eid,
                    'from': from_node,
                    'to': to_node,
                    'speed': f"{speed:.2f}",
                    'numLanes': f"{lanes}",
                    'name': name,
                    'shape': shape_str
                })
                
            # Backward Edge (if dir 2 or 3)
            if travel_dir in [2, 3]:
                eid = edge_id_base + eid_suffix + "_rev"
                edges.append({
                    'id': eid,
                    'from': to_node,
                    'to': from_node,
                    'speed': f"{speed:.2f}",
                    'numLanes': f"{lanes}",
                    'name': name,
                    'shape': shape_rev_str
                })

    print(f"Generated {len(nodes)} nodes and {len(edges)} edges.")

    # Write Nodes XML
    print(f"Writing {OUTPUT_NODES}...")
    root = ET.Element("nodes")
    for coord, nid in nodes.items():
        # SUMO expects x, y
        ET.SubElement(root, "node", id=nid, x=f"{coord[0]:.2f}", y=f"{coord[1]:.2f}")
        
    tree = ET.ElementTree(root)
    # Pretty print or standard write? Minidom for pretty print is slow for large files.
    # Standard write is fine.
    tree.write(OUTPUT_NODES, encoding="UTF-8", xml_declaration=True)

    # Write Edges XML
    print(f"Writing {OUTPUT_EDGES}...")
    root = ET.Element("edges")
    for e in edges:
        ET.SubElement(root, "edge", 
                      id=e['id'], 
                      attrib={'from': e['from'], 
                              'to': e['to']},
                      speed=e['speed'], 
                      numLanes=e['numLanes'],
                      name=e['name'],
                      shape=e['shape'])
                      
    tree = ET.ElementTree(root)
    tree.write(OUTPUT_EDGES, encoding="UTF-8", xml_declaration=True)
    
    # ---------------------------------------------------------
    # Generate Explicit Connections from TURN Table
    # ---------------------------------------------------------
    import fiona
    GDB_PATH = os.path.abspath(os.path.join("data", "RdNet_IRNP.gdb"))
    OUTPUT_CON = os.path.abspath(os.path.join("data", "processed", "hk_irn.con.xml"))
    
    print(f"Reading TURN table from {GDB_PATH}...")
    
    # Build Map: ORIG_FID -> List of generated SUMO Edge IDs
    # Because one FID might result in normal or _rev or suffix edges (if split?)
    # In our logic, 
    # FID (Index) -> ROUTE_ID -> Edge ID(s)
    # But wait, ROUTE_ID is not unique to FID? 
    # In GDB, ROUTE_ID seems to be the main identifier but FID is the handle.
    # inspect output said: CENTERLINE: Found 36058 unique FIDs. Found 36058 unique ROUTE_ID(s).
    # So 1-to-1 mapping is likely.
    
    # Let's map FID -> ROUTE_ID first
    # We can rely on the fact that we preserved ORIG_FID in GeoJSON
    
    fid_to_route = {}
    valid_edge_ids = set()
    
    for _, row in gdf.iterrows():
        fid = int(row['ORIG_FID']) # Assuming integer-like
        rid = str(row['ROUTE_ID'])
        fid_to_route[fid] = rid
        
        # Also track what edges we actually generated
        # We need to predict edge IDs based on logic used above
        # The logic was: `eid = edge_id_base + eid_suffix + ("_rev" if ...)`
        # Since we don't store the generated list in a DF easily, let's just use `edges` list.
    
    # Better: Map ROUTE_ID -> Set of generated EdgeIDs
    route_to_edges = {}
    for e in edges:
        # edge id is like "123" or "123_0" or "123_rev" or "123_0_rev"
        # We know the base ROUTE_ID is the prefix usually.
        # But we can just iterate.
        eid = e['id']
        valid_edge_ids.add(eid)
        
        # Heuristic to find base ROUTE_ID?
        # Or Just use the fid_to_route map we just built to translate Turn(FID) -> Turn(RouteID) -> EdgeIDs
        pass

    connections = []
    
    try:
        with fiona.open(GDB_PATH, layer="TURN") as src:
            count = 0
            for feature in src:
                props = feature['properties']
                # IRN Turn table uses EDGE1FID (From) and EDGE2FID (To)?
                # inspect output: EDGE1FID, EDGE2FID.
                
                fid1 = props.get('EDGE1FID')
                fid2 = props.get('EDGE2FID')
                
                if fid1 is None or fid2 is None: continue
                
                rid1 = fid_to_route.get(fid1)
                rid2 = fid_to_route.get(fid2)
                
                if rid1 is None or rid2 is None: continue
                
                # We have connection from Route1 -> Route2.
                # Now we need to link the specific generated edges.
                # Problem: Route1 might be split into multiple parts (0, 1, 2) if MultiLineString.
                # Usually Turn connects the END (last part) of Route1 to START (first part) of Route2.
                # So we look for edges ending with highest suffix for Route1
                # and edges starting with lowest suffix (0) for Route2.
                
                # Heuristic: Generate ALL combinations of fwd/rev for these routes
                # and let netconvert filter invalid ones.
                
                # Suffixes?
                # We don't track suffixes in fid_to_route.
                # But we can search valid_edge_ids.
                # This is O(N^2) if not careful? No, just generating strings.
                
                # Check possible variations:
                # {rid}
                # {rid}_0, {rid}_1 ... (We need to find the one connected to the node? Hard without node info here)
                # "Permissive Generation":
                # Generate connections for:
                #   {rid1} -> {rid2}
                #   {rid1}_rev -> {rid2}
                #   {rid1} -> {rid2}_rev
                #   {rid1}_rev -> {rid2}_rev
                # AND also handle suffixes?
                # "Suffix problem": If Rid1 is split into Rid1_0, Rid1_1. Turn is from Rid1_1.
                # Rid1 (no suffix) won't exist.
                # We should probably find all edges in `valid_edge_ids` that start with Rid1
                
                # Optimization: Pre-group edges by RouteID
                # (Do this before loop)
                pass
                
                connections.append((rid1, rid2)) # Defer processing
                count += 1
                
            print(f"  Found {count} turns in table.")
            
    except Exception as e:
        print(f"Failed to read TURN table: {e}")
        
    # Group edges by RouteID
    edges_by_route = {}
    for e in edges:
        eid = e['id']
        # Extract base route ID.
        # Format: {RID} or {RID}_{idx} or {RID}_rev or {RID}_{idx}_rev
        # Split by '_'
        parts = eid.split('_')
        # RID is the first part always? Yes, assuming RID is integer-like string.
        base_rid = parts[0]
        if base_rid not in edges_by_route:
            edges_by_route[base_rid] = []
        edges_by_route[base_rid].append(eid)
        
    # Generate XML entries
    con_xml = []
    con_count = 0
    
    unique_cons = set() # Avoid dupes
    
    for rid1, rid2 in connections:
        # Get all variants for From and To
        from_variants = edges_by_route.get(rid1, [])
        to_variants = edges_by_route.get(rid2, [])
        
        for f_id in from_variants:
            for t_id in to_variants:
                # Add check: Don't connect fwd to rev if they are same segment? Self-loop?
                # Let netconvert decide validity.
                
                # Check 2: Valid connection usually involves:
                # A_rev -> B_rev
                # A -> B
                # A_rev -> B
                # A -> B_rev
                # (All combinations might be valid depending on geometry)
                
                sig = f'{f_id}->{t_id}'
                if sig not in unique_cons:
                     con_xml.append(f'<connection from="{f_id}" to="{t_id}"/>')
                     unique_cons.add(sig)
                     con_count += 1

    print(f"Generated {con_count} explicit connections.")
    
    print(f"Writing {OUTPUT_CON}...")
    with open(OUTPUT_CON, 'w', encoding='utf-8') as f:
        f.write('<connections>\n')
        f.write('\n'.join(con_xml))
        f.write('\n</connections>')
        
    print("Done.")

if __name__ == "__main__":
    convert_to_sumo_xml()
