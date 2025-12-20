
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import json
import numpy as np
from shapely.geometry import Point, LineString, shape
from rtree import index

# Constants
DETECTOR_CSV = "data/raw/detector_locations/traffic_speed_volume_occ_info-20251219-175822.csv"
DETECTOR_XML_PATTERN = "data/raw/detector_locations/rawSpeedVol-all-*.xml" 
NETWORK_GEOJSON = "data/processed/hk_irn_edges.geojson"
OUTPUT_ROUTE_FILE = "sumo/routes/background.rou.xml"
SEARCH_RADIUS = 0.0005 # degrees, approx 50m

def load_detectors(csv_path):
    """Load detector metadata: ID -> (Lon, Lat)."""
    # Assuming CSV has 'AID_ID_Number', 'Easting', 'Northing'
    # Based on previous `Get-Content` output:
    # AID_ID_Number,District,...,Easting,Northing,...
    try:
        # Use simple pandas read, skip bad lines if any
        df = pd.read_csv(csv_path, skipinitialspace=True, encoding='utf-8-sig')
        # Clean col names
        df.columns = [c.replace('\ufeff', '').replace('ï»¿', '').strip() for c in df.columns]
        print(f"Columns found: {df.columns.tolist()}")
        
        # Check required columns
        req = ['AID_ID_Number', 'Longitude', 'Latitude']
        if not all(c in df.columns for c in req):
            print(f"Error: CSV missing columns. Available: {df.columns}")
            return {}
            
        detectors = {}
        for _, row in df.iterrows():
            did = str(row['AID_ID_Number']).strip()
            try:
                x = float(row['Longitude'])
                y = float(row['Latitude'])
                detectors[did] = (x, y)
                
                # Validation
                if not (113 < x < 115 and 22 < y < 23):
                     # Maybe HK1980 was read? No, we requested fields.
                     pass 
            except ValueError:
                continue
        return detectors
    except Exception as e:
        print(f"Failed to load detectors CSV: {e}")
        return {}

def load_traffic_data(xml_file):
    """
    Parse XML to get {detector_id: total_volume_30s}.
    Returns volume in vehicles per 30s.
    """
    traffic = {}
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Structure: raw_speed_volume_list -> periods -> period -> detectors -> detector
        # We iterate all 'detector' tags
        for det_node in root.iter():
            if det_node.tag.endswith('detector') and 'detectors' not in det_node.tag:
                did = None
                vol_sum = 0
                
                # Get ID
                for child in det_node:
                    if child.tag.endswith('detector_id'):
                        did = child.text.strip()
                        break
                
                if not did:
                    continue
                    
                # Sum volume from lanes
                lanes_node = None
                for child in det_node:
                    if child.tag.endswith('lanes'):
                        lanes_node = child
                        break
                
                if lanes_node:
                    for lane in lanes_node:
                        # Find volume tag
                        for prop in lane:
                            if prop.tag.endswith('volume'):
                                try:
                                    v = int(prop.text)
                                    vol_sum += v
                                except:
                                    pass
                
                if vol_sum > 0:
                    traffic[did] = vol_sum
                    
    except Exception as e:
        print(f"Error parsing XML {xml_file}: {e}")
        
    return traffic

def build_spatial_index(geojson_path):
    """
    Build R-tree index for edges.
    Returns (idx, edges_dict)
    edges_dict: {id: {'id': edge_id, 'geometry': shape}}
    """
    idx = index.Index()
    edges = {}
    
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geo = json.load(f)
        
    count = 0
    for feat in geo['features']:
        props = feat.get('properties', {})
        # Route ID in IRN usually maps to... wait, we need SUMO Edge ID.
        # Week 1 script used OSM. Week 2 uses IRN. 
        # convert_irn_to_sumo_xml.py used 'ROUTE_ID' as part of edge ID?
        # Let's check the geojson properties.
        # "ROUTE_ID": 275
        # The Edge ID in .edg.xml likely involves ROUTE_ID. 
        # In convert_irn_to_sumo_xml: edge_id = str(route_id) (plus direction suffix potentially)
        
        # Let's assume ROUTE_ID dictates the edge ID base.
        # But wait, IRN has directions. 
        # A single centerline can be bi-directional.
        # We need to be careful. The detector just gives coordinate.
        # We'll map to the nearest feature, then verify valid edges later.
        
        rid = props.get('ROUTE_ID')
        if rid is None: 
            continue
            
        geom = shape(feat['geometry'])
        edges[count] = {'route_id': rid, 'geometry': geom, 'props': props}
        idx.insert(count, geom.bounds)
        count += 1
        
    print(f"Indexed {count} road segments.")
    return idx, edges

def load_valid_edges_from_net(net_xml):
    """Parse net.xml to get a set of valid edge IDs."""
    valid = set()
    try:
        for event, elem in ET.iterparse(net_xml, events=("start",)):
            if elem.tag == "edge":
                eid = elem.get("id")
                if eid and not eid.startswith(":"):
                    valid.add(eid)
                elem.clear()
    except Exception as e:
        print(f"Error reading net xml: {e}")
    print(f"Loaded {len(valid)} valid edges from network.")
    return valid

def generate_route_file(detectors, traffic, idx, edges, output_path, valid_edges):
    """
    Generate SUMO routes, verifying against valid_edges.
    """
    with open(output_path, 'w') as f:
        f.write('<routes>\n')
        f.write('    <vType id="private_car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="50" color="1,1,0"/>\n')
        
        mapped_count = 0
        total_veh = 0
        
        for did, volume in traffic.items():
            if did not in detectors:
                continue
                
            x, y = detectors[did]
            pt = Point(x, y)
            
            # Find nearest edge
            nearest_edge = None
            min_dist = SEARCH_RADIUS
            
            potential = list(idx.intersection((x-SEARCH_RADIUS, y-SEARCH_RADIUS, x+SEARCH_RADIUS, y+SEARCH_RADIUS)))
            
            for pid in potential:
                edge_data = edges[pid]
                dist = edge_data['geometry'].distance(pt)
                if dist < min_dist:
                    min_dist = dist
                    nearest_edge = edge_data
            
            if nearest_edge:
                rid = str(nearest_edge['props'].get('ROUTE_ID'))
                
                # Try explicit mappings
                # Valid patterns: {rid}, {rid}_rev
                
                candidates = []
                # Check forward
                if rid in valid_edges:
                    candidates.append(rid)
                
                # Check backward
                rev_id = f"{rid}_rev"
                if rev_id in valid_edges:
                    candidates.append(rev_id)
                
                if not candidates:
                    # Edge geometry matches, but ID not found? 
                    # Could happen if ROUTE_ID isn't the direct key.
                    # But convert script used ROUTE_ID.
                    continue

                # Distribute flow
                veh_per_hour = volume * 120
                if veh_per_hour <= 0: continue
                
                # Split traffic among valid candidates
                # If detector has directional info, we could use it, but here we just blindly fill the road.
                flow_per_edge = veh_per_hour / len(candidates)
                
                for edge_id in candidates:
                    f.write(f'    <flow id="flow_{did}_{edge_id}" type="private_car" begin="0" end="3600" number="{int(flow_per_edge)}" from="{edge_id}" to="{edge_id}"/>\n')
                    total_veh += int(flow_per_edge)
                
                mapped_count += 1
        
        f.write('</routes>\n')
        print(f"Generated routes for {mapped_count} detectors. Total {total_veh} vehicles scheduled.")


def main():
    # ... (preamble same)
    print("Loading detectors...")
    dets = load_detectors(DETECTOR_CSV)
    
    files = sorted(glob.glob(DETECTOR_XML_PATTERN))
    if not files: return
    xml_file = files[-1]
    traffic = load_traffic_data(xml_file)
    
    print("Building network index...")
    if not os.path.exists(NETWORK_GEOJSON): return
    idx, edges = build_spatial_index(NETWORK_GEOJSON)
    
    # NEW: Load valid (sumo) edges
    net_file = "sumo/net/hk_irn.net.xml"
    if not os.path.exists(net_file):
        print(f"Net file {net_file} missing.")
        return
    valid_edges = load_valid_edges_from_net(net_file)

    print("Generating Route File...")
    os.makedirs(os.path.dirname(OUTPUT_ROUTE_FILE), exist_ok=True)
    generate_route_file(dets, traffic, idx, edges, OUTPUT_ROUTE_FILE, valid_edges)
    print(f"Done. Saved to {OUTPUT_ROUTE_FILE}")

if __name__ == "__main__":
    main()
