import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import xml.etree.ElementTree as ET
import os
import shutil

def remap_stops():
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    STOPS_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml"))
    CSV_FILE = os.path.abspath(os.path.join("data", "processed", "kmb_route_stop_dist.csv"))
    
    # 1. Load Valid Edges from NET file
    print("Loading Network Edges...")
    edges = []
    # Using iterparse to extract edge geometries (simplified as lines)
    # Note: NET file coordinates are already projected (presumably HK Grid or UTM), matching the CSV if handled correctly,
    # BUT wait, the CSV typically has lat/lon. We need to check coordinate systems.
    # The previous `convert_irn_to_sumo_xml.py` likely handled projection or SUMO did.
    # If SUMO net is in meters (likely), and CSV is lat/lon, we need to project CSV points.
    
    # Let's inspect ONE edge in NET to guess the CRS or magnitude.
    # From debug output: "5557.68, 20877.32". These look like HK Grid (1980) truncated or similar.
    # Hong Kong Grid coordinates are usually locally origin-based in some converts.
    # Let's trust that `generate_sumo_stops_irn.py` (which created the original stops) knew the transform.
    # If we simply search for the nearest edge usage *Euclidean distance on these inputs*, we rely on consistent CRS.
    
    # Assuming the NET file coords are consistent with what generated the stops.
    
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'edge':
            # We use the FIRST lane's shape as approximation
            lane = elem.find('lane')
            if lane is not None:
                eid = elem.get('id')
                shape = lane.get('shape')
                # "x1,y1 x2,y2"
                coords = [tuple(map(float, p.split(','))) for p in shape.split()]
                if coords:
                    edges.append({'id': eid, 'geometry': LineString(coords)})
            elem.clear() # clear memory
            
    gdf_edges = gpd.GeoDataFrame(edges, geometry='geometry')
    # Build spatial index
    sindex = gdf_edges.sindex
    print(f"Loaded {len(gdf_edges)} valid edges with geometry.")
    
    # 2. Identify Invalid Stops
    tree = ET.parse(STOPS_FILE)
    root = tree.getroot()
    
    valid_edge_ids = set(gdf_edges['id'])
    stops_to_fix = []
    
    for stop in root.findall('busStop'):
        original_lane = stop.get('lane')
        edge_id = original_lane.split('_')[0]
        if edge_id not in valid_edge_ids:
            stops_to_fix.append(stop)
            
    print(f"Found {len(stops_to_fix)} stops to fix.")
    
    if not stops_to_fix:
        print("No stops to fix.")
        return

    # 3. Load CSV to get Stop Coords
    df_stops = pd.read_csv(CSV_FILE)
    # We need a way to transform lat/lon to Net Coords.
    # Since we lack the exact proj string used, let's look at a VALID stop and see the offset?
    # Or, we can re-use the `convert_irn_to_sumo_xml` projection?
    # Wait, the `debug_connection.py` output showed coords like 5000, 20000.
    # Standard HK Grid is ~800,000, 800,000.
    # This implies the NET is using a localized/offset CRS.
    # I will attempt to use "nearest neighbor in the XML file itself" or just find ANY edge near the declared lane of the invalid stop?
    # No, the invalid stop's declared lane doesn't exist.
    
    # Strategy C: The invalid stops likely belong to the 6 missing stops found earlier.
    # Their IDs: 147D513A708AA741, 60A4F4A41CD61FAC, etc.
    # Look up these IDs in CSV -> Get Lat/Lon.
    # Find a VALID stop in XML that is close to this Lat/Lon? No, that doesn't help map to edge.
    
    # Let's assume the invalid edges were REMOVED because they were outside BBox, 
    # OR they are valid IRN edges that just got renamed/split?
    # If they were clipped out, we need to map to the NEAREST edge that WAS kept (inside the BBox).
    
    # We need the Lat/Lon -> Net Coords transform.
    # I'll create a simple Linear Regression estimator based on valid stops!
    # 1. Take 10 valid stops.
    # 2. Get their CSV Lat/Lon and their XML Lane Shape center.
    # 3. Fit (Lat, Lon) -> (X, Y).
    # 4. Predict (X, Y) for invalid stops using their CSV Lat/Lon.
    # 5. Query nearest edge to that (X, Y) in valid set.
    
    valid_samples = []
    for stop in root.findall('busStop'):
        edge_id = stop.get('lane').split('_')[0]
        stop_id = stop.get('id')
        if edge_id in valid_edge_ids and len(valid_samples) < 20:
            # Look up in CSV
            row = df_stops[df_stops['stop_id'] == stop_id]
            if not row.empty:
                # Get Net Coord roughly (start of lane)
                edge_geom = gdf_edges[gdf_edges['id'] == edge_id].geometry.values[0]
                pt = edge_geom.coords[0] # start point
                valid_samples.append({
                    'lat': row.iloc[0]['lat'], 
                    'lon': row.iloc[0]['long'], 
                    'x': pt[0], 
                    'y': pt[1]
                })

    if not valid_samples:
        print("Cannot build coordinate transform reference.")
        return
        
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    df_sample = pd.DataFrame(valid_samples)
    X_train = df_sample[['lon', 'lat']]
    y_train_x = df_sample['x']
    y_train_y = df_sample['y']
    
    reg_x = LinearRegression().fit(X_train, y_train_x)
    reg_y = LinearRegression().fit(X_train, y_train_y)
    
    print("Coordinate transform model built.")
    
    # 4. Fix Stops
    for stop in stops_to_fix:
        stop_id = stop.get('id')
        row = df_stops[df_stops['stop_id'] == stop_id]
        if row.empty:
            print(f"Stop {stop_id} not found in CSV. Cannot fix.")
            continue
            
        lon, lat = row.iloc[0]['long'], row.iloc[0]['lat']
        pred_x = reg_x.predict([[lon, lat]])[0]
        pred_y = reg_y.predict([[lon, lat]])[0]
        
        # Find nearest edge to (pred_x, pred_y)
        pt = Point(pred_x, pred_y)
        # Using simple distance sort (assumes small network is loaded in memory ok)
        # gdf_edges.distance(pt) is vectorised
        distances = gdf_edges.distance(pt)
        nearest_idx = distances.idxmin()
        nearest_edge = gdf_edges.iloc[nearest_idx]
        
        print(f"Remapping Stop {stop_id} (missing {stop.get('lane')}) -> {nearest_edge['id']}_0 (Dist: {distances.min():.2f}m)")
        
        # Update XML
        # Assuming lane 0 is safe
        stop.set('lane', f"{nearest_edge['id']}_0")
        stop.set('startPos', '0')
        stop.set('endPos', '15') # reduced length
        
    tree.write(STOPS_FILE)
    print("Updated bus_stops.add.xml")

if __name__ == "__main__":
    remap_stops()
