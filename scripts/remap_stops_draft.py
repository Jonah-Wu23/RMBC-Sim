import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import xml.etree.ElementTree as ET
import os

def remap_stops():
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    STOPS_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml"))
    CSV_FILE = os.path.abspath(os.path.join("data", "processed", "kmb_route_stop_dist.csv"))
    OUTPUT_STOPS = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml")) # Overwrite
    
    print("Loading Network Edges...")
    # We need full shapes to find nearest
    # Using simple parsing for speed (or use geopandas if we can convert xml to geojson easily? netconvert can do that!)
    # Let's use simple parsing of the net file for lanes.
    
    edges = []
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'lane':
            eid = elem.get('id')
            shape = elem.get('shape')
            # "x1,y1 x2,y2"
            coords = [tuple(map(float, p.split(','))) for p in shape.split()]
            edges.append({'id': eid, 'geometry': LineString(coords), 'edge_id': elem.get('id').rpartition('_')[0]})
        if elem.tag == 'edge': 
            elem.clear() # clear memory
            
    gdf_edges = gpd.GeoDataFrame(edges, geometry='geometry')
    print(f"Loaded {len(gdf_edges)} lanes.")
    
    print("Loading Stops from CSV...")
    df_stops = pd.read_csv(CSV_FILE)
    # We need to filter for the stops present in the XML mostly, or just re-generate all.
    # But let's just fix the ones in the XML to preserve any manual changes? 
    # Actually, simpler to just iterate the XML stops, look up their ID in CSV to get Lat/Lon (WGS84),
    # transform to Net Coords (assuming Net is UTM 50N? No, net is likely EPSG:32650 or similar Hong Kong grid).
    # Wait, the net file is in meters. The CSV is lat/lon.
    # We need to project the CSV point to the Net's CRS.
    
    # Let's assume the previous generation script handled projection. 
    # We can infer the projection from the NET file location section?
    # Or just use the fact that we have a large existing match.
    
    # Better idea: Just use the `generate_sumo_stops_irn.py` logic but restrict it to ONLY the valid edges.
    # If that script exists and works, we can just run it again with a filter!
    
    pass

if __name__ == "__main__":
    pass
