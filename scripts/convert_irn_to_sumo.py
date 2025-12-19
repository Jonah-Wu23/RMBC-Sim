"""
Script to convert Hong Kong IRN GDB data to a SUMO-compatible format (GeoJSON).
Merges CENTERLINE with SPEED_LIMIT and infers lane counts.

Usage: python scripts/convert_irn_to_sumo.py
"""

import geopandas as gpd
import pandas as pd
import os
import sys

# Define Paths
GDB_PATH = os.path.abspath(os.path.join("data", "RdNet_IRNP.gdb"))
OUTPUT_FILE = os.path.abspath(os.path.join("data", "processed", "hk_irn_edges.geojson"))
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)

def convert_irn():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Reading GDB: {GDB_PATH}")
    
    # 1. Read CENTERLINE
    print("Loading CENTERLINE layer...")
    try:
        gdf_edges = gpd.read_file(GDB_PATH, layer="CENTERLINE")
    except Exception as e:
        print(f"Error reading CENTERLINE: {e}")
        return

    print(f"  Loaded {len(gdf_edges)} edges.")
    
    # 2. Read SPEED_LIMIT
    print("Loading SPEED_LIMIT layer...")
    try:
        gdf_speed = gpd.read_file(GDB_PATH, layer="SPEED_LIMIT")
    except Exception as e:
        print(f"Error reading SPEED_LIMIT: {e}")
        # Proceed without speed limits if fails
        gdf_speed = pd.DataFrame()

    # 3. Merge Speed Attributes
    if not gdf_speed.empty:
        print("Merging Speed Limits...")
        # SPEED_LIMIT layer has 'SPEED_LIMIT' as string like "70 km/h"
        # We need to parse it to integer.
        
        # Helper to parse speed string
        def parse_speed(s):
            if not isinstance(s, str): return 50 # Default
            s = s.lower().replace('km/h', '').strip()
            try:
                return int(s)
            except:
                return 50

        # Prepare speed dictionary: ROAD_ROUTE_ID -> Speed (take max if duplicates exist)
        speed_map = {}
        for _, row in gdf_speed.iterrows():
            rid = row.get('ROAD_ROUTE_ID')
            raw_speed = row.get('SPEED_LIMIT')
            if rid is not None:
                speed_val = parse_speed(raw_speed)
                # Keep the higher speed if duplicates overlap (conservative for capacity)
                if rid not in speed_map or speed_val > speed_map[rid]:
                    speed_map[rid] = speed_val
        
        # Apply to edges
        # Default speed 50 km/h (~13.88 m/s)
        gdf_edges['speed_kmh'] = gdf_edges['ROUTE_ID'].map(speed_map).fillna(50)
    else:
        print("Warning: Skipping Speed Limit merge (empty layer). Defaulting to 50 km/h.")
        gdf_edges['speed_kmh'] = 50

    # 4. Infer Lane Counts (Heuristic)
    print("Inferring Lane Counts...")
    def estimate_lanes(row):
        speed = row['speed_kmh']
        name = str(row.get('STREET_ENAME', '')).lower()
        
        # Heuristic Rule 1: High speed -> more lanes
        if speed >= 100: return 3
        if speed >= 80: return 3
        if speed >= 70: return 2
        
        # Heuristic Rule 2: Keywords
        if "highway" in name or "expressway" in name or "bypass" in name:
            return 3
        if "road" in name and speed >= 50:
            return 2
            
        return 1

    gdf_edges['lanes'] = gdf_edges.apply(estimate_lanes, axis=1)

    # Convert Speed to m/s for SUMO
    gdf_edges['speed_ms'] = gdf_edges['speed_kmh'] / 3.6

    # 5. Coordinate Transformation
    # IRN is EPSG:2326 (HK Grid). SUMO works well with UTM or WGS84.
    # WGS84 (EPSG:4326) is standard for GeoJSON.
    target_crs = "EPSG:4326"
    print(f"Reprojecting to {target_crs}...")
    if gdf_edges.crs != target_crs:
        gdf_edges = gdf_edges.to_crs(target_crs)

    # 6. Save
    print(f"Saving to {OUTPUT_FILE}...")
    
    # Preserve FID (Index)
    # Note: When using pyogrio/fiona, index is usually the FID.
    gdf_edges['ORIG_FID'] = gdf_edges.index
    # Verify max FID matches reasonable count or inspection
    print(f"  Max FID: {gdf_edges['ORIG_FID'].max()}")

    # Keep only useful columns for SUMO import
    columns_to_keep = ['ROUTE_ID', 'STREET_ENAME', 'speed_ms', 'lanes', 'geometry', 'TRAVEL_DIRECTION', 'speed_kmh', 'ORIG_FID']
    gdf_edges = gdf_edges[columns_to_keep]
    
    gdf_edges.to_file(OUTPUT_FILE, driver="GeoJSON")
    print("Done.")

if __name__ == "__main__":
    convert_irn()
