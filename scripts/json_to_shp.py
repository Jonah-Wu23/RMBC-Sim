import geopandas as gpd
import os

INPUT = os.path.abspath(os.path.join("data", "processed", "hk_irn_edges.geojson"))
OUTPUT_DIR = os.path.abspath(os.path.join("data", "processed", "hk_irn_shp"))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hk_irn_edges.shp")

print(f"Reading {INPUT}...")
gdf = gpd.read_file(INPUT)
print(f"Saving to {OUTPUT_FILE}...")
gdf.to_file(OUTPUT_FILE)
print("Done.")
