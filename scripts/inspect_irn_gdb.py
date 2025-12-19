"""
Script to inspect the layers and schema of an Esri File Geodatabase (GDB).
Usage: python scripts/inspect_irn_gdb.py
"""

import sys
import fiona
import os

# Set GDB path
GDB_PATH = os.path.abspath(os.path.join("data", "RdNet_IRNP.gdb"))

def inspect_gdb(gdb_path):
    print(f"Inspecting GDB at: {gdb_path}")
    
    if not os.path.exists(gdb_path):
        print("Error: GDB path does not exist.")
        return

    try:
        # List layers
        layers = fiona.listlayers(gdb_path)
        print(f"Found {len(layers)} layers:")
        for idx, layer_name in enumerate(layers):
            print(f"{idx + 1}. {layer_name}")
        
        print("-" * 40)

        # Verify ID overlap between CENTERLINE and SPEED_LIMIT
        print("\nVerifying ID Overlap:")
        centerline_ids = set()
        speed_limit_ids = set()
        centerline_fids = set()
        
        # Read CENTERLINE IDs
        try:
            travel_dirs = {}
            with fiona.open(gdb_path, layer="CENTERLINE") as src:
                for feature in src:
                    # Capture ROUTE_ID
                    rid = feature['properties'].get('ROUTE_ID')
                    if rid is not None:
                        centerline_ids.add(rid)
                    # Capture FID (the feature id from fiona)
                    centerline_fids.add(feature.id)
                    
                    # Count Travel Direction
                    td = feature['properties'].get('TRAVEL_DIRECTION')
                    travel_dirs[td] = travel_dirs.get(td, 0) + 1
                    
            print(f"  CENTERLINE: Found {len(centerline_ids)} unique ROUTE_ID(s).")
            print(f"  CENTERLINE: Found {len(centerline_fids)} unique FIDs (min: {min(map(int, centerline_fids))}, max: {max(map(int, centerline_fids))}).")
            print(f"  CENTERLINE: TRAVEL_DIRECTION distribution: {travel_dirs}")
        except Exception as e:
            print(f"  Failed to read CENTERLINE: {e}")

        # Read SPEED_LIMIT IDs
        try:
            with fiona.open(gdb_path, layer="SPEED_LIMIT") as src:
                for feature in src:
                    rid = feature['properties'].get('ROAD_ROUTE_ID')
                    if rid is not None:
                        speed_limit_ids.add(rid)
            print(f"  SPEED_LIMIT: Found {len(speed_limit_ids)} unique ROAD_ROUTE_ID(s).")
        except Exception as e:
            print(f"  Failed to read SPEED_LIMIT: {e}")
            
        # Calc Overlap
        overlap = centerline_ids.intersection(speed_limit_ids)
        print(f"  Overlap Count: {len(overlap)}")
        if len(speed_limit_ids) > 0:
            print(f"  Coverage: {len(overlap) / len(speed_limit_ids) * 100:.1f}% of Speed Limit segments map to Centerlines.")
        else:
            print("  Coverage: 0% (No Speed Limit IDs found)")
            
        # Check Turn Table FID reference
        # Turn table uses EDGE1FID which likely refers to OBJECTID/FID of CENTERLINE
        try:
            with fiona.open(gdb_path, layer="TURN") as src:
                turn_ref_fids = set()
                for feature in src:
                     # EDGE1FID is likely the reference
                     fid = feature['properties'].get('EDGE1FID')
                     if fid is not None:
                         turn_ref_fids.add(str(fid)) # fiona IDs are strings usually
                
                print(f"  TURN: Found {len(turn_ref_fids)} unique referenced FIDs.")
                fid_overlap = centerline_fids.intersection(turn_ref_fids)
                print(f"  TURN -> CENTERLINE FID Overlap: {len(fid_overlap)}")
        except Exception as e:
            print(f"  Failed to check TURN table: {e}")


    except Exception as e:
        print(f"Failed to list layers: {e}")

if __name__ == "__main__":
    inspect_gdb(GDB_PATH)
