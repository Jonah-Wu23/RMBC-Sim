
import sys
import fiona
import os

# Set GDB path
GDB_PATH = os.path.abspath(os.path.join("data", "ATC_IRNP.gdb"))

def inspect_atc_gdb(gdb_path):
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

        # Inspect ATC_STATION_PT
        if "ATC_STATION_PT" in layers:
            print("\nInspecting ATC_STATION_PT:")
            try:
                with fiona.open(gdb_path, layer="ATC_STATION_PT") as src:
                    print(f"  Feature Count: {len(src)}")
                    print(f"  Schema: {list(src.schema['properties'].keys())}")
            except Exception as e:
                print(f"  Error reading ATC_STATION_PT: {e}")
                
        # Inspect ATC_STATION_LINE
        if "ATC_STATION_LINE" in layers:
            print("\nInspecting ATC_STATION_LINE:")
            try:
                with fiona.open(gdb_path, layer="ATC_STATION_LINE") as src:
                    print(f"  Feature Count: {len(src)}")
                    print(f"  Schema: {list(src.schema['properties'].keys())}")
            except Exception as e:
                print(f"  Error reading ATC_STATION_LINE: {e}")

        # Try to guess GISDB table names
        years = ['18', '19', '20', '21', '22', '23', '24', '25']
        for yy in years:
            tbl = f"GISDB{yy}"
            if tbl in layers:
                print(f"\n[SUCCESS] Found explicit layer: {tbl}")
            else:
                # Try opening it even if not in list
                try:
                    with fiona.open(gdb_path, layer=tbl) as src:
                         print(f"\n[HIDDEN DISCOVERY] Successfully opened hidden table: {tbl}")
                         print(f"  Row Count: {len(src)}")
                         print(f"  Schema: {list(src.schema['properties'].keys())}")
                         if len(src) > 0:
                             print(f"  Sample: {next(iter(src))['properties']}")
                except Exception:
                    pass # expected if table doesn't exist
                    
    except Exception as e:
        print(f"Failed to inspect GDB: {e}")

if __name__ == "__main__":
    inspect_atc_gdb(GDB_PATH)
