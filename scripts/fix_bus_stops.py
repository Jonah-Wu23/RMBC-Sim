import xml.etree.ElementTree as ET
import sys
import os

def fix_bus_stops(input_file):
    print(f"Processing {input_file}...")
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        modified_count = 0
        min_length = 20.0
        
        # Hardcoded fixes for internal lanes or bad mappings
        # Map: busStopID -> newLane
        manual_fixes = {
            "AB0F79ED382081D0": "96214_rev_0" # Fix YOHO MALL I on internal lane :11698_0_0
        }

        for bus_stop in root.findall('busStop'):
            stop_id = bus_stop.get('id')
            
            # Apply manual fixes first
            if stop_id in manual_fixes:
                old_lane = bus_stop.get('lane')
                new_lane = manual_fixes[stop_id]
                bus_stop.set('lane', new_lane)
                print(f"Manual Fix {stop_id}: lane {old_lane} -> {new_lane}")
                modified_count += 1

            start_pos = float(bus_stop.get('startPos'))
            end_pos = float(bus_stop.get('endPos'))
            
            length = end_pos - start_pos
            
            if length < min_length:
                new_end_pos = start_pos + min_length
                bus_stop.set('endPos', f"{new_end_pos:.2f}")
                # print(f"Fixed length {stop_id}: {length:.2f}m -> {(new_end_pos - start_pos):.2f}m")
                modified_count += 1
                
        if modified_count > 0:
            tree.write(input_file, encoding='utf-8', xml_declaration=True)
            print(f"Successfully modified {modified_count} bus stops.")
        else:
            print("No bus stops needed fixing.")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    file_path = "sumo/additional/bus_stops.add.xml"
    if not os.path.exists(file_path):
        # Try absolute path based on user context if relative fails (just in case)
        file_path = r"d:\Documents\Bus Project\Sorce code\sumo\additional\bus_stops.add.xml"
        
    if os.path.exists(file_path):
        fix_bus_stops(file_path)
    else:
        print(f"Critical Error: Could not find {file_path}")
        sys.exit(1)
