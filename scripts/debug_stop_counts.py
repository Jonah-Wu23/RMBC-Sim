import xml.etree.ElementTree as ET
import csv
import argparse

def check_stops(xml_file, csv_file, vehicle_pattern):
    print(f"Checking {xml_file} for vehicle pattern '{vehicle_pattern}'...")
    
    # 1. Get Start/End Stop IDs from XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        stops = []
        for stop in root.findall('stopinfo'):
            vid = stop.get('id')
            # Check if pattern is in vid (e.g. '960' in 'flow_960_inbound.0')
            if vehicle_pattern in vid:
                stops.append({
                    'id': stop.get('busStop'),
                    'time': float(stop.get('ended')),
                    'vid': vid
                })
        
        # Sort by time
        stops.sort(key=lambda x: x['time'])
        
        if not stops:
            print(f"No stops found for pattern '{vehicle_pattern}'")
            return

        # We might have mixed vehicles if pattern is generic (like '960'). 
        # Let's pick the one with the most stops or just the first specific VID found.
        # Group by VID first.
        unique_vids = set(s['vid'] for s in stops)
        target_vid = sorted(list(unique_vids))[0] # Pick first one
        print(f"Analyzing specific vehicle: {target_vid}")
        
        target_stops = [s for s in stops if s['vid'] == target_vid]
        target_stops.sort(key=lambda x: x['time'])

        start_stop_id = target_stops[0]['id']
        end_stop_id = target_stops[-1]['id']
        print(f"Start Stop ID: {start_stop_id}")
        print(f"End Stop ID:   {end_stop_id}")
        print(f"Stop Count:    {len(target_stops)}")
        
        # 2. Lookup Details in CSV
        print(f"\nLooking up details in {csv_file}...")
        start_info = None
        end_info = None
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['stop_id'] == start_stop_id:
                    start_info = row
                if row['stop_id'] == end_stop_id:
                    end_info = row
        
        # 3. Print Results
        def print_info(label, info):
            if info:
                print(f"\n{label}:")
                # print(f"  ID:        {info.get('stop_id')}")
                print(f"  Name (TC): {info.get('stop_name_tc')}")
                print(f"  Name (EN): {info.get('stop_name_en')}")
                print(f"  Lat/Long:  {info.get('lat')}, {info.get('long')}")
            else:
                print(f"\n{label}: Not found in CSV (ID might be unmapped or hex?)")

        print_info("Start Location", start_info)
        print_info("End Location (Simulation Cutoff)", end_info)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", default="flow_68X_inbound.0", help="Vehicle ID pattern to search")
    args = parser.parse_args()
    
    check_stops("sumo/output/stopinfo_exp2.xml", "data/processed/kmb_route_stop_dist.csv", args.vehicle)
