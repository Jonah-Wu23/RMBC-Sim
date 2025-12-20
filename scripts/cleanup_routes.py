import xml.etree.ElementTree as ET
import sys
import os

def cleanup_routes(route_file, output_file, remove_ids):
    print(f"Cleaning {route_file}...")
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        count = 0
        rem_set = set(remove_ids)
        
        # Iterate over all vehicles AND flows
        for mob in root.iter():
            if mob.tag in ['vehicle', 'flow']:
                stops_to_rem = []
                for stop in mob.findall('stop'):
                    sid = stop.get('busStop')
                    if sid and sid in rem_set:
                        stops_to_rem.append(stop)
                        count += 1
                
                # Also check nested routes? <route> <stop...> </route> 
                # (Less common in generated files but possible)
                for route in mob.findall('route'):
                    for stop in route.findall('stop'):
                         sid = stop.get('busStop')
                         if sid and sid in rem_set:
                            route.remove(stop)
                            count += 1

                for s in stops_to_rem:
                    mob.remove(s)
                    
        # Also clean top-level routes if any exist with stops (SUMO standard)
        for route in root.findall('route'):
             stops_to_rem = []
             for stop in route.findall('stop'):
                 sid = stop.get('busStop')
                 if sid in rem_set:
                     stops_to_rem.append(stop)
                     count += 1
             for s in stops_to_rem:
                 route.remove(s)
        
        if count > 0:
            print(f"Removed {count} stop references.")
        else:
            print("No references found (via robust search).")
            
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    r_file = "sumo/routes/baseline_irn_calibrated.rou.xml"
    o_file = "sumo/routes/baseline_irn_calibrated_clean.rou.xml"
    
    ids = ["447C1D79A10C83E2", "47A2AA98F8E20AE2", "2B3A6C000BB085CD"]
    
    if os.path.exists(r_file):
        cleanup_routes(r_file, o_file, ids)
    else:
        # Fallback
        r_file = r"d:\Documents\Bus Project\Sorce code\sumo\routes\baseline_irn_calibrated.rou.xml"
        o_file = r"d:\Documents\Bus Project\Sorce code\sumo\routes\baseline_irn_calibrated_clean.rou.xml"
        if os.path.exists(r_file):
            cleanup_routes(r_file, o_file, ids)
        else:
            print("Route file not found.")
