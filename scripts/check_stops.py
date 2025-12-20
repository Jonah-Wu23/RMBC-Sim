import xml.etree.ElementTree as ET
import os

def check_stops_validity():
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    STOPS_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops.add.xml"))
    
    if not os.path.exists(NET_FILE) or not os.path.exists(STOPS_FILE):
        print("Files not found.")
        return

    print("Loading valid edges from net file...")
    valid_edges = set()
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'edge':
            valid_edges.add(elem.get('id'))
            elem.clear()
    print(f"Loaded {len(valid_edges)} valid edges.")
    
    print("Checking bus stops...")
    tree = ET.parse(STOPS_FILE)
    root = tree.getroot()
    
    total_stops = 0
    valid_stops = 0
    invalid_stops = 0
    
    for stop in root.findall('busStop'):
        total_stops += 1
        edge_id = stop.get('lane').split('_')[0] # lane id is edge_id_laneindex
        if edge_id in valid_edges:
            valid_stops += 1
        else:
            invalid_stops += 1
            print(f"Invalid Stop {stop.get('id')} on missing edge {edge_id}")
            
    print(f"\nTotal Stops: {total_stops}")
    print(f"Valid Stops: {valid_stops}")
    print(f"Invalid Stops: {invalid_stops}")

if __name__ == "__main__":
    check_stops_validity()
