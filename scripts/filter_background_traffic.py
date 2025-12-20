import xml.etree.ElementTree as ET
import os

def filter_routes():
    EDGE_FILE = os.path.abspath(os.path.join("data", "processed", "hk_irn.edg.xml"))
    INPUT_ROU = os.path.abspath(os.path.join("sumo", "routes", "background.rou.xml"))
    OUTPUT_ROU = os.path.abspath(os.path.join("sumo", "routes", "background_clipped.rou.xml"))
    
    if not os.path.exists(EDGE_FILE):
        print(f"Edge file {EDGE_FILE} not found.")
        return
        
    print(f"Reading valid edges from {EDGE_FILE}...")
    valid_edges = set()
    context = ET.iterparse(EDGE_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'edge':
            valid_edges.add(elem.get('id'))
            elem.clear()
            
    print(f"Found {len(valid_edges)} valid edges.")
    
    print(f"Filtering routes from {INPUT_ROU}...")
    tree = ET.parse(INPUT_ROU)
    root = tree.getroot()
    
    flows_to_remove = []
    
    total_flows = 0
    kept_flows = 0
    
    for flow in root.findall('flow'):
        total_flows += 1
        from_edge = flow.get('from')
        to_edge = flow.get('to')
        
        # Check explicit edges or route string if present (though this file seems to use from/to)
        # Assuming only from/to are used for these flows based on snippet
        
        if from_edge not in valid_edges or to_edge not in valid_edges:
            flows_to_remove.append(flow)
        else:
            kept_flows += 1
            
    for flow in flows_to_remove:
        root.remove(flow)
        
    print(f"Total flows: {total_flows}")
    print(f"Removed: {len(flows_to_remove)}")
    print(f"Kept: {kept_flows}")
    
    print(f"Writing {OUTPUT_ROU}...")
    tree.write(OUTPUT_ROU, encoding='UTF-8', xml_declaration=True)

if __name__ == "__main__":
    filter_routes()
