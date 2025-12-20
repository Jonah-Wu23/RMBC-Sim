
import xml.etree.ElementTree as ET
import sys

NET_FILE = "sumo/net/hk_irn.net.xml"
ROUTE_FILE = "sumo/routes/background.rou.xml"

def check_edges():
    print(f"Checking {NET_FILE} for edge formats...")
    edges = set()
    try:
        # Iterparse to handle large file
        for event, elem in ET.iterparse(NET_FILE, events=("start",)):
            if elem.tag == "edge":
                eid = elem.get("id")
                # Ignore internal edges
                if eid and not eid.startswith(":"):
                    edges.add(eid)
                elem.clear()
            
            if len(edges) > 0 and len(edges) % 1000 == 0:
                 # Just to see progress if running locally, but here strictly getting set
                 pass
                 
    except Exception as e:
        print(f"Error reading net file: {e}")
        return

    print(f"Total non-internal edges found: {len(edges)}")
    sample = list(edges)[:10]
    print(f"Sample edge IDs: {sample}")

    print(f"\nChecking coverage of {ROUTE_FILE}...")
    missing_count = 0
    total_flows = 0
    try:
        tree = ET.parse(ROUTE_FILE)
        root = tree.getroot()
        for flow in root.findall("flow"):
            total_flows += 1
            fid = flow.get("from")
            if fid not in edges:
                missing_count += 1
                if missing_count < 5:
                    print(f"  Missing Edge: {fid}")
    except Exception as e:
        print(f"Error reading route file: {e}")

    print(f"Total flows: {total_flows}")
    print(f"Flows with missing edges: {missing_count}")
    
    if missing_count > 0:
        print("CRITICAL: Mismatch between Route file edges and Net file edges.")

if __name__ == "__main__":
    check_edges()
