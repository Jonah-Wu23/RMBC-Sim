"""
Generate SUMO routes for IRN network.
Uses the generated bus stops to define flows.
"""

import os
import xml.etree.ElementTree as ET

STOP_FILE = os.path.abspath(os.path.join("sumo", "additional", "bus_stops_irn.add.xml"))
OUTPUT_FILE = os.path.abspath(os.path.join("sumo", "routes", "baseline_irn.rou.xml"))

def generate_routes():
    print(f"Reading stops from {STOP_FILE}...")
    tree = ET.parse(STOP_FILE)
    root = tree.getroot()
    
    # Organize stops by sequence logic (assuming naming convention or external data)
    # Actually, we need the route sequence. reading kmb_route_stop_dist.csv again is better.
    import pandas as pd
    CSV_FILE = os.path.abspath(os.path.join("data", "processed", "kmb_route_stop_dist.csv"))
    df = pd.read_csv(CSV_FILE)
    
    # Filter Routes
    routes = df['route'].unique()
    
    flows = []
    
    # Common Bus VType
    vtype = """    <vType id="kmb_double_decker" length="12.0" width="2.55" maxSpeed="20.0" 
           guiShape="bus" color="1,0,0" personCapacity="137"/>"""
    
    flows.append(vtype)
    
    for route in routes:
        for bound in df[df['route'] == route]['bound'].unique():
            # Get stops in order
            subset = df[(df['route'] == route) & (df['bound'] == bound)].sort_values('seq')
            if subset.empty: continue
            
            stop_ids = subset['stop_id'].tolist()
            
            # Find the FIRST and LAST stop elements to determine From/To edges?
            # Or just let SUMO route between stops.
            # Best is to list all stops.
            
            # Create Flow
            # Baseline flow: 1 bus every 10 mins? Or specific schedule?
            # For calibration, we might want a dense flow or specific departures.
            # Let's simple flow: 10 vehicles
            
            flow_id = f"flow_{route}_{bound}"
            
            # Extract first stop edge
            first_stop_id = str(stop_ids[0])
            # Find lane from XML
            # Use XPath or simple search
            try:
                first_stop_elem = root.find(f".//busStop[@id='{first_stop_id}']")
                from_edge = first_stop_elem.attrib['lane'].split('_')[0]
                
                last_stop_id = str(stop_ids[-1])
                last_stop_elem = root.find(f".//busStop[@id='{last_stop_id}']")
                to_edge = last_stop_elem.attrib['lane'].split('_')[0]
            except Exception as e:
                print(f"Error finding edge for stops on {route}-{bound}: {e}")
                continue

            lines = []
            lines.append(f'    <flow id="{flow_id}" type="kmb_double_decker" begin="0" end="3600" number="6" from="{from_edge}" to="{to_edge}">')
            
            for sid in stop_ids:
                # 20s dwell time baseline
                lines.append(f'        <stop busStop="{sid}" duration="20"/>')
                
            lines.append('    </flow>')
            flows.extend(lines)

    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('<routes>\n')
        f.write('\n'.join(flows))
        f.write('\n</routes>')
    print("Done.")

if __name__ == "__main__":
    generate_routes()
