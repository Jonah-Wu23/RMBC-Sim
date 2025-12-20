import os
import subprocess
import sys

# Paths
SUMO_HOME = os.environ.get('SUMO_HOME', r"C:\Program Files (x86)\Eclipse\Sumo")
NET_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_irn_v3.net.xml"
OUTPUT_NET = r"d:\Documents\Bus Project\Sorce code\sumo\net\hk_cropped.net.xml"
ROUTES_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\routes\fixed_routes.rou.xml"
OUTPUT_ROUTES = r"d:\Documents\Bus Project\Sorce code\sumo\routes\fixed_routes_cropped.rou.xml"
BACKGROUND_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\routes\background_clipped.rou.xml"
OUTPUT_BACKGROUND = r"d:\Documents\Bus Project\Sorce code\sumo\routes\background_cropped.rou.xml"

# Bounding Box: [MinX, MinY, MaxX, MaxY]
# City core: From Mei Foo/Sai Ying Pun to Wan Chai/Mong Kok
# Coordinate system is local SUMO coordinates
BBOX = "20000,3000,26000,11000"

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

def crop_network():
    print("--- 1. Cropping Network ---")
    cmd = [
        os.path.join(SUMO_HOME, 'bin', 'netconvert'),
        "-s", NET_FILE,
        "--keep-edges.in-boundary", BBOX,
        "--keep-edges.components", "1", # Keep only largest connected component
        "-o", OUTPUT_NET
    ]
    run_command(cmd)

def cut_routes():
    print("\n--- 2. Cutting Routes ---")
    # Using cutRoutes.py tool from SUMO
    cut_routes_tool = os.path.join(SUMO_HOME, 'tools', 'route', 'cutRoutes.py')
    if not os.path.exists(cut_routes_tool):
        print(f"Tool not found at {cut_routes_tool}")
        return

    cmd = [
        sys.executable, cut_routes_tool,
        OUTPUT_NET,
        ROUTES_FILE
    ]
    
    print(f"Running (redirecting stdout): {' '.join(cmd)} > {OUTPUT_ROUTES}")
    with open(OUTPUT_ROUTES, 'w') as outfile:
        # cutRoutes.py arguments are strict. It seems it doesn't like -o or --output in this version.
        # It prints to stdout by default? Previous error "Valid area contains 13224 edges" suggests it prints info to stdout too?
        # If it prints info AND xml to stdout, we have a problem.
        # Let's check if there is an option --xml-output or similar? No.
        # Wait, the file content "Valid area contains 13224 edges" IS the stdout.
        # This means cutRoutes.py prints log info to stdout!
        # We need to filter it? Or maybe it writes to a file if we don't redirect?
        # The help said "Usage: cutRoutes.py [options] <new_net.xml> <routes> [<routes2> ...]"
        # It didn't list -o.
        # Let's look at the error again: "cutRoutes.py: error: unrecognized arguments: -o" ( inferred from earlier)
        # But wait, I saw "--orig-ids" error.
        
        # If I run it without redirection, where does it go?
        # If it prints log to stdout, we can't redirect trivially.
        # Looking at SUMO tools generally, they often have -o or --output-file.
        # If cutRoutes.py is old, maybe it assumes redirection but prints logs to stderr?
        # "Valid area contains 13224 edges" sounds like log.
        # Let's try to capture stdout and stderr separately, and only write valid XML lines to file?
        # Risky.
        
        # Alternative: The error "unrecognized arguments: ...fixed_routes_cropped.rou.xml" when passed as 3rd arg
        # suggests it treats extra args as input route files.
        
        # Let's try finding the right option.
        # --routes-output? --output-file?
        # Since I cannot see the full help, I will guess it might be mixed.
        # BUT, the file contained "Valid area contains...". This confirms stdout pollution.
        # I will try to filter the output.
        pass

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return

    # Filter lines
    lines = result.stdout.splitlines()
    xml_lines = [l for l in lines if l.strip().startswith('<') or len(l.strip()) == 0]
    # Header might be missing?
    # Actually, cutRoutes usually outputs a full XML.
    # The "Valid area..." line is likely the first line.
    
    # Better approach: Iterate lines, skip until first '<'
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('<'):
            start_idx = i
            break
            
    final_content = '\n'.join(lines[start_idx:])
    
    with open(OUTPUT_ROUTES, 'w') as f:
        f.write(final_content)
        
    print(f"Success: saved filtered output to {OUTPUT_ROUTES}")

def filter_background():
    print("\n--- 3. Filtering Background Traffic ---")
    # Background traffic is typically flows. We can use the same tool or filter manually.
    # Since background_clipped.rou.xml contains simple flows from/to same edge, 
    # we can just keep flows where the edge exists in the new network.
    
    # First, get set of valid edges in cropped net
    import xml.etree.ElementTree as ET
    tree = ET.parse(OUTPUT_NET)
    valid_edges = {e.get('id') for e in tree.findall('edge')}
    
    print(f"Found {len(valid_edges)} valid edges in cropped network.")
    
    bg_tree = ET.parse(BACKGROUND_FILE)
    bg_root = bg_tree.getroot()
    
    kept_count = 0
    removed_count = 0
    
    # Iterate and remove flows that don't belong to valid edges
    for flow in bg_root.findall('flow'):
        from_edge = flow.get('from')
        to_edge = flow.get('to')
        if from_edge in valid_edges and to_edge in valid_edges:
            kept_count += 1
        else:
            bg_root.remove(flow)
            removed_count += 1
            
    print(f"Kept {kept_count} flows, removed {removed_count} out-of-boundary flows.")
    bg_tree.write(OUTPUT_BACKGROUND, encoding="UTF-8", xml_declaration=True)

if __name__ == "__main__":
    crop_network()
    # Check if network was generated
    if os.path.exists(OUTPUT_NET):
        cut_routes()
        filter_background()
        print("\nCropping complete!")
    else:
        print("Failed to generate cropped network.")
