import pandas as pd
import os
import subprocess

def generate_network(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    # Filter for Route 68X, inbound
    # In a real scenario we might want both, but for a simple baseline corridor let's take one direction or just the unique stops.
    route_data = df[df['route'] == '68X'].sort_values(by=['bound', 'seq'])
    
    if route_data.empty:
        print("Error: Route 68X not found in data.")
        return

    # Create nodes
    nod_xml = os.path.join(output_dir, "hk_baseline.nod.xml")
    with open(nod_xml, "w") as f:
        f.write('<nodes>\n')
        # We'll create nodes for each stop
        for idx, row in route_data.iterrows():
            node_id = f"stop_{row['stop_id']}_{row['seq']}"
            f.write(f'    <node id="{node_id}" x="{row["long"]}" y="{row["lat"]}" type="priority"/>\n')
        f.write('</nodes>\n')

    # Create edges
    edg_xml = os.path.join(output_dir, "hk_baseline.edg.xml")
    with open(edg_xml, "w") as f:
        f.write('<edges>\n')
        # Connect stops in sequence for each bound
        for bound in route_data['bound'].unique():
            bound_data = route_data[route_data['bound'] == bound].sort_values(by='seq')
            nodes = bound_data.apply(lambda row: f"stop_{row['stop_id']}_{row['seq']}", axis=1).tolist()
            for i in range(len(nodes) - 1):
                edge_id = f"edge_{bound}_{i}"
                f.write(f'    <edge id="{edge_id}" from="{nodes[i]}" to="{nodes[i+1]}" priority="1" numLanes="2" speed="13.89"/>\n')
        f.write('</edges>\n')

    # Run netconvert
    net_xml = os.path.join(output_dir, "hk_baseline.net.xml")
    try:
        subprocess.run([
            "netconvert", 
            "--node-files", nod_xml, 
            "--edge-files", edg_xml, 
            "--output-file", net_xml,
            "--proj.utm", "true" # Important for GPS to local coords
        ], check=True)
        print(f"Successfully generated {net_xml}")
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")

if __name__ == "__main__":
    csv_file = r"d:\Documents\Bus Project\Sorce code\data\processed\kmb_route_stop_dist.csv"
    output_folder = r"d:\Documents\Bus Project\Sorce code\sumo\net"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generate_network(csv_file, output_folder)
