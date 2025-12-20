import re
import os

def parse_log_and_generate_connections(log_file, output_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        content = f.read()

    # Regex to find no connection warnings
    # Warning: No connection between edge '106831_rev' and edge '105817' found.
    pattern = re.compile(r"Warning: No connection between edge '(.+?)' and edge '(.+?)' found\.")
    
    matches = pattern.findall(content)
    unique_connections = sorted(list(set(matches)))
    
    print(f"Found {len(unique_connections)} unique missing connections.")
    
    with open(output_file, 'w') as f:
        f.write('<connections>\n')
        for from_edge, to_edge in unique_connections:
            # Defaulting to connecting lane 0 to lane 0. 
            # In complex scenarios, this might need refinement, but it solves basic discontinuity.
            f.write(f'    <connection from="{from_edge}" to="{to_edge}" fromLane="0" toLane="0"/>\n')
        f.write('</connections>\n')
    
    print(f"Written {len(unique_connections)} connections to {output_file}")

if __name__ == "__main__":
    LOG_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\output\simulation_clipped.log"
    OUTPUT_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\net\fixed_connections.con.xml"
    parse_log_and_generate_connections(LOG_FILE, OUTPUT_FILE)
