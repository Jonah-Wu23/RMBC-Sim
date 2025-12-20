import re
import os

def filter_failed_connections():
    LOG_FILE = r"d:\Documents\Bus Project\Sorce code\netconvert_v2.log"
    CON_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\net\fixed_connections.con.xml"
    OUTPUT_FILE = r"d:\Documents\Bus Project\Sorce code\sumo\net\fixed_connections_v2.con.xml"
    
    if not os.path.exists(LOG_FILE) or not os.path.exists(CON_FILE):
        print("Files not found.")
        return

    # Extract failed connections
    # Error: Could not insert connection between '106944_rev' and '2468_rev' after build.
    failed_pairs = set()
    with open(LOG_FILE, 'r') as f:
        for line in f:
            match = re.search(r"Error: Could not insert connection between '(.+?)' and '(.+?)' after build", line)
            if match:
                failed_pairs.add((match.group(1), match.group(2)))
    
    print(f"Found {len(failed_pairs)} failed connections to remove.")
    
    # Filter XML
    with open(CON_FILE, 'r') as f:
        lines = f.readlines()
        
    with open(OUTPUT_FILE, 'w') as f:
        for line in lines:
            # Check if this line defines a connection in failed_pairs
            # Line format: <connection from="A" to="B" .../>
            match = re.search(r'from="(.+?)" to="(.+?)"', line)
            if match:
                f_edge, t_edge = match.group(1), match.group(2)
                if (f_edge, t_edge) in failed_pairs:
                    print(f"Removing connection {f_edge} -> {t_edge}")
                    continue # Skip writing this line
            
            f.write(line)
            
    print(f"Written filtered connections to {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_failed_connections()
