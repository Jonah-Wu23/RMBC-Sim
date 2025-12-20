import xml.etree.ElementTree as ET
import sys

def check_connections(net_file):
    print(f"Loading {net_file}...")
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Custom checks for "fixed" connections
        check_pairs = [
            ("5272_rev", "5291"),
            ("93563", "261287_rev"),
            ("97116", "95356")
        ]
        
        print("\n--- Connection Existence Check ---")
        for from_e, to_e in check_pairs:
            found = []
            for conn in root.findall('connection'):
                if conn.get('from') == from_e and conn.get('to') == to_e:
                    found.append(f"{conn.get('fromLane')}->{conn.get('toLane')}")
            print(f"{from_e} -> {to_e}: {found}")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_connections(r"sumo/net/hk_irn_v3.net.xml")
