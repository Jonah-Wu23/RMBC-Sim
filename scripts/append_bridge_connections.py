import xml.etree.ElementTree as ET
import os

def append_connections():
    CON_FILE = os.path.abspath(os.path.join("sumo", "net", "fixed_connections_v2.con.xml"))
    
    if not os.path.exists(CON_FILE):
        print(f"{CON_FILE} not found.")
        return

    tree = ET.parse(CON_FILE)
    root = tree.getroot()
    
    # New connections to add
    new_conns = [
        {'from': '260430', 'to': 'bridge_GAP_68X_IN'},
        {'from': '94664_rev', 'to': 'bridge_GAP_960_OUT'},
        {'from': '5291', 'to': 'bridge_GAP_960_IN'}
    ]
    
    added = 0
    for conn in new_conns:
        # Check if already exists? (Unlikely given previous check)
        # Just append.
        # <connection from="..." to="..." />
        # Default lane 0 to 0 is implied if not specified? 
        # Netconvert usually handles "from lane" -> "to lane" guess if omitted, 
        # BUT for explicit connection file, we usually specify specific lanes or let it guess.
        # Let's specify fromLane="0" toLane="0" to be safe.
        
        c = ET.SubElement(root, 'connection')
        c.set('from', conn['from'])
        c.set('to', conn['to'])
        c.set('fromLane', '0')
        c.set('toLane', '0')
        added += 1
        print(f"Added connection {conn['from']} -> {conn['to']}")
        
    tree.write(CON_FILE)
    print(f"Updated {CON_FILE} with {added} connections.")

if __name__ == "__main__":
    append_connections()
