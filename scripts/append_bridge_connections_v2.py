import xml.etree.ElementTree as ET
import os

def append_connections():
    CON_FILE = os.path.abspath(os.path.join("sumo", "net", "fixed_connections_v2.con.xml"))
    
    if not os.path.exists(CON_FILE):
        print(f"{CON_FILE} not found.")
        return

    tree = ET.parse(CON_FILE)
    root = tree.getroot()
    
    # Existing + New connections to ensure they are all present
    # We append blindly, netconvert handles dupes usually or we check.
    # To be safe, I'll just check if it's there before appending?
    # Simple set check.
    
    existing = set()
    for c in root.findall('connection'):
        existing.add((c.get('from'), c.get('to')))
        
    conns = [
        # Phase 1
        ('260430', 'bridge_GAP_68X_IN'),
        ('94664_rev', 'bridge_GAP_960_OUT'),
        ('5291', 'bridge_GAP_960_IN'),
        
        # Phase 2 (New)
        ('97153_rev', 'bridge_68X_GAP_2'),
        ('97403', 'bridge_68X_GAP_3'),
        ('115546', 'bridge_960_GAP_2'),
        ('94266', 'bridge_960_GAP_3'),
        
        # Phase 3
        ('96253', 'bridge_68X_GAP_4'),
        ('96300', 'bridge_68X_GAP_5'),
        ('121514_rev', 'bridge_960_GAP_4'),
        ('94457', 'bridge_960_GAP_5'),
        
        # Phase 4
        ('95550', 'bridge_68X_GAP_6'),
        ('142955', 'bridge_68X_GAP_7'),
        ('95208', 'bridge_960_GAP_6'),
        ('96999', 'bridge_960_GAP_7'),
        
        # Phase 5
        ('94947_rev', 'bridge_68X_GAP_8'),
        ('97070', 'bridge_68X_GAP_9'),
        ('97018', 'bridge_960_GAP_8'),
        ('96344', 'bridge_960_GAP_9'),
        
        # Phase 6
        ('8000_rev', 'bridge_68X_GAP_10'),
        
        # Phase 7
        ('94664_rev', 'bridge_960_GAP_10'),
        ('96965', 'bridge_960_GAP_11'),
        
        # Phase 8
        ('106944_rev', 'bridge_960_GAP_12')
    ]
    
    added = 0
    for f, t in conns:
        if (f, t) not in existing:
            c = ET.SubElement(root, 'connection')
            c.set('from', f)
            c.set('to', t)
            c.set('fromLane', '0')
            c.set('toLane', '0')
            added += 1
            print(f"Added connection {f} -> {t}")
        else:
            print(f"Skipping existing {f} -> {t}")
            
    tree.write(CON_FILE)
    print(f"Updated {CON_FILE} with {added} new connections.")

if __name__ == "__main__":
    append_connections()
