import xml.etree.ElementTree as ET
import os

def check_bridge_connections():
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    CONNECTIONS = [
        # Phase 1
        ('260430', 'bridge_GAP_68X_IN'),
        ('bridge_GAP_68X_IN', '97403'),
        
        ('94664_rev', 'bridge_GAP_960_OUT'),
        ('bridge_GAP_960_OUT', '12401'), # Fixed destination to node-based? No, this script checks edge connections.
        ('bridge_GAP_960_OUT', '94266'), # Old dest was 94266? Step 388 says bridge->94266 OK.
        
        ('5291', 'bridge_GAP_960_IN'),
        ('bridge_GAP_960_IN', '2731'),
        
        # Phase 2
        ('97153_rev', 'bridge_68X_GAP_2'),
        ('bridge_68X_GAP_2', '96253'),
        
        ('97403', 'bridge_68X_GAP_3'),
        ('bridge_68X_GAP_3', '96300'),
        
        ('115546', 'bridge_960_GAP_2'),
        ('bridge_960_GAP_2', '4833'),
        
        ('94266', 'bridge_960_GAP_3'),
        ('bridge_960_GAP_3', '94642'),
        
        # Phase 3
        ('96253', 'bridge_68X_GAP_4'),
        ('bridge_68X_GAP_4', '95550'),
        
        ('96300', 'bridge_68X_GAP_5'),
        ('bridge_68X_GAP_5', '142955'),
        
        ('121514_rev', 'bridge_960_GAP_4'),
        ('bridge_960_GAP_4', '95208'),
        
        ('94457', 'bridge_960_GAP_5'),
        ('bridge_960_GAP_5', '94968_rev'),
        
        # Phase 4
        ('95550', 'bridge_68X_GAP_6'),
        ('bridge_68X_GAP_6', '96273_rev'),
        
        ('142955', 'bridge_68X_GAP_7'),
        ('bridge_68X_GAP_7', '97070'),
        
        ('95208', 'bridge_960_GAP_6'),
        ('bridge_960_GAP_6', '96999'),
        
        ('96999', 'bridge_960_GAP_7'),
        ('bridge_960_GAP_7', '96344'),
        
        # Phase 5
        ('94947_rev', 'bridge_68X_GAP_8'),
        ('bridge_68X_GAP_8', '142955'),
        
        ('97070', 'bridge_68X_GAP_9'),
        ('bridge_68X_GAP_9', '97407'),
        
        ('97018', 'bridge_960_GAP_8'),
        ('bridge_960_GAP_8', '94642'),
        
        ('96344', 'bridge_960_GAP_9'),
        ('bridge_960_GAP_9', '93953_rev'),
        
        # Phase 6
        ('8000_rev', 'bridge_68X_GAP_10'),
        ('bridge_68X_GAP_10', '106606'),
        
        # Phase 7
        ('94664_rev', 'bridge_960_GAP_10'),
        ('bridge_960_GAP_10', '266384'),
        
        ('96965', 'bridge_960_GAP_11'),
        ('bridge_960_GAP_11', '97116'),
        
        # Phase 8
        ('106944_rev', 'bridge_960_GAP_12'),
        ('bridge_960_GAP_12', '2468_rev')
    ]
    
    print("Loading connections from net file...")
    found_connections = set()
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'connection':
            f = elem.get('from')
            t = elem.get('to')
            found_connections.add((f, t))
            elem.clear()
            
    print(f"Loaded {len(found_connections)} connections.")
    
    missing = []
    for f, t in CONNECTIONS:
        if (f, t) in found_connections:
            print(f"[OK] {f} -> {t}")
        else:
            print(f"[MISSING] {f} -> {t}")
            missing.append((f, t))
            
    return missing

if __name__ == "__main__":
    check_bridge_connections()
