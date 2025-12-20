import xml.etree.ElementTree as ET
import os

def inspect_disconnection(edge_id_from, edge_id_to):
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    
    if not os.path.exists(NET_FILE):
        print(f"Net file {NET_FILE} not found.")
        return

    print(f"Inspecting connection: {edge_id_from} -> {edge_id_to}")
    
    context = ET.iterparse(NET_FILE, events=('end',))
    edge_from_info = None
    edge_to_info = None
    
    for event, elem in context:
        if elem.tag == 'edge':
            if elem.get('id') == edge_id_from:
                edge_from_info = {
                    'id': elem.get('id'),
                    'from': elem.get('from'),
                    'to': elem.get('to'),
                    'priority': elem.get('priority'),
                    'lanes': []
                }
                for lane in elem.findall('lane'):
                    edge_from_info['lanes'].append({
                        'id': lane.get('id'),
                        'shape': lane.get('shape')
                    })
            elif elem.get('id') == edge_id_to:
                edge_to_info = {
                    'id': elem.get('id'),
                    'from': elem.get('from'),
                    'to': elem.get('to'),
                    'priority': elem.get('priority'),
                    'lanes': []
                }
                for lane in elem.findall('lane'):
                    edge_to_info['lanes'].append({
                        'id': lane.get('id'),
                        'shape': lane.get('shape')
                    })
            elem.clear()
            
    if edge_from_info:
        print(f"\nEdge FROM ({edge_id_from}):")
        print(f"  Nodes: {edge_from_info['from']} -> {edge_from_info['to']}")
        for lane in edge_from_info['lanes']:
            print(f"  Lane {lane['id']} Shape: {lane['shape']}")
            
    if edge_to_info:
        print(f"\nEdge TO ({edge_id_to}):")
        print(f"  Nodes: {edge_to_info['from']} -> {edge_to_info['to']}")
        for lane in edge_to_info['lanes']:
            print(f"  Lane {lane['id']} Shape: {lane['shape']}")

    if edge_from_info and edge_to_info:
        print("\n--- Junction Analysis ---")
        if edge_from_info['to'] == edge_to_info['from']:
            print(f"Common Junction Node: {edge_from_info['to']}")
        else:
            print(f"MISMATCHED Junction Nodes: {edge_from_info['to']} (from-end) vs {edge_to_info['from']} (to-start)")
            print("This suggests the edges do not physically meet at the same node ID.")

if __name__ == "__main__":
    print("Inspecting new gaps...")
    inspect_disconnection('97153_rev', '96253')
    inspect_disconnection('97403', '96300')
    inspect_disconnection('115546', '4833')
    inspect_disconnection('94266', '94642')
