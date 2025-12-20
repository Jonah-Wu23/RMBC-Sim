import xml.etree.ElementTree as ET
import os

def get_node_ids(pairs):
    NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn_v2.net.xml"))
    
    # Pre-scan edges to dict for speed
    edge_nodes = {} # id -> (from, to)
    context = ET.iterparse(NET_FILE, events=('end',))
    for event, elem in context:
        if elem.tag == 'edge':
            edge_nodes[elem.get('id')] = (elem.get('from'), elem.get('to'))
            elem.clear()
            
    for f_edge, t_edge in pairs:
        f_info = edge_nodes.get(f_edge)
        t_info = edge_nodes.get(t_edge)
        
        if f_info and t_info:
            print(f"PAIR: {f_edge} -> {t_edge}")
            print(f"  Gap: {f_info[1]} -> {t_info[0]}") # From's To -> To's From
            
        else:
            print(f"PAIR: {f_edge} -> {t_edge} : Edge not found")

if __name__ == "__main__":
    pairs = [
        ('106944_rev', '2468_rev')
    ]
    get_node_ids(pairs)
