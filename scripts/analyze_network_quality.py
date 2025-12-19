"""
Analyze the quality of the generated SUMO network.
Calculates:
1. Total Nodes/Edges
2. Largest Strongly Connected Component (SCC) coverage.
3. Number of disconnected subgraphs.
"""

import os
import sys
import networkx as nx

# Add SUMO_HOME/tools to path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

NET_FILE = os.path.abspath(os.path.join("sumo", "net", "hk_irn.net.xml"))

def analyze_net():
    print(f"Loading network: {NET_FILE}")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        print(f"Failed to load network: {e}")
        return

    # Build Graph using NetworkX
    G = nx.DiGraph()
    
    edges = net.getEdges()
    nodes = net.getNodes()
    
    print(f"Network Stats:")
    print(f"  Total Nodes: {len(nodes)}")
    print(f"  Total Edges: {len(edges)}")
    
    for e in edges:
        from_node = e.getFromNode().getID()
        to_node = e.getToNode().getID()
        length = e.getLength()
        G.add_edge(from_node, to_node, weight=length)
        
    # Analyze Connectivity
    scc = list(nx.strongly_connected_components(G))
    wcc = list(nx.weakly_connected_components(G))
    
    largest_scc = max(scc, key=len)
    largest_scc_size = len(largest_scc)
    
    print(f"\nConnectivity Analysis:")
    print(f"  Strongly Connected Components (SCC) Count: {len(scc)}")
    print(f"  Weakly Connected Components (WCC) Count: {len(wcc)}")
    print(f"  Largest SCC Size (Nodes): {largest_scc_size}")
    print(f"  Network Coverage (SCC): {largest_scc_size / len(nodes) * 100:.2f}%")
    
    # Check if key stops are in the largest SCC
    # Only if we had stop mapping info here, but generic coverage is a good proxy.
    
    if len(wcc) > 1:
        print("\nWarning: The network is not fully connected (Weakly).")
        print(f"  Largest WCC: {len(max(wcc, key=len))} nodes.")
        
    # Check Dead ends
    dead_ends = [n for n, d in G.out_degree() if d == 0]
    print(f"\nTopology Issues:")
    print(f"  Dead-end Nodes: {len(dead_ends)}")

if __name__ == "__main__":
    analyze_net()
