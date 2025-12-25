"""判别脚本：检查目标边是否出现在 edgedata 文件（哪怕 sampledSeconds=0）"""
import xml.etree.ElementTree as ET
import pandas as pd
import json

mapping_df = pd.read_csv('config/calibration/link_edge_mapping.csv')
targets = set()
for link_id in [20,32,33,42,43,44]:
    edge_ids = json.loads(mapping_df[mapping_df['observation_id']==link_id]['edge_ids'].values[0])
    for e in edge_ids:
        targets.add(e)
        targets.add(e.replace('_rev',''))
        targets.add(e+'_rev' if not e.endswith('_rev') else e[:-4])

print(f"目标边数量（含变体）: {len(targets)}")

tree = ET.parse('sumo/output/ies_runs/iter01_run00/edgedata.out.xml')
root = tree.getroot()

seen = {}
for interval in root.findall('.//interval'):
    for edge in interval.findall('edge'):
        eid = edge.get('id')
        if eid in targets:
            seen[eid] = float(edge.get('sampledSeconds','0'))

print(f"出现过的目标边数量: {len(seen)}")
print(f"其中 sampledSeconds > 0 的: {sum(1 for v in seen.values() if v > 0)}")
print(f"示例: {list(seen.items())[:10]}")

# 检查 68X_outbound route 前20条边是否在 sampled_edges
print("\n--- 检查 68X_outbound route 前后段采样情况 ---")
tree2 = ET.parse('sumo/routes/fixed_routes.rou.xml')
root2 = tree2.getroot()

all_sampled_edges = set()
for interval in root.findall('.//interval'):
    for edge in interval.findall('edge'):
        eid = edge.get('id')
        if float(edge.get('sampledSeconds','0')) > 0:
            all_sampled_edges.add(eid)

for vehicle in root2.findall('.//vehicle'):
    vid = vehicle.get('id', '')
    if 'flow_68X_outbound' in vid:
        route = vehicle.find('route')
        if route is not None:
            edges = route.get('edges', '').split()
            front_20 = edges[:20]
            back_20 = edges[-20:]
            front_sampled = sum(1 for e in front_20 if e in all_sampled_edges)
            back_sampled = sum(1 for e in back_20 if e in all_sampled_edges)
            print(f"68X_outbound 总共 {len(edges)} 条边")
            print(f"  前20条边采样率: {front_sampled}/20")
            print(f"  后20条边采样率: {back_sampled}/20")
            break
