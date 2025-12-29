#!/usr/bin/env python
"""分析单向性问题：为何正向绕远"""
import sumolib

net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')

# 查看从 4541 到 105735_rev 的短路径
path, cost = net.getOptimalPath(net.getEdge('105673_rev'), net.getEdge('105735_rev'), vClass='passenger')
print('4541 -> 105735_rev 的短路径 (706m):')
print(f'  边: {[e.getID() for e in path]}')

print()

# 查看从 105735_rev 到 105673 的长路径
path2, cost2 = net.getOptimalPath(net.getEdge('105735_rev'), net.getEdge('105673'), vClass='passenger')
print('105735_rev -> 4541 的长路径 (2311m):')
print(f'  前10边: {[e.getID() for e in path2[:10]]}')
print(f'  后10边: {[e.getID() for e in path2[-10:]]}')

# 检查短路径上的边是否有反向边
print()
print('短路径边的反向边检查:')
for e in path:
    eid = e.getID()
    if eid.endswith('_rev'):
        rev_eid = eid[:-4]
    else:
        rev_eid = eid + '_rev'
    exists = net.hasEdge(rev_eid)
    print(f'  {eid} -> {rev_eid}: {"存在" if exists else "不存在"}')
