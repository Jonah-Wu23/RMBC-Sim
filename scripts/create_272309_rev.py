#!/usr/bin/env python
"""获取 272309 详情并生成反向边定义"""
import sumolib
import xml.etree.ElementTree as ET

# 获取 272309 的详细信息
net = sumolib.net.readNet('sumo/net/hk_cropped.net.xml')
e = net.getEdge('272309')

print('272309 边详情:')
print(f'  from: {e.getFromNode().getID()}')
print(f'  to: {e.getToNode().getID()}')
print(f'  length: {e.getLength():.2f}')
print(f'  numLanes: {e.getLaneNumber()}')
print(f'  speed: {e.getSpeed():.2f}')

# 获取 shape
shape = e.getShape()
print(f'  shape points: {len(shape)}')

# 反向 shape
rev_shape = list(reversed(shape))
rev_shape_str = ' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in rev_shape)

# 生成反向边的 XML
print('\n生成 272309_rev 边定义:')
edge_xml = f'''
<edge id="272309_rev" from="{e.getToNode().getID()}" to="{e.getFromNode().getID()}" 
      priority="6" numLanes="{e.getLaneNumber()}" speed="{e.getSpeed():.2f}">
    <lane index="0" speed="{e.getSpeed():.2f}" length="{e.getLength():.2f}" 
          shape="{rev_shape_str}"/>
</edge>
'''
print(edge_xml)

# 保存到补丁文件
patch_file = 'tmp/patch_272309_rev.edg.xml'
root = ET.Element('edges')
edge_elem = ET.SubElement(root, 'edge')
edge_elem.set('id', '272309_rev')
edge_elem.set('from', e.getToNode().getID())
edge_elem.set('to', e.getFromNode().getID())
edge_elem.set('priority', '6')
edge_elem.set('numLanes', str(e.getLaneNumber()))
edge_elem.set('speed', f'{e.getSpeed():.2f}')

lane_elem = ET.SubElement(edge_elem, 'lane')
lane_elem.set('index', '0')
lane_elem.set('speed', f'{e.getSpeed():.2f}')
lane_elem.set('length', f'{e.getLength():.2f}')
lane_elem.set('shape', rev_shape_str)

tree = ET.ElementTree(root)
ET.indent(tree, space='    ')
tree.write(patch_file, encoding='utf-8', xml_declaration=True)
print(f'\n补丁文件已保存: {patch_file}')
