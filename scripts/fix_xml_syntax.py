#!/usr/bin/env python
"""修复 edg.xml 中的 XML 语法错误"""

with open('tmp/hk_v3_plain.edg.xml', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific broken pattern: 3.50"/ disallow="bus"> to 3.50" disallow="bus"/>
old = '3.50"/ disallow="bus">'
new = '3.50" disallow="bus"/>'
content = content.replace(old, new)

with open('tmp/hk_v3_plain.edg.xml', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed XML syntax')
