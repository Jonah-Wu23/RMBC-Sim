#!/usr/bin/env python
"""
为指定的 bridge edges 添加 disallow="bus" 属性。
修改 tmp/hk_v3_plain.edg.xml 文件，然后使用 netconvert 重建网络。
"""

import re
from pathlib import Path

BRIDGES_TO_DISABLE = ['bridge_68X_GAP_7']  # 需要禁用的 bridge edges

def main():
    edg_file = Path('tmp/hk_v3_plain.edg.xml')
    
    # 读取文件
    content = edg_file.read_text(encoding='utf-8')
    
    for bridge_id in BRIDGES_TO_DISABLE:
        # 查找这条边
        pattern = rf'(<edge id="{bridge_id}"[^>]*)(/>|>)'
        match = re.search(pattern, content)
        
        if match:
            prefix = match.group(1)
            suffix = match.group(2)
            
            # 检查是否已有 disallow 属性
            if 'disallow=' in prefix:
                print(f'{bridge_id}: 已有 disallow 属性，跳过')
                continue
                
            # 添加 disallow="bus" 属性
            new_prefix = prefix + ' disallow="bus"'
            content = content.replace(match.group(0), new_prefix + suffix)
            print(f'{bridge_id}: 添加 disallow="bus" ✓')
        else:
            print(f'{bridge_id}: 未在文件中找到')
    
    # 写回文件
    edg_file.write_text(content, encoding='utf-8')
    print(f'\n已更新 {edg_file}')

if __name__ == '__main__':
    main()
