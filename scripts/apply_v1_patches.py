#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_v1_patches.py
===================
将V1 bridge edge补丁插入导出的plain文件，并调用netconvert重建网络
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "tmp"
NET_DIR = PROJECT_ROOT / "sumo" / "net"

def insert_before_closing_tag(file_path: Path, closing_tag: str, content: str):
    """在闭合标签之前插入内容"""
    text = file_path.read_text(encoding="utf-8")
    
    # 找到闭合标签的位置
    idx = text.rfind(closing_tag)
    if idx == -1:
        raise ValueError(f"找不到闭合标签 {closing_tag} in {file_path}")
    
    # 在闭合标签前插入内容
    new_text = text[:idx] + content + "\n" + text[idx:]
    file_path.write_text(new_text, encoding="utf-8")
    print(f"✅ 已插入到 {file_path.name}")

def main():
    # 1. 读取补丁内容
    edges_patch = (TMP_DIR / "v1_bridge_edges.xml").read_text(encoding="utf-8")
    conns_patch = (TMP_DIR / "v1_bridge_connections.xml").read_text(encoding="utf-8")
    
    # 移除XML注释行，只保留实际元素
    edges_content = "\n".join([
        line for line in edges_patch.split("\n")
        if line.strip() and "<edge" in line or "<!--" in line
    ])
    conns_content = "\n".join([
        line for line in conns_patch.split("\n")
        if line.strip() and "<connection" in line
    ])
    
    print("📝 Edge补丁内容:")
    print(edges_content)
    print("\n📝 Connection补丁内容:")
    print(conns_content)
    
    # 2. 插入到plain文件
    insert_before_closing_tag(TMP_DIR / "hk_plain.edg.xml", "</edges>", edges_content)
    insert_before_closing_tag(TMP_DIR / "hk_plain.con.xml", "</connections>", conns_content)
    
    # 3. 运行netconvert重建网络
    output_net = NET_DIR / "hk_irn_v3_patched_v1.net.xml"
    log_file = PROJECT_ROOT / "logs" / "netconvert_v1_patch.log"
    
    cmd = [
        "netconvert",
        "--node-files", str(TMP_DIR / "hk_plain.nod.xml"),
        "--edge-files", str(TMP_DIR / "hk_plain.edg.xml"),
        "--connection-files", str(TMP_DIR / "hk_plain.con.xml"),
        "--tllogic-files", str(TMP_DIR / "hk_plain.tll.xml"),
        "-o", str(output_net),
    ]
    
    print(f"\n🔧 运行 netconvert...")
    print(f"   输出: {output_net}")
    
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, capture_output=True, text=True)
        f.write(result.stdout)
        f.write(result.stderr)
    
    if result.returncode == 0:
        print(f"✅ 网络重建成功!")
        print(f"   日志: {log_file}")
        
        # 检查输出文件大小
        if output_net.exists():
            size_mb = output_net.stat().st_size / (1024 * 1024)
            print(f"   文件大小: {size_mb:.1f} MB")
    else:
        print(f"❌ netconvert 失败!")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
